#!/usr/bin/env python3
"""
量子化 module 別 ablation — どの module を INT8 化すると金融 PPL が壊れるか。

phase5(金融特化)で INT8 dynamic が finance PPL を +47% 劣化させた。
本スクリプトは module グループ(head / experts / attention / ffn_dense / router)を
個別に fp32 に残したときの finance PPL とサイズを測り、主犯を特定する。
→ 主犯だけ fp32 に残す「混合精度」でサイズ削減を保ちつつ品質を回復できるか見る。

選択的量子化: quantize_dynamic に量子化対象の module 名 set を渡す。

使い方:
    python training/exp_quantize_ablation.py --ckpt checkpoints/finance_a100_v2/phase5_final.pt \
        --eval_max_chunks 30
"""

import argparse
import io
import sys
from pathlib import Path

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training.eval_perplexity import (
    load_model, _build_gpt2_tokenizer, compute_perplexity, load_wikitext103,
)
from training.finance_pretrain import _CACHE_VERSION


# module グループ(名前で判定)
GROUPS = {
    "head":      lambda n: n == "head",
    "experts":   lambda n: ("routed_experts" in n) or ("shared_experts" in n),
    "attn":      lambda n: ".attn." in n,
    "ffn_dense": lambda n: (".ffn." in n) and ("experts" not in n) and ("router" not in n),
    "router":    lambda n: "router" in n,
}


def _state_dict_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def _quantize_names(model, names):
    """指定した名前集合の nn.Linear だけ INT8 dynamic 量子化(他は fp32)。"""
    try:
        from torch.ao.quantization import quantize_dynamic
    except Exception:
        from torch.quantization import quantize_dynamic
    if not names:
        return model
    return quantize_dynamic(model, set(names), dtype=torch.qint8)


def _eval_ppl(model, cfg, ids, device, max_chunks):
    model.eval()
    ppl, _ = compute_perplexity(model, cfg, ids, device, seq_len=1024,
                                n_loops=8, max_chunks=max_chunks)
    return ppl


def _load_token_ids(cache_dir, name, vocab, cap=200_000):
    cp = Path(cache_dir) / f"{name}_{vocab}_{_CACHE_VERSION}.pt"
    if not cp.exists():
        return None
    obj = torch.load(cp, weights_only=True)
    ids = obj["ids"] if isinstance(obj, dict) else obj
    return ids[:cap].clone()


def main():
    p = argparse.ArgumentParser(description="量子化 module 別 ablation")
    p.add_argument("--ckpt", default="checkpoints/finance_a100_v2/phase5_final.pt")
    p.add_argument("--eval_max_chunks", type=int, default=30)
    p.add_argument("--finance_eval", default="financial_news_gpt2")
    p.add_argument("--cache_dir", default=".cache")
    p.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = p.parse_args()
    if args.eval_max_chunks <= 0:
        p.error(f"--eval_max_chunks must be > 0 (got {args.eval_max_chunks})")

    device = torch.device("cpu")
    print("Device: cpu (INT8 dynamic is CPU-only)")

    model, cfg = load_model(args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint)
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    # 各グループの param サイズ(fp32 で残したときのサイズ寄与)
    group_params = {}
    for g, pred in GROUPS.items():
        group_params[g] = sum(
            sum(x.numel() for x in m.parameters())
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and pred(n)
        )

    # 金融ドメイン評価(financial_news)。Phase3+ では学習分布と重なりうる。
    fin_ids = _load_token_ids(args.cache_dir, args.finance_eval, cfg.vocab_size)
    if fin_ids is None:
        print("[error] 金融評価キャッシュが無いので ablation できません"); sys.exit(1)

    all_set = set(linear_names)

    # 評価する構成: fp32 / INT8全部 / 各グループを fp32 に残す
    configs = [("fp32 (baseline)", set())]               # 何も量子化しない
    configs.append(("INT8 (all Linear)", all_set))       # 全量子化
    for g, pred in GROUPS.items():
        keep = {n for n in linear_names if pred(n)}
        configs.append((f"INT8 except {g}", all_set - keep))

    print(f"\nLinear={len(linear_names)}  group params(M): "
          + "  ".join(f"{g}={group_params[g]/1e6:.1f}" for g in GROUPS))

    results = []
    for label, qnames in configs:
        qmodel = _quantize_names(model, qnames)  # inplace=False: model は fp32 のまま
        sz = _state_dict_mb(qmodel)
        fin = _eval_ppl(qmodel, cfg, fin_ids, device, args.eval_max_chunks)
        results.append((label, len(qnames), sz, fin))
        print(f"  {label:<22} quantized={len(qnames):>3}  size={sz:>4.0f}MB  finance PPL={fin:.2f}")
        del qmodel

    # ── サマリ ───────────────────────────────────────────────
    fp32_fin = results[0][3]
    int8_fin = results[1][3]
    print("\n" + "=" * 70)
    print("  量子化 module 別 ablation — finance PPL(低いほど良い)")
    print("=" * 70)
    print(f"  {'config':<22}{'#quant':>7}{'size MB':>9}{'finance PPL':>13}{'Δ vs fp32':>11}")
    for label, nq, sz, fin in results:
        dpct = (fin - fp32_fin) / fp32_fin * 100
        print(f"  {label:<22}{nq:>7}{sz:>9.0f}{fin:>13.2f}{dpct:>10.1f}%")
    print("=" * 70)
    print("  読み方: 『INT8 except G』で PPL が大きく回復する G ほど、量子化に弱い主犯。")
    print(f"  全量子化の劣化: {fp32_fin:.1f} → {int8_fin:.1f} ({(int8_fin-fp32_fin)/fp32_fin*100:+.1f}%)")
    print("  注: 部分評価(max_chunks)。財務ドメイン評価は Phase3+ で学習分布と重なりうる。")


if __name__ == "__main__":
    main()
