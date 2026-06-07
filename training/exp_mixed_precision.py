#!/usr/bin/env python3
"""
最小混合精度の確定 — どれだけ少ない fp32 で INT8 劣化を回復できるか。

ablation で「MLA の KV 潜在 × recurrent loop 増幅」が INT8 劣化の正体と判明した。
ここでは fp32 に残す集合を段階的に絞り、**最小の fp32 集合で最大の回復**を探す。

各構成は「指定 module だけ fp32、残りは全 INT8」。finance PPL とサイズ・fp32 param で比較。

使い方:
    python training/exp_mixed_precision.py --ckpt checkpoints/finance_a100_v2/phase5_final.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training.eval_perplexity import (
    load_model, load_wikitext103, _build_gpt2_tokenizer,
)
from training.exp_quantize_ablation import (
    _state_dict_mb, _quantize_names, _eval_ppl, _load_token_ids,
)


# fp32 に残す候補(要点に絞った最小化系列)。MLA(kv_down/kv_up)対応。
KEEP_FP32 = {
    "all attn":              lambda n: ".attn." in n,
    "recurrent KV (2層)":    lambda n: n.startswith("recurrent.block.attn.") and
                                       (".kv_down" in n or ".kv_up" in n),
    "recurrent kv_down only":lambda n: n == "recurrent.block.attn.kv_down",
    "recurrent kv_up only":  lambda n: n == "recurrent.block.attn.kv_up",
}


def main():
    p = argparse.ArgumentParser(description="最小混合精度の確定")
    p.add_argument("--ckpt", default="checkpoints/finance_a100_v2/phase5_final.pt")
    p.add_argument("--eval_max_chunks", type=int, default=30)
    p.add_argument("--finance_eval", default="financial_news_gpt2")
    p.add_argument("--cache_dir", default=".cache")
    p.add_argument("--also_wikitext", action="store_true",
                   help="WikiText(汎用)PPL も測り、回復が finance 特有でないことを確認する")
    p.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = p.parse_args()
    if args.eval_max_chunks <= 0:
        p.error(f"--eval_max_chunks must be > 0 (got {args.eval_max_chunks})")

    device = torch.device("cpu")
    print("Device: cpu (INT8 dynamic is CPU-only)")
    model, cfg = load_model(args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint)
    linear = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    all_names = set(linear)

    fin_ids = _load_token_ids(args.cache_dir, args.finance_eval, cfg.vocab_size)
    if fin_ids is None:
        print("[error] 金融評価キャッシュが無い"); sys.exit(1)

    wt_ids = None
    if args.also_wikitext:
        print("\n[Eval data] WikiText-103 validation(汎用)…")
        wt_ids = load_wikitext103("validation", _build_gpt2_tokenizer(), 1024)

    def fp32_params(keep_names):
        return sum(sum(x.numel() for x in linear[n].parameters()) for n in keep_names)

    # 構成: (label, fp32に残す名前集合)
    configs = [("fp32 (baseline)", all_names), ("INT8 (all)", set())]
    for label, pred in KEEP_FP32.items():
        keep = {n for n in all_names if pred(n)}
        if not keep:
            print(f"  [warn] '{label}' matched 0 Linear modules"); continue
        configs.append((f"keep {label} fp32", keep))

    results = []
    for label, keep in configs:
        qnames = all_names - keep                 # keep 以外を量子化
        qmodel = _quantize_names(model, qnames)
        sz = _state_dict_mb(qmodel)
        fin = _eval_ppl(qmodel, cfg, fin_ids, device, args.eval_max_chunks)
        wt = _eval_ppl(qmodel, cfg, wt_ids, device, args.eval_max_chunks) if wt_ids is not None else None
        kp = fp32_params(keep) / 1e6
        results.append((label, len(keep), kp, sz, fin, wt))
        msg = f"  {label:<28} fp32層={len(keep):>3} ({kp:.2f}M)  size={sz:>4.0f}MB  finance={fin:.2f}"
        if wt is not None:
            msg += f"  WikiText={wt:.2f}"
        print(msg)
        del qmodel

    fp32_fin, int8_fin = results[0][4], results[1][4]
    rec_fin = int8_fin - fp32_fin
    fp32_wt = results[0][5]; int8_wt = results[1][5]
    rec_wt = (int8_wt - fp32_wt) if wt_ids is not None else None

    print("\n" + "=" * 86)
    print("  最小混合精度 — PPL(低いほど良い)/ 回復率 = INT8 からどれだけ fp32 側へ戻したか")
    print("=" * 86)
    head = f"  {'config':<28}{'fp32 M':>8}{'finance':>9}{'回復':>6}"
    if wt_ids is not None:
        head += f"{'WikiText':>10}{'回復':>6}"
    print(head)
    for label, nk, kp, sz, fin, wt in results:
        fr = (int8_fin - fin) / rec_fin * 100 if rec_fin > 0 else 0.0
        line = f"  {label:<28}{kp:>8.2f}{fin:>9.2f}{fr:>5.0f}%"
        if wt is not None:
            wr = (int8_wt - wt) / rec_wt * 100 if rec_wt and rec_wt > 0 else 0.0
            line += f"{wt:>10.2f}{wr:>5.0f}%"
        print(line)
    print("=" * 86)
    print(f"  全INT8 finance: {fp32_fin:.1f}→{int8_fin:.1f} ({(int8_fin-fp32_fin)/fp32_fin*100:+.1f}%)"
          + (f" / WikiText: {fp32_wt:.1f}→{int8_wt:.1f} ({(int8_wt-fp32_wt)/fp32_wt*100:+.1f}%)" if wt_ids is not None else ""))
    print("  狙い: 最小の fp32 params で高い回復率。finance/WikiText 両方で効けば層特有(ドメイン非依存)。")
    print("  注: 部分評価(max_chunks)・n=1。state_dict サイズは実行時メモリ/速度とは別。")


if __name__ == "__main__":
    main()
