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

from training.eval_perplexity import load_model
from training.exp_quantize_ablation import (
    _state_dict_mb, _quantize_names, _eval_ppl, _load_token_ids,
)


# fp32 に残す候補(段階的に縮小)。MLA(kv_down/kv_up)対応。
KEEP_FP32 = {
    "all attn":              lambda n: ".attn." in n,
    "recurrent attn":        lambda n: n.startswith("recurrent.block.attn."),
    "all KV (kv_down/up)":   lambda n: (".attn.kv_down" in n or ".attn.kv_up" in n),
    "recurrent KV (minimal)":lambda n: n.startswith("recurrent.block.attn.") and
                                       (".kv_down" in n or ".kv_up" in n),
    "recurrent kv_up only":  lambda n: n == "recurrent.block.attn.kv_up",
    "recurrent kv_down only":lambda n: n == "recurrent.block.attn.kv_down",
}


def main():
    p = argparse.ArgumentParser(description="最小混合精度の確定")
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
    linear = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    all_names = set(linear)

    fin_ids = _load_token_ids(args.cache_dir, args.finance_eval, cfg.vocab_size)
    if fin_ids is None:
        print("[error] 金融評価キャッシュが無い"); sys.exit(1)

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
        kp = fp32_params(keep) / 1e6
        results.append((label, len(keep), kp, sz, fin))
        print(f"  {label:<28} fp32層={len(keep):>3} ({kp:.2f}M)  size={sz:>4.0f}MB  finance PPL={fin:.2f}")
        del qmodel

    fp32_fin = results[0][4]
    int8_fin = results[1][4]
    recover = int8_fin - fp32_fin  # 全劣化幅
    print("\n" + "=" * 78)
    print("  最小混合精度 — finance PPL(低いほど良い)。回復率 = INT8からどれだけ戻したか")
    print("=" * 78)
    print(f"  {'config':<28}{'fp32層':>7}{'fp32 M':>8}{'size':>7}{'PPL':>9}{'Δfp32':>8}{'回復率':>8}")
    for label, nk, kp, sz, fin in results:
        d = (fin - fp32_fin) / fp32_fin * 100
        rec = (int8_fin - fin) / recover * 100 if recover > 0 else 0.0
        print(f"  {label:<28}{nk:>7}{kp:>8.2f}{sz:>6.0f}M{fin:>9.2f}{d:>7.1f}%{rec:>7.0f}%")
    print("=" * 78)
    print(f"  全INT8: {fp32_fin:.1f} → {int8_fin:.1f} ({(int8_fin-fp32_fin)/fp32_fin*100:+.1f}%)")
    print("  回復率: (INT8_PPL − config_PPL) / (INT8_PPL − fp32_PPL)。100%=fp32相当まで回復。")
    print("  狙い: 最小の fp32 層数/params で高い回復率を出す構成。")


if __name__ == "__main__":
    main()
