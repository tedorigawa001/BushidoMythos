#!/usr/bin/env python3
"""
推論量子化トラック — INT8 dynamic quantization の品質/サイズ影響を測る。

学習トラック(8-bit optimizer)とは別。ここは**推論**のためにモデル重みを量子化し、
PPL 劣化とモデルサイズ削減のトレードオフを見る。

PyTorch の dynamic quantization(`quantize_dynamic`)を使う:
  - nn.Linear の重みを INT8 に量子化、活性化は実行時に動的量子化。
  - 較正データ不要・CPU で動作・追加依存なし。
  - 注意: nn.Embedding は対象外(入力埋め込みは量子化されない)。MLA/MoE 等の
    Linear 投影・expert・lm_head が INT8 化される。

評価軸:
  WikiText PPL(品質、低いほど良い) と state_dict サイズ(MB)。

使い方:
    python training/exp_quantize.py --ckpt checkpoints/finance_a100_v2/phase1_final.pt \
        --eval_max_chunks 40
"""

import argparse
import io
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training.eval_perplexity import (
    load_model, load_wikitext103, _build_gpt2_tokenizer, compute_perplexity,
)


def _state_dict_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def _count_linear(model) -> int:
    return sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))


def _quantize_dynamic_int8(model):
    """nn.Linear を INT8 dynamic 量子化したモデルを返す。"""
    try:
        from torch.ao.quantization import quantize_dynamic
    except Exception:
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def _eval_ppl(model, cfg, ids, device, max_chunks):
    model.eval()
    ppl, _ = compute_perplexity(model, cfg, ids, device, seq_len=1024,
                                n_loops=8, max_chunks=max_chunks)
    return ppl


def main():
    p = argparse.ArgumentParser(description="推論 INT8 量子化の品質/サイズ比較")
    p.add_argument("--ckpt", default="checkpoints/finance_a100_v2/phase1_final.pt")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--eval_max_chunks", type=int, default=40)
    p.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = p.parse_args()

    # dynamic quantization の INT8 演算は CPU 実行
    device = torch.device("cpu")
    print("Device: cpu (INT8 dynamic quantization is CPU-only)")

    tok = _build_gpt2_tokenizer()
    wt_ids = load_wikitext103(args.split, tok, 1024)

    # ── fp32 ──────────────────────────────────────────────────
    print("\n=== fp32 (baseline) ===")
    model, cfg = load_model(args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint)
    n_lin = _count_linear(model)
    nparams = sum(p.numel() for p in model.parameters())
    sz32 = _state_dict_mb(model)
    ppl32 = _eval_ppl(model, cfg, wt_ids, device, args.eval_max_chunks)
    print(f"  params={nparams/1e6:.1f}M  Linear層={n_lin}  size={sz32:.0f}MB  WikiText PPL={ppl32:.2f}")

    # ── INT8 dynamic ──────────────────────────────────────────
    print("\n=== INT8 dynamic quantization ===")
    qmodel = _quantize_dynamic_int8(model)
    sz8 = _state_dict_mb(qmodel)
    ppl8 = _eval_ppl(qmodel, cfg, wt_ids, device, args.eval_max_chunks)
    print(f"  size={sz8:.0f}MB  WikiText PPL={ppl8:.2f}")

    # ── サマリ ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  推論量子化 (INT8 dynamic) — params={nparams/1e6:.1f}M, Linear={n_lin}")
    print("=" * 60)
    print(f"  {'':<10}{'size MB':>10}{'WikiText PPL':>14}")
    print(f"  {'fp32':<10}{sz32:>10.0f}{ppl32:>14.2f}")
    print(f"  {'INT8':<10}{sz8:>10.0f}{ppl8:>14.2f}")
    print(f"  サイズ削減: {sz32 - sz8:.0f}MB ({(1 - sz8/sz32)*100:.0f}%)")
    ppl_delta_pct = (ppl8 - ppl32) / ppl32 * 100
    print(f"  PPL 変化: {ppl32:.2f} → {ppl8:.2f} ({ppl_delta_pct:+.1f}%)")
    print("=" * 60)
    print("  注: nn.Embedding は dynamic quant 対象外(埋め込みは fp32 のまま)。")
    print("      部分評価(max_chunks)。GPU 推論で使うなら別方式(bitsandbytes/GPTQ等)。")


if __name__ == "__main__":
    main()
