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
from training.finance_pretrain import _CACHE_VERSION


def _load_token_ids(cache_dir, name, vocab, cap=200_000):
    """トークンキャッシュから評価用 1-D id を得る(先頭 cap 個のみ常駐)。"""
    cp = Path(cache_dir) / f"{name}_{vocab}_{_CACHE_VERSION}.pt"
    if not cp.exists():
        return None
    obj = torch.load(cp, weights_only=True)
    ids = obj["ids"] if isinstance(obj, dict) else obj
    return ids[:cap].clone()


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
    p.add_argument("--finance_eval", default="financial_news_gpt2",
                   help="金融 held-out 評価のキャッシュ名(無ければ金融評価をスキップ)")
    p.add_argument("--cache_dir", default=".cache")
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

    # 金融 held-out(学習に未使用の財務テキスト)。キャッシュが無ければスキップ。
    fin_ids = _load_token_ids(args.cache_dir, args.finance_eval, cfg.vocab_size)
    if fin_ids is None:
        print(f"  [note] 金融 held-out キャッシュ無し ({args.finance_eval}); WikiText のみ評価")

    wt32 = _eval_ppl(model, cfg, wt_ids, device, args.eval_max_chunks)
    fin32 = _eval_ppl(model, cfg, fin_ids, device, args.eval_max_chunks) if fin_ids is not None else None
    print(f"  params={nparams/1e6:.1f}M  Linear層={n_lin}  size={sz32:.0f}MB  "
          f"WikiText PPL={wt32:.2f}" + (f"  finance PPL={fin32:.2f}" if fin32 else ""))

    # ── INT8 dynamic ──────────────────────────────────────────
    print("\n=== INT8 dynamic quantization ===")
    qmodel = _quantize_dynamic_int8(model)
    sz8 = _state_dict_mb(qmodel)
    wt8 = _eval_ppl(qmodel, cfg, wt_ids, device, args.eval_max_chunks)
    fin8 = _eval_ppl(qmodel, cfg, fin_ids, device, args.eval_max_chunks) if fin_ids is not None else None
    print(f"  size={sz8:.0f}MB  WikiText PPL={wt8:.2f}" + (f"  finance PPL={fin8:.2f}" if fin8 else ""))

    # ── サマリ ───────────────────────────────────────────────
    def _pct(a, b):
        return f"{(b - a) / a * 100:+.1f}%"
    print("\n" + "=" * 64)
    print(f"  推論量子化 (INT8 dynamic) — params={nparams/1e6:.1f}M, Linear={n_lin}")
    print("=" * 64)
    hdr = f"  {'':<8}{'size MB':>10}{'WikiText↓':>12}"
    if fin32:
        hdr += f"{'finance↓':>12}"
    print(hdr)
    row32 = f"  {'fp32':<8}{sz32:>10.0f}{wt32:>12.2f}"
    row8 = f"  {'INT8':<8}{sz8:>10.0f}{wt8:>12.2f}"
    if fin32:
        row32 += f"{fin32:>12.2f}"
        row8 += f"{fin8:>12.2f}"
    print(row32); print(row8)
    print(f"  サイズ削減: {sz32 - sz8:.0f}MB ({(1 - sz8/sz32)*100:.0f}%)")
    print(f"  WikiText PPL: {wt32:.2f} → {wt8:.2f} ({_pct(wt32, wt8)})")
    if fin32:
        print(f"  finance  PPL: {fin32:.2f} → {fin8:.2f} ({_pct(fin32, fin8)})")
    print("=" * 64)
    print("  注: nn.Embedding は dynamic quant 対象外(埋め込みは fp32 のまま)。")
    print("      部分評価(max_chunks)。GPU 推論で使うなら別方式(bitsandbytes/GPTQ等)。")


if __name__ == "__main__":
    main()
