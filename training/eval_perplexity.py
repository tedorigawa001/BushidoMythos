#!/usr/bin/env python3
"""
training/eval_perplexity.py — WikiText-103 validation/test perplexity evaluation.

学習済みチェックポイントの perplexity を WikiText-103 で計測し、
GPT-2 公開ベースラインと比較する。

使い方:
    # Phase 1 終了時点の評価（デフォルト: validation split）
    python training/eval_perplexity.py \\
        --ckpt checkpoints/finance_a100_v2/phase1_final.pt

    # テストセットで評価
    python training/eval_perplexity.py \\
        --ckpt checkpoints/finance_a100_v2/phase1_final.pt --split test

    # 全フェーズのチェックポイントを順に比較
    python training/eval_perplexity.py \\
        --ckpt_dir checkpoints/finance_a100_v2 --compare

    # stride 付きスライディングウィンドウ（より正確、より時間がかかる）
    python training/eval_perplexity.py \\
        --ckpt checkpoints/finance_a100_v2/phase1_final.pt --stride 512

Note:
    ここで計測する PPL は validation/test セットへの perplexity であり、
    学習中に記録した training loss とは異なる。
    GPT-2 ベースラインも test-set PPL で報告されているが、tokenizer・stride・
    前処理が異なる場合があるため rough reference として扱うこと（not strictly apples-to-apples）。
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import MythosConfig, BushidoMythos

# ---------------------------------------------------------------------------
# GPT-2 公開ベースライン（WikiText-103 test-set PPL）
# 出典: Radford et al. 2019 / Hugging Face model cards
# ---------------------------------------------------------------------------
GPT2_BASELINES = {
    "GPT-2 small  (117M)":  29.41,
    "GPT-2 medium (345M)":  22.76,
    "GPT-2 large  (762M)":  19.93,
    "GPT-2 XL    (1.5B)":   17.48,
}

# compare モードの表示順（学習の進行順 = phase1 → phase5）
_PHASE_ORDER = [
    "phase1_final.pt",
    "phase2_final.pt",
    "phase3_final.pt",
    "phase4_final.pt",
    "phase5_final.pt",
    "final.pt",
]

# --ckpt 未指定時の自動選択優先順位（最も学習が進んだものを優先）
_AUTO_SELECT_ORDER = [
    "phase5_final.pt",
    "phase4_final.pt",
    "phase3_final.pt",
    "phase2_final.pt",
    "phase1_final.pt",
    "final.pt",
]


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def _strip_compile_prefix(sd: dict) -> dict:
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def load_model(ckpt_path: str, device: torch.device, allow_unsafe: bool = False):
    print(f"Loading: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as safe_err:
        if not allow_unsafe:
            raise RuntimeError(
                f"weights_only=True でのロードに失敗しました: {safe_err}\n"
                "信頼できるチェックポイントの場合は --allow_unsafe_checkpoint を指定してください "
                "(weights_only=False は任意コード実行のリスクがあります)。"
            ) from safe_err
        print("  [warn] weights_only=True 失敗 → --allow_unsafe_checkpoint により "
              "weights_only=False で再ロードします。")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = MythosConfig(**ckpt["cfg"])
    model = BushidoMythos(cfg).to(device)

    sd = _strip_compile_prefix(ckpt.get("model_state", ckpt.get("model", {})))
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params={n_params:.1f}M  vocab={cfg.vocab_size:,}  "
          f"dim={cfg.dim}  loops={cfg.max_loop_iters}  step={ckpt.get('step', '?')}")
    return model, cfg


# ---------------------------------------------------------------------------
# データ準備
# ---------------------------------------------------------------------------

def load_wikitext103(split: str, tokenizer, seq_len: int) -> torch.Tensor:
    """WikiText-103 を tokenize して 1-D long tensor を返す。"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets が必要です: pip install datasets")

    print(f"Loading WikiText-103 ({split})...")
    # 新しい datasets/huggingface_hub は bare 名 "wikitext" を拒否し namespace/name を要求する。
    # namespaced("Salesforce/wikitext")を優先し、旧版向けに bare 名へフォールバックする。
    ds = None
    _errs = []
    for _repo in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(_repo, "wikitext-103-v1", split=split)
            break
        except Exception as _e:  # noqa: BLE001 — どの repo 名が通るか試す
            _errs.append(f"{_repo}: {type(_e).__name__}: {_e}")
    if ds is None:
        raise RuntimeError("WikiText-103 のロードに失敗しました:\n" + "\n".join(_errs))

    # 全テキストを結合して tokenize
    text = "\n\n".join(ex["text"] for ex in ds if ex["text"].strip())
    print(f"  Tokenizing {len(text):,} characters...")
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"  {len(ids):,} tokens")
    return torch.tensor(ids, dtype=torch.long)


def _build_gpt2_tokenizer():
    """GPT-2 tokenizer を返す（ネットワーク不要なら local_files_only=True）。"""
    try:
        from transformers import GPT2Tokenizer
        try:
            return GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
        except Exception:
            return GPT2Tokenizer.from_pretrained("gpt2")
    except ImportError:
        raise RuntimeError("transformers が必要です: pip install transformers")


# ---------------------------------------------------------------------------
# Perplexity 計算
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: BushidoMythos,
    cfg: MythosConfig,
    token_ids: torch.Tensor,
    device: torch.device,
    seq_len: int = 1024,
    stride: Optional[int] = None,
    n_loops: int = 8,
    amp_dtype: Optional[torch.dtype] = None,
    max_chunks: Optional[int] = None,
) -> tuple[float, float]:
    """WikiText-103 PPL を計算する。

    Args:
        stride:     None = 非重複チャンク（高速）。int = スライディングウィンドウ（より正確）。
        amp_dtype:  torch.autocast に使う dtype。None = autocast なし（CPU / float32）。
        max_chunks: None = 全チャンク。int = 先頭 N チャンクで打ち切り（高速な部分評価）。
                    部分評価のため絶対値は full PPL と一致しないが、相対比較には十分。

    Returns:
        (ppl, avg_nll): perplexity と平均 negative log-likelihood。
    """
    # 入力検証（stride=0 は無限ループ、seq_len/n_loops<=0 は不正）
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if n_loops <= 0:
        raise ValueError(f"n_loops must be > 0, got {n_loops}")
    if stride is not None and stride <= 0:
        raise ValueError(f"stride must be None or > 0, got {stride}")

    if stride is None:
        stride = seq_len  # 非重複

    # cfg.vocab_size より大きい token ID を clamp する。
    # 学習側と同じ処理（chat.py の _GPT2Tok.encode と同等）。
    token_ids = token_ids.clamp(max=cfg.vocab_size - 1).to(device)
    n_tokens = token_ids.shape[0]

    total_nll = 0.0
    total_counted = 0
    n_chunks = 0
    t0 = time.time()

    autocast_device = device.type if device.type == "cuda" else "cpu"
    use_amp = amp_dtype is not None and amp_dtype != torch.float32

    with torch.no_grad():
        pos = 0
        while pos < n_tokens - 1:
            end = min(pos + seq_len, n_tokens - 1)
            chunk = token_ids[pos : end + 1]  # +1 for target
            x = chunk[:-1].unsqueeze(0)  # (1, T)
            y = chunk[1:].unsqueeze(0)   # (1, T)

            if x.shape[1] == 0:
                break

            with torch.autocast(autocast_device, dtype=amp_dtype, enabled=use_amp):
                logits = model(x, n_loops=n_loops)  # (1, T, vocab)

            # stride 分だけ後ろをカウント対象にする（先頭はコンテキストのみ）。
            # 末尾の短い chunk では実長 x.shape[1] を基準にして取りこぼしを防ぐ。
            count_from = max(0, x.shape[1] - stride)
            logits_count = logits[:, count_from:, :]
            y_count = y[:, count_from:]

            if y_count.shape[1] > 0:
                nll = F.cross_entropy(
                    logits_count.reshape(-1, cfg.vocab_size),
                    y_count.reshape(-1),
                    reduction="sum",
                ).item()
                total_nll += nll
                total_counted += y_count.numel()

            pos += stride
            n_chunks += 1

            if max_chunks is not None and n_chunks >= max_chunks:
                break

            if n_chunks % 20 == 0:
                elapsed = time.time() - t0
                ppl_so_far = math.exp(total_nll / max(total_counted, 1))
                progress = pos / n_tokens * 100
                print(f"  {progress:.0f}%  chunks={n_chunks}  "
                      f"ppl={ppl_so_far:.2f}  elapsed={elapsed:.0f}s", end="\r")

    print()
    avg_nll = total_nll / max(total_counted, 1)
    ppl = math.exp(avg_nll)
    return ppl, avg_nll


# ---------------------------------------------------------------------------
# 表示
# ---------------------------------------------------------------------------

def _print_comparison(our_name: str, our_ppl: float, our_params_m: float) -> None:
    """GPT-2 ベースラインと並べて表示する。"""
    print()
    print("=" * 60)
    print("  WikiText-103 Perplexity Comparison")
    print("  (lower is better; GPT-2 baselines = test-set)")
    print("=" * 60)
    print(f"  {'Model':<30} {'Params':>8}  {'PPL':>8}")
    print("-" * 60)
    for name, ppl in GPT2_BASELINES.items():
        params = name.split("(")[1].split(")")[0] if "(" in name else "—"
        print(f"  {name:<30} {params:>8}  {ppl:>8.2f}")
    print("-" * 60)
    print(f"  {our_name:<30} {our_params_m:>6.0f}M  {our_ppl:>8.2f}  ← our model")
    print("=" * 60)
    print()
    print("Note: GPT-2 baselines are test-set PPL (Radford et al. 2019).")
    print("      Our PPL is on the validation set unless --split test is used.")
    print("      Tokenizer / stride / preprocessing may differ — treat as rough reference,")
    print("      not a strictly apples-to-apples comparison.")
    print("      Phase 1 trains on WikiText-103; Phase 2-5 may raise PPL on this benchmark.")


# ---------------------------------------------------------------------------
# compare モード（全フェーズを順に評価）
# ---------------------------------------------------------------------------

def run_compare(args: argparse.Namespace, device: torch.device,
                token_ids: torch.Tensor,
                amp_dtype: Optional[torch.dtype] = None) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    results = []
    for name in _PHASE_ORDER:
        path = ckpt_dir / name
        if not path.exists():
            continue
        model, cfg = load_model(str(path), device, allow_unsafe=args.allow_unsafe_checkpoint)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        ppl, nll = compute_perplexity(
            model, cfg, token_ids, device,
            seq_len=args.seq_len, stride=args.stride,
            n_loops=args.n_loops, amp_dtype=amp_dtype,
            max_chunks=args.max_chunks,
        )
        results.append((name.replace("_final.pt", "").replace(".pt", ""), n_params, ppl))
        print(f"  {name}: PPL = {ppl:.2f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _bar_scale = 40 / max(r[2] for r in results) if results else 1.0
    print()
    print("=" * 60)
    print(f"  Phase-by-Phase PPL (WikiText-103 {args.split})")
    print("=" * 60)
    for name, params, ppl in results:
        bar = "█" * max(1, int(ppl * _bar_scale))  # scale to max 40 chars
        print(f"  {name:<20} {ppl:6.2f}  {bar}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WikiText-103 perplexity evaluation")
    p.add_argument("--ckpt",     default=None,
                   help="評価するチェックポイントパス")
    p.add_argument("--ckpt_dir", default="checkpoints/finance_a100_v2",
                   help="--compare 時のチェックポイントディレクトリ")
    p.add_argument("--compare",  action="store_true",
                   help="全フェーズのチェックポイントを順に評価して比較")
    p.add_argument("--split",    default="validation",
                   choices=["validation", "test"],
                   help="評価する WikiText-103 のスプリット (default: validation)")
    p.add_argument("--seq_len",  type=int, default=1024,
                   help="チャンクサイズ (default: 1024)")
    p.add_argument("--stride",   type=int, default=None,
                   help="スライディングウィンドウの stride。None=非重複 (default: None)")
    p.add_argument("--n_loops",  type=int, default=8,
                   help="推論ループ数 (default: 8)")
    p.add_argument("--dtype",    default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--allow_unsafe_checkpoint", action="store_true",
                   help="weights_only=True 失敗時に weights_only=False で再ロードを許可する"
                        "（信頼できるチェックポイントのみ。任意コード実行のリスクあり）")
    p.add_argument("--max_chunks", type=int, default=None,
                   help="先頭 N チャンクで評価を打ち切る（高速な部分評価）。相対比較向け。"
                        "未指定=全チャンク")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cc = torch.cuda.get_device_properties(0).major
        if args.dtype == "auto":
            amp_dtype = torch.bfloat16 if cc >= 8 else torch.float16
        else:
            amp_dtype = {"float32": torch.float32,
                         "float16": torch.float16,
                         "bfloat16": torch.bfloat16}[args.dtype]
        print(f"Device: {torch.cuda.get_device_name(0)}  dtype={amp_dtype}")
    else:
        device = torch.device("cpu")
        amp_dtype = None  # CPU は float32 で評価（autocast なし）
        print("Device: CPU (CUDA not available — evaluation will be slow)")

    # Tokenizer & data
    tok = _build_gpt2_tokenizer()
    token_ids = load_wikitext103(args.split, tok, args.seq_len)

    if args.compare:
        run_compare(args, device, token_ids, amp_dtype=amp_dtype)
        return

    if not args.ckpt:
        # 自動選択
        ckpt_dir = Path(args.ckpt_dir)
        for name in _AUTO_SELECT_ORDER:
            if (ckpt_dir / name).exists():
                args.ckpt = str(ckpt_dir / name)
                break
        if not args.ckpt:
            print("Error: --ckpt を指定するか --ckpt_dir にチェックポイントを置いてください。")
            sys.exit(1)

    model, cfg = load_model(args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    stride_desc = f"stride={args.stride}" if args.stride else "non-overlapping"
    print(f"\nEvaluating on WikiText-103 {args.split} "
          f"(seq_len={args.seq_len}, {stride_desc}, n_loops={args.n_loops})\n")

    ppl, avg_nll = compute_perplexity(
        model, cfg, token_ids, device,
        seq_len=args.seq_len, stride=args.stride, n_loops=args.n_loops,
        amp_dtype=amp_dtype, max_chunks=args.max_chunks,
    )

    ckpt_label = Path(args.ckpt).stem
    print(f"\nResult: PPL = {ppl:.2f}  (avg NLL = {avg_nll:.4f})")

    _print_comparison(f"BushidoMythos {ckpt_label}", ppl, n_params)


if __name__ == "__main__":
    main()
