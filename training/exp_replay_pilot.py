#!/usr/bin/env python3
"""
記憶リプレイ pilot 実験 — general-language replay は忘却を抑えるか。

phase1_final（一般言語で学習済み、WikiText PPL が低い状態）から、金融 SFT
(trading_qa) を flat-LR で短く学習する。これを replay_ratio を変えて複数回行い、
学習後の WikiText-103 PPL を比較する。

  仮説: replay_ratio > 0（WikiText アンカーをインターリーブ）なら、金融特化中の
        WikiText PPL 劣化（破滅的忘却）が小さくなる。

注意:
  - ローカル pilot。flat-LR・短ステップ・部分評価(--eval_max_chunks)で「方向性」を見る。
    絶対値や効果量は本番(A100・フルフェーズ)とは異なる。
  - 公平性: 各 run は phase1_final を再ロードして同一の初期重みから開始し、
    torch シードを揃えるので、差は replay の有無のみに由来する。

使い方:
    python training/exp_replay_pilot.py --steps 400 --replay_ratios 0.0 0.2
    python training/exp_replay_pilot.py --device cpu --eval_max_chunks 40
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training.finance_pretrain import (
    run_phase, TextDataset, SFTDataset, _CACHE_VERSION,
)
from training.eval_perplexity import (
    load_model, load_wikitext103, _build_gpt2_tokenizer, compute_perplexity,
)


def _sft_from_cache(name, vocab, seq_len, bs, device, cache_dir):
    """トークンキャッシュから直接 SFTDataset を構築（HF ロードを回避）。"""
    cp = Path(cache_dir) / f"{name}_{vocab}_{_CACHE_VERSION}.pt"
    if not cp.exists():
        raise FileNotFoundError(
            f"トークンキャッシュが見つかりません: {cp}\n"
            "先に通常の学習を一度走らせてキャッシュを生成してください。")
    return SFTDataset([], vocab, seq_len, bs, device, cp)


def _text_from_cache(name, vocab, seq_len, bs, device, cache_dir):
    """トークンキャッシュから直接 TextDataset を構築（HF ロードを回避）。"""
    cp = Path(cache_dir) / f"{name}_{vocab}_{_CACHE_VERSION}.pt"
    if not cp.exists():
        raise FileNotFoundError(
            f"トークンキャッシュが見つかりません: {cp}\n"
            "先に通常の学習を一度走らせてキャッシュを生成してください。")
    return TextDataset([], vocab, seq_len, bs, device, cp)


def _run_args(replay_ratio, seq_len, batch_size):
    return argparse.Namespace(
        grad_accum_steps=1, grad_clip=1.0, log_every=100, save_every=10_000_000,
        mem_log_every=0, batch_size=batch_size, seq_len=seq_len,
        loop_schedule="off", replay_ratio=replay_ratio, loop_seed=0,
        allow_unsafe_checkpoint=True,
    )


def _eval_wikitext(model, cfg, wt_ids, device, max_chunks):
    model.eval()
    ppl, nll = compute_perplexity(model, cfg, wt_ids, device, seq_len=1024,
                                  n_loops=8, max_chunks=max_chunks)
    return ppl


def main():
    p = argparse.ArgumentParser(description="記憶リプレイ pilot 実験")
    p.add_argument("--ckpt", default="checkpoints/finance_a100_v2/phase1_final.pt")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq_len", type=int, default=256, help="学習時 seq_len")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--replay_ratios", type=float, nargs="+", default=[0.0, 0.2])
    p.add_argument("--eval_max_chunks", type=int, default=40)
    p.add_argument("--cache_dir", default=".cache")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  steps={args.steps}  lr={args.lr}  "
          f"seq_len={args.seq_len}  eval_max_chunks={args.eval_max_chunks}")

    # ── 評価用 WikiText（1回ロード）─────────────────────────────
    print("\n[Eval data] WikiText-103 validation …")
    tok = _build_gpt2_tokenizer()
    wt_ids = load_wikitext103("validation", tok, 1024)

    # ── 金融 SFT データ + リプレイ用 WikiText アンカー（キャッシュから直接構築）──
    # build_* は HF を先にロードしハングするため、トークンキャッシュから直接構築する。
    print("\n[Train data] trading_qa (finance SFT) — from token cache …")
    cfg0 = load_model(args.ckpt, device, allow_unsafe=True)[1]
    ds_finance = _sft_from_cache("trading_qa_sft", cfg0.vocab_size, args.seq_len,
                                 args.batch_size, device, args.cache_dir)
    print("\n[Replay anchor] WikiText-103 (general language) — from token cache …")
    replay_ds = _text_from_cache("wikitext103_gpt2", cfg0.vocab_size, args.seq_len,
                                 args.batch_size, device, args.cache_dir)

    # ── ベースライン: 学習前の phase1_final の WikiText PPL ──────
    base_model, base_cfg = load_model(args.ckpt, device, allow_unsafe=True)
    ppl_before = _eval_wikitext(base_model, base_cfg, wt_ids, device, args.eval_max_chunks)
    print(f"\n=== Baseline (phase1_final, before finance SFT): WikiText PPL = {ppl_before:.2f} ===")
    del base_model

    # ── 各 replay_ratio で学習 → WikiText PPL 測定 ──────────────
    results = []
    for ratio in args.replay_ratios:
        print(f"\n{'#'*60}\n# Run: replay_ratio={ratio}\n{'#'*60}")
        torch.manual_seed(0)  # 金融バッチ順を揃える（差は replay のみ）
        model, cfg = load_model(args.ckpt, device, allow_unsafe=True)
        opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        sch = LambdaLR(opt, lambda s: 1.0)  # flat LR（短時間で確実に学習させる）
        rp_args = _run_args(ratio, args.seq_len, args.batch_size)
        tmp = Path(tempfile.mkdtemp())
        run_phase("ReplayPilot", ds_finance, model, cfg, opt, sch, rp_args,
                  tmp, 0, args.steps, "pilot_final.pt", device, torch.float32,
                  replay_dataset=(replay_ds if ratio > 0 else None))
        ppl_after = _eval_wikitext(model, cfg, wt_ids, device, args.eval_max_chunks)
        delta = ppl_after - ppl_before
        results.append((ratio, ppl_after, delta))
        print(f"\n=== replay_ratio={ratio}: WikiText PPL {ppl_before:.2f} → {ppl_after:.2f} "
              f"(Δ={delta:+.2f}) ===")
        del model, opt, sch

    # ── サマリ ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Memory-replay pilot — WikiText forgetting after finance SFT")
    print("=" * 60)
    print(f"  baseline (phase1_final)      PPL = {ppl_before:.2f}")
    for ratio, ppl_after, delta in results:
        print(f"  replay_ratio={ratio:<5}          PPL = {ppl_after:.2f}  (Δ={delta:+.2f})")
    print("=" * 60)
    print("  期待: replay_ratio が大きいほど Δ(WikiText 劣化) が小さい = 忘却を抑制")
    print("  注: 部分評価(max_chunks)・短ステップの pilot。絶対値は本番と異なる。")


if __name__ == "__main__":
    main()
