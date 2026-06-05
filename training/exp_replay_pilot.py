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
  - 公平性: 各 run は phase1_final を再ロードして同一の初期重みから開始し、torch
    シードも揃える。つまり「同一初期重み・同一シードで replay_ratio だけを変える」比較。
    ただし replay 有効時は金融SFTバッチの一部が WikiText に置き換わるため、モデルが
    見る金融バッチ列も変化する（「replay 以外は完全に同一」ではない点に注意）。

評価2軸:
    WikiText PPL  = 忘却（汎用言語の劣化、低いほど良い）
    finance PPL   = ドメイン学習（held-out 金融テキスト、低いほど良い）

2つの実験:
    [1] replay_ratio スイープ（置換モード）:
        python training/exp_replay_pilot.py --replay_ratios 0.0 0.05 0.1 0.2
    [2] 「置換」vs「追加」比較（金融ステップを一定に保つ）:
        python training/exp_replay_pilot.py --replay_ratios 0.2 --keep_finance_steps
        → これを [1] の置換 0.2 と比べ、forgetting 抑制が金融学習の犠牲かを切り分ける。
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

# pilot は in-memory で評価するので、run_phase のチェックポイント保存(~1.1GB/run)は不要。
# 無効化してディスク枯渇を防ぐ（run_phase は module グローバルの save_checkpoint を呼ぶ）。
import training.finance_pretrain as _fp
_fp.save_checkpoint = lambda *a, **k: None


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


def _eval_ppl(model, cfg, ids, device, max_chunks):
    model.eval()
    ppl, _ = compute_perplexity(model, cfg, ids, device, seq_len=1024,
                                n_loops=8, max_chunks=max_chunks)
    return ppl


def _load_token_ids(cache_dir, name, vocab, cap=200_000):
    """トークンキャッシュから評価用 1-D id テンソルを得る（先頭 cap 個のみ常駐）。

    巨大キャッシュ（financial_news ~1.5GB）の全常駐を避けるため先頭を切り出す。
    部分評価(max_chunks)には十分。
    """
    cp = Path(cache_dir) / f"{name}_{vocab}_{_CACHE_VERSION}.pt"
    if not cp.exists():
        raise FileNotFoundError(f"トークンキャッシュなし: {cp}")
    obj = torch.load(cp, weights_only=True)
    ids = obj["ids"] if isinstance(obj, dict) else obj  # SFT は dict, Text は tensor
    return ids[:cap].clone()  # clone で親テンソルを解放


def main():
    p = argparse.ArgumentParser(description="記憶リプレイ pilot 実験")
    p.add_argument("--ckpt", default="checkpoints/finance_a100_v2/phase1_final.pt")
    p.add_argument("--steps", type=int, default=400, help="金融学習ステップ（置換時は総ステップ）")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq_len", type=int, default=256, help="学習時 seq_len")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--replay_ratios", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--eval_max_chunks", type=int, default=40)
    p.add_argument("--finance_eval", default="financial_news_gpt2",
                   help="金融 held-out 評価のキャッシュ名（学習に使っていない金融テキスト）")
    p.add_argument("--keep_finance_steps", action="store_true",
                   help="『追加』モード: 総ステップを 1/(1-ratio) 倍し、金融ステップ数を一定に保つ。"
                        "（既定=『置換』モード: 総ステップ固定で金融ステップが ratio 分減る）")
    p.add_argument("--cache_dir", default=".cache")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    mode = "ADD(金融ステップ一定)" if args.keep_finance_steps else "REPLACE(総ステップ固定)"
    print(f"Device: {device}  finance_steps≈{args.steps}  lr={args.lr}  "
          f"seq_len={args.seq_len}  eval_max_chunks={args.eval_max_chunks}  mode={mode}")

    # ── 評価データ（WikiText=忘却, 金融 held-out=学習）──────────
    print("\n[Eval data] WikiText-103 validation …")
    tok = _build_gpt2_tokenizer()
    wt_ids = load_wikitext103("validation", tok, 1024)

    cfg0 = load_model(args.ckpt, device, allow_unsafe=True)[1]
    print(f"\n[Eval data] finance held-out ({args.finance_eval}) …")
    fin_ids = _load_token_ids(args.cache_dir, args.finance_eval, cfg0.vocab_size)

    # ── 学習データ（金融 SFT）+ リプレイ用 WikiText アンカー ─────
    print("\n[Train data] trading_qa (finance SFT) — from token cache …")
    ds_finance = _sft_from_cache("trading_qa_sft", cfg0.vocab_size, args.seq_len,
                                 args.batch_size, device, args.cache_dir)
    print("\n[Replay anchor] WikiText-103 — from token cache …")
    replay_ds = _text_from_cache("wikitext103_gpt2", cfg0.vocab_size, args.seq_len,
                                 args.batch_size, device, args.cache_dir)

    # ── ベースライン（学習前）──────────────────────────────────
    base_model, base_cfg = load_model(args.ckpt, device, allow_unsafe=True)
    wt0 = _eval_ppl(base_model, base_cfg, wt_ids, device, args.eval_max_chunks)
    fin0 = _eval_ppl(base_model, base_cfg, fin_ids, device, args.eval_max_chunks)
    print(f"\n=== Baseline (phase1_final): WikiText PPL={wt0:.2f}  finance PPL={fin0:.2f} ===")
    del base_model

    # ── 各 replay_ratio で学習 → WikiText / finance PPL 測定 ─────
    results = []
    for ratio in args.replay_ratios:
        total = round(args.steps / (1.0 - ratio)) if (args.keep_finance_steps and ratio < 1.0) else args.steps
        fin_steps = round(total * (1.0 - ratio))
        print(f"\n{'#'*60}\n# Run: replay_ratio={ratio}  total_steps={total}  "
              f"≈finance_steps={fin_steps}  ({mode})\n{'#'*60}")
        torch.manual_seed(0)  # 金融バッチ順を揃える（同一初期重み・同一seedで ratio だけ変える）
        model, cfg = load_model(args.ckpt, device, allow_unsafe=True)
        opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        sch = LambdaLR(opt, lambda s: 1.0)  # flat LR
        rp_args = _run_args(ratio, args.seq_len, args.batch_size)
        tmp = Path(tempfile.mkdtemp())
        run_phase("ReplayPilot", ds_finance, model, cfg, opt, sch, rp_args,
                  tmp, 0, total, "pilot_final.pt", device, torch.float32,
                  replay_dataset=(replay_ds if ratio > 0 else None))
        wt = _eval_ppl(model, cfg, wt_ids, device, args.eval_max_chunks)
        fin = _eval_ppl(model, cfg, fin_ids, device, args.eval_max_chunks)
        results.append((ratio, total, fin_steps, wt, fin))
        print(f"\n=== ratio={ratio}: WikiText {wt0:.1f}→{wt:.1f}  finance {fin0:.1f}→{fin:.1f} ===")
        del model, opt, sch

    # ── サマリ ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  Memory-replay sweep — mode={mode}")
    print("  WikiText PPL = 忘却(低いほど良い)  /  finance PPL = ドメイン学習(低いほど良い)")
    print("=" * 72)
    print(f"  {'ratio':>6} {'total':>6} {'fin_steps':>9} {'WikiText↓':>10} {'finance↓':>9}")
    print(f"  {'base':>6} {'-':>6} {'-':>9} {wt0:>10.1f} {fin0:>9.1f}")
    for ratio, total, fin_steps, wt, fin in results:
        print(f"  {ratio:>6} {total:>6} {fin_steps:>9} {wt:>10.1f} {fin:>9.1f}")
    print("=" * 72)
    print("  trade-off: replay↑ で WikiText(忘却)↓ だが finance(学習)↑ になりうる。")
    print("  ADD モードなら金融ステップを一定に保つので、その犠牲を切り分けられる。")
    print("  注: 部分評価(max_chunks)・短ステップの pilot。絶対値は本番と異なる。")


if __name__ == "__main__":
    main()
