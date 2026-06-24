"""
replay_ratio スイープ
=====================
finance ドメイン学習時の記憶リプレイ(--replay_ratio)を複数値で自動的に回し、
各設定での finance PPL と WikiText PPL(標準 sliding-window)を集計する。

狙い: 「忘却(WikiText 劣化)をどこまで戻すと finance をどれだけ失うか」の
トレードオフ曲線を 1 コマンドで得る。grad_checkpoint は常時 ON(メモリ削減・厳密勾配)。

各 ratio について:
  1) finance_pretrain.py を subprocess 起動して学習(指定フェーズを STEPS step)。
  2) 生成された phase{PHASE}_final.pt を読み、compute_perplexity(sliding-window)で
     finance と WikiText の PPL を in-process 計測。
  3) 表に集計し、Markdown / CSV で保存。

実行例(Colab, GPU):
  python3 training/exp_replay_sweep.py \
    --base_ckpt checkpoints/finance_a100_v2/phase1_final.pt \
    --phase 3 --steps 3000 --ratios 0,0.05,0.1,0.2 \
    --seq_len 1024 --batch_size 4 --lr 1e-4 --stride 512 \
    --eval_max_chunks 30 --device cuda \
    --outdir checkpoints/replay_sweep --report training/report/replay_sweep.md

注意:
- replay_ratio は Phase2 以降で有効。一般言語の基盤を持つ base(例 phase1_final.pt)
  から finance フェーズ(--phase 3)を回すのが意図に合う。
- WikiText 評価は datasets のダウンロードが必要(Colab 等オンライン環境で実行)。
- 学習はフェーズあたり STEPS step を実走するため、ratio 数だけ GPU 時間がかかる。
"""

import argparse
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _phase_step_args(phase: int, steps: int) -> list:
    """対象フェーズだけ STEPS、他は 0(累積 step 設計で実 N step に揃える)。"""
    out = []
    for p in range(1, 6):
        out += [f"--phase{p}_steps", str(steps if p == phase else 0)]
    return out


def train_one(args, ratio: float, ckpt_dir: str) -> str:
    """1 つの replay_ratio で学習。生成された phaseN_final.pt のパスを返す。"""
    cmd = [
        sys.executable, "-u", "training/finance_pretrain.py",
        "--phase", str(args.phase),
        *_phase_step_args(args.phase, args.steps),
        "--base_ckpt", args.base_ckpt,
        "--ckpt_dir", ckpt_dir,
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--warmup_steps", str(args.warmup_steps),
        "--seed", str(args.seed),
        "--loop_seed", str(args.loop_seed),
        "--replay_ratio", str(ratio),
        "--grad_checkpoint",                 # メモリ削減(常時 ON)
        "--cache_dir", args.cache_dir,
        "--save_every", "100000",            # 中間保存を抑制
    ]
    if args.optim8bit:
        cmd.append("--optim8bit")
    if args.dtype:
        cmd += ["--dtype", args.dtype]

    final = os.path.join(ckpt_dir, f"phase{args.phase}_final.pt")
    print("\n" + "=" * 70)
    print(f"[train] replay_ratio={ratio}  -> {final}")
    print("RUN:", " ".join(cmd))
    print("=" * 70)
    if args.dry_run:
        return final
    if args.skip_existing and os.path.exists(final):
        print(f"[skip] 既存の {final} を再利用(--skip_existing)")
        return final
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    if not os.path.exists(final):
        raise RuntimeError(f"学習後に {final} が見つかりません。finance_pretrain の保存先を確認してください。")
    return final


def eval_ppls(ckpt: str, args):
    """finance / WikiText の PPL を標準 sliding-window で計測して返す。"""
    import torch
    from training.eval_perplexity import (
        load_model, compute_perplexity, load_wikitext103, _build_gpt2_tokenizer,
    )
    from training.exp_quantize_ablation import _load_token_ids

    dev = torch.device(args.device)
    model, cfg = load_model(ckpt, dev)

    # finance(キャッシュ・オフライン可)
    fin_ids = _load_token_ids(args.cache_dir, args.finance_eval, cfg.vocab_size)
    if fin_ids is None:
        raise SystemExit(f"finance 評価キャッシュが無い: {args.cache_dir}/{args.finance_eval}_*.pt")
    fin_ppl, _ = compute_perplexity(
        model, cfg, fin_ids, dev, seq_len=args.seq_len, stride=args.stride,
        n_loops=args.n_loops, max_chunks=args.eval_max_chunks)

    # WikiText(要ダウンロード)
    tok = _build_gpt2_tokenizer()
    wk_ids = load_wikitext103("test", tok, args.seq_len)
    wk_ppl, _ = compute_perplexity(
        model, cfg, wk_ids, dev, seq_len=args.seq_len, stride=args.stride,
        n_loops=args.n_loops, max_chunks=args.eval_max_chunks)

    del model
    return fin_ppl, wk_ppl


def render(rows, args) -> str:
    """rows: [(ratio, finance_ppl, wikitext_ppl), ...] -> Markdown 文字列。"""
    base = rows[0] if rows else None
    lines = [
        "# replay_ratio スイープ結果\n",
        f"base_ckpt={args.base_ckpt}  phase={args.phase}  steps={args.steps}  "
        f"seq_len={args.seq_len}  stride={args.stride}  n_loops={args.n_loops}  "
        f"eval_max_chunks={args.eval_max_chunks}\n",
        "| replay_ratio | finance PPL | WikiText PPL | finance Δ | WikiText Δ |",
        "|---|---|---|---|---|",
    ]
    for ratio, fin, wk in rows:
        if base and base[1] and base[2]:
            df = f"{(fin-base[1])/base[1]*100:+.1f}%"
            dw = f"{(wk-base[2])/base[2]*100:+.1f}%"
        else:
            df = dw = "—"
        lines.append(f"| {ratio} | {fin:.2f} | {wk:.2f} | {df} | {dw} |")
    lines.append("\n読み方: replay_ratio↑ で WikiText(一般言語)PPL が下がる(忘却が戻る)一方、"
                 "finance PPL は上がりやすい(置換のため実 finance 学習量が減る)。"
                 "Δ は先頭行(最小 ratio)基準。")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="replay_ratio スイープ(学習→標準PPL集計)")
    p.add_argument("--base_ckpt", default="checkpoints/finance_a100_v2/phase1_final.pt",
                   help="学習の起点。一般言語基盤を持つ ckpt(例 phase1_final.pt)推奨")
    p.add_argument("--phase", type=int, default=3, help="学習フェーズ(replay は Phase2 以降で有効)")
    p.add_argument("--steps", type=int, default=3000, help="フェーズあたりの学習ステップ数")
    p.add_argument("--ratios", default="0,0.05,0.1,0.2", help="replay_ratio のカンマ区切り")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loop_seed", type=int, default=0)
    p.add_argument("--optim8bit", action="store_true", help="8-bit AdamW(bitsandbytes/CUDA)")
    p.add_argument("--dtype", default="", help="finance_pretrain の --dtype(空=既定/auto)")
    # 評価
    p.add_argument("--stride", type=int, default=512, help="標準 sliding-window stride(既定 512)")
    p.add_argument("--n_loops", type=int, default=8)
    p.add_argument("--eval_max_chunks", type=int, default=30)
    p.add_argument("--finance_eval", default="financial_news_gpt2")
    p.add_argument("--cache_dir", default=".cache")
    p.add_argument("--device", default="cuda", help="学習/評価デバイス。fp32 PPL なので cuda 可")
    # 出力/制御
    p.add_argument("--outdir", default="checkpoints/replay_sweep", help="各 ratio の ckpt 置き場")
    p.add_argument("--report", default="training/report/replay_sweep.md", help="集計 Markdown の保存先")
    p.add_argument("--skip_existing", action="store_true", help="phaseN_final.pt があれば学習をスキップ")
    p.add_argument("--dry_run", action="store_true", help="学習/評価せず実行計画だけ表示")
    args = p.parse_args()

    ratios = [float(x) for x in args.ratios.split(",") if x.strip() != ""]
    rows = []
    for ratio in ratios:
        ckpt_dir = os.path.join(args.outdir, f"replay_{ratio}")
        final = train_one(args, ratio, ckpt_dir)
        if args.dry_run:
            print(f"[dry_run] eval 予定: {final}(finance + wikitext, stride={args.stride})")
            continue
        fin, wk = eval_ppls(final, args)
        print(f"[result] replay_ratio={ratio}  finance PPL={fin:.2f}  WikiText PPL={wk:.2f}")
        rows.append((ratio, fin, wk))

    if args.dry_run:
        print("\n[dry_run] 実行計画のみ。実走するには --dry_run を外してください。")
        return

    report = render(rows, args)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[saved] {args.report}")


if __name__ == "__main__":
    main()
