#!/usr/bin/env python3
"""
training/plot_loss.py — BushidoMythos 学習ログから loss curve を再生成する。

使い方:
    python training/plot_loss.py
    python training/plot_loss.py --log checkpoints/finance_a100_v2/train.log
    python training/plot_loss.py --log checkpoints/finance_a100_v2/train.log --out loss_curve.png
    python training/plot_loss.py --smooth 0.2  # EMA 係数を変更 (0=生データ, 1=強スムース)
    python training/plot_loss.py --light        # ライトテーマ
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"\[Phase(\d+)-(\S+?)\] step\s+(\d+)/\d+\s+loss=([\d.]+)"
    r"(?:\s+ce=([\d.]+))?(?:.*?ppl=([\d.]+))?"
)

PHASE_META = {
    1: ("Phase 1 — WikiText-103",        "#4C72B0"),
    2: ("Phase 2 — ReasoningMix",        "#55A868"),
    3: ("Phase 3 — FinanceDomain",       "#C44E52"),
    4: ("Phase 4 — TradingMethodology",  "#DD8452"),
    5: ("Phase 5 — TradingQA",           "#8172B2"),
}


def parse_log(path: str) -> dict[int, tuple[list[int], list[float]]]:
    data: dict[int, tuple[list[int], list[float]]] = {}
    with open(path) as f:
        for line in f:
            m = _LINE_RE.search(line)
            if not m:
                continue
            ph   = int(m.group(1))
            step = int(m.group(3))
            loss = float(m.group(4))
            if ph not in data:
                data[ph] = ([], [])
            data[ph][0].append(step)
            data[ph][1].append(loss)
    return data


# ---------------------------------------------------------------------------
# Smooth
# ---------------------------------------------------------------------------

def ema(values: list[float], alpha: float) -> list[float]:
    if alpha <= 0:
        return list(values)
    out, s = [], values[0]
    for v in values:
        s = alpha * v + (1.0 - alpha) * s
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(phase_data: dict, out: str, smooth: float, light: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    bg     = "#ffffff" if light else "#0f1117"
    fg     = "#111111" if light else "#ffffff"
    grid_c = "#cccccc" if light else "#333333"
    leg_bg = "#f5f5f5" if light else "#1c1f26"

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    legend_patches = []
    boundaries = []   # (step, phase_num) for vlines

    for ph in sorted(phase_data):
        steps, losses = phase_data[ph]
        if not steps:
            continue
        label, color = PHASE_META.get(ph, (f"Phase {ph}", "#aaaaaa"))

        # Clamp spike at very beginning (step 1 may have loss > 10)
        losses_c = [min(l, 9.0) for l in losses]

        ax.scatter(steps, losses_c, s=5, alpha=0.3, color=color, zorder=2)
        ax.plot(steps, ema(losses_c, smooth), color=color, linewidth=2.0, zorder=3)

        boundaries.append((steps[0], ph))
        legend_patches.append(mpatches.Patch(color=color, label=label))

    # Phase boundary lines + labels
    for bstep, ph in boundaries[1:]:
        ax.axvline(x=bstep, color=fg, linewidth=0.7, linestyle="--", alpha=0.35, zorder=1)
        ax.text(bstep + 150, ax.get_ylim()[1] * 0.97,
                f"Ph{ph}", color=fg, fontsize=7.5, alpha=0.6, va="top")

    # Final loss annotation
    last_ph = max(phase_data)
    last_steps, last_losses = phase_data[last_ph]
    if last_steps:
        lx, ly = last_steps[-1], last_losses[-1]
        ax.annotate(
            f"Final loss: {ly:.3f}",
            xy=(lx, ly),
            xytext=(-130, 35), textcoords="offset points",
            color=fg, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=PHASE_META.get(last_ph, ("", "#aaa"))[1], lw=1.4),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=leg_bg,
                      edgecolor=PHASE_META.get(last_ph, ("", "#aaa"))[1]),
        )

    # Axis styling
    ax.set_xlabel("Global Step", color=fg, fontsize=11)
    ax.set_ylabel("Training Loss", color=fg, fontsize=11)
    ax.set_title("BushidoMythos — 5-Phase Financial Training Loss",
                 color=fg, fontsize=13, pad=12)
    ax.set_ylim(bottom=0)
    ax.tick_params(colors=fg, labelsize=9)
    ax.grid(axis="y", color=grid_c, linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_c)

    ax.legend(handles=legend_patches, loc="upper right", fontsize=8.5,
              facecolor=leg_bg, edgecolor=grid_c, labelcolor=fg, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Plot training loss curve from finance_pretrain.py log")
    p.add_argument("--log",    default="checkpoints/finance_a100_v2/train.log",
                   help="Path to train.log (default: checkpoints/finance_a100_v2/train.log)")
    p.add_argument("--out",    default="loss_curve.png",
                   help="Output image path (default: loss_curve.png)")
    p.add_argument("--smooth", type=float, default=0.25,
                   help="EMA smoothing coefficient 0–1 (default: 0.25)")
    p.add_argument("--light",  action="store_true",
                   help="Use a white background instead of dark")
    args = p.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib が見つかりません。先にインストールしてください:")
        print("  pip install matplotlib")
        sys.exit(1)

    log_path = args.log
    if not Path(log_path).exists():
        print(f"ログファイルが見つかりません: {log_path}")
        sys.exit(1)

    phase_data = parse_log(log_path)
    if not phase_data:
        print("ログからステップデータを読み取れませんでした。フォーマットを確認してください。")
        sys.exit(1)

    n_records = sum(len(v[0]) for v in phase_data.values())
    print(f"読み込み完了: {n_records} レコード, フェーズ {sorted(phase_data.keys())}")

    plot(phase_data, args.out, args.smooth, args.light)


if __name__ == "__main__":
    main()
