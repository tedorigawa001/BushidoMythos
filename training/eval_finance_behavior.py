#!/usr/bin/env python3
"""
金融ふるまい評価 — Phase 1 vs Phase 5 を固定プロンプトで比較する。

WikiText-103 の perplexity では「金融特化が機能しているか」を判断できないため、
このスクリプトは生成出力そのものを 3 つの軸で定量化する:

  ② 固定プロンプト比較   : 同じプロンプトを各チェックポイントに与え、出力を並べて見せる。
  ③ フォーマット追従率   : ### Instruction: に対し ### Response: 的に整理された
                            （非退化・ターン境界 ### を出す）出力をどれだけ生成できるか。
  ④ リスク言及率         : 損切り / ポジションサイズ / レバレッジ / 流動性 /
                            イベントリスク / 不確実性 を出力にどれだけ含むか。

生成ロジック・チェックポイントロード・トークナイザは chat.py を再利用する
（finance_mode の ### Instruction / ### Response 整形と ### 境界停止を含む）。

使い方:
    python3 training/eval_finance_behavior.py
    python3 training/eval_finance_behavior.py \
        --ckpts checkpoints/finance_a100_v2/phase1_final.pt \
                checkpoints/finance_a100_v2/phase5_final.pt \
        --max_tokens 96 --loops 8 --out training/finance_behavior_report.md

Note:
    キーワード一致は「言及の有無」を測るものであり、推論の正しさは保証しない
    （mention != correct）。あくまで挙動の方向性を見る rough な指標。
"""

import argparse
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# chat.py の生成・ロード資産を再利用する
import chat
from chat import (
    build_tokenizer,
    generate,
    get_device,
    load_model,
)
from chat import (
    _INSTRUCT_PREFIX,
    _INSTRUCT_RESPONSE,
    _INSTRUCT_STOP,
    _FINANCE_RISK_SUFFIX,
)


# ---------------------------------------------------------------------------
# 既定プロンプト集（ユーザー指定の 4 件 + 補助）
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "high leverage risk",
    "overnight volatile position",
    "Fed rate and inflation",
    "position sizing",
    "Should I add to a losing position?",
    "Explain stop-loss placement.",
    "How do earnings announcements affect trading risk?",
    "What is liquidity risk in trading?",
]


# ---------------------------------------------------------------------------
# ④ リスク言及: 概念バケットと同義語
# ---------------------------------------------------------------------------

RISK_CONCEPTS = {
    "stop_loss": ["stop loss", "stop-loss", "stoploss", "cut the loss",
                  "cut losses", "invalidation", "invalidate", "損切"],
    "position_sizing": ["position siz", "position size", "sizing",
                        "risk per trade", "% of", "percent of",
                        "ポジションサイズ"],
    "leverage": ["leverage", "leveraged", "margin", "レバレッジ"],
    "liquidity": ["liquidity", "slippage", "spread", "thinly traded", "流動性"],
    "event_risk": ["event risk", "catalyst", "earnings", "fed", "fomc",
                   "rate decision", "economic data", "news", "イベント"],
    # 「不確実性」: 硬い語に加え、一般的なヘッジ表現も拾う
    # （acknowledge uncertainty を学習目標にしているため、ヘッジ語はこの概念の核）
    "uncertainty": ["uncertain", "uncertainty", "not guaranteed", "no guarantee",
                    "not sure", "depends", "may ", "might", "could ", "probabilit",
                    "downside", "cannot predict", "can't predict", "no one knows",
                    "verify", "不確実", "わからない"],
}

# ③ 構造化推論の proxy: 接続詞・命令・例示など「整理された応答」に現れる手がかり
STRUCTURE_CUES = [
    "because", "however", "therefore", "should", "consider", "instead",
    "for example", "in summary", "first", "second", "note that",
    "keep in mind", "if ", "on the other hand",
]


# ---------------------------------------------------------------------------
# 生成（raw 文字列も取得するため chat.generate の薄いラッパ）
# ---------------------------------------------------------------------------

def generate_with_raw(model, cfg, tokenizer, prompt, args, device):
    """trimmed（表示用）と raw（境界判定用）の両方を返す。

    chat.generate と同じ整形だが、### 境界で切る前の raw も取得する。
    """
    full_prompt = _INSTRUCT_PREFIX + prompt + _FINANCE_RISK_SUFFIX + _INSTRUCT_RESPONSE
    ids = tokenizer.encode(full_prompt)
    if not ids:
        ids = [0]

    # chat.generate と同じガード: max_new_tokens が文脈長以上なら半分に調整
    max_new = args.max_tokens
    max_prompt_len = cfg.max_seq_len - max_new
    if max_prompt_len <= 0:
        max_new = cfg.max_seq_len // 2
        max_prompt_len = cfg.max_seq_len - max_new
    if len(ids) > max_prompt_len:
        ids = ids[-max_prompt_len:]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            n_loops=args.loops,
            temperature=args.temp,
            top_k=min(args.top_k, cfg.vocab_size) if args.top_k > 0 else 0,
            repetition_penalty=args.rep_penalty,
        )
    raw = tokenizer.decode(out[0, len(ids):].tolist())

    trimmed = raw
    stop_idx = trimmed.find(_INSTRUCT_STOP)
    if stop_idx != -1:
        trimmed = trimmed[:stop_idx]
    return trimmed.strip(), raw


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------

def repetition_rate(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0
    return 1.0 - len(set(words)) / len(words)


def structure_cues_in(text: str) -> int:
    low = text.lower()
    return sum(1 for cue in STRUCTURE_CUES if cue in low)


def quality_metrics(trimmed: str, raw: str) -> dict:
    """③(フォーマット追従）と品質の proxy を返す。

    注意:
      - non_degenerate は「長さがあり繰り返しが少ない」だけの弱い品質指標で、
        ### Response 形式に整理されたかは測らない（フォーマット追従そのものではない）。
      - 真のフォーマット追従は emitted_boundary（### ターン境界を出したか）。
      - structured は接続詞/箇条書きによる「整理された推論」の粗い proxy。
    """
    words = trimmed.split()
    rep = repetition_rate(trimmed)
    n_cues = structure_cues_in(trimmed)
    has_bullets = any(m in trimmed for m in ("\n- ", "\n* ", "\n1.", "\n2.", "- ", "1. "))
    return {
        "n_words": len(words),
        "rep_rate": rep,
        # 非退化応答（品質の弱い proxy）
        "non_degenerate": len(words) >= 5 and rep < 0.5,
        # 構造化推論の proxy: 構造語が複数、または箇条書き
        "n_structure_cues": n_cues,
        "structured": n_cues >= 2 or has_bullets,
        # ③ 真のフォーマット追従: ### ターン境界を出した = 学習形式を内在化
        "emitted_boundary": _INSTRUCT_STOP.strip() in raw or raw.strip().startswith("###"),
    }


def risk_concepts_in(text: str) -> set:
    """④ 出力に含まれるリスク概念の集合。"""
    low = text.lower()
    found = set()
    for concept, kws in RISK_CONCEPTS.items():
        if any(kw in low for kw in kws):
            found.add(concept)
    return found


# ---------------------------------------------------------------------------
# 評価本体
# ---------------------------------------------------------------------------

def _seed_all(seed: int, device: torch.device) -> None:
    """device に応じてシードする。

    torch.manual_seed は CPU/CUDA をシードするが MPS の RNG には効かないため、
    MPS では torch.mps.manual_seed も呼ぶ（run 間の再現性を確保）。
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def evaluate_ckpt(ckpt_path, prompts, args, device):
    model, cfg = load_model(ckpt_path, device, allow_unsafe=args.allow_unsafe_checkpoint)
    tokenizer = build_tokenizer(cfg.vocab_size, mode=args.tokenizer)

    rows = []
    for i, prompt in enumerate(prompts):
        _seed_all(args.seed, device)  # プロンプト間・run 間で再現性を揃える
        trimmed, raw = generate_with_raw(model, cfg, tokenizer, prompt, args, device)
        fm = quality_metrics(trimmed, raw)
        concepts = risk_concepts_in(trimmed)
        rows.append({
            "prompt": prompt, "output": trimmed,
            "concepts": concepts, **fm,
        })
        print(f"  [{i+1}/{len(prompts)}] {prompt[:40]:<40} "
              f"concepts={len(concepts)} structured={fm['structured']} "
              f"nondegen={fm['non_degenerate']}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def aggregate(rows):
    n = len(rows)
    if n == 0:
        return {}
    per_concept = {c: sum(1 for r in rows if c in r["concepts"]) / n
                   for c in RISK_CONCEPTS}
    return {
        "n": n,
        "boundary_rate": sum(r["emitted_boundary"] for r in rows) / n,
        "non_degenerate_rate": sum(r["non_degenerate"] for r in rows) / n,
        "structured_rate": sum(r["structured"] for r in rows) / n,
        "avg_words": sum(r["n_words"] for r in rows) / n,
        "avg_rep_rate": sum(r["rep_rate"] for r in rows) / n,
        "avg_concepts": sum(len(r["concepts"]) for r in rows) / n,
        "per_concept": per_concept,
    }


# ---------------------------------------------------------------------------
# レポート出力
# ---------------------------------------------------------------------------

def _label(ckpt_path: str) -> str:
    return Path(ckpt_path).stem


def make_unique_labels(ckpts):
    """stem が衝突する場合は親ディレクトリ名を前置、それでも衝突すれば連番付与。"""
    stems = [Path(c).stem for c in ckpts]
    if len(set(stems)) == len(stems):
        return stems
    labels = [f"{Path(c).parent.name}/{Path(c).stem}" for c in ckpts]
    if len(set(labels)) == len(labels):
        return labels
    return [f"{lab}#{i}" for i, lab in enumerate(labels)]


def build_report(results, agg, args) -> str:
    """Markdown レポート（ブログ流用可）を組み立てる。"""
    labels = args.labels
    L = []
    L.append("# Finance behavior eval — fixed-prompt comparison\n")
    L.append(f"- prompts: {len(args.prompts_used)}  | max_tokens={args.max_tokens} "
             f"loops={args.loops} temp={args.temp} top_k={args.top_k} seed={args.seed}\n")
    L.append("> mention != correct: キーワード一致は言及の有無のみを測る rough な指標 (n が小さい点にも注意)。\n")

    # 集計表
    L.append("\n## Aggregate metrics\n")
    L.append("| metric | " + " | ".join(labels) + " |")
    L.append("|---|" + "---|" * len(labels))
    def row(name, key, pct=False, lower=False):
        vals = []
        for lab in labels:
            v = agg[lab][key]
            vals.append(f"{v*100:.0f}%" if pct else f"{v:.2f}")
        arrow = " ↓" if lower else (" ↑" if pct or key in ("avg_concepts",) else "")
        return f"| {name}{arrow} | " + " | ".join(vals) + " |"
    L.append(row("③ format adherence (### boundary)", "boundary_rate", pct=True))
    L.append(row("non-degenerate rate", "non_degenerate_rate", pct=True))
    L.append(row("structured-reasoning rate", "structured_rate", pct=True))
    L.append(row("④ avg risk concepts (0-6)", "avg_concepts"))
    L.append(row("avg words", "avg_words"))
    L.append(row("repetition rate", "avg_rep_rate", lower=True))

    # ④ 概念別カバレッジ
    L.append("\n## ④ Risk-concept coverage\n")
    L.append("| concept | " + " | ".join(labels) + " |")
    L.append("|---|" + "---|" * len(labels))
    for c in RISK_CONCEPTS:
        vals = [f"{agg[lab]['per_concept'][c]*100:.0f}%" for lab in labels]
        L.append(f"| {c} | " + " | ".join(vals) + " |")

    # ② 固定プロンプト出力
    L.append("\n## ② Fixed-prompt outputs\n")
    for i, prompt in enumerate(args.prompts_used):
        L.append(f"\n### {i+1}. `{prompt}`\n")
        for lab in labels:
            r = results[lab][i]  # noqa: index aligned with prompts_used
            tag = " ".join(sorted(r["concepts"])) or "—"
            L.append(f"**{lab}** (concepts: {tag})")
            L.append("")
            L.append("```")
            L.append(r["output"] if r["output"] else "(empty)")
            L.append("```")
    return "\n".join(L) + "\n"


def print_console_summary(results, agg, args):
    labels = args.labels
    print("\n" + "=" * 64)
    print("  Finance behavior — Phase comparison")
    print("=" * 64)
    hdr = f"  {'metric':<28}" + "".join(f"{lab:>16}" for lab in labels)
    print(hdr)
    print("-" * 64)
    def line(name, key, pct=False):
        vals = "".join(
            (f"{agg[lab][key]*100:>15.0f}%" if pct else f"{agg[lab][key]:>16.2f}")
            for lab in labels)
        print(f"  {name:<28}{vals}")
    line("③ format adherence (###)", "boundary_rate", pct=True)
    line("non-degenerate rate", "non_degenerate_rate", pct=True)
    line("structured-reasoning rate", "structured_rate", pct=True)
    line("④ avg risk concepts", "avg_concepts")
    line("avg words", "avg_words")
    line("repetition rate", "avg_rep_rate")
    print("-" * 64)
    print("  ④ per-concept coverage")
    for c in RISK_CONCEPTS:
        vals = "".join(f"{agg[lab]['per_concept'][c]*100:>15.0f}%" for lab in labels)
        print(f"    {c:<24}{vals}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="金融ふるまい評価 (Phase 比較)")
    p.add_argument("--ckpts", nargs="+", default=[
        "checkpoints/finance_a100_v2/phase1_final.pt",
        "checkpoints/finance_a100_v2/phase5_final.pt",
    ], help="比較するチェックポイント（複数可、表示はこの順）")
    p.add_argument("--prompts", nargs="+", default=None,
                   help="プロンプト集（省略時は既定 8 件）")
    p.add_argument("--tokenizer", default="auto", choices=["auto", "gpt2", "mythos"])
    p.add_argument("--max_tokens", type=int, default=96)
    p.add_argument("--loops", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--rep_penalty", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                   help="auto=自動検出。MPS はサンプリングが run 間で非再現のため、"
                        "再現可能な数値が必要なら cpu を指定（決定的・ただし低速）")
    p.add_argument("--out", default="training/report/finance_behavior_report.md",
                   help="Markdown レポート出力先（空文字で無効。親ディレクトリは自動作成）")
    p.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = p.parse_args()

    if args.temp <= 0:
        p.error("--temp は 0 より大きい値")
    if args.max_tokens <= 0:
        p.error("--max_tokens は 1 以上")
    if args.loops <= 0:
        p.error("--loops は 1 以上")
    if args.rep_penalty <= 0:
        p.error("--rep_penalty は 0 より大きい値 (1.0=無効、>1.0 で繰り返し抑制)")
    return args


def main() -> None:
    args = parse_args()
    args.prompts_used = args.prompts or DEFAULT_PROMPTS

    device = torch.device(args.device) if args.device != "auto" else get_device()
    if device.type == "mps":
        print("[note] MPS はサンプリングが run 間で非再現です。再現値が必要なら --device cpu。")
    print(f"Device: {device}")
    print(f"Prompts: {len(args.prompts_used)}  max_tokens={args.max_tokens} loops={args.loops}\n")

    # 存在するものだけに絞り、一意ラベルを割り当てる（同名 stem の衝突対策）
    for c in args.ckpts:
        if not Path(c).exists():
            print(f"[skip] not found: {c}")
    args.ckpts = [c for c in args.ckpts if Path(c).exists()]
    if not args.ckpts:
        print("評価できるチェックポイントがありません。")
        sys.exit(1)
    args.labels = make_unique_labels(args.ckpts)

    results = {}
    agg = {}
    for ckpt, label in zip(args.ckpts, args.labels):
        print(f"\n=== {label} ===")
        rows = evaluate_ckpt(ckpt, args.prompts_used, args, device)
        results[label] = rows
        agg[label] = aggregate(rows)

    print_console_summary(results, agg, args)

    if args.out:
        report = build_report(results, agg, args)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
