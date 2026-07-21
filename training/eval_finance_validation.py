#!/usr/bin/env python3
"""Teacher-forced Finance QA validation and prompt-binding checkpoint selection."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from chat import get_device, load_model
from training.eval_finance_qa import make_labels, resolve_dtype
from training.finance_pretrain import _fit_sft_example, _get_gpt2_tokenizer, _tokenize_sft


DEFAULT_VALIDATION = Path(__file__).parent / "eval_data/finance_qa_curated_v2_validation.json"


def resolve_device(requested: str) -> torch.device:
    return get_device() if requested == "auto" else torch.device(requested)


def load_validation(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    examples = payload.get("examples")
    if payload.get("split") != "validation" or not isinstance(examples, list) or not examples:
        raise ValueError("validation corpus must contain a non-empty validation split")
    seen = set()
    families = defaultdict(list)
    for example in examples:
        example_id = str(example.get("id") or "").strip()
        family = str(example.get("scenario_family") or "").strip()
        if not example_id or example_id in seen:
            raise ValueError(f"validation id must be present and unique: {example_id!r}")
        if not family or not example.get("instruction") or not example.get("response"):
            raise ValueError(f"{example_id}: family, instruction, and response are required")
        seen.add(example_id)
        families[family].append(example)
    too_small = [family for family, rows in families.items() if len(rows) < 2]
    if too_small:
        raise ValueError(f"binding evaluation needs at least two rows per family: {too_small}")
    return payload, hashlib.sha256(raw).hexdigest()


def build_binding_rows(examples: list[dict]) -> list[dict]:
    """Pair each prompt with a different response from the same scenario family."""
    families = defaultdict(list)
    for example in examples:
        families[example["scenario_family"]].append(example)
    rows = []
    for family in sorted(families):
        selected = sorted(families[family], key=lambda item: item["id"])
        for index, example in enumerate(selected):
            distractor = selected[(index + 1) % len(selected)]
            rows.append({
                "id": example["id"],
                "category": example["category"],
                "scenario_family": family,
                "is_calculation": "calculation" in example,
                "instruction": example["instruction"],
                "context": example.get("context", ""),
                "correct_response": example["response"],
                "distractor_id": distractor["id"],
                "distractor_response": distractor["response"],
            })
    return rows


def _tokenize_rows(rows: list[dict], tokenizer, vocab_size: int, seq_len: int) -> dict:
    ids_rows = []
    mask_rows = []
    metadata = []
    for row in rows:
        for response_kind, response in (
            ("correct", row["correct_response"]),
            ("distractor", row["distractor_response"]),
        ):
            ids, mask = _tokenize_sft(
                tokenizer, row["instruction"], response, row["context"], vocab_size
            )
            fitted = _fit_sft_example(ids, mask, seq_len + 1)
            if fitted is None:
                raise ValueError(f"validation example cannot be tokenized: {row['id']}")
            fitted_ids, fitted_mask = fitted
            ids_rows.append(fitted_ids)
            mask_rows.append(fitted_mask)
            metadata.append({
                "id": row["id"],
                "category": row["category"],
                "scenario_family": row["scenario_family"],
                "is_calculation": row["is_calculation"],
                "response_kind": response_kind,
                "distractor_id": row["distractor_id"],
            })
    return {
        "ids": torch.tensor(ids_rows, dtype=torch.long),
        "mask": torch.tensor(mask_rows, dtype=torch.bool),
        "metadata": metadata,
    }


def _evaluate_tokenized(model, tokenized: dict, args, device, dtype) -> list[dict]:
    ids = tokenized["ids"]
    masks = tokenized["mask"]
    output = []
    use_amp = device.type == "cuda" and dtype != torch.float32
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.inference_mode():
        for start in range(0, len(ids), args.batch_size):
            end = min(start + args.batch_size, len(ids))
            batch = ids[start:end].to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            loss_mask = masks[start:end, 1:].to(device)
            with torch.autocast(autocast_device, dtype=dtype, enabled=use_amp):
                logits = model(x, n_loops=args.loops)
                per_token = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="none",
                ).reshape_as(y)
            predictions = logits.argmax(dim=-1)
            token_counts = loss_mask.sum(dim=1)
            loss_sums = (per_token.float() * loss_mask).sum(dim=1)
            correct_counts = ((predictions == y) & loss_mask).sum(dim=1)
            for local_index in range(end - start):
                count = int(token_counts[local_index].item())
                if count <= 0:
                    raise ValueError("validation row has no response tokens after fitting")
                meta = tokenized["metadata"][start + local_index]
                output.append({
                    **meta,
                    "response_tokens": count,
                    "loss_sum": float(loss_sums[local_index].item()),
                    "nll": float(loss_sums[local_index].item()) / count,
                    "correct_tokens": int(correct_counts[local_index].item()),
                })
    return output


def summarize_rows(rows: list[dict]) -> dict:
    correct = [row for row in rows if row["response_kind"] == "correct"]
    distractors = {
        row["id"]: row for row in rows if row["response_kind"] == "distractor"
    }
    correct_by_id = {row["id"]: row for row in correct}
    total_tokens = sum(row["response_tokens"] for row in correct)
    total_loss = sum(row["loss_sum"] for row in correct)
    total_correct = sum(row["correct_tokens"] for row in correct)
    category_nll = {}
    for category in sorted({row["category"] for row in correct}):
        selected = [row for row in correct if row["category"] == category]
        tokens = sum(row["response_tokens"] for row in selected)
        category_nll[category] = sum(row["loss_sum"] for row in selected) / tokens
    margins = {
        row_id: distractors[row_id]["nll"] - row["nll"]
        for row_id, row in correct_by_id.items()
    }
    calculation_ids = [
        row["id"] for row in correct if row["is_calculation"]
    ]

    def binding_summary(ids: list[str]) -> dict:
        selected = [margins[row_id] for row_id in ids]
        return {
            "accuracy": sum(value > 0 for value in selected) / len(selected),
            "mean_margin": sum(selected) / len(selected),
            "examples": len(selected),
        }

    return {
        "response_nll": total_loss / total_tokens,
        "response_perplexity": math.exp(min(total_loss / total_tokens, 20.0)),
        "response_token_accuracy": total_correct / total_tokens,
        "response_tokens": total_tokens,
        "category_nll": category_nll,
        "prompt_binding": binding_summary(sorted(correct_by_id)),
        "calculation_binding": binding_summary(calculation_ids),
    }


def checkpoint_gate(
    baseline: dict,
    candidate: dict,
    min_nll_improvement: float,
    min_binding_accuracy: float,
    max_category_regression: float,
) -> dict:
    reasons = []
    scalar_values = [
        candidate["response_nll"],
        candidate["prompt_binding"]["accuracy"],
        candidate["calculation_binding"]["accuracy"],
        *candidate["category_nll"].values(),
    ]
    if not all(math.isfinite(value) for value in scalar_values):
        return {"passed": False, "reasons": ["non-finite validation metric"]}
    required_nll = baseline["response_nll"] * (1.0 - min_nll_improvement)
    if candidate["response_nll"] > required_nll:
        reasons.append(
            f"response_nll={candidate['response_nll']:.6f} > {required_nll:.6f}"
        )
    for metric in ("prompt_binding", "calculation_binding"):
        floor = max(min_binding_accuracy, baseline[metric]["accuracy"])
        value = candidate[metric]["accuracy"]
        if value < floor:
            reasons.append(f"{metric}_accuracy={value:.3f} < {floor:.3f}")
    for category, baseline_nll in baseline["category_nll"].items():
        limit = baseline_nll * (1.0 + max_category_regression)
        value = candidate["category_nll"][category]
        if value > limit:
            reasons.append(f"{category}_nll={value:.6f} > {limit:.6f}")
    return {"passed": not reasons, "reasons": reasons}


def select_checkpoint(results: list[dict], args) -> tuple[str | None, dict]:
    baseline = results[0]
    gates = {}
    for result in results[1:]:
        gates[result["label"]] = checkpoint_gate(
            baseline["summary"], result["summary"],
            args.min_nll_improvement, args.min_binding_accuracy,
            args.max_category_regression,
        )
    eligible = [
        result for result in results[1:] if gates[result["label"]]["passed"]
    ]
    selected = min(eligible, key=lambda item: item["summary"]["response_nll"]) if eligible else None
    return (selected["checkpoint"] if selected else None), gates


def evaluate_checkpoint(path: str, label: str, validation: dict, args, device) -> dict:
    model, cfg = load_model(path, device, allow_unsafe=args.allow_unsafe_checkpoint)
    model.set_act_compute_skip(False)
    dtype = resolve_dtype(device, args.dtype)
    tokenizer = _get_gpt2_tokenizer()
    binding_rows = build_binding_rows(validation["examples"])
    tokenized = _tokenize_rows(binding_rows, tokenizer, cfg.vocab_size, args.seq_len)
    rows = _evaluate_tokenized(model, tokenized, args, device, dtype)
    summary = summarize_rows(rows)
    print(
        f"  {label}: nll={summary['response_nll']:.6f} "
        f"binding={summary['prompt_binding']['accuracy']:.1%} "
        f"calc_binding={summary['calculation_binding']['accuracy']:.1%}"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"label": label, "checkpoint": path, "summary": summary}


def build_report(payload: dict) -> str:
    lines = [
        "# Finance QA Validation Checkpoint Selection", "",
        f"- validation: `{payload['validation']['version']}` (`{payload['validation']['sha256']}`)",
        f"- selected checkpoint: `{payload['recommended_checkpoint']}`",
        "", "## Metrics", "",
        "| checkpoint | response NLL | token accuracy | prompt binding | calculation binding | gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for index, result in enumerate(payload["results"]):
        summary = result["summary"]
        gate = "baseline" if index == 0 else (
            "pass" if payload["selection_gates"][result["label"]]["passed"] else "fail"
        )
        lines.append(
            f"| {result['label']} | {summary['response_nll']:.6f} | "
            f"{summary['response_token_accuracy']:.1%} | "
            f"{summary['prompt_binding']['accuracy']:.1%} | "
            f"{summary['calculation_binding']['accuracy']:.1%} | {gate} |"
        )
    lines += ["", "## Gate Details", ""]
    if payload["recommended_checkpoint"] is None:
        lines.append("**No candidate passed validation selection. Final held-out evaluation must not run.**")
    for label, gate in payload["selection_gates"].items():
        lines.append(f"- `{label}`: " + ("passed" if gate["passed"] else "; ".join(gate["reasons"])))
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", nargs="+", required=True,
                        help="Phase 3 baseline first, followed by candidate checkpoints")
    parser.add_argument("--validation_data", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--min_nll_improvement", type=float, default=0.05)
    parser.add_argument("--min_binding_accuracy", type=float, default=0.60)
    parser.add_argument("--max_category_regression", type=float, default=0.10)
    parser.add_argument("--json_out", type=Path, default=Path("training/report/finance_validation.json"))
    parser.add_argument("--md_out", type=Path, default=Path("training/report/finance_validation.md"))
    parser.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = parser.parse_args()
    if len(args.ckpts) < 2:
        parser.error("--ckpts requires a baseline and at least one candidate")
    for path in args.ckpts:
        if not Path(path).is_file():
            parser.error(f"checkpoint not found: {path}")
    if not args.validation_data.is_file():
        parser.error(f"validation corpus not found: {args.validation_data}")
    if args.batch_size <= 0 or args.seq_len <= 0 or args.loops <= 0:
        parser.error("batch size, sequence length, and loops must be positive")
    if not (0.0 <= args.min_nll_improvement < 1.0):
        parser.error("--min_nll_improvement must be in [0,1)")
    if not (0.0 <= args.min_binding_accuracy <= 1.0):
        parser.error("--min_binding_accuracy must be in [0,1]")
    if args.max_category_regression < 0.0:
        parser.error("--max_category_regression must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    validation, sha256 = load_validation(args.validation_data)
    device = resolve_device(args.device)
    labels = make_labels(args.ckpts)
    results = [
        evaluate_checkpoint(path, label, validation, args, device)
        for path, label in zip(args.ckpts, labels)
    ]
    recommended, gates = select_checkpoint(results, args)
    payload = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "device": str(device)},
        "validation": {"path": str(args.validation_data), "version": validation["version"], "sha256": sha256},
        "config": {
            "checkpoints": args.ckpts, "batch_size": args.batch_size,
            "seq_len": args.seq_len, "loops": args.loops, "dtype": args.dtype,
            "min_nll_improvement": args.min_nll_improvement,
            "min_binding_accuracy": args.min_binding_accuracy,
            "max_category_regression": args.max_category_regression,
        },
        "results": results,
        "selection_gates": gates,
        "recommended_checkpoint": recommended,
    }
    _write_json(args.json_out, payload)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(build_report(payload), encoding="utf-8")
    print(f"selected={recommended}")
    print(f"JSON: {args.json_out}")
    print(f"Markdown: {args.md_out}")


if __name__ == "__main__":
    main()
