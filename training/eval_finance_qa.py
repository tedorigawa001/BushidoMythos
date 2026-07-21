#!/usr/bin/env python3
"""Reference-backed finance QA evaluation for checkpoint comparisons."""

import argparse
import datetime
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from chat import build_tokenizer, generate, get_device, load_model
from training.eval_finance_behavior import _seed_all, repetition_rate


DEFAULT_SUITE = Path(__file__).parent / "eval_data" / "finance_qa_v2.json"
DEFAULT_CKPTS = [
    "checkpoints/finance_a100_v3_full/phase2_final.pt",
    "checkpoints/finance_a100_v3_full/phase3_final.pt",
    "checkpoints/finance_a100_v3_full/phase4_final.pt",
    "checkpoints/finance_a100_v3_full/phase5_final.pt",
]
ADOPTION_THRESHOLDS = {
    "pass_rate": 0.80,
    "concept_recall": 0.75,
    "numeric_accuracy": 0.90,
    "unsafe_rate": 0.0,
    "max_exact_response_fraction": 0.20,
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def load_suite(path: Path) -> tuple:
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation suite must contain a non-empty 'cases' list")
    seen = set()
    for case in cases:
        case_id = case.get("id")
        if not case_id or case_id in seen:
            raise ValueError(f"case id must be present and unique: {case_id!r}")
        seen.add(case_id)
        if not case.get("question") or not case.get("reference_answer"):
            raise ValueError(f"{case_id}: question and reference_answer are required")
        concepts = case.get("required_concepts")
        if not isinstance(concepts, list) or not concepts:
            raise ValueError(f"{case_id}: required_concepts must be non-empty")
        for concept in concepts:
            if not concept.get("name") or not concept.get("aliases"):
                raise ValueError(f"{case_id}: every concept needs name and aliases")
        topic_anchors = case.get("topic_anchors", [])
        if not isinstance(topic_anchors, list) or any(
            not isinstance(anchor, str) or not anchor.strip() for anchor in topic_anchors
        ):
            raise ValueError(f"{case_id}: topic_anchors must be a list of non-empty strings")
    return payload, hashlib.sha256(raw).hexdigest()


def _concept_matches(response: str, case: dict) -> list:
    normalized = _normalize(response)
    matches = []
    for concept in case["required_concepts"]:
        aliases = concept["aliases"]
        matched = next((alias for alias in aliases if _normalize(alias) in normalized), None)
        matches.append({
            "name": concept["name"],
            "matched": matched is not None,
            "matched_alias": matched,
        })
    return matches


def _topic_match(response: str, case: dict) -> dict:
    anchors = case.get("topic_anchors", [])
    if not anchors:
        return {"required": False, "matched": True, "matched_anchor": None}
    normalized = _normalize(response)
    matched = next(
        (
            anchor for anchor in anchors
            if re.search(rf"(?<!\w){re.escape(_normalize(anchor))}(?!\w)", normalized)
        ),
        None,
    )
    return {
        "required": True,
        "matched": matched is not None,
        "matched_anchor": matched,
    }


_NUMBER_RE = re.compile(r"(?<![\w.])[$£€]?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(%)?")


def _numbers_in(response: str) -> list:
    values = []
    for match in _NUMBER_RE.finditer(response):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        values.append({"value": value, "kind": "percent" if match.group(2) else "number"})
    return values


def _numeric_matches(response: str, case: dict) -> list:
    observed = _numbers_in(response)
    matches = []
    for expected in case.get("numeric_expectations", []):
        target = float(expected["value"])
        tolerance = float(expected.get("tolerance", 0.01))
        kind = expected.get("kind", "number")
        found = next(
            (
                item for item in observed
                if item["kind"] == kind
                and math.isclose(item["value"], target, rel_tol=0.0, abs_tol=tolerance)
            ),
            None,
        )
        matches.append({
            "name": expected["name"],
            "expected": target,
            "kind": kind,
            "matched": found is not None,
        })
    return matches


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 40):start]
    return re.search(
        r"\b(no|not|never|cannot|can't|isn't|aren't|doesn't|won't|without)\b[^.!?]{0,30}$",
        prefix,
    ) is not None


def _unsafe_matches(response: str, case: dict) -> list:
    normalized = _normalize(response)
    hits = []
    for phrase in case.get("forbidden_claims", []):
        needle = _normalize(phrase)
        start = normalized.find(needle)
        while start != -1:
            if not _is_negated(normalized, start):
                hits.append(phrase)
                break
            start = normalized.find(needle, start + len(needle))
    return hits


def score_response(response: str, case: dict) -> dict:
    concept_matches = _concept_matches(response, case)
    topic_match = _topic_match(response, case)
    numeric_matches = _numeric_matches(response, case)
    unsafe_matches = _unsafe_matches(response, case)
    concept_recall = sum(item["matched"] for item in concept_matches) / len(concept_matches)
    numeric_accuracy = (
        sum(item["matched"] for item in numeric_matches) / len(numeric_matches)
        if numeric_matches else None
    )
    words = response.split()
    rep_rate = repetition_rate(response)
    non_degenerate = len(words) >= 8 and rep_rate < 0.5
    safety = not unsafe_matches
    if numeric_accuracy is None:
        score = 0.75 * concept_recall + 0.15 * float(safety) + 0.10 * float(non_degenerate)
    else:
        score = (
            0.55 * concept_recall
            + 0.25 * numeric_accuracy
            + 0.10 * float(safety)
            + 0.10 * float(non_degenerate)
        )
    threshold = float(case.get("min_concept_recall", 0.6))
    passed = (
        topic_match["matched"]
        and concept_recall >= threshold
        and (numeric_accuracy is None or numeric_accuracy == 1.0)
        and safety
        and non_degenerate
    )
    return {
        "score": score,
        "passed": passed,
        "concept_recall": concept_recall,
        "concept_matches": concept_matches,
        "topic_relevant": topic_match["matched"],
        "topic_match": topic_match,
        "numeric_accuracy": numeric_accuracy,
        "numeric_matches": numeric_matches,
        "unsafe": not safety,
        "unsafe_matches": unsafe_matches,
        "non_degenerate": non_degenerate,
        "word_count": len(words),
        "repetition_rate": rep_rate,
    }


def resolve_dtype(device: torch.device, requested: str) -> torch.dtype:
    if device.type != "cuda":
        if requested not in ("auto", "float32"):
            raise ValueError(f"--dtype {requested} requires CUDA")
        return torch.float32
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def make_labels(paths: list) -> list:
    labels = [Path(path).stem for path in paths]
    if len(labels) != len(set(labels)):
        labels = [f"{Path(path).parent.name}/{Path(path).stem}" for path in paths]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels are not unique")
    return labels


def evaluate_checkpoint(path: str, label: str, suite: dict, args, device, dtype) -> dict:
    model, cfg = load_model(path, device, allow_unsafe=args.allow_unsafe_checkpoint)
    model.set_act_compute_skip(True)
    tokenizer = build_tokenizer(cfg.vocab_size, mode=args.tokenizer)
    rows = []
    for case_index, case in enumerate(suite["cases"]):
        for seed in args.seeds:
            effective_seed = seed + case_index * 1009
            _seed_all(effective_seed, device)
            response = generate(
                model,
                cfg,
                tokenizer,
                case["question"],
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=min(args.top_k, cfg.vocab_size) if args.top_k > 0 else 0,
                n_loops=args.loops,
                device=device,
                finance_mode=True,
                repetition_penalty=args.rep_penalty,
                compute_dtype=dtype,
            )
            scoring = score_response(response, case)
            rows.append({
                "case_id": case["id"],
                "category": case["category"],
                "seed": seed,
                "effective_seed": effective_seed,
                "response": response,
                **scoring,
            })
            print(
                f"  [{label}] {case['id']:<27} seed={seed} "
                f"score={scoring['score']:.3f} pass={str(scoring['passed']).lower()}"
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"label": label, "checkpoint": path, "rows": rows}


def summarize(result: dict) -> dict:
    rows = result["rows"]
    numeric = [row for row in rows if row["numeric_accuracy"] is not None]
    response_counts = Counter(_normalize(row.get("response", "")) for row in rows)
    categories = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        categories[category] = {
            "score": sum(row["score"] for row in selected) / len(selected),
            "pass_rate": sum(row["passed"] for row in selected) / len(selected),
        }
    return {
        "score": sum(row["score"] for row in rows) / len(rows),
        "pass_rate": sum(row["passed"] for row in rows) / len(rows),
        "concept_recall": sum(row["concept_recall"] for row in rows) / len(rows),
        "topic_relevance_rate": sum(row.get("topic_relevant", True) for row in rows) / len(rows),
        "numeric_accuracy": (
            sum(row["numeric_accuracy"] for row in numeric) / len(numeric)
            if numeric else None
        ),
        "unsafe_rate": sum(row["unsafe"] for row in rows) / len(rows),
        "non_degenerate_rate": sum(row["non_degenerate"] for row in rows) / len(rows),
        "max_exact_response_fraction": max(response_counts.values()) / len(rows),
        "categories": categories,
    }


def adoption_gate(summary: dict) -> dict:
    reasons = []
    for key in ("pass_rate", "concept_recall", "numeric_accuracy"):
        value = summary[key]
        threshold = ADOPTION_THRESHOLDS[key]
        if value is None or value < threshold:
            reasons.append(f"{key}={value} < {threshold}")
    if summary["unsafe_rate"] > ADOPTION_THRESHOLDS["unsafe_rate"]:
        reasons.append(
            f"unsafe_rate={summary['unsafe_rate']} > {ADOPTION_THRESHOLDS['unsafe_rate']}"
        )
    collapse = summary["max_exact_response_fraction"]
    collapse_limit = ADOPTION_THRESHOLDS["max_exact_response_fraction"]
    if collapse > collapse_limit:
        reasons.append(
            f"max_exact_response_fraction={collapse} > {collapse_limit}"
        )
    return {"passed": not reasons, "reasons": reasons}


def build_report(payload: dict, suite: dict) -> str:
    labels = [result["label"] for result in payload["results"]]
    summaries = payload["summaries"]
    lines = [
        "# Finance QA Evaluation - Phase 2-5",
        "",
        f"- suite: `{payload['suite']['version']}` (`{payload['suite']['sha256']}`)",
        f"- seeds: {payload['config']['seeds']} | loops={payload['config']['loops']} "
        f"max_tokens={payload['config']['max_tokens']} temp={payload['config']['temperature']}",
        "",
        "> Rule-based rubric: concept coverage and numeric checks are deterministic, but do not prove factual correctness. Review the sampled outputs before adopting a checkpoint.",
        "",
        "## Aggregate",
        "",
        "| metric | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]

    def metric_row(name, key, percent=False):
        values = []
        for label in labels:
            value = summaries[label][key]
            values.append("-" if value is None else (f"{value * 100:.1f}%" if percent else f"{value:.3f}"))
        lines.append(f"| {name} | " + " | ".join(values) + " |")

    metric_row("overall score", "score")
    metric_row("pass rate", "pass_rate", percent=True)
    metric_row("topic relevance", "topic_relevance_rate", percent=True)
    metric_row("required-concept recall", "concept_recall", percent=True)
    metric_row("numeric accuracy", "numeric_accuracy", percent=True)
    metric_row("unsafe-claim rate", "unsafe_rate", percent=True)
    metric_row("non-degenerate rate", "non_degenerate_rate", percent=True)
    metric_row("largest exact-response share", "max_exact_response_fraction", percent=True)

    lines += ["", "## Adoption Gate", ""]
    if payload["recommended_checkpoint"] is None:
        lines.append("**No checkpoint passed the absolute adoption gate.**")
    else:
        lines.append(f"Recommended checkpoint: **{payload['recommended_checkpoint']}**")
    lines += [
        "",
        "Thresholds: pass rate >= 80%, required-concept recall >= 75%, numeric accuracy >= 90%, unsafe-claim rate = 0%, largest exact-response share <= 20%.",
        "",
        "| checkpoint | passed | reasons |",
        "|---|---|---|",
    ]
    for label in labels:
        gate = payload["adoption_gates"][label]
        reasons = "; ".join(gate["reasons"]) or "-"
        lines.append(f"| {label} | {str(gate['passed']).lower()} | {reasons} |")

    lines += ["", "## Per Case", "", "| case | category | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    result_by_label = {result["label"]: result for result in payload["results"]}
    for case in suite["cases"]:
        cells = []
        for label in labels:
            rows = [row for row in result_by_label[label]["rows"] if row["case_id"] == case["id"]]
            score = sum(row["score"] for row in rows) / len(rows)
            passed = sum(row["passed"] for row in rows) / len(rows)
            cells.append(f"{score:.3f} / {passed * 100:.0f}%")
        lines.append(f"| `{case['id']}` | {case['category']} | " + " | ".join(cells) + " |")

    sample_seed = payload["config"]["seeds"][0]
    lines += ["", f"## Sample Outputs (seed {sample_seed})", ""]
    for case in suite["cases"]:
        lines += [f"### `{case['id']}`", "", f"**Question:** {case['question']}", "", f"**Reference:** {case['reference_answer']}", ""]
        for label in labels:
            row = next(
                row for row in result_by_label[label]["rows"]
                if row["case_id"] == case["id"] and row["seed"] == sample_seed
            )
            lines += [f"**{label}** - score={row['score']:.3f}, pass={str(row['passed']).lower()}", ""]
            response_lines = row["response"].splitlines() or ["(empty)"]
            lines.extend("    " + part for part in response_lines)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference-backed finance QA checkpoint comparison")
    parser.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--tokenizer", choices=["auto", "gpt2", "mythos"], default="auto")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--rep_penalty", type=float, default=1.3)
    parser.add_argument("--json_out", type=Path, default=Path("training/report/finance_qa_phase2_5.json"))
    parser.add_argument("--md_out", type=Path, default=Path("training/report/finance_qa_phase2_5.md"))
    parser.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("--seeds requires at least one seed")
    if args.max_tokens <= 0 or args.loops <= 0 or args.temp <= 0 or args.rep_penalty <= 0:
        parser.error("max_tokens, loops, temp, and rep_penalty must be positive")
    missing = [path for path in args.ckpts if not Path(path).is_file()]
    if missing:
        parser.error("requested checkpoints are missing: " + ", ".join(missing))
    return args


def main() -> None:
    args = parse_args()
    suite, suite_hash = load_suite(args.suite)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    labels = make_labels(args.ckpts)
    print(f"Device: {device} dtype={dtype} suite={suite['version']} cases={len(suite['cases'])}")
    results = [
        evaluate_checkpoint(path, label, suite, args, device, dtype)
        for path, label in zip(args.ckpts, labels)
    ]
    summaries = {result["label"]: summarize(result) for result in results}
    adoption_gates = {label: adoption_gate(summary) for label, summary in summaries.items()}
    eligible = [
        label for label in labels if adoption_gates[label]["passed"]
    ]
    recommended = (
        max(eligible, key=lambda label: summaries[label]["score"])
        if eligible else None
    )
    payload = {
        "created_at": datetime.datetime.now().isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "dtype": str(dtype),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "suite": {"path": str(args.suite), "version": suite["version"], "sha256": suite_hash},
        "config": {
            "checkpoints": args.ckpts,
            "seeds": args.seeds,
            "loops": args.loops,
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
            "top_k": args.top_k,
            "repetition_penalty": args.rep_penalty,
        },
        "summaries": summaries,
        "adoption_thresholds": ADOPTION_THRESHOLDS,
        "adoption_gates": adoption_gates,
        "recommended_checkpoint": recommended,
        "results": results,
    }
    _write_json(args.json_out, payload)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(build_report(payload, suite), encoding="utf-8")
    print(f"JSON: {args.json_out}")
    print(f"Markdown: {args.md_out}")


if __name__ == "__main__":
    main()
