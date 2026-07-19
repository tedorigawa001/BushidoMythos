#!/usr/bin/env python3
"""Summarize and compare finance training wall-clock reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_RUNTIME_KEYS = (
    "dtype",
    "compile",
    "grouped_moe",
    "liger_fused_ce",
    "optimizer_backend",
)


def _number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def summarize_report(payload: dict, *, label: str) -> dict:
    phases = payload.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"{label}: phases must be a non-empty list")

    total_wall = _number(payload.get("total_wall_seconds"), "total_wall_seconds")
    phase_wall = sum(
        _number(phase.get("wall_seconds"), "phase.wall_seconds")
        for phase in phases
    )
    data_wait = sum(
        _number(phase.get("data_wait_seconds", 0), "phase.data_wait_seconds")
        for phase in phases
    )
    optimizer = sum(
        _number(phase.get("optimizer_seconds", 0), "phase.optimizer_seconds")
        for phase in phases
    )
    tokens = sum(
        int(_number(phase.get("tokens_processed"), "phase.tokens_processed"))
        for phase in phases
    )
    dataset = sum(
        _number(item.get("seconds"), "dataset_build.seconds")
        for item in payload.get("dataset_builds", [])
    )
    serializations = payload.get("checkpoint_serializations", [])
    serialize = sum(
        _number(item.get("seconds"), "checkpoint_serialization.seconds")
        for item in serializations
    )
    phase_names = {phase.get("name") for phase in phases}
    phase_serialize = sum(
        _number(item.get("seconds"), "checkpoint_serialization.seconds")
        for item in serializations
        if item.get("phase") in phase_names
    )

    async_copy = payload.get("async_checkpoint_copy") or {"enabled": False}
    async_enabled = bool(async_copy.get("enabled", False))
    copy_seconds = _number(async_copy.get("copy_seconds", 0), "copy_seconds")
    bytes_copied = int(_number(async_copy.get("bytes_copied", 0), "bytes_copied"))
    pending = int(_number(async_copy.get("pending", 0), "pending"))
    errors = async_copy.get("errors", [])
    if not isinstance(errors, list):
        raise ValueError(f"{label}: async_checkpoint_copy.errors must be a list")
    if async_enabled and (pending or errors):
        raise ValueError(
            f"{label}: asynchronous copy incomplete: pending={pending} errors={errors}"
        )

    runtime = payload.get("runtime") or {}
    phase_signature = [
        {
            "name": phase.get("name"),
            "start_step": phase.get("start_step"),
            "end_step": phase.get("end_step"),
            "tokens_processed": phase.get("tokens_processed"),
        }
        for phase in phases
    ]
    phase_breakdown = []
    for phase in phases:
        name = phase.get("name")
        wall = _number(phase.get("wall_seconds"), "phase.wall_seconds")
        wait = _number(phase.get("data_wait_seconds", 0), "phase.data_wait_seconds")
        opt = _number(phase.get("optimizer_seconds", 0), "phase.optimizer_seconds")
        phase_tokens = int(
            _number(phase.get("tokens_processed"), "phase.tokens_processed")
        )
        phase_save = sum(
            _number(item.get("seconds"), "checkpoint_serialization.seconds")
            for item in serializations
            if item.get("phase") == name
        )
        phase_breakdown.append(
            {
                "name": name,
                "wall_seconds": wall,
                "tokens_processed": phase_tokens,
                "effective_tokens_per_second": phase_tokens / max(wall, 1e-9),
                "data_wait_seconds": wait,
                "optimizer_seconds": opt,
                "checkpoint_serialize_seconds": phase_save,
                "compute_and_other_seconds": max(
                    wall - wait - opt - phase_save, 0.0
                ),
            }
        )
    accounted_phase = data_wait + optimizer + phase_serialize
    return {
        "label": label,
        "total_wall_seconds": total_wall,
        "phase_wall_seconds": phase_wall,
        "setup_and_finalize_seconds": max(total_wall - phase_wall, 0.0),
        "dataset_build_seconds": dataset,
        "data_wait_seconds": data_wait,
        "optimizer_seconds": optimizer,
        "checkpoint_serialize_seconds": serialize,
        "phase_compute_and_other_seconds": max(phase_wall - accounted_phase, 0.0),
        "tokens_processed": tokens,
        "effective_tokens_per_second": tokens / max(phase_wall, 1e-9),
        "async_copy": {
            "enabled": async_enabled,
            "files_copied": int(async_copy.get("files_copied", 0)),
            "bytes_copied": bytes_copied,
            "copy_seconds": copy_seconds,
            "copy_mib_per_second": (
                bytes_copied / 1024**2 / copy_seconds if copy_seconds else 0.0
            ),
            "max_queue_depth": int(async_copy.get("max_queue_depth", 0)),
            "pending": pending,
            "errors": errors,
        },
        "runtime": {key: runtime.get(key) for key in _RUNTIME_KEYS},
        "phase_signature": phase_signature,
        "phases": phase_breakdown,
    }


def compare_reports(baseline: dict, candidate: dict) -> dict:
    mismatches = []
    if baseline["phase_signature"] != candidate["phase_signature"]:
        mismatches.append("phase_signature")
    for key in _RUNTIME_KEYS:
        if baseline["runtime"].get(key) != candidate["runtime"].get(key):
            mismatches.append(f"runtime.{key}")
    if mismatches:
        raise ValueError(
            f"{candidate['label']}: incomparable with {baseline['label']}: "
            + ", ".join(mismatches)
        )
    return {
        "baseline": baseline["label"],
        "candidate": candidate["label"],
        "wall_clock_speedup": baseline["total_wall_seconds"]
        / max(candidate["total_wall_seconds"], 1e-9),
        "total_wall_seconds_delta": candidate["total_wall_seconds"]
        - baseline["total_wall_seconds"],
        "effective_tokens_per_second_ratio": candidate[
            "effective_tokens_per_second"
        ]
        / max(baseline["effective_tokens_per_second"], 1e-9),
        "checkpoint_serialize_seconds_delta": candidate[
            "checkpoint_serialize_seconds"
        ]
        - baseline["checkpoint_serialize_seconds"],
        "data_wait_seconds_delta": candidate["data_wait_seconds"]
        - baseline["data_wait_seconds"],
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _print_summary(summary: dict) -> None:
    copy = summary["async_copy"]
    print(
        f"{summary['label']}: total={summary['total_wall_seconds']:.2f}s "
        f"phase={summary['phase_wall_seconds']:.2f}s "
        f"effective={summary['effective_tokens_per_second']:.1f} tok/s"
    )
    print(
        f"  dataset={summary['dataset_build_seconds']:.2f}s "
        f"data_wait={summary['data_wait_seconds']:.2f}s "
        f"optimizer={summary['optimizer_seconds']:.2f}s "
        f"serialize={summary['checkpoint_serialize_seconds']:.2f}s "
        f"compute_other={summary['phase_compute_and_other_seconds']:.2f}s"
    )
    print(
        f"  async_copy={str(copy['enabled']).lower()} "
        f"files={copy['files_copied']} copy={copy['copy_seconds']:.2f}s "
        f"rate={copy['copy_mib_per_second']:.1f} MiB/s "
        f"max_queue={copy['max_queue_depth']} pending={copy['pending']}"
    )
    for phase in summary["phases"]:
        print(
            f"  phase={phase['name']} wall={phase['wall_seconds']:.2f}s "
            f"effective={phase['effective_tokens_per_second']:.1f} tok/s "
            f"data_wait={phase['data_wait_seconds']:.2f}s "
            f"optimizer={phase['optimizer_seconds']:.2f}s "
            f"serialize={phase['checkpoint_serialize_seconds']:.2f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels matching report order; defaults to each file stem",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare every report after the first with the first baseline",
    )
    parser.add_argument("--json_out", type=Path)
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.reports):
        parser.error("--labels count must match the number of reports")
    if args.compare and len(args.reports) < 2:
        parser.error("--compare requires at least two reports")

    labels = args.labels or [path.stem for path in args.reports]
    summaries = []
    for path, label in zip(args.reports, labels):
        with path.open(encoding="utf-8") as handle:
            summaries.append(summarize_report(json.load(handle), label=label))
    for summary in summaries:
        _print_summary(summary)

    comparisons = []
    if args.compare:
        comparisons = [
            compare_reports(summaries[0], candidate)
            for candidate in summaries[1:]
        ]
        for comparison in comparisons:
            print(
                f"compare {comparison['candidate']} vs {comparison['baseline']}: "
                f"wall_speedup={comparison['wall_clock_speedup']:.3f}x "
                f"wall_delta={comparison['total_wall_seconds_delta']:+.2f}s "
                f"effective_tok_s={comparison['effective_tokens_per_second_ratio']:.3f}x"
            )

    if args.json_out:
        _write_json_atomic(
            args.json_out,
            {"summaries": summaries, "comparisons": comparisons},
        )


if __name__ == "__main__":
    main()
