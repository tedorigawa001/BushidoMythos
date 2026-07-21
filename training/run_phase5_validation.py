#!/usr/bin/env python3
"""Train Phase 5, select on validation only, then evaluate final held-out once."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training import run_phase5_pilot as pilot


def run_checked(command: list, label: str) -> None:
    """Stream child output and repeat its tail in the raised error."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail = deque(maxlen=100)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        diagnostic = "\n".join(tail)
        raise RuntimeError(
            f"{label} failed with exit code {return_code}. "
            f"Last {len(tail)} output lines:\n{diagnostic}"
        )


def build_validation_command(args, candidates: list[Path]) -> list:
    checkpoints = [args.phase3_ckpt, *candidates]
    command = [sys.executable, "training/eval_finance_validation.py", "--ckpts"]
    command += [str(path) for path in checkpoints]
    command += [
        "--validation_data", str(args.validation_data),
        "--device", args.eval_device,
        "--dtype", args.dtype,
        "--batch_size", str(args.eval_batch_size),
        "--seq_len", str(args.seq_len),
        "--loops", str(args.eval_loops),
        "--min_nll_improvement", str(args.min_nll_improvement),
        "--min_binding_accuracy", str(args.min_binding_accuracy),
        "--max_category_regression", str(args.max_category_regression),
        "--json_out", str(args.output_root / "finance_validation_selection.json"),
        "--md_out", str(args.output_root / "finance_validation_selection.md"),
    ]
    if args.allow_unsafe_checkpoint:
        command.append("--allow_unsafe_checkpoint")
    return command


def build_final_command(args, selected: Path) -> list:
    command = [
        sys.executable, "training/eval_finance_qa.py", "--ckpts",
        str(args.phase3_ckpt), str(selected),
        "--suite", str(args.held_out_suite),
        "--device", args.eval_device,
        "--dtype", args.dtype,
        "--seeds", *[str(seed) for seed in args.final_seeds],
        "--loops", str(args.eval_loops),
        "--max_tokens", str(args.final_max_tokens),
        "--json_out", str(args.output_root / "finance_qa_final_selected.json"),
        "--md_out", str(args.output_root / "finance_qa_final_selected.md"),
    ]
    if args.allow_unsafe_checkpoint:
        command.append("--allow_unsafe_checkpoint")
    return command


def candidate_checkpoints(output_dir: Path) -> list[Path]:
    periodic = sorted(output_dir.glob("step_*.pt"))
    final = output_dir / "phase5_final.pt"
    return [*periodic, *([final] if final.is_file() else [])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3_ckpt", type=Path, required=True)
    parser.add_argument("--base_ckpt", type=Path)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--local_root", type=Path)
    parser.add_argument("--run_name", default="curated_v2")
    parser.add_argument("--cache_dir", type=Path, default=Path("/content/cache"))
    parser.add_argument(
        "--curated_data", type=Path,
        default=Path("training/train_data/finance_qa_curated_v2_train.json"),
    )
    parser.add_argument(
        "--validation_data", type=Path,
        default=Path("training/eval_data/finance_qa_curated_v2_validation.json"),
    )
    parser.add_argument(
        "--held_out_suite", type=Path,
        default=Path("training/eval_data/finance_qa_v2.json"),
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--phase1_steps", type=int, default=30000)
    parser.add_argument("--phase2_steps", type=int, default=8000)
    parser.add_argument("--phase3_steps", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--mem_log_every", type=int, default=25)
    parser.add_argument("--max_similarity", type=float, default=0.80)
    parser.add_argument("--max_response_share", type=float, default=0.10)
    parser.add_argument("--loop_tail_max", type=int, default=12)
    parser.add_argument("--loop_tail_p", type=float, default=0.2)
    parser.add_argument("--loop_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay_ratio", type=float, default=0.05)
    parser.add_argument("--act_anchor_step", type=int, default=0)
    parser.add_argument("--act_threshold_start", type=float, default=0.5)
    parser.add_argument("--act_warmup_frac", type=float, default=0.73)
    parser.add_argument("--ponder_weight_start", type=float, default=0.03)
    parser.add_argument("--ponder_weight_end", type=float, default=0.0)
    parser.add_argument("--eval_device", choices=["cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_loops", type=int, default=8)
    parser.add_argument("--min_nll_improvement", type=float, default=0.05)
    parser.add_argument("--min_binding_accuracy", type=float, default=0.60)
    parser.add_argument("--max_category_regression", type=float, default=0.10)
    parser.add_argument("--final_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--final_max_tokens", type=int, default=128)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grouped_moe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused_optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--act_curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_final_eval", action="store_true")
    parser.add_argument("--allow_unsafe_checkpoint", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    required = [
        ("Phase 3 checkpoint", args.phase3_ckpt),
        ("training corpus", args.curated_data),
        ("validation corpus", args.validation_data),
        ("final held-out suite", args.held_out_suite),
    ]
    if not args.skip_train:
        required.append(("base checkpoint", args.base_ckpt))
    for label, path in required:
        if path is None or not path.is_file():
            parser.error(f"{label} not found: {path}")
    if args.steps <= 0 or args.save_every <= 0:
        parser.error("--steps and --save_every must be positive")
    if args.seq_len != 256:
        parser.error(
            "this controlled validation experiment requires --seq_len 256; "
            "using the notebook-wide sequence length changes memory use and the comparison"
        )
    if args.eval_batch_size <= 0:
        parser.error("--eval_batch_size must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.comparison_ckpts = []
    args.eval_suite = args.held_out_suite
    args.eval_seeds = args.final_seeds
    args.eval_max_tokens = args.final_max_tokens
    args.keep_last_n_steps = 0
    output_dir = args.output_root / args.run_name
    train_command = pilot.build_train_command(args)
    if args.dry_run:
        print(json.dumps({
            "train": None if args.skip_train else train_command,
            "validation_pattern": str(output_dir / "step_*.pt"),
            "final_policy": "run only when validation selects a checkpoint",
        }, indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.local_root:
        args.local_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        print("\n=== Phase 5 curated v2 training ===", flush=True)
        run_checked(train_command, "Phase 5 training")
    candidates = candidate_checkpoints(output_dir)
    if not candidates:
        raise RuntimeError(f"no Phase 5 candidate checkpoints found under {output_dir}")
    print(f"\n=== Validation-only selection ({len(candidates)} candidates) ===", flush=True)
    validation_command = build_validation_command(args, candidates)
    run_checked(validation_command, "Finance validation selection")
    selection_path = args.output_root / "finance_validation_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("recommended_checkpoint")
    if not selected:
        print("No checkpoint passed validation. Final held-out evaluation skipped.")
        return
    if args.no_final_eval:
        print(f"Validation selected: {selected}. Final held-out evaluation disabled.")
        return
    final_result = args.output_root / "finance_qa_final_selected.json"
    if final_result.exists():
        print(
            f"Final held-out result already exists: {final_result}. "
            "Not running it again; use a new --output_root for a new experiment."
        )
        return
    print("\n=== One-time final held-out evaluation ===", flush=True)
    run_checked(build_final_command(args, Path(selected)), "Final held-out evaluation")
    print(f"Selected checkpoint: {selected}")
    print(f"Validation: {selection_path}")
    print(f"Final: {final_result}")


if __name__ == "__main__":
    main()
