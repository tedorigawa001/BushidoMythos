#!/usr/bin/env python3
"""Run an aligned Phase 5 optimizer-step dose ablation from one Phase 3 state."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training import run_phase5_pilot as pilot


def variant_name(steps: int) -> str:
    return f"aligned_steps_{steps:04d}"


def build_train_commands(args) -> list[list]:
    commands = []
    for steps in args.step_variants:
        variant_args = copy.copy(args)
        variant_args.steps = steps
        variant_args.run_name = variant_name(steps)
        commands.append(pilot.build_train_command(variant_args))
    return commands


def build_eval_command(args) -> list:
    checkpoints = [args.phase3_ckpt, *args.comparison_ckpts]
    checkpoints += [
        args.output_root / variant_name(steps) / "phase5_final.pt"
        for steps in args.step_variants
    ]
    report_stem = getattr(args, "eval_report_stem", "finance_qa_phase5_step_ablation")
    command = [sys.executable, "training/eval_finance_qa.py", "--ckpts"]
    command += [str(path) for path in checkpoints]
    command += [
        "--suite", str(args.eval_suite),
        "--device", args.eval_device,
        "--dtype", args.dtype,
        "--seeds", *[str(seed) for seed in args.eval_seeds],
        "--loops", str(args.eval_loops),
        "--max_tokens", str(args.eval_max_tokens),
        "--json_out", str(args.output_root / f"{report_stem}.json"),
        "--md_out", str(args.output_root / f"{report_stem}.md"),
    ]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aligned Phase 5 step-dose ablation")
    parser.add_argument("--phase3_ckpt", type=Path, required=True)
    parser.add_argument("--base_ckpt", type=Path)
    parser.add_argument("--comparison_ckpts", nargs="*", type=Path, default=[])
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--local_root", type=Path)
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
        "--eval_suite", type=Path,
        default=Path("training/eval_data/finance_qa_v2.json"),
    )
    parser.add_argument("--step_variants", nargs="+", type=int, default=[10, 50, 200])
    parser.add_argument("--warmup_steps", type=int, default=50)
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
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--mem_log_every", type=int, default=10)
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
    parser.add_argument("--eval_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval_loops", type=int, default=8)
    parser.add_argument("--eval_max_tokens", type=int, default=128)
    parser.add_argument(
        "--eval_report_stem", default="finance_qa_phase5_step_ablation",
        help="basename for evaluation JSON and Markdown outputs",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="skip training and evaluate existing step-variant checkpoints",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grouped_moe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused_optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--act_curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    required_paths = [
        ("Phase 3 checkpoint", args.phase3_ckpt),
        ("evaluation suite", args.eval_suite),
    ]
    if not args.eval_only:
        required_paths += [
            ("base checkpoint", args.base_ckpt),
            ("curated data", args.curated_data),
            ("validation data", args.validation_data),
        ]
    for label, path in required_paths:
        if path is None:
            parser.error(f"{label} is required unless --eval_only is set")
        if not path.is_file():
            parser.error(f"{label} not found: {path}")
    for path in args.comparison_ckpts:
        if not path.is_file():
            parser.error(f"comparison checkpoint not found: {path}")
    if any(steps <= 0 for steps in args.step_variants):
        parser.error("--step_variants values must be positive")
    if len(set(args.step_variants)) != len(args.step_variants):
        parser.error("--step_variants values must be unique")
    if not args.eval_report_stem or Path(args.eval_report_stem).name != args.eval_report_stem:
        parser.error("--eval_report_stem must be a non-empty filename stem")
    if args.eval_only:
        for steps in args.step_variants:
            path = args.output_root / variant_name(steps) / "phase5_final.pt"
            if not path.is_file():
                parser.error(f"step-variant checkpoint not found: {path}")
    if args.warmup_steps < 0:
        parser.error("--warmup_steps must be non-negative")
    if not (0.0 < args.max_similarity <= 1.0):
        parser.error("--max_similarity must be in (0,1]")
    if not (0.0 < args.max_response_share <= 1.0):
        parser.error("--max_response_share must be in (0,1]")
    return args


def main() -> None:
    args = parse_args()
    train_commands = [] if args.eval_only else build_train_commands(args)
    eval_command = build_eval_command(args)
    if args.dry_run:
        print(json.dumps([*train_commands, eval_command], indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.local_root:
        args.local_root.mkdir(parents=True, exist_ok=True)
    for steps, command in zip(args.step_variants, train_commands):
        print(f"\n=== Phase 5 aligned dose: {steps} steps ===", flush=True)
        subprocess.run(command, check=True)
    print("\n=== Finance QA dose comparison ===", flush=True)
    subprocess.run(eval_command, check=True)
    print(f"JSON: {args.output_root / (args.eval_report_stem + '.json')}")
    print(f"Markdown: {args.output_root / (args.eval_report_stem + '.md')}")


if __name__ == "__main__":
    main()
