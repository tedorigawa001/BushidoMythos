#!/usr/bin/env python3
"""Train and evaluate an audited curated Phase 5 pilot from Phase 3."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _append_switch(command: list, enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def _run_name(args) -> str:
    return getattr(args, "run_name", "curated")


def build_train_command(args) -> list:
    run_name = _run_name(args)
    output_dir = args.output_root / run_name
    command = [
        sys.executable,
        "training/finance_pretrain.py",
        "--phase", "5",
        "--base_ckpt", str(args.base_ckpt),
        "--resume", str(args.phase3_ckpt),
        "--ckpt_dir", str(output_dir),
        "--cache_dir", str(args.cache_dir),
        "--phase1_steps", str(args.phase1_steps),
        "--phase2_steps", str(args.phase2_steps),
        "--phase3_steps", str(args.phase3_steps),
        "--phase4_steps", "0",
        "--phase5_steps", str(args.steps),
        "--phase5_data_mode", "curated",
        "--phase5_curated_path", str(args.curated_data),
        "--phase5_validation_path", str(args.validation_data),
        "--phase5_eval_suite", str(args.eval_suite),
        "--phase5_max_similarity", str(args.max_similarity),
        "--phase5_max_response_share", str(args.max_response_share),
        "--phase5_audit_json", str(output_dir / "phase5_data_audit.json"),
        "--batch_size", str(args.batch_size),
        "--grad_accum_steps", str(args.grad_accum_steps),
        "--seq_len", str(args.seq_len),
        "--dtype", args.dtype,
        "--lr", str(args.lr),
        "--warmup_steps", str(args.warmup_steps),
        "--save_every", str(args.save_every),
        "--log_every", str(args.log_every),
        "--mem_log_every", str(args.mem_log_every),
        "--keep_last_n_steps", str(getattr(args, "keep_last_n_steps", 1)),
        "--loop_schedule", "curriculum",
        "--loop_tail_max", str(args.loop_tail_max),
        "--loop_tail_p", str(args.loop_tail_p),
        "--loop_seed", str(args.loop_seed),
        "--seed", str(args.seed),
        "--replay_ratio", str(args.replay_ratio),
        "--act_anchor_step", str(args.act_anchor_step),
        "--act_threshold_start", str(args.act_threshold_start),
        "--act_warmup_frac", str(args.act_warmup_frac),
        "--ponder_weight_start", str(args.ponder_weight_start),
        "--ponder_weight_end", str(args.ponder_weight_end),
    ]
    if args.local_root:
        command += ["--local_ckpt_dir", str(args.local_root / run_name)]
    _append_switch(command, args.compile, "--compile")
    _append_switch(command, args.grad_checkpoint, "--grad_checkpoint")
    _append_switch(command, args.grouped_moe, "--grouped_moe")
    _append_switch(command, args.fused_optimizer, "--fused_optimizer")
    _append_switch(command, args.act_curriculum, "--act_curriculum")
    return command


def build_eval_command(args) -> list:
    checkpoints = [args.phase3_ckpt, *args.comparison_ckpts]
    checkpoints.append(args.output_root / _run_name(args) / "phase5_final.pt")
    command = [sys.executable, "training/eval_finance_qa.py", "--ckpts"]
    command += [str(path) for path in checkpoints]
    command += [
        "--suite", str(args.eval_suite),
        "--device", args.eval_device,
        "--dtype", args.dtype,
        "--seeds", *[str(seed) for seed in args.eval_seeds],
        "--loops", str(args.eval_loops),
        "--max_tokens", str(args.eval_max_tokens),
        "--json_out", str(args.output_root / "finance_qa_phase5_pilot.json"),
        "--md_out", str(args.output_root / "finance_qa_phase5_pilot.md"),
    ]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audited curated Phase 5 pilot")
    parser.add_argument("--phase3_ckpt", type=Path, required=True)
    parser.add_argument("--base_ckpt", type=Path, required=True)
    parser.add_argument("--comparison_ckpts", nargs="*", type=Path, default=[])
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--local_root", type=Path)
    parser.add_argument("--cache_dir", type=Path, default=Path("/content/cache"))
    parser.add_argument(
        "--curated_data",
        type=Path,
        default=Path("training/train_data/finance_qa_curated_v2_train.json"),
    )
    parser.add_argument(
        "--validation_data",
        type=Path,
        default=Path("training/eval_data/finance_qa_curated_v2_validation.json"),
    )
    parser.add_argument(
        "--eval_suite",
        type=Path,
        default=Path("training/eval_data/finance_qa_v2.json"),
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--phase1_steps", type=int, default=30000)
    parser.add_argument("--phase2_steps", type=int, default=8000)
    parser.add_argument("--phase3_steps", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--mem_log_every", type=int, default=50)
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
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grouped_moe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused_optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--act_curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    for label, path in (
        ("Phase 3 checkpoint", args.phase3_ckpt),
        ("base checkpoint", args.base_ckpt),
        ("curated data", args.curated_data),
        ("validation data", args.validation_data),
        ("evaluation suite", args.eval_suite),
    ):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")
    missing_comparisons = [path for path in args.comparison_ckpts if not path.is_file()]
    if missing_comparisons:
        parser.error(f"comparison checkpoint not found: {missing_comparisons[0]}")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not (0.0 < args.max_similarity <= 1.0):
        parser.error("--max_similarity must be in (0,1]")
    if not (0.0 < args.max_response_share <= 1.0):
        parser.error("--max_response_share must be in (0,1]")
    return args


def main() -> None:
    args = parse_args()
    commands = [build_train_command(args), build_eval_command(args)]
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.local_root:
        args.local_root.mkdir(parents=True, exist_ok=True)
    print("\n=== Phase 5 curated pilot ===", flush=True)
    subprocess.run(commands[0], check=True)
    print("\n=== Finance QA comparison ===", flush=True)
    subprocess.run(commands[1], check=True)
    print(f"JSON: {args.output_root / 'finance_qa_phase5_pilot.json'}")
    print(f"Markdown: {args.output_root / 'finance_qa_phase5_pilot.md'}")


if __name__ == "__main__":
    main()
