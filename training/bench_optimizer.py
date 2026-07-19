#!/usr/bin/env python3
"""Compare standard, fused, and 8-bit AdamW on the same training workload."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import grouped_moe_runtime_status
from training.eval_perplexity import load_model
from training.finance_pretrain import make_optimizer, optimizer_backend


_EXPECTED_BACKEND = {
    "adamw": "torch_adamw",
    "fused": "torch_adamw_fused",
    "8bit": "bitsandbytes_adamw8bit",
}


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _make_batches(args, vocab_size: int, device: torch.device):
    generator = torch.Generator().manual_seed(args.seed)
    batches = []
    for _ in range(args.num_batches):
        x = torch.randint(
            0, vocab_size, (args.batch_size, args.seq_len), generator=generator
        ).to(device)
        y = torch.randint(
            0, vocab_size, (args.batch_size, args.seq_len), generator=generator
        ).to(device)
        batches.append((x, y))
    return batches


def _train_step(model, optimizer, batch, cfg, args, amp_dtype, timed=False):
    x, y = batch
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=amp_dtype):
        logits = model(x, n_loops=args.n_loops)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
        ) + model._last_aux_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    start_event = end_event = None
    if timed:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    optimizer.step()
    if timed:
        end_event.record()
    return loss.detach(), start_event, end_event


def _run_mode(mode, args, device, amp_dtype, batches):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, cfg = load_model(
        args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint
    )
    cfg.use_gradient_checkpointing = args.grad_checkpoint
    model.cfg.use_gradient_checkpointing = args.grad_checkpoint
    model.set_grouped_moe(args.grouped_moe, amp_dtype)
    model.train()
    optimizer = make_optimizer(
        model.parameters(),
        args.lr,
        optim8bit=mode == "8bit",
        fused=mode == "fused",
        log_status=False,
    )
    backend = optimizer_backend(optimizer)
    active = backend == _EXPECTED_BACKEND[mode]
    print(
        f"[optimizer] requested={mode} active={str(active).lower()} "
        f"backend={backend}"
    )
    if not active:
        raise RuntimeError(
            f"optimizer mode {mode!r} requested but backend is {backend!r}"
        )

    if args.compile:
        model = torch.compile(model)
    for index in range(args.warmup):
        _train_step(
            model, optimizer, batches[index % len(batches)], cfg, args, amp_dtype
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    loss_tensors = []
    events = []
    started = time.perf_counter()
    for index in range(args.steps):
        loss, start_event, end_event = _train_step(
            model,
            optimizer,
            batches[index % len(batches)],
            cfg,
            args,
            amp_dtype,
            timed=True,
        )
        loss_tensors.append(loss.float())
        events.append((start_event, end_event))
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    losses = [float(loss.cpu()) for loss in loss_tensors]
    optimizer_seconds = sum(
        start.elapsed_time(end) for start, end in events
    ) / 1000.0
    tokens = args.steps * args.batch_size * args.seq_len
    result = {
        "active": True,
        "backend": backend,
        "seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "optimizer_seconds": optimizer_seconds,
        "optimizer_ms_per_step": optimizer_seconds * 1000.0 / args.steps,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--modes", nargs="+", choices=tuple(_EXPECTED_BACKEND), default=list(_EXPECTED_BACKEND))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--n_loops", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--grouped_moe", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--allow_unsafe_checkpoint", action="store_true")
    parser.add_argument("--json_out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("optimizer benchmark requires CUDA")
    for name in ("steps", "batch_size", "seq_len", "num_batches", "n_loops"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be at least 0")

    amp_dtype = _dtype(args.dtype, device)
    if args.grouped_moe:
        active, reason = grouped_moe_runtime_status(device, amp_dtype)
        print(
            "[grouped_moe] requested=true "
            f"active={str(active).lower()} reason={reason}"
        )
        if not active:
            raise RuntimeError("grouped MoE requested but inactive: " + reason)

    # Read once only to size deterministic batches; every mode reloads clean weights.
    probe_model, probe_cfg = load_model(
        args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint
    )
    parameter_count = sum(parameter.numel() for parameter in probe_model.parameters())
    batches = _make_batches(args, probe_cfg.vocab_size, device)
    del probe_model
    torch.cuda.empty_cache()

    results = {}
    for mode in args.modes:
        print(f"\n=== {mode} ===")
        results[mode] = _run_mode(mode, args, device, amp_dtype, batches)
        result = results[mode]
        print(
            f"{mode:>6}: {result['seconds']:.3f}s  "
            f"{result['tokens_per_second']:.1f} tok/s  "
            f"optimizer={result['optimizer_ms_per_step']:.3f} ms/step  "
            f"peak={result['peak_allocated_mib']:.1f} MiB  "
            f"loss={result['first_loss']:.5f}->{result['final_loss']:.5f}"
        )

    reference = results.get("8bit") or results[args.modes[0]]
    comparisons = {}
    for mode, result in results.items():
        comparisons[mode] = {
            "throughput_vs_reference": (
                result["tokens_per_second"] / reference["tokens_per_second"]
            ),
            "peak_mib_delta_vs_reference": (
                result["peak_allocated_mib"] - reference["peak_allocated_mib"]
            ),
            "max_loss_delta_vs_reference": max(
                abs(left - right)
                for left, right in zip(result["losses"], reference["losses"])
            ),
        }
    payload = {
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(device)
            ),
            "dtype": str(amp_dtype).removeprefix("torch."),
        },
        "parameter_count": parameter_count,
        "config": {
            key: getattr(args, key)
            for key in (
                "modes", "steps", "warmup", "batch_size", "seq_len",
                "num_batches", "n_loops", "lr", "grad_clip", "seed",
                "grad_checkpoint", "grouped_moe", "compile",
            )
        },
        "results": results,
        "reference": "8bit" if "8bit" in results else args.modes[0],
        "comparisons": comparisons,
    }
    if args.json_out:
        _write_json_atomic(args.json_out, payload)
        print(f"JSON: {args.json_out}")


if __name__ == "__main__":
    main()
