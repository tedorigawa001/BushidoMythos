#!/usr/bin/env python3
"""Benchmark ACT curriculum training steps with and without torch.compile.

The benchmark isolates model forward/backward from dataset and optimizer I/O.
It updates ACT threshold/ponder buffers every step, matching the dynamic part of
the production curriculum while using identical random batches for both modes.

Examples:
    python training/bench_act_compile.py --device cuda --ckpt checkpoints/finance_a100_v2/phase1_final.pt
    python training/bench_act_compile.py --device cpu --tiny --steps 2 --warmup 1 --seq_len 8 --n_loops 1
"""

import argparse
import gc
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import (
    BushidoMythos,
    MythosConfig,
    chunked_linear_cross_entropy,
    grouped_moe_runtime_status,
)
from bushido_mythos.main import _GroupedLinear
from training.eval_perplexity import load_model


@dataclass
class BenchResult:
    mode: str
    seconds: float
    tokens_per_second: float
    peak_memory_mb: float
    first_loss: float
    last_loss: float
    warmup_unique_graphs: int
    warmup_graph_breaks: int
    measured_unique_graphs: int
    measured_graph_breaks: int
    unique_graphs: int
    graph_breaks: int
    grouped_moe_active: bool


def _tiny_cfg() -> MythosConfig:
    return MythosConfig(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=32,
        lora_rank=2,
    )


def _resolve_dtype(device: torch.device, requested: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device.type != "cuda":
        return torch.float32
    major = torch.cuda.get_device_properties(device).major
    return torch.bfloat16 if major >= 8 else torch.float16


def _load_benchmark_model(
    ckpt: Optional[str], device: torch.device, allow_unsafe: bool
) -> Tuple[BushidoMythos, MythosConfig]:
    if ckpt:
        return load_model(ckpt, device, allow_unsafe=allow_unsafe)
    cfg = _tiny_cfg()
    return BushidoMythos(cfg).to(device), cfg


def _make_batches(
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
        batches.append((x.to(device), y.to(device)))
    return batches


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dynamo_counts() -> Tuple[int, int]:
    if not hasattr(torch, "_dynamo"):
        return 0, 0
    counters = torch._dynamo.utils.counters
    unique_graphs = int(counters.get("stats", {}).get("unique_graphs", 0))
    graph_breaks = sum(int(v) for v in counters.get("graph_break", {}).values())
    return unique_graphs, graph_breaks


def _validate_grouped_linear(device: torch.device) -> dict:
    """Compare native grouped_mm forward/backward with non-square F.linear."""
    generator = torch.Generator(device=device).manual_seed(1729)
    counts = torch.tensor([3, 0, 5], device=device, dtype=torch.int32)
    offsets = counts.cumsum(0).to(dtype=torch.int32)
    x = torch.randn(
        8,
        16,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    weight = torch.randn(
        3,
        24,
        16,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    output = _GroupedLinear.apply(x, weight, offsets)
    boundaries = [0] + offsets.tolist()
    reference = torch.cat(
        [
            F.linear(
                x_ref[boundaries[group] : boundaries[group + 1]],
                weight_ref[group],
            )
            for group in range(weight.shape[0])
        ],
        dim=0,
    )
    grad_output = torch.randn(
        output.shape, device=device, dtype=output.dtype, generator=generator
    )
    grads = torch.autograd.grad(output, (x, weight), grad_output)
    ref_grads = torch.autograd.grad(reference, (x_ref, weight_ref), grad_output)

    metrics = {
        "output_max_abs_delta": float(
            (output - reference).detach().abs().max()
        ),
        "input_grad_max_abs_delta": float(
            (grads[0] - ref_grads[0]).detach().abs().max()
        ),
        "weight_grad_max_abs_delta": float(
            (grads[1] - ref_grads[1]).detach().abs().max()
        ),
    }
    try:
        torch.testing.assert_close(output, reference, atol=0.05, rtol=0.03)
        torch.testing.assert_close(grads[0], ref_grads[0], atol=0.05, rtol=0.03)
        torch.testing.assert_close(grads[1], ref_grads[1], atol=0.05, rtol=0.03)
    except AssertionError as error:
        raise RuntimeError(
            "native grouped MoE numerical probe failed: " + str(metrics)
        ) from error
    return metrics


def _benchmark_mode(
    model: BushidoMythos,
    cfg: MythosConfig,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_dtype: torch.dtype,
    n_loops: int,
    warmup: int,
    compile_model: bool,
    seed: int,
    compile_backend: str = "inductor",
    ce_chunk_size: int = 0,
) -> BenchResult:
    if compile_model:
        if not hasattr(torch, "compile") or not torch._dynamo.is_dynamo_supported():
            raise RuntimeError("torch.compile is not supported by this Python/PyTorch build")
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        model = torch.compile(model, backend=compile_backend)

    model.train()
    target = getattr(model, "_orig_mod", model)
    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    total_steps = len(batches)

    def run_step(index: int, x: torch.Tensor, y: torch.Tensor) -> float:
        progress = index / max(total_steps - 1, 1)
        threshold = 0.5 + (0.99 - 0.5) * progress
        ponder_weight = 0.03 * (1.0 - progress)
        target.set_act_curriculum_values(threshold, ponder_weight)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed + index)
        with torch.autocast(
            autocast_device, dtype=amp_dtype, enabled=use_amp
        ):
            if ce_chunk_size > 0:
                hidden = model(x, n_loops=n_loops, return_hidden=True)
                ce_loss = chunked_linear_cross_entropy(
                    hidden, target.head.weight, y, ce_chunk_size
                )
            else:
                logits = model(x, n_loops=n_loops)
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
                )
            loss = ce_loss + model._last_aux_loss
        loss.backward()
        return float(loss.detach())

    for index in range(warmup):
        x, y = batches[index % total_steps]
        run_step(index, x, y)

    _sync(device)
    warmup_graphs, warmup_breaks = (
        _dynamo_counts() if compile_model else (0, 0)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    losses = []
    started = time.perf_counter()
    for index, (x, y) in enumerate(batches):
        losses.append(run_step(index, x, y))
    _sync(device)
    seconds = time.perf_counter() - started

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    unique_graphs, graph_breaks = _dynamo_counts() if compile_model else (0, 0)
    measured_graphs = max(unique_graphs - warmup_graphs, 0)
    measured_breaks = max(graph_breaks - warmup_breaks, 0)
    tokens = len(batches) * batches[0][0].numel()
    return BenchResult(
        mode="compile" if compile_model else "eager",
        seconds=seconds,
        tokens_per_second=tokens / max(seconds, 1e-9),
        peak_memory_mb=peak_mb,
        first_loss=losses[0],
        last_loss=losses[-1],
        warmup_unique_graphs=warmup_graphs,
        warmup_graph_breaks=warmup_breaks,
        measured_unique_graphs=measured_graphs,
        measured_graph_breaks=measured_breaks,
        unique_graphs=unique_graphs,
        graph_breaks=graph_breaks,
        grouped_moe_active=any(
            module.use_grouped_moe
            for module in target.modules()
            if hasattr(module, "use_grouped_moe")
        ),
    )


def _runtime_info(device: torch.device, dtype: torch.dtype) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            gpu=torch.cuda.get_device_name(device),
            compute_capability=f"{props.major}.{props.minor}",
            vram_gb=props.total_memory / (1024**3),
        )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", help="Checkpoint path; omit only with --tiny")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny random model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_loops", type=int, default=8)
    parser.add_argument(
        "--grad_checkpoint", action="store_true",
        help="Enable recurrent-loop gradient checkpointing for both modes",
    )
    parser.add_argument(
        "--compile_backend",
        default="inductor",
        choices=["inductor", "eager", "aot_eager"],
        help="Use eager only for local graph/recompile smoke checks",
    )
    parser.add_argument(
        "--ce_chunk_size",
        type=int,
        default=0,
        help="Checkpoint tied LM-head CE in token chunks; 0 uses full logits",
    )
    parser.add_argument(
        "--grouped_moe",
        action="store_true",
        help="Use native BF16 grouped GEMM for routed experts when available",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json_out")
    parser.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = parser.parse_args()

    if not args.tiny and not args.ckpt:
        parser.error("--ckpt is required unless --tiny is set")
    if min(args.steps, args.batch_size, args.seq_len, args.n_loops) < 1:
        parser.error("--steps, --batch_size, --seq_len, and --n_loops must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.ce_chunk_size < 0:
        parser.error("--ce_chunk_size must be >= 0")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        if not args.tiny:
            raise RuntimeError("CUDA was requested but is not available")
        requested_device = torch.device("cpu")
    device = requested_device
    amp_dtype = _resolve_dtype(device, args.dtype)

    grouped_moe_active, grouped_moe_reason = grouped_moe_runtime_status(
        device, amp_dtype
    )
    if args.grouped_moe:
        print(
            "[grouped_moe] requested=true "
            f"active={str(grouped_moe_active).lower()} reason={grouped_moe_reason}"
        )
        if not grouped_moe_active:
            if args.json_out:
                Path(args.json_out).write_text(
                    json.dumps(
                        {
                            "runtime": _runtime_info(device, amp_dtype),
                            "config": {
                                "grouped_moe": True,
                                "grouped_moe_reason": grouped_moe_reason,
                            },
                            "results": [],
                            "grouped_moe_request_valid": False,
                            "steady_state_valid": False,
                            "error": "grouped_moe_inactive",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            raise RuntimeError(
                "--grouped_moe requested but inactive: " + grouped_moe_reason
            )

    grouped_moe_probe = None
    if args.grouped_moe:
        try:
            grouped_moe_probe = _validate_grouped_linear(device)
        except RuntimeError as error:
            if args.json_out:
                Path(args.json_out).write_text(
                    json.dumps(
                        {
                            "runtime": _runtime_info(device, amp_dtype),
                            "config": {
                                "grouped_moe": True,
                                "grouped_moe_reason": grouped_moe_reason,
                                "grouped_moe_probe": None,
                            },
                            "results": [],
                            "grouped_moe_request_valid": False,
                            "steady_state_valid": False,
                            "error": "grouped_moe_probe_failed",
                            "error_detail": str(error),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            raise

    model0, cfg = _load_benchmark_model(
        None if args.tiny else args.ckpt, device, args.allow_unsafe_checkpoint
    )
    cfg.use_gradient_checkpointing = args.grad_checkpoint
    if args.seq_len > cfg.max_seq_len:
        raise ValueError(
            f"seq_len={args.seq_len} exceeds checkpoint max_seq_len={cfg.max_seq_len}"
        )
    if args.n_loops > cfg.max_loop_iters:
        raise ValueError(
            f"n_loops={args.n_loops} exceeds max_loop_iters={cfg.max_loop_iters}"
        )
    initial_state = {k: v.detach().cpu().clone() for k, v in model0.state_dict().items()}
    batches = _make_batches(
        args.steps, args.batch_size, args.seq_len, cfg.vocab_size, device, args.seed
    )
    del model0

    print(json.dumps(_runtime_info(device, amp_dtype), ensure_ascii=False, indent=2))
    results = []
    for compile_model in (False, True):
        model = BushidoMythos(cfg).to(device)
        model.load_state_dict(initial_state)
        model.set_grouped_moe(args.grouped_moe, amp_dtype)
        result = _benchmark_mode(
            model, cfg, batches, device, amp_dtype, args.n_loops,
            args.warmup, compile_model, args.seed, args.compile_backend,
            args.ce_chunk_size,
        )
        results.append(result)
        print(
            f"{result.mode:>7}: {result.seconds:.3f}s  "
            f"{result.tokens_per_second:.1f} tok/s  "
            f"peak={result.peak_memory_mb:.1f} MiB  "
            f"graphs={result.unique_graphs} "
            f"(warmup={result.warmup_unique_graphs}, measured={result.measured_unique_graphs})  "
            f"breaks={result.graph_breaks} "
            f"(warmup={result.warmup_graph_breaks}, measured={result.measured_graph_breaks})  "
            f"grouped_moe_active={str(result.grouped_moe_active).lower()}"
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eager, compiled = results
    speedup = eager.seconds / max(compiled.seconds, 1e-9)
    max_loss_delta = max(
        abs(eager.first_loss - compiled.first_loss),
        abs(eager.last_loss - compiled.last_loss),
    )
    print(f"speedup={speedup:.3f}x  max_loss_delta={max_loss_delta:.6g}")

    payload = {
        "runtime": _runtime_info(device, amp_dtype),
        "config": {
            "checkpoint": args.ckpt,
            "steps": args.steps,
            "warmup": args.warmup,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_loops": args.n_loops,
            "grad_checkpoint": args.grad_checkpoint,
            "compile_backend": args.compile_backend,
            "ce_chunk_size": args.ce_chunk_size,
            "grouped_moe": args.grouped_moe,
            "grouped_moe_reason": (
                grouped_moe_reason if args.grouped_moe else "not_requested"
            ),
            "grouped_moe_probe": grouped_moe_probe,
            "seed": args.seed,
        },
        "results": [asdict(result) for result in results],
        "grouped_moe_request_valid": (
            not args.grouped_moe
            or all(result.grouped_moe_active for result in results)
        ),
        "steady_state_valid": (
            all(result.measured_unique_graphs == 0 for result in results)
            and (
                not args.grouped_moe
                or all(result.grouped_moe_active for result in results)
            )
        ),
        "speedup": speedup,
        "max_loss_delta": max_loss_delta,
    }
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
