#!/usr/bin/env python3
"""Benchmark expanded-KV SDPA against PyTorch native GQA SDPA on CUDA."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos.main import (
    _gqa_scaled_dot_product_attention,
    native_gqa_sdpa_runtime_status,
)


def _dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _run_step(q, k, v, grad_output, use_native_gqa: bool) -> torch.Tensor:
    for tensor in (q, k, v):
        tensor.grad = None
    output = _gqa_scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        use_native_gqa=use_native_gqa,
    )
    output.backward(grad_output)
    return output


def _validate(q, k, v, grad_output) -> dict:
    legacy = _run_step(q, k, v, grad_output, use_native_gqa=False)
    legacy_grads = tuple(tensor.grad.detach().clone() for tensor in (q, k, v))
    native = _run_step(q, k, v, grad_output, use_native_gqa=True)
    native_grads = tuple(tensor.grad.detach().clone() for tensor in (q, k, v))
    return {
        "output_max_abs_delta": float(
            (native.detach() - legacy.detach()).abs().max().float().cpu()
        ),
        "q_grad_max_abs_delta": float(
            (native_grads[0] - legacy_grads[0]).abs().max().float().cpu()
        ),
        "k_grad_max_abs_delta": float(
            (native_grads[1] - legacy_grads[1]).abs().max().float().cpu()
        ),
        "v_grad_max_abs_delta": float(
            (native_grads[2] - legacy_grads[2]).abs().max().float().cpu()
        ),
    }


def _benchmark(q, k, v, grad_output, use_native_gqa, warmup, steps) -> dict:
    for _ in range(warmup):
        _run_step(q, k, v, grad_output, use_native_gqa)
    torch.cuda.synchronize(q.device)
    torch.cuda.reset_peak_memory_stats(q.device)
    started = time.perf_counter()
    for _ in range(steps):
        _run_step(q, k, v, grad_output, use_native_gqa)
    torch.cuda.synchronize(q.device)
    seconds = time.perf_counter() - started
    tokens = steps * q.shape[0] * q.shape[-2]
    return {
        "seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(q.device) / 1024**2,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json_out", type=Path)
    args = parser.parse_args()

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native GQA benchmark requires a CUDA device")
    if args.n_heads % args.n_kv_heads:
        parser.error("--n_heads must be divisible by --n_kv_heads")
    for name in ("batch_size", "seq_len", "n_heads", "n_kv_heads", "head_dim"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be at least 0")
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    device = torch.device("cuda")
    active, reason = native_gqa_sdpa_runtime_status(device)
    print(f"[native_gqa] requested=true active={str(active).lower()} reason={reason}")
    if not active:
        raise RuntimeError("native GQA requested but inactive: " + reason)

    dtype = _dtype_from_name(args.dtype, device)
    torch.manual_seed(args.seed)
    shape_q = (args.batch_size, args.n_heads, args.seq_len, args.head_dim)
    shape_kv = (args.batch_size, args.n_kv_heads, args.seq_len, args.head_dim)
    q = torch.randn(shape_q, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape_kv, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape_kv, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(shape_q, device=device, dtype=dtype)

    validation = _validate(q, k, v, grad_output)
    legacy = _benchmark(q, k, v, grad_output, False, args.warmup, args.steps)
    native = _benchmark(q, k, v, grad_output, True, args.warmup, args.steps)
    speedup = native["tokens_per_second"] / legacy["tokens_per_second"]
    payload = {
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(device)
            ),
            "dtype": str(dtype).removeprefix("torch."),
        },
        "config": {
            key: getattr(args, key)
            for key in (
                "batch_size", "seq_len", "n_heads", "n_kv_heads", "head_dim",
                "warmup", "steps", "seed",
            )
        },
        "native_gqa_active": active,
        "native_gqa_reason": reason,
        "validation": validation,
        "results": {"legacy": legacy, "native": native},
        "speedup": speedup,
    }

    print(
        f" legacy: {legacy['seconds']:.3f}s  "
        f"{legacy['tokens_per_second']:.1f} tok/s  "
        f"peak={legacy['peak_allocated_mib']:.1f} MiB"
    )
    print(
        f" native: {native['seconds']:.3f}s  "
        f"{native['tokens_per_second']:.1f} tok/s  "
        f"peak={native['peak_allocated_mib']:.1f} MiB  "
        "native_gqa_active=true"
    )
    print(
        f"speedup={speedup:.3f}x  "
        f"output_delta={validation['output_max_abs_delta']:.6g}  "
        f"max_grad_delta={max(validation[key] for key in validation if 'grad' in key):.6g}"
    )
    if args.json_out:
        _write_json_atomic(args.json_out, payload)
        print(f"JSON: {args.json_out}")


if __name__ == "__main__":
    main()
