#!/usr/bin/env python3
"""Benchmark batch-1 chat generation with legacy and native grouped MoE."""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from chat import apply_grouped_moe, load_model


@dataclass
class ChatBenchResult:
    mode: str
    seconds: float
    generated_tokens: int
    tokens_per_second: float
    peak_memory_mb: float
    grouped_moe_active: bool
    output_ids: list[int]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_mode(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    compute_dtype: torch.dtype,
    n_loops: int,
    max_new_tokens: int,
    warmup: int,
    repeats: int,
    grouped: bool,
) -> ChatBenchResult:
    model.set_grouped_moe(False, compute_dtype)
    if grouped:
        apply_grouped_moe(model, True, device, compute_dtype)

    use_amp = device.type == "cuda" and compute_dtype != torch.float32

    def generate_once() -> torch.Tensor:
        torch.manual_seed(0)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=compute_dtype, enabled=use_amp
        ):
            return model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                n_loops=n_loops,
                temperature=1.0,
                top_k=1,
                repetition_penalty=1.0,
                eos_token_id=None,
            )

    for _ in range(warmup):
        generate_once()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    output = input_ids
    started = time.perf_counter()
    for _ in range(repeats):
        output = generate_once()
    _sync(device)
    seconds = time.perf_counter() - started
    generated = repeats * (output.shape[1] - input_ids.shape[1])
    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    active = any(
        bool(getattr(module, "use_grouped_moe", False))
        for module in model.modules()
    )
    return ChatBenchResult(
        mode="grouped" if grouped else "legacy",
        seconds=seconds,
        generated_tokens=generated,
        tokens_per_second=generated / max(seconds, 1e-9),
        peak_memory_mb=peak_mb,
        grouped_moe_active=active,
        output_ids=output[0].tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--n_loops", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json_out")
    parser.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = parser.parse_args()

    if min(
        args.prompt_len,
        args.max_new_tokens,
        args.n_loops,
        args.repeats,
    ) < 1:
        parser.error("prompt_len, max_new_tokens, n_loops, and repeats must be >= 1")
    if args.warmup < 0:
        parser.error("warmup must be >= 0")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("grouped chat benchmark requires CUDA")
    compute_dtype = torch.bfloat16
    model, cfg = load_model(
        args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint
    )
    if args.prompt_len + args.max_new_tokens > cfg.max_seq_len:
        raise ValueError(
            "prompt_len + max_new_tokens exceeds checkpoint max_seq_len "
            f"({args.prompt_len} + {args.max_new_tokens} > {cfg.max_seq_len})"
        )

    generator = torch.Generator().manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        cfg.vocab_size,
        (1, args.prompt_len),
        generator=generator,
    ).to(device)

    results = []
    for grouped in (False, True):
        result = _run_mode(
            model,
            input_ids,
            device,
            compute_dtype,
            args.n_loops,
            args.max_new_tokens,
            args.warmup,
            args.repeats,
            grouped,
        )
        results.append(result)
        print(
            f"{result.mode:>7}: {result.seconds:.3f}s  "
            f"{result.tokens_per_second:.2f} generated tok/s  "
            f"peak={result.peak_memory_mb:.1f} MiB  "
            f"grouped_moe_active={str(result.grouped_moe_active).lower()}"
        )

    legacy, grouped = results
    speedup = legacy.seconds / max(grouped.seconds, 1e-9)
    outputs_match = legacy.output_ids == grouped.output_ids
    print(f"speedup={speedup:.3f}x  outputs_match={str(outputs_match).lower()}")
    payload = {
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "dtype": "bfloat16",
        },
        "config": {
            "checkpoint": args.ckpt,
            "batch_size": 1,
            "prompt_len": args.prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "n_loops": args.n_loops,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "results": [asdict(result) for result in results],
        "speedup": speedup,
        "outputs_match": outputs_match,
        "grouped_moe_request_valid": grouped.grouped_moe_active,
    }
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
