import torch
import pytest

from training.bench_act_compile import (
    _benchmark_mode,
    _make_batches,
    _resolve_dtype,
    _tiny_cfg,
    _validate_grouped_linear,
)
from bushido_mythos import BushidoMythos
from bushido_mythos import main as mythos_main


def test_resolve_dtype_uses_float32_on_cpu_for_auto():
    assert _resolve_dtype(torch.device("cpu"), "auto") == torch.float32


def test_grouped_linear_probe_uses_non_square_weights(monkeypatch):
    def fake_grouped_mm(a, b, *, offs):
        if offs.dtype != torch.int32:
            raise RuntimeError("Offsets have to be int32")
        if b.ndim == 2 and a.stride(0) * a.element_size() % 16 != 0:
            raise RuntimeError("strides should be multiple of 16 bytes")
        boundaries = [0] + offs.tolist()
        if b.ndim == 3:
            return torch.cat(
                [
                    a[boundaries[e] : boundaries[e + 1]] @ b[e]
                    for e in range(b.shape[0])
                ]
            )
        return torch.stack(
            [
                a[:, boundaries[e] : boundaries[e + 1]]
                @ b[boundaries[e] : boundaries[e + 1]]
                for e in range(len(boundaries) - 1)
            ]
        )

    monkeypatch.setattr(mythos_main, "_NATIVE_GROUPED_MM", fake_grouped_mm)
    metrics = _validate_grouped_linear(torch.device("cpu"))
    assert metrics["output_max_abs_delta"] == 0.0
    assert metrics["input_grad_max_abs_delta"] == 0.0
    assert metrics["weight_grad_max_abs_delta"] == 0.0


def test_eager_benchmark_cpu_smoke():
    cfg = _tiny_cfg()
    device = torch.device("cpu")
    model = BushidoMythos(cfg)
    batches = _make_batches(2, 1, 4, cfg.vocab_size, device, seed=0)

    result = _benchmark_mode(
        model=model,
        cfg=cfg,
        batches=batches,
        device=device,
        amp_dtype=torch.float32,
        n_loops=1,
        warmup=0,
        compile_model=False,
        seed=0,
    )

    assert result.mode == "eager"
    assert result.seconds > 0
    assert result.tokens_per_second > 0
    assert result.first_loss > 0
    assert result.warmup_unique_graphs == 0
    assert result.measured_unique_graphs == 0
    assert result.grouped_moe_active is False


def test_chunked_ce_benchmark_cpu_smoke():
    cfg = _tiny_cfg()
    device = torch.device("cpu")
    model = BushidoMythos(cfg)
    batches = _make_batches(1, 1, 4, cfg.vocab_size, device, seed=0)

    result = _benchmark_mode(
        model=model,
        cfg=cfg,
        batches=batches,
        device=device,
        amp_dtype=torch.float32,
        n_loops=1,
        warmup=0,
        compile_model=False,
        seed=0,
        ce_chunk_size=2,
    )

    assert result.mode == "eager"
    assert result.first_loss > 0


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile (Dynamo) is unavailable",
)
def test_compile_eager_backend_cpu_smoke():
    cfg = _tiny_cfg()
    device = torch.device("cpu")
    model = BushidoMythos(cfg)
    batches = _make_batches(1, 1, 4, cfg.vocab_size, device, seed=0)

    result = _benchmark_mode(
        model=model,
        cfg=cfg,
        batches=batches,
        device=device,
        amp_dtype=torch.float32,
        n_loops=1,
        warmup=1,
        compile_model=True,
        seed=0,
        compile_backend="eager",
    )

    assert result.mode == "compile"
    assert result.unique_graphs > 0
    assert result.grouped_moe_active is False
    assert result.unique_graphs == (
        result.warmup_unique_graphs + result.measured_unique_graphs
    )
    assert result.graph_breaks == (
        result.warmup_graph_breaks + result.measured_graph_breaks
    )
    assert result.first_loss > 0
