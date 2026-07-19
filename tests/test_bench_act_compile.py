import torch
import pytest

from training.bench_act_compile import (
    _benchmark_mode,
    _make_batches,
    _resolve_dtype,
    _tiny_cfg,
)
from bushido_mythos import BushidoMythos


def test_resolve_dtype_uses_float32_on_cpu_for_auto():
    assert _resolve_dtype(torch.device("cpu"), "auto") == torch.float32


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
    assert result.first_loss > 0
