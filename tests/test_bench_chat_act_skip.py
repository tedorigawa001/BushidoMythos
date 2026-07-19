import torch

from bushido_mythos import BushidoMythos, MythosConfig
from training.bench_chat_act_skip import _run_mode


def test_act_skip_chat_benchmark_cpu_smoke_and_exact_output():
    cfg = MythosConfig(
        vocab_size=64,
        dim=16,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=16,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=8,
        lora_rank=2,
    )
    model = BushidoMythos(cfg).eval()
    with torch.no_grad():
        model.recurrent.act.halt.weight.zero_()
        model.recurrent.act.halt.bias.fill_(20.0)
    input_ids = torch.tensor([[1, 2]])
    kwargs = dict(
        model=model,
        input_ids=input_ids,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        n_loops=3,
        max_new_tokens=1,
        warmup=0,
        repeats=1,
    )
    legacy = _run_mode(enable_skip=False, **kwargs)
    skipped = _run_mode(enable_skip=True, **kwargs)

    assert legacy.output_ids == skipped.output_ids
    assert legacy.executed_loops == 3
    assert legacy.cache_only_loops == 0
    assert skipped.executed_loops == 1
    assert skipped.cache_only_loops == 2
    assert skipped.compute_skip_fraction == 2 / 3
