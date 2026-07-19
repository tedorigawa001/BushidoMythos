import torch

from bushido_mythos import BushidoMythos, MythosConfig
from training.bench_chat_grouped_moe import _run_mode


def test_legacy_chat_benchmark_cpu_smoke():
    cfg = MythosConfig(
        vocab_size=64,
        dim=16,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=16,
        max_loop_iters=1,
        prelude_layers=1,
        coda_layers=1,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=8,
        lora_rank=2,
    )
    model = BushidoMythos(cfg)
    input_ids = torch.tensor([[1, 2]])
    result = _run_mode(
        model=model,
        input_ids=input_ids,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        n_loops=1,
        max_new_tokens=1,
        warmup=0,
        repeats=1,
        grouped=False,
    )
    assert result.mode == "legacy"
    assert result.generated_tokens == 1
    assert result.tokens_per_second > 0
    assert result.grouped_moe_active is False
