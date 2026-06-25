import torch
import pytest

from bushido_mythos.moda import (
    MoDAConfig,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_emb,
    DeepSeekMoE,
    MoDAAttention,
    MoDABlock,
    MoDAModel,
)


def test_moda_config_defaults():
    cfg = MoDAConfig()
    assert cfg.vocab_size == 32000
    assert cfg.d_model == 2048


def test_rmsnorm():
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_rotary_embedding():
    dim = 64
    max_seq_len = 128
    rope = RotaryEmbedding(dim, max_seq_len)
    # x shape should be [B, H, T, d] = [2, 8, 10, 64]
    x = torch.randn(2, 8, 10, dim)
    cos, sin = rope(10)
    
    assert cos.shape == (1, 1, 10, dim)
    assert sin.shape == (1, 1, 10, dim)
    
    out = apply_rotary_emb(x, cos, sin)
    assert out.shape == x.shape


def test_deepseek_moe_forward():
    cfg = MoDAConfig(
        d_model=64,
        n_shared_experts=2,
        n_routed_experts=8,
        n_activated_experts=2,
        expert_hidden_dim=32,
        moe_balance_alpha=0.01,
    )
    moe = DeepSeekMoE(cfg)
    x = torch.randn(2, 10, 64)
    out, aux_loss = moe(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    # Check the balance loss is a float tensor
    assert aux_loss is not None
    assert isinstance(aux_loss.item(), float)


def test_moda_attention_forward():
    cfg = MoDAConfig(
        d_model=64,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=16,
    )
    # create a rope first
    rope = RotaryEmbedding(16, 128)
    attn = MoDAAttention(cfg)
    x = torch.randn(2, 10, 64)
    cos, sin = rope(10)
    
    depth_k_cache = []
    depth_v_cache = []
    
    # Simulate past layers in kv cache for depth attention
    for layer in range(3):
        # K, V shape for cross-layer depth attention is expected to match internal depth-write shape
        # Let's use the actual block to get the correct shape or just [B, T, n_heads_kv, head_dim]
        depth_k_cache.append(torch.randn(2, 2, 10, 16))
        depth_v_cache.append(torch.randn(2, 2, 10, 16))

    out = attn(x, depth_k_cache, depth_v_cache, cos, sin)
    assert out.shape == x.shape
    
    # Run with empty cache (start of inference)
    out2 = attn(x, [], [], cos, sin)
    assert out2.shape == x.shape


def test_transformer_block():
    cfg = MoDAConfig(
        d_model=64,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=16,
        n_shared_experts=2,
        n_routed_experts=8,
        n_activated_experts=2,
        expert_hidden_dim=32,
    )
    block = MoDABlock(cfg)
    rope = RotaryEmbedding(16, 128)
    x = torch.randn(2, 10, 64)
    cos, sin = rope(10)
    
    depth_k_cache = []
    depth_v_cache = []
    out, k_out, v_out, aux_loss = block(x, depth_k_cache, depth_v_cache, cos, sin)
    assert out.shape == x.shape


def test_moda_language_model():
    cfg = MoDAConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=16,
        n_shared_experts=2,
        n_routed_experts=4,
        n_activated_experts=2,
        expert_hidden_dim=32,
    )
    model = MoDAModel(cfg)
    
    input_ids = torch.randint(0, 100, (2, 10))
    logits, model_aux_loss = model(input_ids)
    
    assert logits.shape == (2, 10, 100)
    
    if model_aux_loss is not None:
        assert isinstance(model_aux_loss.item(), float)
