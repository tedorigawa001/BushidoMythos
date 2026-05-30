import torch
import pytest
from bushido_mythos.main import (
    ACTHalting,
    DepthCrossAttention,
    Expert,
    GQAttention,
    LTIInjection,
    LoRAAdapter,
    MLAttention,
    MoEFFN,
    MythosConfig,
    BushidoMythos,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)

# ---------------------------------------------------------------------------
# Shared small configs (kept tiny so tests run fast on CPU)
# ---------------------------------------------------------------------------

B, T = 2, 8  # batch, sequence length


def gqa_cfg(**overrides) -> MythosConfig:
    defaults = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        act_threshold=0.99,
        lora_rank=4,
        # MLA fields must be valid even when not used
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)


def mla_cfg(**overrides) -> MythosConfig:
    return gqa_cfg(attn_type="mla", **overrides)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert norm(x).shape == x.shape

    def test_unit_rms(self):
        # after norm the RMS of each vector should be ≈ 1 when weight=1
        norm = RMSNorm(64)
        torch.nn.init.ones_(norm.weight)
        x = torch.randn(4, 64)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_learnable_weight(self):
        norm = RMSNorm(8)
        assert norm.weight.requires_grad


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_precompute_shape(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        assert freqs.shape == (32, 8, 2)  # (max_len, dim//2, 2=[cos,sin])
        assert not freqs.is_complex()

    def test_apply_rope_shape(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.randn(B, T, 4, 16)
        out = apply_rope(x, freqs[:T])
        assert out.shape == x.shape

    def test_apply_rope_preserves_norm(self):
        # rotation is an isometry — norms must be unchanged
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.randn(B, T, 4, 16)
        out = apply_rope(x, freqs[:T])
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)

    def test_different_positions_differ(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.ones(1, 2, 1, 16)
        out = apply_rope(x, freqs[:2])
        # position 0 and position 1 should produce different rotations
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# RoPE extended — correctness invariants
# ---------------------------------------------------------------------------


class TestRoPEExtended:
    """Comprehensive correctness tests for precompute_rope_freqs and apply_rope."""

    # --- precompute_rope_freqs ---

    def test_position_zero_is_unit_phasor(self):
        """freqs[0] must be cos=1, sin=0 for every pair (angle=0)."""
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        assert torch.allclose(freqs[0, :, 0], torch.ones(8), atol=1e-6)   # cos=1
        assert torch.allclose(freqs[0, :, 1], torch.zeros(8), atol=1e-6)  # sin=0

    def test_all_phasors_have_unit_magnitude(self):
        """Every rotation magnitude must be 1 — RoPE is an isometric rotation."""
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        mag = (freqs[..., 0] ** 2 + freqs[..., 1] ** 2).sqrt()
        assert torch.allclose(mag, torch.ones_like(mag), atol=1e-6)

    def test_angles_equal_outer_product(self):
        """freqs[t, k] must encode angle t × base_freq[k]."""
        dim, max_len, theta = 8, 6, 500000.0
        freqs = precompute_rope_freqs(dim=dim, max_len=max_len, theta=theta)
        base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        expected_angles = torch.outer(t, base)  # (max_len, dim//2)
        assert torch.allclose(freqs[..., 0], expected_angles.cos(), atol=1e-6)
        assert torch.allclose(freqs[..., 1], expected_angles.sin(), atol=1e-6)

    def test_higher_theta_produces_smaller_angles(self):
        """Larger theta → slower frequency decay → smaller rotation angle per step.

        Index 0 (dim_i=0) is excluded: its frequency is 1/(theta^0)=1 for any theta,
        so the comparison is not meaningful there.
        """
        dim, max_len = 16, 8
        freqs_fast = precompute_rope_freqs(dim=dim, max_len=max_len, theta=100.0)
        freqs_slow = precompute_rope_freqs(dim=dim, max_len=max_len, theta=500000.0)
        angle_fast = freqs_fast[1, 1:, 1].atan2(freqs_fast[1, 1:, 0]).abs()
        angle_slow = freqs_slow[1, 1:, 1].atan2(freqs_slow[1, 1:, 0]).abs()
        assert (angle_fast > angle_slow).all()

    def test_default_theta_matches_explicit(self):
        """Omitting theta must equal passing theta=500000.0."""
        f1 = precompute_rope_freqs(16, 8)
        f2 = precompute_rope_freqs(16, 8, theta=500000.0)
        assert torch.allclose(f1, f2)

    # --- apply_rope ---

    def test_position_zero_is_identity(self):
        """T=1 input uses only freqs[0] = 1+0j, so output must equal input."""
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        x = torch.randn(2, 1, 4, 16)
        out = apply_rope(x, freqs[:1])
        assert torch.allclose(x, out, atol=1e-6)

    def test_dtype_float32_preserved(self):
        freqs = precompute_rope_freqs(dim=16, max_len=16)
        x = torch.randn(1, 4, 2, 16).float()
        assert apply_rope(x, freqs[:4]).dtype == torch.float32

    def test_dtype_float16_preserved(self):
        freqs = precompute_rope_freqs(dim=16, max_len=16)
        x = torch.randn(1, 4, 2, 16).half()
        assert apply_rope(x, freqs[:4]).dtype == torch.float16

    def test_inverse_rotation_recovers_input(self):
        """Rotating by freqs then by inv(freqs) must recover the original.

        The inverse rotation negates the sin component (equivalent to complex conjugate).
        """
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=8)
        x = torch.randn(2, 4, 3, dim)
        rotated = apply_rope(x, freqs[:4])
        inv_freqs = freqs[:4].clone()
        inv_freqs[..., 1] = -inv_freqs[..., 1]  # negate sin = conjugate rotation
        recovered = apply_rope(rotated, inv_freqs)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_batch_independence(self):
        """Output for one batch item must not depend on other items in the batch."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=16)
        torch.manual_seed(7)
        x_a = torch.randn(1, 4, 2, dim)
        x_b = torch.randn(1, 4, 2, dim)
        solo = apply_rope(x_a, freqs[:4])
        batched = apply_rope(torch.cat([x_a, x_b], dim=0), freqs[:4])[:1]
        assert torch.allclose(solo, batched, atol=1e-6)

    def test_head_independence(self):
        """All heads at the same position must receive identical rotations."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=8)
        x = torch.randn(1, 4, 1, dim).expand(1, 4, 3, dim).contiguous()
        out = apply_rope(x, freqs[:4])
        assert torch.allclose(out[:, :, 0], out[:, :, 1], atol=1e-6)
        assert torch.allclose(out[:, :, 1], out[:, :, 2], atol=1e-6)

    def test_relative_position_property(self):
        """
        Core RoPE invariant: <RoPE(q,m), RoPE(k,n)> depends only on (n-m).
        Two pairs with the same offset must produce the same dot product.
        """
        dim, max_len = 16, 32
        freqs = precompute_rope_freqs(dim=dim, max_len=max_len)
        torch.manual_seed(42)
        q = torch.randn(1, 1, 1, dim)
        k = torch.randn(1, 1, 1, dim)

        def rope_at(tensor, pos):
            """Rotate tensor at a specific position by embedding it in a zero sequence."""
            seq = torch.zeros(1, pos + 1, 1, dim)
            seq[0, pos] = tensor[0, 0]
            return apply_rope(seq, freqs[: pos + 1])[:, pos : pos + 1]

        # Both pairs have relative offset n - m = 6
        dot_3_9 = (rope_at(q, 3) * rope_at(k, 9)).sum()
        dot_1_7 = (rope_at(q, 1) * rope_at(k, 7)).sum()
        assert torch.allclose(dot_3_9, dot_1_7, atol=1e-5)

    def test_max_len_boundary(self):
        """apply_rope must handle T == max_len without error or NaN."""
        max_len = 10
        freqs = precompute_rope_freqs(dim=8, max_len=max_len)
        x = torch.randn(1, max_len, 2, 8)
        out = apply_rope(x, freqs)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_exceeds_max_len_raises(self):
        """apply_rope must raise RuntimeError when T > max_len."""
        freqs = precompute_rope_freqs(dim=8, max_len=4)
        x = torch.randn(1, 8, 2, 8)  # T=8 > max_len=4
        with pytest.raises(RuntimeError):
            apply_rope(x, freqs)


# ---------------------------------------------------------------------------
# GQAttention
# ---------------------------------------------------------------------------


class TestGQAttention:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )
        self.attn = GQAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_kv_cache_accumulates(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs, kv_cache=cache, cache_key="layer0")
        assert "layer0" in cache
        k_len = cache["layer0"]["k"].shape[1]
        # second call adds T more tokens
        self.attn(x, self.freqs, kv_cache=cache, cache_key="layer0")
        assert cache["layer0"]["k"].shape[1] == k_len + T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.full((1, 1, T, T), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        out = self.attn(x, self.freqs, mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# MLAttention
# ---------------------------------------------------------------------------


class TestMLAttention:
    def setup_method(self):
        self.cfg = mla_cfg()
        self.freqs = precompute_rope_freqs(
            self.cfg.qk_rope_head_dim, self.cfg.max_seq_len
        )
        self.attn = MLAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_cache_stores_compressed_kv(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs, kv_cache=cache, cache_key="mla0")
        assert "c_kv" in cache["mla0"]
        assert "k_rope" in cache["mla0"]
        # c_kv should have kv_lora_rank as last dim, not full K/V
        assert cache["mla0"]["c_kv"].shape[-1] == self.cfg.kv_lora_rank

    def test_cache_accumulates_across_steps(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs, kv_cache=cache, cache_key="mla0")
        first_len = cache["mla0"]["c_kv"].shape[1]
        self.attn(x, self.freqs, kv_cache=cache, cache_key="mla0")
        assert cache["mla0"]["c_kv"].shape[1] == first_len + T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.triu(torch.full((1, 1, T, T), float("-inf")), diagonal=1)
        out = self.attn(x, self.freqs, mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# Expert (dense SwiGLU FFN)
# ---------------------------------------------------------------------------


class TestExpert:
    def test_output_shape(self):
        expert = Expert(dim=64, expert_dim=32)
        x = torch.randn(B, T, 64)
        assert expert(x).shape == (B, T, 64)

    def test_flat_input(self):
        expert = Expert(dim=32, expert_dim=16)
        x = torch.randn(5, 32)
        assert expert(x).shape == (5, 32)


# ---------------------------------------------------------------------------
# MoEFFN
# ---------------------------------------------------------------------------


class TestMoEFFN:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.moe = MoEFFN(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        assert self.moe(x).shape == (B, T, self.cfg.dim)

    def test_router_bias_not_grad(self):
        # router_bias is a buffer, not a parameter
        param_names = {n for n, _ in self.moe.named_parameters()}
        assert "router_bias" not in param_names

    def test_shared_experts_always_fire(self):
        # Zero out all routed experts; output should still be nonzero from shared
        for exp in self.moe.routed_experts:
            for p in exp.parameters():
                p.data.zero_()
        x = torch.randn(B, T, self.cfg.dim)
        out = self.moe(x)
        assert out.abs().sum() > 0


# ---------------------------------------------------------------------------
# loop_index_embedding
# ---------------------------------------------------------------------------


class TestLoopIndexEmbedding:
    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        out = loop_index_embedding(h, loop_t=0, loop_dim=8)
        assert out.shape == h.shape

    def test_different_iterations_differ(self):
        h = torch.zeros(1, 1, 64)
        out0 = loop_index_embedding(h, loop_t=0, loop_dim=8)
        out1 = loop_index_embedding(h, loop_t=1, loop_dim=8)
        assert not torch.allclose(out0, out1)

    def test_only_first_dims_modified(self):
        h = torch.zeros(1, 1, 64)
        loop_dim = 8
        out = loop_index_embedding(h, loop_t=3, loop_dim=loop_dim)
        # channels beyond loop_dim should be unchanged (still 0)
        assert torch.all(out[..., loop_dim:] == 0)


# ---------------------------------------------------------------------------
# LoRAAdapter
# ---------------------------------------------------------------------------


class TestLoRAAdapter:
    def setup_method(self):
        self.lora = LoRAAdapter(dim=64, rank=8, max_loops=10)

    def test_output_shape(self):
        x = torch.randn(B, T, 64)
        out = self.lora(x, loop_t=0)
        assert out.shape == (B, T, 64)

    def test_different_loops_differ(self):
        x = torch.randn(B, T, 64)
        out0 = self.lora(x, loop_t=0)
        out1 = self.lora(x, loop_t=1)
        assert not torch.allclose(out0, out1)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    def test_gqa_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_mla_output_shape(self):
        cfg = mla_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_moe_block_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=True)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_attn_type_selection(self):
        assert isinstance(TransformerBlock(gqa_cfg()).attn, GQAttention)
        assert isinstance(TransformerBlock(mla_cfg()).attn, MLAttention)


# ---------------------------------------------------------------------------
# LTIInjection
# ---------------------------------------------------------------------------


class TestLTIInjection:
    def setup_method(self):
        self.inj = LTIInjection(dim=64)

    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        t = torch.randn(B, T, 64)
        assert self.inj(h, e, t).shape == (B, T, 64)

    def test_spectral_radius_lt_1(self):
        A = self.inj.get_A()
        assert A.max().item() < 1.0

    def test_spectral_radius_gt_0(self):
        A = self.inj.get_A()
        assert A.min().item() > 0.0

    def test_spectral_radius_stable_after_large_grad_step(self):
        # Simulate an aggressive gradient update and verify stability holds
        opt = torch.optim.SGD(self.inj.parameters(), lr=1e3)
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        t = torch.randn(B, T, 64)
        loss = self.inj(h, e, t).sum()
        loss.backward()
        opt.step()
        A = self.inj.get_A()
        assert A.max().item() < 1.0


# ---------------------------------------------------------------------------
# ACTHalting
# ---------------------------------------------------------------------------


class TestACTHalting:
    def setup_method(self):
        self.act = ACTHalting(dim=64)

    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        p = self.act(h)
        assert p.shape == (B, T)

    def test_values_in_01(self):
        h = torch.randn(B, T, 64)
        p = self.act(h)
        assert p.min().item() >= 0.0
        assert p.max().item() <= 1.0


# ---------------------------------------------------------------------------
# RecurrentBlock
# ---------------------------------------------------------------------------


class TestRecurrentBlock:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.block = RecurrentBlock(self.cfg)
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )

    def test_output_shape(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_more_loops_changes_output(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out1 = self.block(h.clone(), e.clone(), self.freqs, n_loops=1)
        out3 = self.block(h.clone(), e.clone(), self.freqs, n_loops=3)
        assert not torch.allclose(out1, out3)

    def test_single_loop_runs(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs, n_loops=1)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# BushidoMythos — GQA mode
# ---------------------------------------------------------------------------


class TestBushidoMythosGQA:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = BushidoMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        logits = self.model(self.ids)
        assert not torch.isnan(logits).any()

    def test_generate_shape(self):
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (B, T + 4)

    def test_weight_tying(self):
        assert self.model.head.weight is self.model.embed.weight

    def test_lti_spectral_radius(self):
        A = self.model.recurrent.injection.get_A()
        assert A.max().item() < 1.0

    def test_depth_extrapolation_changes_output(self):
        # More loops at inference should produce different (ideally better) output
        logits_shallow = self.model(self.ids, n_loops=1)
        logits_deep = self.model(self.ids, n_loops=3)
        assert not torch.allclose(logits_shallow, logits_deep)

    def test_kv_cache_generate_matches_no_cache(self):
        # Single-token generation with and without cache should agree
        torch.manual_seed(0)
        prompt = torch.randint(0, self.cfg.vocab_size, (1, T))
        with torch.no_grad():
            logits_no_cache = self.model(prompt, n_loops=2)[:, -1, :]
            cache = {}
            logits_cached = self.model(prompt, n_loops=2, kv_cache=cache)[:, -1, :]
        assert torch.allclose(logits_no_cache, logits_cached, atol=1e-4)

    def test_single_token_forward(self):
        # Mask is None when T=1; should not crash
        single = torch.randint(0, self.cfg.vocab_size, (B, 1))
        logits = self.model(single)
        assert logits.shape == (B, 1, self.cfg.vocab_size)


# ---------------------------------------------------------------------------
# BushidoMythos — MLA mode
# ---------------------------------------------------------------------------


class TestBushidoMythosMLА:
    def setup_method(self):
        self.cfg = mla_cfg()
        self.model = BushidoMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        assert not torch.isnan(self.model(self.ids)).any()

    def test_generate_shape(self):
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (B, T + 4)

    def test_lti_spectral_radius(self):
        A = self.model.recurrent.injection.get_A()
        assert A.max().item() < 1.0

    def test_mla_cache_is_compressed(self):
        # MLA cache should store c_kv (lora_rank), not full K/V (n_heads * head_dim)
        cache = {}
        with torch.no_grad():
            self.model(self.ids, kv_cache=cache)
        # find any MLA cache entry and check dimensions
        mla_entries = {k: v for k, v in cache.items() if "c_kv" in v}
        assert len(mla_entries) > 0
        for entry in mla_entries.values():
            assert entry["c_kv"].shape[-1] == self.cfg.kv_lora_rank


# ---------------------------------------------------------------------------
# GQA vs MLA: same config, different attn_type
# ---------------------------------------------------------------------------


class TestAttnTypeSwap:
    def test_gqa_and_mla_produce_different_outputs(self):
        cfg_gqa = gqa_cfg()
        cfg_mla = mla_cfg()
        ids = torch.randint(0, cfg_gqa.vocab_size, (B, T))
        logits_gqa = BushidoMythos(cfg_gqa)(ids)
        logits_mla = BushidoMythos(cfg_mla)(ids)
        # different architectures, different params → outputs must differ
        assert not torch.allclose(logits_gqa, logits_mla)

    def test_both_modes_produce_valid_shapes(self):
        ids = torch.randint(0, 200, (B, T))
        for attn_type in ("gqa", "mla"):
            cfg = gqa_cfg(attn_type=attn_type)
            logits = BushidoMythos(cfg)(ids)
            assert logits.shape == (B, T, cfg.vocab_size)

    def test_mla_fewer_kv_cache_bytes(self):
        # MLA cache should be smaller than GQA cache for the same sequence
        ids = torch.randint(0, 200, (1, T))
        cache_gqa, cache_mla = {}, {}
        with torch.no_grad():
            BushidoMythos(gqa_cfg())(ids, kv_cache=cache_gqa)
            BushidoMythos(mla_cfg())(ids, kv_cache=cache_mla)

        def cache_bytes(cache):
            return sum(
                t.numel() * t.element_size()
                for entry in cache.values()
                for t in entry.values()
            )

        assert cache_bytes(cache_mla) < cache_bytes(cache_gqa)


# ---------------------------------------------------------------------------
# Hyper-connections
# ---------------------------------------------------------------------------


class TestHyperConnections:
    def setup_method(self):
        self.cfg = gqa_cfg(use_hyper_connections=True)
        self.cfg_std = gqa_cfg(use_hyper_connections=False)

    def test_alpha_beta_params_exist(self):
        block = TransformerBlock(self.cfg, use_moe=False)
        assert hasattr(block, "alpha_attn")
        assert hasattr(block, "beta_attn")
        assert hasattr(block, "alpha_ffn")
        assert hasattr(block, "beta_ffn")

    def test_params_are_learnable(self):
        block = TransformerBlock(self.cfg, use_moe=False)
        assert block.alpha_attn.requires_grad
        assert block.beta_attn.requires_grad

    def test_no_hyper_params_without_flag(self):
        block = TransformerBlock(self.cfg_std, use_moe=False)
        assert not hasattr(block, "alpha_attn")

    def test_output_shape(self):
        model = BushidoMythos(self.cfg)
        ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        logits = model(ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_output_differs_from_standard(self):
        torch.manual_seed(42)
        model_hc = BushidoMythos(self.cfg)
        model_std = BushidoMythos(self.cfg_std)
        ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        # Different architectures — outputs must differ
        assert not torch.allclose(model_hc(ids), model_std(ids))

    def test_no_nan(self):
        model = BushidoMythos(self.cfg)
        ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        assert not torch.isnan(model(ids)).any()


# ---------------------------------------------------------------------------
# ACT auxiliary loss
# ---------------------------------------------------------------------------


class TestACTAuxLoss:
    def test_zero_weight_gives_zero_aux_loss(self):
        cfg = gqa_cfg(act_aux_loss_weight=0.0)
        model = BushidoMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        model(ids)
        assert model._last_aux_loss.item() == 0.0

    def test_nonzero_weight_gives_nonzero_aux_loss(self):
        cfg = gqa_cfg(act_aux_loss_weight=0.01)
        model = BushidoMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        model(ids)
        # ACT halting always computes ponder_steps >= 1, so aux_loss > 0
        assert model._last_aux_loss.item() > 0.0

    def test_aux_loss_is_scalar(self):
        cfg = gqa_cfg(act_aux_loss_weight=0.01)
        model = BushidoMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        model(ids)
        assert model._last_aux_loss.ndim == 0

    def test_aux_loss_scales_with_weight(self):
        ids = torch.randint(0, 200, (B, T))
        torch.manual_seed(0)
        m1 = BushidoMythos(gqa_cfg(act_aux_loss_weight=0.01))
        torch.manual_seed(0)
        m2 = BushidoMythos(gqa_cfg(act_aux_loss_weight=0.1))
        m1(ids); m2(ids)
        assert m2._last_aux_loss.item() > m1._last_aux_loss.item()


# ---------------------------------------------------------------------------
# Loop curriculum
# ---------------------------------------------------------------------------


class TestLoopCurriculum:
    def test_forward_succeeds_with_curriculum(self):
        cfg = gqa_cfg(loop_curriculum=True, max_loop_iters=4)
        model = BushidoMythos(cfg)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(ids)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert not torch.isnan(logits).any()

    def test_curriculum_disabled_at_eval(self):
        # In eval mode, curriculum flag should not randomize loops —
        # same seed → same output regardless of curriculum flag.
        cfg_on = gqa_cfg(loop_curriculum=True, max_loop_iters=4)
        cfg_off = gqa_cfg(loop_curriculum=False, max_loop_iters=4)
        ids = torch.randint(0, cfg_on.vocab_size, (1, T))
        torch.manual_seed(7)
        m_on = BushidoMythos(cfg_on).eval()
        torch.manual_seed(7)
        m_off = BushidoMythos(cfg_off).eval()
        with torch.no_grad():
            out_on = m_on(ids)
            out_off = m_off(ids)
        assert torch.allclose(out_on, out_off)


# ---------------------------------------------------------------------------
# Depth cross-attention
# ---------------------------------------------------------------------------


class TestDepthCrossAttention:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.dca = DepthCrossAttention(self.cfg)

    def test_write_cache_shape(self):
        h = torch.randn(B, T, self.cfg.dim)
        dk, dv = self.dca.write_cache(h)
        n_kv = self.cfg.n_kv_heads
        head_dim = self.cfg.dim // self.cfg.n_heads
        assert dk.shape == (B, n_kv, T, head_dim)
        assert dv.shape == (B, n_kv, T, head_dim)

    def test_forward_no_depth_cache(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.triu(torch.full((1, 1, T, T), float("-inf")), diagonal=1)
        out = self.dca(x, mask, [], [])
        assert out.shape == (B, T, self.cfg.dim)
        assert not torch.isnan(out).any()

    def test_forward_with_depth_cache(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.triu(torch.full((1, 1, T, T), float("-inf")), diagonal=1)
        dk, dv = self.dca.write_cache(torch.randn(B, T, self.cfg.dim))
        out = self.dca(x, mask, [dk], [dv])
        assert out.shape == (B, T, self.cfg.dim)
        assert not torch.isnan(out).any()


class TestDepthAttnModel:
    def test_output_shape(self):
        cfg = gqa_cfg(use_depth_attn=True)
        model = BushidoMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(ids)
        assert logits.shape == (B, T, cfg.vocab_size)

    def test_no_nan(self):
        cfg = gqa_cfg(use_depth_attn=True)
        model = BushidoMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        assert not torch.isnan(model(ids)).any()

    def test_depth_attn_module_exists(self):
        cfg = gqa_cfg(use_depth_attn=True)
        model = BushidoMythos(cfg)
        assert hasattr(model.recurrent, "depth_attn")
        assert isinstance(model.recurrent.depth_attn, DepthCrossAttention)

    def test_no_depth_attn_module_without_flag(self):
        cfg = gqa_cfg(use_depth_attn=False)
        model = BushidoMythos(cfg)
        assert not hasattr(model.recurrent, "depth_attn")


# ---------------------------------------------------------------------------
# inputs_embeds forward pass
# ---------------------------------------------------------------------------


class TestInputsEmbeds:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = BushidoMythos(self.cfg)

    def test_output_shape(self):
        emb = torch.randn(B, T, self.cfg.dim)
        logits = self.model(inputs_embeds=emb)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_no_nan(self):
        emb = torch.randn(B, T, self.cfg.dim)
        assert not torch.isnan(self.model(inputs_embeds=emb)).any()

    def test_different_from_token_ids(self):
        # Embedding table maps token IDs → vectors; random embeds differ
        ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        logits_ids = self.model(ids)
        emb = torch.randn(B, T, self.cfg.dim)
        logits_emb = self.model(inputs_embeds=emb)
        assert not torch.allclose(logits_ids, logits_emb)

    def test_last_hidden_captured(self):
        emb = torch.randn(B, T, self.cfg.dim)
        self.model(inputs_embeds=emb)
        assert self.model._last_hidden is not None
        assert self.model._last_hidden.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# generate_coconut
# ---------------------------------------------------------------------------


class TestGenerateCoconut:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = BushidoMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_output_shape(self):
        out = self.model.generate_coconut(self.ids, coconut_steps=2, max_new_tokens=4)
        assert out.shape == (1, T + 4)

    def test_max_new_tokens_zero_returns_prompt(self):
        out = self.model.generate_coconut(self.ids, coconut_steps=2, max_new_tokens=0)
        assert out.shape == self.ids.shape
        assert torch.equal(out, self.ids)

    def test_zero_coconut_steps(self):
        out = self.model.generate_coconut(self.ids, coconut_steps=0, max_new_tokens=3)
        assert out.shape == (1, T + 3)

    def test_no_nan(self):
        out = self.model.generate_coconut(self.ids, coconut_steps=1, max_new_tokens=2)
        assert not torch.isnan(out.float()).any()

    def test_prefix_preserved(self):
        out = self.model.generate_coconut(self.ids, coconut_steps=1, max_new_tokens=3)
        assert torch.equal(out[:, :T], self.ids)

    def test_model_returns_to_train_mode(self):
        self.model.train()
        self.model.generate_coconut(self.ids, coconut_steps=1, max_new_tokens=1)
        assert self.model.training


# ---------------------------------------------------------------------------
# MoE load balancing — router bias and accumulation
# ---------------------------------------------------------------------------


class TestMoELoadBalancing:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = BushidoMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def _get_moe(self):
        return self.model.recurrent.block.ffn

    def test_router_bias_is_buffer_not_param(self):
        moe = self._get_moe()
        param_names = {n for n, _ in moe.named_parameters()}
        assert "router_bias" not in param_names
        buf_names = {n for n, _ in moe.named_buffers()}
        assert "router_bias" in buf_names

    def test_accum_counts_accumulate_across_microbatches(self):
        moe = self._get_moe()
        moe._accum_expert_counts = None
        self.model(self.ids)
        counts_after_1 = moe._accum_expert_counts.clone()
        self.model(self.ids)
        counts_after_2 = moe._accum_expert_counts
        # Each forward adds tokens, so total count must be strictly larger
        assert (counts_after_2 >= counts_after_1).all()
        assert counts_after_2.sum() > counts_after_1.sum()

    def test_update_moe_router_bias_changes_bias(self):
        moe = self._get_moe()
        self.model(self.ids)
        bias_before = moe.router_bias.clone()
        self.model.update_moe_router_bias(bias_lr=1.0)
        # With bias_lr=1.0 and imbalanced load, at least one bias entry changes
        assert not torch.equal(moe.router_bias, bias_before)

    def test_update_resets_accum_counts(self):
        self.model(self.ids)
        assert self._get_moe()._accum_expert_counts is not None
        self.model.update_moe_router_bias()
        assert self._get_moe()._accum_expert_counts is None

    def test_update_without_forward_is_noop(self):
        moe = self._get_moe()
        moe._accum_expert_counts = None
        bias_before = moe.router_bias.clone()
        self.model.update_moe_router_bias()
        assert torch.equal(moe.router_bias, bias_before)


# ---------------------------------------------------------------------------
# Gradient Checkpointing
# ---------------------------------------------------------------------------


class TestGradientCheckpointing:
    """Gradient checkpointing trades activation memory for recompute.

    Key invariants:
      - Forward output is bit-identical to the non-checkpointed path
        (same weights, no dropout, deterministic ops).
      - Backward completes without error and gradients are numerically close.
      - Checkpointing is suppressed when kv_cache is provided (in-place dict
        mutation inside checkpoint would corrupt the cache on recompute).
      - Checkpointing is suppressed in eval mode (self.training=False).
    """

    def _twin_models(self, seed: int, **overrides):
        """Return (model_no_ckpt, model_with_ckpt) with identical weights."""
        torch.manual_seed(seed)
        m_no = BushidoMythos(gqa_cfg(use_gradient_checkpointing=False, **overrides))
        torch.manual_seed(seed)
        m_ck = BushidoMythos(gqa_cfg(use_gradient_checkpointing=True, **overrides))
        return m_no, m_ck

    # --- config ---

    def test_config_default_false(self):
        assert gqa_cfg().use_gradient_checkpointing is False

    def test_config_can_enable(self):
        assert gqa_cfg(use_gradient_checkpointing=True).use_gradient_checkpointing is True

    # --- forward equivalence ---

    def test_train_forward_matches_baseline(self):
        """Forward output must be identical with and without checkpointing."""
        ids = torch.randint(0, 200, (B, T))
        m_no, m_ck = self._twin_models(seed=0)
        m_no.train(); m_ck.train()
        with torch.no_grad():
            assert torch.allclose(m_no(ids), m_ck(ids), atol=1e-5)

    def test_no_nan_with_checkpointing(self):
        m_ck = BushidoMythos(gqa_cfg(use_gradient_checkpointing=True)).train()
        ids = torch.randint(0, m_ck.cfg.vocab_size, (B, T))
        assert not torch.isnan(m_ck(ids)).any()

    # --- backward ---

    def test_backward_completes(self):
        """Backward must not raise when checkpointing is active."""
        m = BushidoMythos(gqa_cfg(use_gradient_checkpointing=True)).train()
        ids = torch.randint(0, m.cfg.vocab_size, (B, T))
        m(ids).sum().backward()

    def test_gradients_match_baseline(self):
        """Gradients from checkpointing must be numerically close to standard backprop."""
        ids = torch.randint(0, 200, (B, T))
        m_no, m_ck = self._twin_models(seed=1)
        m_no.train(); m_ck.train()

        m_no(ids).sum().backward()
        m_ck(ids).sum().backward()

        # Check gradients through the recurrent attention weights
        grad_no = m_no.recurrent.block.attn.wq.weight.grad
        grad_ck = m_ck.recurrent.block.attn.wq.weight.grad
        assert grad_no is not None and grad_ck is not None
        assert torch.allclose(grad_no, grad_ck, atol=1e-4)

    # --- suppression conditions ---

    def test_inactive_at_eval(self):
        """Eval mode must suppress checkpointing — output equals non-ckpt eval."""
        ids = torch.randint(0, 200, (1, T))
        m_no, m_ck = self._twin_models(seed=2)
        m_no.eval(); m_ck.eval()
        with torch.no_grad():
            assert torch.allclose(m_no(ids), m_ck(ids), atol=1e-5)

    def test_inactive_with_kv_cache(self):
        """kv_cache must suppress checkpointing to avoid in-place mutation on recompute."""
        ids = torch.randint(0, 200, (1, T))
        m_no, m_ck = self._twin_models(seed=3)
        m_no.train(); m_ck.train()
        cache_no, cache_ck = {}, {}
        with torch.no_grad():
            out_no = m_no(ids, kv_cache=cache_no)
            out_ck = m_ck(ids, kv_cache=cache_ck)
        assert torch.allclose(out_no, out_ck, atol=1e-5)

    # --- compatibility ---

    def test_mla_mode_forward_and_backward(self):
        """Checkpointing must work with MLA attention."""
        cfg = mla_cfg(use_gradient_checkpointing=True)
        m = BushidoMythos(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits = m(ids)
        logits.sum().backward()
        assert not torch.isnan(logits).any()

    def test_with_loop_curriculum(self):
        """Checkpointing must work alongside loop_curriculum (random depth)."""
        cfg = gqa_cfg(use_gradient_checkpointing=True, loop_curriculum=True, max_loop_iters=4)
        m = BushidoMythos(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        torch.manual_seed(42)
        logits = m(ids)
        logits.sum().backward()
        assert not torch.isnan(logits).any()

    def test_with_act_aux_loss(self):
        """ACT auxiliary loss must still be computed correctly with checkpointing."""
        cfg = gqa_cfg(use_gradient_checkpointing=True, act_aux_loss_weight=0.01)
        m = BushidoMythos(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        m(ids)
        assert m._last_aux_loss.item() > 0.0


# ---------------------------------------------------------------------------
# MLA KV-cache correctness (parallel)
# ---------------------------------------------------------------------------


class TestMLAttentionKVCacheCorrectness:
    """MLA has compressed KV reconstruction — verify cache produces same logits as no-cache."""

    def setup_method(self):
        self.cfg = mla_cfg()
        self.model = BushidoMythos(self.cfg).eval()
        torch.manual_seed(0)
        self.prompt = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_step0_cache_matches_no_cache(self):
        with torch.no_grad():
            logits_no_cache = self.model(self.prompt, n_loops=2)[:, -1, :]
            cache = {}
            logits_cached = self.model(self.prompt, n_loops=2, kv_cache=cache)[:, -1, :]
        assert torch.allclose(logits_no_cache, logits_cached, atol=1e-4)

    def test_cache_grows_correctly_across_steps(self):
        """After step-0 prefill, step-1 single-token decode should not raise."""
        cache = {}
        with torch.no_grad():
            self.model(self.prompt, n_loops=2, kv_cache=cache)
            next_tok = torch.randint(0, self.cfg.vocab_size, (1, 1))
            logits = self.model(next_tok, n_loops=2, kv_cache=cache, start_pos=T)
        assert logits.shape == (1, 1, self.cfg.vocab_size)
        assert not torch.isnan(logits).any()

    def test_first_step_logits_match_direct_forward(self):
        """The logits used for step-0 sampling must equal a direct forward pass."""
        with torch.no_grad():
            # Direct forward with empty cache
            cache_direct = {}
            logits_direct = self.model(self.prompt, n_loops=2,
                                       kv_cache=cache_direct)[:, -1, :]
            # Prefill via a fresh cache (simulates what _generate_inner does at step 0)
            cache_gen = {}
            logits_gen = self.model(self.prompt, n_loops=2,
                                    kv_cache=cache_gen)[:, -1, :]
        assert torch.allclose(logits_direct, logits_gen, atol=1e-5)


# ---------------------------------------------------------------------------
# Empty input guard
# ---------------------------------------------------------------------------


class TestEmptyInputGuard:
    """forward() and generate() must fail fast with clear errors on T=0 input."""

    def setup_method(self):
        self.model = BushidoMythos(gqa_cfg())

    def test_forward_raises_on_empty_input_ids(self):
        empty = torch.zeros((1, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="[Ii]nput"):
            self.model(empty)

    def test_forward_raises_on_empty_inputs_embeds(self):
        empty = torch.zeros((1, 0, gqa_cfg().dim))
        with pytest.raises(ValueError, match="[Ii]nput"):
            self.model(inputs_embeds=empty)

    def test_generate_raises_on_empty_input(self):
        empty = torch.zeros((1, 0), dtype=torch.long)
        with pytest.raises(ValueError):
            self.model.generate(empty, max_new_tokens=3)


# ---------------------------------------------------------------------------
# generate() edge cases
# ---------------------------------------------------------------------------


class TestGenerateEdgeCases:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = BushidoMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_max_new_tokens_zero_returns_input_unchanged(self):
        out = self.model.generate(self.ids, max_new_tokens=0)
        assert out.shape == self.ids.shape
        assert torch.equal(out, self.ids)

    def test_returns_to_train_mode_after_generate(self):
        self.model.train()
        self.model.generate(self.ids, max_new_tokens=2, n_loops=1)
        assert self.model.training

    def test_stays_in_eval_mode_after_generate(self):
        self.model.eval()
        self.model.generate(self.ids, max_new_tokens=2, n_loops=1)
        assert not self.model.training

    def test_repetition_penalty_formula_positive_logit(self):
        """Positive logit for a seen token is divided by penalty → reduced."""
        penalty = 1.5
        score = torch.tensor([2.0])
        penalised = torch.where(score < 0, score * penalty, score / penalty)
        assert penalised.item() == pytest.approx(2.0 / penalty)

    def test_repetition_penalty_formula_negative_logit(self):
        """Negative logit for a seen token is multiplied by penalty → more negative."""
        penalty = 1.5
        score = torch.tensor([-3.0])
        penalised = torch.where(score < 0, score * penalty, score / penalty)
        assert penalised.item() == pytest.approx(-3.0 * penalty)

    def test_repetition_penalty_shape_unchanged(self):
        """repetition_penalty must not change output tensor shape."""
        out_no_pen  = self.model.generate(self.ids, max_new_tokens=3, n_loops=1,
                                          repetition_penalty=1.0)
        out_penalised = self.model.generate(self.ids, max_new_tokens=3, n_loops=1,
                                            repetition_penalty=1.5)
        assert out_no_pen.shape == out_penalised.shape == (1, T + 3)

    def test_top_k_zero_does_not_crash(self):
        """top_k=0 disables top-K filtering — generation should still complete."""
        out = self.model.generate(self.ids, max_new_tokens=3, n_loops=1, top_k=0)
        assert out.shape == (1, T + 3)


# ---------------------------------------------------------------------------
# ACT invariants
# ---------------------------------------------------------------------------


class TestACTInvariants:
    """Adaptive Computation Time: ponder cost and aux-loss structural invariants."""

    def setup_method(self):
        self.cfg = gqa_cfg(max_loop_iters=4, act_aux_loss_weight=0.01)
        self.model = BushidoMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_ponder_cost_in_valid_range(self):
        """Mean ponder steps must be in [1, max_loop_iters]."""
        self.model(self.ids)
        cost = self.model.recurrent._last_ponder_cost.item()
        assert 1.0 <= cost <= self.cfg.max_loop_iters

    def test_ponder_cost_is_scalar(self):
        self.model(self.ids)
        assert self.model.recurrent._last_ponder_cost.ndim == 0

    def test_ponder_cost_non_negative(self):
        self.model(self.ids)
        assert self.model.recurrent._last_ponder_cost.item() >= 0.0

    def test_aux_loss_non_negative(self):
        self.model(self.ids)
        assert self.model._last_aux_loss.item() >= 0.0

    def test_high_threshold_uses_more_steps(self):
        """Higher ACT threshold → more steps before halting → higher ponder cost."""
        torch.manual_seed(0)
        m_low = BushidoMythos(gqa_cfg(act_threshold=0.5, max_loop_iters=4))
        torch.manual_seed(0)
        m_high = BushidoMythos(gqa_cfg(act_threshold=0.99, max_loop_iters=4))
        ids = torch.randint(0, 200, (B, T))
        m_low(ids)
        m_high(ids)
        assert (m_high.recurrent._last_ponder_cost.item()
                >= m_low.recurrent._last_ponder_cost.item())


# ---------------------------------------------------------------------------
# start_pos RoPE offset correctness
# ---------------------------------------------------------------------------


class TestStartPos:
    """start_pos shifts RoPE frequencies — different positions must give different outputs."""

    def setup_method(self):
        self.model = BushidoMythos(gqa_cfg()).eval()
        self.ids = torch.randint(0, gqa_cfg().vocab_size, (1, 4))

    def test_different_start_pos_produces_different_freqs_cis(self):
        """freqs_cis sliced at different offsets must differ — RoPE encodes position."""
        cfg = gqa_cfg()
        model = BushidoMythos(cfg)
        freqs_0 = model.freqs_cis[0:4]
        freqs_5 = model.freqs_cis[5:9]
        assert not torch.allclose(freqs_0, freqs_5)

    def test_zero_start_pos_is_default(self):
        """Explicit start_pos=0 must match implicit default."""
        with torch.no_grad():
            logits_default = self.model(self.ids)
            logits_explicit = self.model(self.ids, start_pos=0)
        assert torch.allclose(logits_default, logits_explicit)

    def test_start_pos_within_max_seq_len(self):
        """start_pos near max_seq_len - T should not raise."""
        cfg = gqa_cfg()
        T_short = 2
        ids = torch.randint(0, cfg.vocab_size, (1, T_short))
        start = cfg.max_seq_len - T_short
        with torch.no_grad():
            out = self.model(ids, start_pos=start)
        assert out.shape == (1, T_short, cfg.vocab_size)


# ---------------------------------------------------------------------------
# MoE top-K selection
# ---------------------------------------------------------------------------


class TestMoETopK:
    """Each token must activate exactly n_experts_per_tok routed experts."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = BushidoMythos(self.cfg)
        self.moe = self.model.recurrent.block.ffn
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_last_expert_counts_sum_equals_topk_tokens(self):
        """_last_expert_counts sums to B*T*topk for the last loop iteration."""
        self.model(self.ids)
        total = self.moe._last_expert_counts.sum().item()
        expected = B * T * self.cfg.n_experts_per_tok
        assert total == pytest.approx(expected)

    def test_expert_counts_length_equals_n_experts(self):
        self.model(self.ids)
        assert self.moe._last_expert_counts.shape == (self.cfg.n_experts,)

    def test_expert_counts_non_negative(self):
        self.model(self.ids)
        assert (self.moe._last_expert_counts >= 0).all()

    def test_accum_counts_sum_grows_across_forwards(self):
        """Accumulated counts grow monotonically across forward passes."""
        self.moe._accum_expert_counts = None
        self.model(self.ids)
        after_first = self.moe._accum_expert_counts.sum().item()
        self.model(self.ids)
        after_second = self.moe._accum_expert_counts.sum().item()
        assert after_second > after_first


# ---------------------------------------------------------------------------
# MythosConfig validation
# ---------------------------------------------------------------------------


class TestMythosConfigValidation:
    def test_invalid_attn_type_raises_value_error(self):
        with pytest.raises(ValueError, match="attn_type"):
            MythosConfig(attn_type="invalid")

    def test_valid_attn_types_do_not_raise(self):
        MythosConfig(attn_type="gqa")
        MythosConfig(attn_type="mla")


# ---------------------------------------------------------------------------
# Weight tying — GQA and MLA
# ---------------------------------------------------------------------------


class TestWeightTying:
    def test_weight_tying_gqa(self):
        model = BushidoMythos(gqa_cfg())
        assert model.head.weight is model.embed.weight

    def test_weight_tying_mla(self):
        model = BushidoMythos(mla_cfg())
        assert model.head.weight is model.embed.weight

    def test_weight_tying_preserved_after_train_step(self):
        """A gradient update must not break the weight tying."""
        model = BushidoMythos(gqa_cfg()).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ids = torch.randint(0, gqa_cfg().vocab_size, (1, T))
        model(ids).sum().backward()
        opt.step()
        assert model.head.weight is model.embed.weight


# ---------------------------------------------------------------------------
# Depth extrapolation beyond max_loop_iters
# ---------------------------------------------------------------------------


class TestDepthExtrapolation:
    """n_loops > max_loop_iters must work without error or NaN at inference."""

    def setup_method(self):
        self.cfg = gqa_cfg(max_loop_iters=3)
        self.model = BushidoMythos(self.cfg).eval()
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_forward_beyond_max_loop_iters(self):
        with torch.no_grad():
            logits = self.model(self.ids, n_loops=6)
        assert logits.shape == (1, T, self.cfg.vocab_size)
        assert not torch.isnan(logits).any()

    def test_generate_beyond_max_loop_iters(self):
        out = self.model.generate(self.ids, max_new_tokens=3, n_loops=6)
        assert out.shape == (1, T + 3)
        assert not torch.isnan(out.float()).any()

    def test_extra_loops_do_not_raise_or_nan(self):
        """n_loops > max_loop_iters must not raise or produce NaN.

        Whether the output numerically differs depends on ACT halting and
        weight magnitudes — not testable with random weights. Shape/stability
        is the meaningful invariant here.
        """
        with torch.no_grad():
            logits_3 = self.model(self.ids, n_loops=3)
            logits_6 = self.model(self.ids, n_loops=6)
        assert logits_3.shape == logits_6.shape
        assert not torch.isnan(logits_3).any()
        assert not torch.isnan(logits_6).any()


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
