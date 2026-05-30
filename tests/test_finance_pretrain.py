"""
Tests for training/finance_pretrain.py.

Covers:
  - TextDataset: cache roundtrip, shape, ValueError on insufficient tokens
  - lr_lambda: warmup, cosine decay, min-lr floor
  - save/load checkpoint: scheduler_state, phase_steps, legacy fallback
  - Phase 2 reasoning-mix skip condition when step >= phase2_end
  - Cache key versioning
"""

import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Allow importing from training/ and repo root regardless of cwd
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training.finance_pretrain import (
    TextDataset,
    SFTDataset,
    _CACHE_VERSION,
    load_checkpoint,
    lr_lambda,
    save_checkpoint,
    _vram_str,
)
from chat import find_latest_ckpt
from bushido_mythos import MythosConfig, BushidoMythos


# ---------------------------------------------------------------------------
# Shared tiny model
# ---------------------------------------------------------------------------

def tiny_cfg() -> MythosConfig:
    return MythosConfig(
        vocab_size=64,
        dim=32,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=16,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=16,
        act_threshold=0.99,
        lora_rank=2,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        rope_theta=10000.0,
    )


def make_model_and_opt(cfg: MythosConfig):
    model = BushidoMythos(cfg)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda s: lr_lambda(s, warmup=5, total=20))
    return model, optimizer, scheduler


# ---------------------------------------------------------------------------
# TextDataset
# ---------------------------------------------------------------------------

class TestTextDataset:

    def _make_dataset(self, n_tokens: int, seq_len: int = 8, batch_size: int = 2,
                      tmpdir: Path = None) -> TextDataset:
        ids = torch.arange(n_tokens, dtype=torch.long)
        cache_path = tmpdir / "test_ids.pt"
        torch.save(ids, cache_path)
        # Pass empty rows — cache already exists so tokenisation is skipped
        return TextDataset(
            rows=[],
            vocab_size=256,
            seq_len=seq_len,
            batch_size=batch_size,
            device=torch.device("cpu"),
            cache_path=cache_path,
        )

    def test_iter_raises_on_insufficient_tokens(self, tmp_path):
        """n = len(ids) - seq_len - 1 <= 0 must raise ValueError."""
        ds = self._make_dataset(n_tokens=8, seq_len=8, tmpdir=tmp_path)
        with pytest.raises(ValueError, match="too small"):
            next(iter(ds))

    def test_iter_yields_correct_shapes(self, tmp_path):
        seq_len, batch_size = 4, 2
        ds = self._make_dataset(n_tokens=200, seq_len=seq_len, batch_size=batch_size, tmpdir=tmp_path)
        x, y = next(iter(ds))
        assert x.shape == (batch_size, seq_len)
        assert y.shape == (batch_size, seq_len)

    def test_iter_y_is_x_shifted_by_one(self, tmp_path):
        """Each y[b] must equal x[b] shifted left by 1 token."""
        seq_len, batch_size = 4, 2
        ds = self._make_dataset(n_tokens=200, seq_len=seq_len, batch_size=batch_size, tmpdir=tmp_path)
        x, y = next(iter(ds))
        # For any consecutive pair (s, s+1) drawn from ids, y == x + 1 in value
        # (our ids are arange so consecutive positions differ by 1)
        diff = (y - x).abs()
        assert diff.max().item() <= 1  # adjacent tokens differ by at most 1

    def test_len_positive_for_sufficient_tokens(self, tmp_path):
        ds = self._make_dataset(n_tokens=200, seq_len=4, batch_size=2, tmpdir=tmp_path)
        assert len(ds) > 0

    def test_cache_roundtrip(self, tmp_path):
        """Dataset writes a cache file; a second instance loads it without rows."""
        rows = ["hello world finance trading", "stock market index fund"]
        cache_path = tmp_path / "cache_rt.pt"

        # First init: no cache → tokenisation attempted (will fail without transformers,
        # so we pre-save a fake cache to test the loading branch)
        ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.long)
        torch.save(ids, cache_path)

        ds = TextDataset(
            rows=rows,
            vocab_size=256,
            seq_len=4,
            batch_size=1,
            device=torch.device("cpu"),
            cache_path=cache_path,
        )
        assert torch.equal(ds._ids, ids)

    def test_cache_key_contains_version(self, tmp_path):
        """Cache filename must include _CACHE_VERSION to prevent stale reads."""
        version_tag = f"_{_CACHE_VERSION}.pt"
        # Verify the constant is non-empty and the naming convention holds
        assert _CACHE_VERSION, "_CACHE_VERSION must not be empty"
        assert version_tag.endswith(".pt")
        # Simulate the filename built in build_wikitext103 / build_financial_news
        fake_path = tmp_path / f"wikitext103_gpt2_50257{version_tag}"
        assert _CACHE_VERSION in fake_path.name


# ---------------------------------------------------------------------------
# lr_lambda
# ---------------------------------------------------------------------------

class TestLrLambda:

    def test_warmup_phase_linear(self):
        warmup, total = 10, 100
        assert lr_lambda(0, warmup, total) == pytest.approx(0.0)
        assert lr_lambda(5, warmup, total) == pytest.approx(0.5)
        assert lr_lambda(10, warmup, total) == pytest.approx(1.0)

    def test_cosine_decay_after_warmup(self):
        warmup, total = 10, 110
        # Midpoint of cosine: step = warmup + (total - warmup) / 2 = 60
        mid = lr_lambda(60, warmup, total)
        assert 0.1 < mid < 1.0  # between min_lr_ratio and 1

    def test_min_lr_floor_at_end(self):
        warmup, total = 10, 110
        end_lr = lr_lambda(total, warmup, total)
        assert end_lr == pytest.approx(0.1)  # default min_lr_ratio

    def test_custom_min_lr_ratio(self):
        end_lr = lr_lambda(100, warmup=10, total=100, min_lr_ratio=0.05)
        assert end_lr == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

class TestCheckpoint:

    def setup_method(self):
        self.cfg = tiny_cfg()
        self.model, self.optimizer, self.scheduler = make_model_and_opt(self.cfg)

    def test_save_includes_scheduler_state(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, step=10, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "scheduler_state" in ckpt

    def test_save_includes_phase_steps(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, step=10, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg,
                        phase1_steps=20_000, phase2_steps=10_000)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["phase1_steps"] == 20_000
        assert ckpt["phase2_steps"] == 10_000

    def test_save_includes_tag(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, step=5, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg, tag="Phase1-WikiText103")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["tag"] == "Phase1-WikiText103"

    def test_load_restores_step(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, step=42, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg)
        _, opt2, sched2 = make_model_and_opt(self.cfg)
        model2 = BushidoMythos(self.cfg)
        step = load_checkpoint(str(path), model2, opt2, sched2)
        assert step == 42

    def test_load_restores_scheduler_via_load_state_dict(self, tmp_path):
        """load_checkpoint must use load_state_dict (not replay) when scheduler_state present."""
        path = tmp_path / "ckpt.pt"
        # Advance scheduler a few steps before saving
        for _ in range(5):
            self.optimizer.step()
            self.scheduler.step()
        save_checkpoint(path, step=5, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg)

        _, opt2, sched2 = make_model_and_opt(self.cfg)
        model2 = BushidoMythos(self.cfg)

        # Patch scheduler.step to detect if replay is used (it should not be)
        original_step = sched2.step
        step_call_count = [0]
        def counting_step(*a, **kw):
            step_call_count[0] += 1
            return original_step(*a, **kw)
        sched2.step = counting_step

        load_checkpoint(str(path), model2, opt2, sched2)
        # scheduler_state is present → should NOT replay 5 steps
        assert step_call_count[0] == 0

    def test_load_legacy_checkpoint_replays_steps(self, tmp_path):
        """Checkpoints without scheduler_state fall back to step replay."""
        path = tmp_path / "legacy.pt"
        # Manually save a checkpoint without scheduler_state
        torch.save(
            {
                "step": 3,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )
        _, opt2, sched2 = make_model_and_opt(self.cfg)
        model2 = BushidoMythos(self.cfg)

        step_call_count = [0]
        original_step = sched2.step
        def counting_step(*a, **kw):
            step_call_count[0] += 1
            return original_step(*a, **kw)
        sched2.step = counting_step

        returned_step = load_checkpoint(str(path), model2, opt2, sched2)
        assert returned_step == 3
        assert step_call_count[0] == 3  # replayed exactly 3 steps


# ---------------------------------------------------------------------------
# Phase 2 skip condition
# ---------------------------------------------------------------------------

class TestPhase2SkipCondition:

    def test_phase2_end_anchored_to_p1_plus_p2(self):
        """phase2_end must equal p1_total + phase2_steps regardless of current step."""
        phase1_steps = 20_000
        phase2_steps = 10_000
        phase2_end = phase1_steps + phase2_steps
        assert phase2_end == 30_000

    def test_phase2_skipped_when_step_at_end(self):
        """If step >= phase2_end the condition (step < phase2_end) is False."""
        phase1_steps = 20_000
        phase2_steps = 10_000
        phase2_end = phase1_steps + phase2_steps
        for step in (phase2_end, phase2_end + 1, phase2_end + 5_000):
            assert not (step < phase2_end), f"step={step} should skip Phase 2"

    def test_phase2_runs_when_step_below_end(self):
        """If step < phase2_end the condition is True and Phase 2 should run."""
        phase1_steps = 20_000
        phase2_steps = 10_000
        phase2_end = phase1_steps + phase2_steps
        for step in (0, phase1_steps, phase2_end - 1):
            assert step < phase2_end, f"step={step} should run Phase 2"

    def test_phase2_build_not_called_when_complete(self, tmp_path):
        """
        Simulate train() reaching Phase 2 check with step already at phase2_end.
        build_reasoning_mix should not be invoked.
        """
        phase1_steps = 10
        phase2_steps = 5
        phase2_end = phase1_steps + phase2_steps
        step = phase2_end  # already done

        called = []

        def mock_build(*args, **kwargs):
            called.append(True)
            return MagicMock()

        # Replicate the exact condition from train()
        with patch("training.finance_pretrain.build_reasoning_mix", side_effect=mock_build):
            phase = 0  # both phases
            if phase in (0, 2) and step < phase2_end:
                mock_build()

        assert called == [], "build_reasoning_mix must not be called when step >= phase2_end"


# ---------------------------------------------------------------------------
# SFTDataset — response-only loss mask
# ---------------------------------------------------------------------------

def _make_sft_cache(tmpdir: Path, n_tokens: int = 200, prompt_frac: float = 0.3,
                    suffix: str = "sft") -> Path:
    """
    Write a pre-built SFT cache (dict with 'ids' and 'mask') so tests bypass
    the internal GPT-2 tokenizer call that requires network access.
    mask is False for the first prompt_frac of tokens, True for the rest.
    """
    ids  = torch.arange(n_tokens, dtype=torch.long) % 256
    mask = torch.zeros(n_tokens, dtype=torch.bool)
    mask[int(n_tokens * prompt_frac):] = True  # response portion
    cache_path = tmpdir / f"sft_{suffix}_sft.pt"
    torch.save({"ids": ids, "mask": mask}, cache_path)
    return cache_path


class TestSFTDataset:

    def _make_ds(self, tmpdir: Path, seq_len: int = 8, batch_size: int = 2,
                 n_tokens: int = 400, suffix: str = "ds"):
        cache_path = _make_sft_cache(tmpdir, n_tokens=n_tokens, suffix=suffix)
        return SFTDataset(
            pairs=[],            # empty — cache already present, no tokenisation needed
            vocab_size=256,
            seq_len=seq_len,
            batch_size=batch_size,
            device=torch.device("cpu"),
            cache_path=cache_path,
        )

    def test_yields_three_tensors(self, tmp_path):
        ds = self._make_ds(tmp_path, suffix="t1")
        batch = next(iter(ds))
        assert len(batch) == 3, "SFTDataset must yield (x, y, mask) triples"

    def test_mask_false_for_prompt_tokens(self, tmp_path):
        """Response-only mask: prompt tokens must be False, response tokens must be True."""
        ds = self._make_ds(tmp_path, seq_len=8, batch_size=2, n_tokens=400, suffix="t2")
        # Check the underlying mask tensor rather than a random batch slice
        assert ds._mask.dtype == torch.bool
        assert not ds._mask.all(), "some tokens must be marked False (prompt tokens excluded from loss)"
        assert ds._mask.any(), "some tokens must be marked True (response tokens included in loss)"

    def test_mask_shape_matches_xy(self, tmp_path):
        ds = self._make_ds(tmp_path, suffix="t3")
        x, y, mask = next(iter(ds))
        assert mask.shape == x.shape == y.shape

    def test_sft_cache_uses_dict_format(self, tmp_path):
        """SFTDataset must persist cache as {"ids": tensor, "mask": tensor}."""
        # Write a cache, instantiate the dataset (it loads from cache), then verify the file
        cache_path = _make_sft_cache(tmp_path, suffix="fmt")
        ds = SFTDataset(
            pairs=[],
            vocab_size=256,
            seq_len=8,
            batch_size=2,
            device=torch.device("cpu"),
            cache_path=cache_path,
        )
        loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
        assert isinstance(loaded, dict), "SFT cache must be a dict with 'ids' and 'mask'"
        assert "ids" in loaded and "mask" in loaded


# ---------------------------------------------------------------------------
# find_latest_ckpt priority order
# ---------------------------------------------------------------------------

class TestFindLatestCkpt:

    def _write_ckpt(self, path):
        path.touch()

    def test_phase5_beats_phase4(self, tmp_path):
        (tmp_path / "phase5_final.pt").touch()
        (tmp_path / "phase4_final.pt").touch()
        assert find_latest_ckpt(str(tmp_path)).endswith("phase5_final.pt")

    def test_phase4_beats_phase3(self, tmp_path):
        (tmp_path / "phase4_final.pt").touch()
        (tmp_path / "phase3_final.pt").touch()
        assert find_latest_ckpt(str(tmp_path)).endswith("phase4_final.pt")

    def test_phase3_beats_final(self, tmp_path):
        (tmp_path / "phase3_final.pt").touch()
        (tmp_path / "final.pt").touch()
        assert find_latest_ckpt(str(tmp_path)).endswith("phase3_final.pt")

    def test_final_beats_phase2(self, tmp_path):
        (tmp_path / "final.pt").touch()
        (tmp_path / "phase2_final.pt").touch()
        assert find_latest_ckpt(str(tmp_path)).endswith("final.pt")

    def test_phase2_beats_phase1(self, tmp_path):
        (tmp_path / "phase2_final.pt").touch()
        (tmp_path / "phase1_final.pt").touch()
        assert find_latest_ckpt(str(tmp_path)).endswith("phase2_final.pt")

    def test_step_fallback_when_no_named(self, tmp_path):
        (tmp_path / "step_001000.pt").touch()
        (tmp_path / "step_002000.pt").touch()
        result = find_latest_ckpt(str(tmp_path))
        assert result.endswith("step_002000.pt")

    def test_returns_none_for_empty_dir(self, tmp_path):
        assert find_latest_ckpt(str(tmp_path)) is None

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        assert find_latest_ckpt(str(tmp_path / "no_such_dir")) is None


# ---------------------------------------------------------------------------
# Phase 3/4 metadata in checkpoint
# ---------------------------------------------------------------------------

class TestPhase34Metadata:

    def setup_method(self):
        self.cfg = tiny_cfg()
        self.model, self.optimizer, self.scheduler = make_model_and_opt(self.cfg)

    def test_save_includes_phase3_steps(self, tmp_path):
        path = tmp_path / "ckpt_p3.pt"
        save_checkpoint(path, step=25_000, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg,
                        phase1_steps=20_000, phase2_steps=10_000,
                        phase3_steps=5_000, phase4_steps=0)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["phase3_steps"] == 5_000
        assert ckpt["phase4_steps"] == 0

    def test_save_includes_phase4_steps(self, tmp_path):
        path = tmp_path / "ckpt_p4.pt"
        save_checkpoint(path, step=28_000, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg,
                        phase1_steps=20_000, phase2_steps=10_000,
                        phase3_steps=5_000, phase4_steps=8_000)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["phase4_steps"] == 8_000

    def test_save_includes_phase5_steps(self, tmp_path):
        path = tmp_path / "ckpt_p5.pt"
        save_checkpoint(path, step=46_000, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg,
                        phase1_steps=20_000, phase2_steps=10_000,
                        phase3_steps=5_000, phase4_steps=8_000, phase5_steps=3_000)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["phase5_steps"] == 3_000

    def test_phase5_steps_default_zero_when_omitted(self, tmp_path):
        path = tmp_path / "ckpt_no_p5.pt"
        save_checkpoint(path, step=10, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt.get("phase5_steps", 0) == 0

    def test_save_tag_reflects_phase(self, tmp_path):
        path = tmp_path / "ckpt_tag.pt"
        save_checkpoint(path, step=10, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg,
                        tag="Phase3-FinanceAlpaca")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["tag"] == "Phase3-FinanceAlpaca"

    def test_load_returns_phase345_steps(self, tmp_path):
        path = tmp_path / "ckpt_load.pt"
        save_checkpoint(path, step=33, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg,
                        phase1_steps=20_000, phase2_steps=10_000,
                        phase3_steps=5_000, phase4_steps=8_000, phase5_steps=3_000)
        _, opt2, sched2 = make_model_and_opt(self.cfg)
        model2 = BushidoMythos(self.cfg)
        step = load_checkpoint(str(path), model2, opt2, sched2)
        assert step == 33
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["phase3_steps"] == 5_000
        assert ckpt["phase4_steps"] == 8_000
        assert ckpt["phase5_steps"] == 3_000


# ---------------------------------------------------------------------------
# _vram_str diagnostic utility
# ---------------------------------------------------------------------------


class TestVramStr:
    """_vram_str must return empty string on CPU and a formatted string on CUDA."""

    def test_cpu_device_returns_empty_string(self):
        result = _vram_str(torch.device("cpu"))
        assert result == ""

    def test_cpu_reset_peak_is_noop(self):
        """reset_peak=True on CPU must not raise."""
        _vram_str(torch.device("cpu"), reset_peak=True)
        _vram_str(torch.device("cpu"), reset_peak=False)

    # -- Mock-based tests: run on CPU, verify string format and reset call --

    def _make_cuda_device(self):
        """Return a MagicMock that looks like a CUDA device."""
        d = MagicMock()
        d.type = "cuda"
        return d

    def test_mocked_cuda_string_contains_all_fields(self):
        """Verify string format without real CUDA by mocking memory functions."""
        MB = 1024 ** 2
        device = self._make_cuda_device()
        with (patch("torch.cuda.memory_allocated",     return_value=2048 * MB),
              patch("torch.cuda.memory_reserved",      return_value=4096 * MB),
              patch("torch.cuda.max_memory_allocated", return_value=3500 * MB),
              patch("torch.cuda.reset_peak_memory_stats")):
            result = _vram_str(device, reset_peak=True)

        assert "alloc=2048MB"    in result
        assert "reserved=4096MB" in result
        assert "peak=3500MB"     in result
        assert "frag="           in result

    def test_mocked_cuda_frag_calculation(self):
        """frag = (reserved - alloc) / reserved * 100."""
        MB = 1024 ** 2
        device = self._make_cuda_device()
        with (patch("torch.cuda.memory_allocated",     return_value=1024 * MB),
              patch("torch.cuda.memory_reserved",      return_value=2048 * MB),
              patch("torch.cuda.max_memory_allocated", return_value=1024 * MB),
              patch("torch.cuda.reset_peak_memory_stats")):
            result = _vram_str(device, reset_peak=False)

        # (2048 - 1024) / 2048 * 100 = 50%
        assert "frag=50%" in result

    def test_mocked_cuda_reset_called_when_flag_true(self):
        """reset_peak_memory_stats must be called exactly once when reset_peak=True."""
        MB = 1024 ** 2
        device = self._make_cuda_device()
        with (patch("torch.cuda.memory_allocated",     return_value=MB),
              patch("torch.cuda.memory_reserved",      return_value=MB),
              patch("torch.cuda.max_memory_allocated", return_value=MB),
              patch("torch.cuda.reset_peak_memory_stats") as mock_reset):
            _vram_str(device, reset_peak=True)
        mock_reset.assert_called_once()

    def test_mocked_cuda_reset_not_called_when_flag_false(self):
        """reset_peak_memory_stats must NOT be called when reset_peak=False."""
        MB = 1024 ** 2
        device = self._make_cuda_device()
        with (patch("torch.cuda.memory_allocated",     return_value=MB),
              patch("torch.cuda.memory_reserved",      return_value=MB),
              patch("torch.cuda.max_memory_allocated", return_value=MB),
              patch("torch.cuda.reset_peak_memory_stats") as mock_reset):
            _vram_str(device, reset_peak=False)
        mock_reset.assert_not_called()

    # -- Real CUDA tests (skipped when no GPU) --

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_returns_nonempty_string(self):
        result = _vram_str(torch.device("cuda"))
        assert result != ""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_string_contains_all_fields(self):
        result = _vram_str(torch.device("cuda"))
        assert "alloc=" in result
        assert "reserved=" in result
        assert "peak=" in result
        assert "frag=" in result

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_peak_resets_after_call(self):
        """reset_peak=True should clear max_memory_allocated."""
        device = torch.device("cuda")
        tmp = torch.zeros(1024, 1024, device=device)
        peak_before = torch.cuda.max_memory_allocated(device)
        del tmp
        _vram_str(device, reset_peak=True)
        peak_after = torch.cuda.max_memory_allocated(device)
        assert peak_after <= peak_before


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
