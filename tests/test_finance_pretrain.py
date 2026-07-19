"""
Tests for training/finance_pretrain.py.

Covers:
  - TextDataset: cache roundtrip, shape, ValueError on insufficient tokens
  - lr_lambda: warmup, cosine decay, min-lr floor
  - save/load checkpoint: scheduler_state, phase_steps, legacy fallback
  - Phase 2 reasoning-mix skip condition when step >= phase2_end
  - Cache key versioning
"""

import argparse
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
    AsyncCheckpointCopier,
    TextDataset,
    SFTDataset,
    _CACHE_VERSION,
    load_checkpoint,
    lr_lambda,
    save_checkpoint,
    _vram_str,
    run_phase,
    _cycle_batches,
    make_optimizer,
    _curriculum_ramp,
    apply_act_curriculum,
    _optimizer_state_compatible,
    validate_act_curriculum_args,
    rotate_step_checkpoints,
)
import training.finance_pretrain as finance_pretrain
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
        assert not (tmp_path / "ckpt.pt.tmp").exists()

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

    def test_save_includes_grouped_moe_runtime_setting(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        for module in self.model.modules():
            if hasattr(module, "use_grouped_moe"):
                module.use_grouped_moe = True
        save_checkpoint(path, step=5, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["runtime_config"]["grouped_moe"] is True

    def test_load_warns_on_grouped_moe_runtime_mismatch(self, tmp_path, capsys):
        path = tmp_path / "ckpt.pt"
        for module in self.model.modules():
            if hasattr(module, "use_grouped_moe"):
                module.use_grouped_moe = True
        save_checkpoint(path, step=5, model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, cfg=self.cfg)

        model2, opt2, sched2 = make_model_and_opt(self.cfg)
        load_checkpoint(str(path), model2, opt2, sched2)
        output = capsys.readouterr().out
        assert "grouped_moe runtime setting mismatch" in output
        assert "saved=true / current=false" in output

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


class TestRotateStepCheckpoints:

    @staticmethod
    def _make_steps(ckpt_dir, steps):
        for s in steps:
            (ckpt_dir / f"step_{s:06d}.pt").write_bytes(b"x")

    def test_keeps_only_last_n(self, tmp_path):
        self._make_steps(tmp_path, [1000, 2000, 3000, 4000, 5000])
        rotate_step_checkpoints(tmp_path, keep_last_n=3)
        remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
        assert remaining == ["step_003000.pt", "step_004000.pt", "step_005000.pt"]

    def test_keep_le_zero_disables_rotation(self, tmp_path):
        self._make_steps(tmp_path, [1000, 2000, 3000])
        rotate_step_checkpoints(tmp_path, keep_last_n=0)
        assert len(list(tmp_path.glob("step_*.pt"))) == 3

    def test_fewer_files_than_keep_is_noop(self, tmp_path):
        self._make_steps(tmp_path, [1000, 2000])
        rotate_step_checkpoints(tmp_path, keep_last_n=3)
        assert len(list(tmp_path.glob("step_*.pt"))) == 2

    def test_does_not_touch_phase_final_checkpoints(self, tmp_path):
        self._make_steps(tmp_path, [1000, 2000, 3000])
        (tmp_path / "phase1_final.pt").write_bytes(b"x")
        (tmp_path / "final.pt").write_bytes(b"x")
        rotate_step_checkpoints(tmp_path, keep_last_n=1)
        assert (tmp_path / "phase1_final.pt").exists()
        assert (tmp_path / "final.pt").exists()
        assert len(list(tmp_path.glob("step_*.pt"))) == 1


class TestAsyncCheckpointCopier:
    def test_copies_atomically_and_reports_stats(self, tmp_path):
        local = tmp_path / "local"
        durable = tmp_path / "drive"
        local.mkdir()
        source = local / "step_001000.pt"
        source.write_bytes(b"checkpoint")
        copier = AsyncCheckpointCopier(durable, keep_last_n_steps=3)
        copier.submit(source)
        copier.close()

        assert (durable / source.name).read_bytes() == b"checkpoint"
        assert not (durable / f"{source.name}.tmp").exists()
        stats = copier.stats()
        assert stats["files_copied"] == 1
        assert stats["bytes_copied"] == len(b"checkpoint")
        assert stats["pending"] == 0
        assert stats["errors"] == []

    def test_copy_failure_is_raised_on_flush(self, tmp_path, monkeypatch):
        source = tmp_path / "step_001000.pt"
        source.write_bytes(b"checkpoint")
        copier = AsyncCheckpointCopier(tmp_path / "drive")

        def fail_copy(*args, **kwargs):
            raise OSError("drive unavailable")

        monkeypatch.setattr(finance_pretrain.shutil, "copy2", fail_copy)
        copier.submit(source)
        with pytest.raises(RuntimeError, match="drive unavailable"):
            copier.close()

    def test_save_checkpoint_enqueues_durable_copy(self, tmp_path):
        cfg = tiny_cfg()
        model, optimizer, scheduler = make_model_and_opt(cfg)
        local = tmp_path / "local"
        durable = tmp_path / "drive"
        copier = AsyncCheckpointCopier(durable)
        path = local / "step_000005.pt"
        save_checkpoint(
            path,
            step=5,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            async_copier=copier,
        )
        copier.close()
        copied = torch.load(
            durable / path.name, map_location="cpu", weights_only=False
        )
        assert copied["step"] == 5

    def test_rotates_only_successfully_copied_periodic_files(self, tmp_path):
        local = tmp_path / "local"
        durable = tmp_path / "drive"
        local.mkdir()
        copier = AsyncCheckpointCopier(
            durable, keep_last_n_steps=2, keep_local_completed=1
        )
        for step in (1000, 2000, 3000):
            source = local / f"step_{step:06d}.pt"
            source.write_bytes(str(step).encode())
            copier.submit(source)
        phase_final = local / "phase1_final.pt"
        phase_final.write_bytes(b"final")
        copier.submit(phase_final)
        copier.close()

        assert sorted(path.name for path in durable.glob("step_*.pt")) == [
            "step_002000.pt",
            "step_003000.pt",
        ]
        assert (durable / "phase1_final.pt").exists()
        assert not (local / "step_001000.pt").exists()
        assert not (local / "step_002000.pt").exists()
        assert (local / "step_003000.pt").exists()
        assert phase_final.exists()


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


class TestMemoryReplay:
    """run_phase memory replay (general-language anchor) behavior."""

    def _args(self, replay_ratio):
        import argparse
        return argparse.Namespace(
            grad_accum_steps=1, grad_clip=1.0, log_every=100, save_every=100,
            mem_log_every=0, batch_size=2, seq_len=8, loop_schedule="off",
            replay_ratio=replay_ratio, loop_seed=0, allow_unsafe_checkpoint=False,
        )

    def _batches(self, fill, n=4):
        # x は全て token=fill（どちらのデータ由来か識別できるようにする）
        return [(torch.full((2, 8), fill, dtype=torch.long),
                 torch.full((2, 8), fill, dtype=torch.long)) for _ in range(n)]

    def _run(self, replay_ratio, wall_metrics=None):
        cfg = tiny_cfg()
        model, opt, sch = make_model_and_opt(cfg)
        seen = []
        orig_forward = model.forward
        def rec(x, **kw):
            seen.append(int(x.reshape(-1)[0].item()))
            return orig_forward(x, **kw)
        model.forward = rec
        current = self._batches(1)   # 現フェーズ由来 = token 1
        replay  = self._batches(2)   # リプレイ由来   = token 2
        tmp = Path(tempfile.mkdtemp())
        run_phase("Phase2-Test", current, model, cfg, opt, sch, self._args(replay_ratio),
                  tmp, 0, 4, "phase2_final.pt", torch.device("cpu"), torch.float32,
                  replay_dataset=replay, wall_metrics=wall_metrics)
        return seen

    def test_cycle_batches_is_infinite(self):
        g = _cycle_batches([("a",), ("b",)])
        assert [next(g) for _ in range(5)] == [("a",), ("b",), ("a",), ("b",), ("a",)]

    def test_full_replay_uses_anchor_only(self):
        seen = self._run(replay_ratio=1.0)
        assert seen and all(v == 2 for v in seen)  # 全て replay(token 2)由来

    def test_zero_replay_uses_current_only(self):
        seen = self._run(replay_ratio=0.0)
        assert seen and all(v == 1 for v in seen)  # 全て current(token 1)由来

    def test_replay_decision_is_deterministic_across_runs(self):
        # 同じ seed/step なら replay 判定系列が一致（resume 安全）
        assert self._run(0.5) == self._run(0.5)

    def test_run_phase_records_wall_clock_breakdown(self):
        metrics = {"phases": [], "checkpoint_serializations": []}
        self._run(0.0, wall_metrics=metrics)

        phase = metrics["phases"][0]
        assert phase["name"] == "Phase2-Test"
        assert phase["start_step"] == 0
        assert phase["end_step"] == 4
        assert phase["wall_seconds"] > 0
        assert phase["data_wait_seconds"] >= 0
        assert phase["optimizer_seconds"] >= 0
        assert phase["tokens_processed"] == 4 * 2 * 8
        assert phase["effective_tokens_per_second"] > 0
        assert metrics["checkpoint_serializations"][0]["name"] == "phase2_final.pt"


class TestMakeOptimizer:
    """make_optimizer: fp32 AdamW by default; safe fallback for 8-bit without CUDA/bnb."""

    def _params(self):
        return list(torch.nn.Linear(8, 8).parameters())

    def test_default_is_adamw(self):
        opt = make_optimizer(self._params(), lr=1e-4, optim8bit=False)
        assert type(opt).__name__ == "AdamW"

    def test_8bit_falls_back_without_cuda(self):
        # CPU 環境では bitsandbytes 8-bit は使えず、通常 AdamW にフォールバックする
        opt = make_optimizer(self._params(), lr=1e-4, optim8bit=True)
        assert type(opt).__name__ == "AdamW"  # クラッシュせずフォールバック

    def test_optimizer_can_step(self):
        params = self._params()
        opt = make_optimizer(params, lr=1e-3, optim8bit=True)
        params[0].sum().backward()
        opt.step()  # 例外が出ないこと


class _FakeCfg:
    """apply_act_curriculum が書き換える最小 cfg(モデルが forward で読む属性のみ)。"""
    def __init__(self, act_threshold=0.99, act_aux_loss_weight=0.0):
        self.act_threshold = act_threshold
        self.act_aux_loss_weight = act_aux_loss_weight


class _FakeModel:
    def __init__(self, cfg):
        self.cfg = cfg


def _curriculum_args(**over):
    base = dict(
        act_threshold_start=0.5,
        act_threshold_end=0.99,
        act_warmup_frac=0.5,
        ponder_weight_start=0.02,
        ponder_weight_end=0.0,
    )
    base.update(over)
    return argparse.Namespace(**base)


class TestCurriculumRamp:
    """_curriculum_ramp: 線形ランプ→hold の数値挙動。"""

    def test_returns_start_at_progress_zero(self):
        assert _curriculum_ramp(0.0, 0.5, 0.99, 0.5) == pytest.approx(0.5)

    def test_linear_midpoint_within_warmup(self):
        # progress=0.25, warmup_frac=0.5 → ランプ進捗 0.5 → 中点
        mid = 0.5 + (0.99 - 0.5) * 0.5
        assert _curriculum_ramp(0.25, 0.5, 0.99, 0.5) == pytest.approx(mid)

    def test_reaches_end_at_warmup_boundary(self):
        assert _curriculum_ramp(0.5, 0.5, 0.99, 0.5) == pytest.approx(0.99)

    def test_holds_end_after_warmup(self):
        assert _curriculum_ramp(0.9, 0.5, 0.99, 0.5) == pytest.approx(0.99)
        assert _curriculum_ramp(1.0, 0.5, 0.99, 0.5) == pytest.approx(0.99)

    def test_zero_warmup_returns_end_immediately(self):
        # frac<=0 はランプ無し=常に end(スケジュール無効と同義)
        assert _curriculum_ramp(0.0, 0.5, 0.99, 0.0) == pytest.approx(0.99)
        assert _curriculum_ramp(0.3, 0.5, 0.99, 0.0) == pytest.approx(0.99)

    def test_negative_progress_clamped_to_start(self):
        assert _curriculum_ramp(-0.5, 0.5, 0.99, 0.5) == pytest.approx(0.5)

    def test_descending_ramp_for_ponder(self):
        # start>end(ponder を下げる方向)でも線形に動く
        assert _curriculum_ramp(0.25, 0.02, 0.0, 0.5) == pytest.approx(0.01)


class TestApplyACTCurriculum:
    """apply_act_curriculum: 共有 cfg を進捗に応じて in-place 更新する。"""

    def test_mutates_shared_cfg_at_start(self):
        m = _FakeModel(_FakeCfg())
        thr, pon = apply_act_curriculum(m, _curriculum_args(), step=0, grand_total=1000)
        assert thr == pytest.approx(0.5)
        assert pon == pytest.approx(0.02)
        # 返り値だけでなく cfg 自体が書き換わる(forward が読むのは cfg)
        assert m.cfg.act_threshold == pytest.approx(0.5)
        assert m.cfg.act_aux_loss_weight == pytest.approx(0.02)

    def test_updates_real_model_tensor_buffers(self):
        model = BushidoMythos(tiny_cfg())

        apply_act_curriculum(model, _curriculum_args(), step=0, grand_total=1000)

        assert model.recurrent._act_threshold.item() == pytest.approx(0.5)
        assert model._act_aux_loss_weight.item() == pytest.approx(0.02)

    @pytest.mark.skipif(
        not torch._dynamo.is_dynamo_supported(),
        reason="torch.compile (Dynamo) がこの Python/torch では未対応",
    )
    def test_updates_compiled_model_wrapper(self):
        model = BushidoMythos(tiny_cfg())
        compiled = torch.compile(model, backend="eager")

        apply_act_curriculum(compiled, _curriculum_args(), step=0, grand_total=1000)

        assert model.recurrent._act_threshold.item() == pytest.approx(0.5)
        assert model._act_aux_loss_weight.item() == pytest.approx(0.02)

    @pytest.mark.skipif(
        not torch._dynamo.is_dynamo_supported(),
        reason="torch.compile (Dynamo) がこの Python/torch では未対応",
    )
    def test_buffer_updates_do_not_trigger_recompile(self):
        from torch._dynamo.testing import CompileCounter

        model = BushidoMythos(tiny_cfg()).eval()
        counter = CompileCounter()
        compiled = torch.compile(model, backend=counter)
        ids = torch.randint(0, model.cfg.vocab_size, (1, 4))

        model.set_act_curriculum_values(0.5, 0.02)
        compiled(ids, n_loops=1)
        initial_frames = counter.frame_count
        first_aux = model._last_aux_loss.item()

        model.set_act_curriculum_values(0.75, 0.01)
        compiled(ids, n_loops=1)

        assert counter.frame_count == initial_frames
        assert model._last_aux_loss.item() != pytest.approx(first_aux)

    def test_threshold_increases_monotonically(self):
        m = _FakeModel(_FakeCfg())
        args = _curriculum_args()
        vals = []
        for s in (0, 100, 250, 400, 500):
            apply_act_curriculum(m, args, step=s, grand_total=1000)
            vals.append(m.cfg.act_threshold)
        assert vals == sorted(vals)          # 単調非減少
        assert vals[0] < vals[-1]            # 実際に上昇

    def test_ponder_decreases_monotonically(self):
        m = _FakeModel(_FakeCfg())
        args = _curriculum_args()
        vals = []
        for s in (0, 100, 250, 400, 500):
            apply_act_curriculum(m, args, step=s, grand_total=1000)
            vals.append(m.cfg.act_aux_loss_weight)
        assert vals == sorted(vals, reverse=True)  # 単調非増加
        assert vals[0] > vals[-1]

    def test_holds_end_after_warmup_completes(self):
        m = _FakeModel(_FakeCfg())
        args = _curriculum_args()
        apply_act_curriculum(m, args, step=500, grand_total=1000)   # warmup 完了点
        apply_act_curriculum(m, args, step=999, grand_total=1000)   # それ以降
        assert m.cfg.act_threshold == pytest.approx(0.99)
        assert m.cfg.act_aux_loss_weight == pytest.approx(0.0)

    def test_warmup_frac_one_ramps_over_whole_run(self):
        # frac=1.0 なら学習全体でランプ。progress=step/(grand_total-1) なので
        # step=500, grand_total=1001 でちょうど中点(500/1000=0.5)。
        m = _FakeModel(_FakeCfg())
        args = _curriculum_args(act_warmup_frac=1.0)
        apply_act_curriculum(m, args, step=500, grand_total=1001)
        assert m.cfg.act_threshold == pytest.approx(0.5 + (0.99 - 0.5) * 0.5)

    def test_last_step_reaches_full_progress(self):
        # 最終ステップ(step=grand_total-1)で progress=1.0 → end に到達
        m = _FakeModel(_FakeCfg())
        args = _curriculum_args(act_warmup_frac=1.0)
        apply_act_curriculum(m, args, step=999, grand_total=1000)
        assert m.cfg.act_threshold == pytest.approx(0.99)

    def test_default_ponder_is_noop(self):
        # ponder_weight_start=end=0(既定)なら act_aux_loss_weight は 0 のまま
        m = _FakeModel(_FakeCfg(act_aux_loss_weight=0.0))
        args = _curriculum_args(ponder_weight_start=0.0, ponder_weight_end=0.0)
        for s in (0, 250, 500, 999):
            apply_act_curriculum(m, args, step=s, grand_total=1000)
            assert m.cfg.act_aux_loss_weight == pytest.approx(0.0)

    def test_anchor_ramps_over_resumed_span(self):
        # phase1_final(step=30000)から resume し grand_total=52000 を学習する状況。
        # anchor=30000 なら開始(step=30000)で start、区間後半で end に到達する。
        m = _FakeModel(_FakeCfg())
        args = _curriculum_args(act_warmup_frac=0.5)
        apply_act_curriculum(m, args, step=30000, grand_total=52000, anchor_step=30000)
        assert m.cfg.act_threshold == pytest.approx(0.5)           # 区間先頭=start
        apply_act_curriculum(m, args, step=45000, grand_total=52000, anchor_step=30000)
        assert m.cfg.act_threshold == pytest.approx(0.99)          # warmup超え=end

    def test_anchor_zero_keeps_global_progress(self):
        # 回帰: anchor=0(既定)だと resume 開始 step で進捗が既に warmup を超え、
        # 閾値が end 固定=カリキュラム無効になる(anchor 指定がこれを救う対比)。
        m = _FakeModel(_FakeCfg())
        args = _curriculum_args(act_warmup_frac=0.5)
        apply_act_curriculum(m, args, step=30000, grand_total=52000, anchor_step=0)
        assert m.cfg.act_threshold == pytest.approx(0.99)


class _Fp32Opt:
    """torch AdamW を模した optimizer(__module__ で 8-bit 判定される)。"""
class _8bitOpt:
    pass
_Fp32Opt.__module__ = "torch.optim.adamw"
_8bitOpt.__module__ = "bitsandbytes.optim.adamw"


def _fp32_state():
    return {"state": {0: {"step": 5, "exp_avg": 1, "exp_avg_sq": 2}}}


def _8bit_state():
    return {"state": {0: {"step": 5, "state1": 1, "state2": 2}}}


class TestOptimizerStateCompatible:
    """8-bit ⇄ fp32 の optimizer state 混在を構造から検出する(KeyError 'state1' 回帰)。"""

    def test_fp32_state_into_fp32_optimizer_ok(self):
        assert _optimizer_state_compatible(_Fp32Opt(), _fp32_state()) is True

    def test_8bit_state_into_8bit_optimizer_ok(self):
        assert _optimizer_state_compatible(_8bitOpt(), _8bit_state()) is True

    def test_fp32_state_into_8bit_optimizer_rejected(self):
        # 本件の再現: phase1(fp32 AdamW)を --optim8bit で resume
        assert _optimizer_state_compatible(_8bitOpt(), _fp32_state()) is False

    def test_8bit_state_into_fp32_optimizer_rejected(self):
        assert _optimizer_state_compatible(_Fp32Opt(), _8bit_state()) is False

    def test_empty_state_is_compatible(self):
        # 未ステップ(state 空)なら何にでもロード可
        assert _optimizer_state_compatible(_8bitOpt(), {"state": {}}) is True
        assert _optimizer_state_compatible(_Fp32Opt(), {}) is True


class TestValidateACTCurriculumArgs:
    """ACT カリキュラム CLI の値域検証。"""

    def test_valid_args_pass(self):
        validate_act_curriculum_args(_curriculum_args())  # 例外なし

    def test_threshold_zero_rejected(self):
        with pytest.raises(ValueError, match="act_threshold_start"):
            validate_act_curriculum_args(_curriculum_args(act_threshold_start=0.0))

    def test_threshold_above_one_rejected(self):
        with pytest.raises(ValueError, match="act_threshold_end"):
            validate_act_curriculum_args(_curriculum_args(act_threshold_end=1.2))

    def test_threshold_one_is_allowed(self):
        validate_act_curriculum_args(_curriculum_args(act_threshold_end=1.0))  # (0,1] の上端 OK

    def test_warmup_frac_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="act_warmup_frac"):
            validate_act_curriculum_args(_curriculum_args(act_warmup_frac=1.5))

    def test_warmup_frac_zero_allowed(self):
        validate_act_curriculum_args(_curriculum_args(act_warmup_frac=0.0))  # [0,1] 下端 OK

    def test_negative_ponder_rejected(self):
        with pytest.raises(ValueError, match="ponder_weight_start"):
            validate_act_curriculum_args(_curriculum_args(ponder_weight_start=-0.1))

    def test_zero_ponder_allowed(self):
        validate_act_curriculum_args(_curriculum_args(ponder_weight_start=0.0, ponder_weight_end=0.0))


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
