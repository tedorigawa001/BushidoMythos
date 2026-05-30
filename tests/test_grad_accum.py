"""
Tests for gradient-accumulation correctness in training/finance_pretrain.py.

Verifies that with grad_accum_steps=N:
  - optimizer.step() is called exactly total_steps times
    (measured via AdamW internal step counter, no monkey-patching)
  - scheduler.step() is called the same number of times
    (measured via LambdaLR.last_epoch)
  - the returned step counter equals total_steps
"""

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import MythosConfig, BushidoMythos
from training.finance_pretrain import run_phase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg() -> MythosConfig:
    return MythosConfig(
        vocab_size=64, dim=32, n_heads=2, n_kv_heads=1,
        max_seq_len=32, max_loop_iters=2, prelude_layers=1, coda_layers=1,
        attn_type="gqa", n_experts=4, n_shared_experts=1, n_experts_per_tok=2,
        expert_dim=16, lora_rank=4,
        kv_lora_rank=16, q_lora_rank=32, qk_rope_head_dim=8,
        qk_nope_head_dim=8, v_head_dim=8,
    )


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        batch_size=1,
        grad_accum_steps=1,
        seq_len=8,
        log_every=99999,
        save_every=99999,
        grad_clip=1.0,
        allow_unsafe_checkpoint=False,
        mem_log_every=0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_dataset(cfg: MythosConfig, n_batches: int, seq_len: int = 8):
    torch.manual_seed(42)
    return [
        (
            torch.randint(0, cfg.vocab_size, (1, seq_len)),
            torch.randint(0, cfg.vocab_size, (1, seq_len)),
        )
        for _ in range(n_batches)
    ]


def _optimizer_step_count(opt: AdamW) -> int:
    """Return how many times optimizer.step() was called.

    AdamW maintains a per-parameter 'step' counter in its state dict.
    All parameters share the same step count (one per optimizer.step() call).
    Returns 0 if no parameters have been updated yet.
    """
    state = opt.state_dict()["state"]
    if not state:
        return 0
    # All params are updated together; pick the first one
    first = next(iter(state.values()))
    step = first.get("step", 0)
    # PyTorch ≥ 2.0 stores step as a 0-dim tensor; older stores as int
    return int(step) if not isinstance(step, int) else step


def _run(cfg, grad_accum, total_steps, tmp_path):
    """Run run_phase and return (opt, sched, final_step)."""
    model = BushidoMythos(cfg)
    opt = AdamW(model.parameters(), lr=1e-4)
    sched = LambdaLR(opt, lr_lambda=lambda _: 1.0)
    args = _make_args(grad_accum_steps=grad_accum)
    n_micro = total_steps * grad_accum + grad_accum  # a few extra micro-batches
    dataset = _make_dataset(cfg, n_micro)

    final_step, _ = run_phase(
        phase_name="TestGradAccum",
        dataset=dataset,
        model=model,
        cfg=cfg,
        optimizer=opt,
        scheduler=sched,
        args=args,
        ckpt_dir=tmp_path,
        start_step=0,
        total_steps=total_steps,
        phase_final_name="final.pt",
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
    )
    return opt, sched, final_step


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGradAccumSteps:

    def setup_method(self):
        self.cfg = _tiny_cfg()

    def test_step_counter_reaches_total_steps(self, tmp_path):
        """Returned step counter must equal total_steps."""
        _, _, final = _run(self.cfg, grad_accum=3, total_steps=2, tmp_path=tmp_path)
        assert final == 2

    def test_optimizer_called_exactly_total_steps_times(self, tmp_path):
        """AdamW internal step counter must equal total_steps after run."""
        opt, _, final = _run(self.cfg, grad_accum=3, total_steps=2, tmp_path=tmp_path)
        assert _optimizer_step_count(opt) == final

    def test_scheduler_steps_match_optimizer(self, tmp_path):
        """LambdaLR.last_epoch must equal the number of optimizer steps."""
        opt, sched, final = _run(self.cfg, grad_accum=3, total_steps=2, tmp_path=tmp_path)
        assert sched.last_epoch == _optimizer_step_count(opt)

    def test_grad_accum_1_steps_every_microbatch(self, tmp_path):
        """With grad_accum=1 the step counter must equal total_steps."""
        opt, sched, final = _run(self.cfg, grad_accum=1, total_steps=3, tmp_path=tmp_path)
        assert final == 3
        assert _optimizer_step_count(opt) == 3
        assert sched.last_epoch == 3

    def test_step_counter_consistent_across_accum_values(self, tmp_path):
        """Regardless of grad_accum, returned step counter must equal total_steps."""
        for grad_accum in (1, 2, 4):
            opt, sched, final = _run(self.cfg, grad_accum=grad_accum,
                                     total_steps=3, tmp_path=tmp_path)
            assert final == 3, f"grad_accum={grad_accum}: expected final_step=3"
            assert _optimizer_step_count(opt) == 3
            assert sched.last_epoch == 3

    def test_scheduler_and_optimizer_always_in_sync(self, tmp_path):
        """scheduler.last_epoch must always equal optimizer step count."""
        for grad_accum in (1, 3):
            opt, sched, _ = _run(self.cfg, grad_accum=grad_accum,
                                  total_steps=2, tmp_path=tmp_path)
            assert sched.last_epoch == _optimizer_step_count(opt), (
                f"grad_accum={grad_accum}: scheduler and optimizer out of sync"
            )


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
