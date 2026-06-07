"""
Tests for training/make_mixed_int8.py — mixed-precision INT8 export/load.

Covers:
  - load_mixed_int8 trusted gate (refuses without trusted=True)
  - format validation (rejects unknown payload format)
  - save -> load roundtrip preserves forward output (lossless reload)
"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import training.make_mixed_int8 as mx
from bushido_mythos import MythosConfig, BushidoMythos


def _tiny_cfg() -> MythosConfig:
    return MythosConfig(
        vocab_size=64, dim=16, n_heads=2, n_kv_heads=1, max_seq_len=16,
        max_loop_iters=2, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=2, n_shared_experts=1, n_experts_per_tok=1, expert_dim=8,
        lora_rank=2, kv_lora_rank=8, q_lora_rank=16, qk_rope_head_dim=4,
        qk_nope_head_dim=4, v_head_dim=4,
    )


def test_trusted_gate_refuses_without_flag():
    # trusted=False は torch.load 前に拒否するのでファイル不要
    with pytest.raises(RuntimeError, match="trusted=True"):
        mx.load_mixed_int8("does_not_matter.pt", trusted=False)


def test_format_check_rejects_unknown():
    with pytest.raises(ValueError, match="format"):
        mx._build_mixed_from_payload({"format": "something_else"}, torch.device("cpu"))


def test_save_load_roundtrip_is_lossless(tmp_path):
    cfg = _tiny_cfg()
    model = BushidoMythos(cfg)
    model.eval()

    # head を fp32 に残し、それ以外の nn.Linear を INT8 量子化
    qnames = mx._quant_names(model, ["head"])
    assert len(qnames) > 0
    kept = [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and n not in qnames]
    assert "head" in kept

    qmodel = mx._quantize(model, qnames)
    payload = {
        "quant_state": qmodel.state_dict(),
        "cfg": cfg.__dict__,
        "keep_fp32": ["head"],
        "quant_names": sorted(qnames),
        "format": mx._FORMAT,
    }
    path = tmp_path / "mixed.pt"
    torch.save(payload, path)

    # 再ロード(trusted)→ forward が一致
    qmodel2, cfg2 = mx.load_mixed_int8(path, torch.device("cpu"), trusted=True)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        y1 = qmodel(x, n_loops=cfg.max_loop_iters)
        y2 = qmodel2(x, n_loops=cfg.max_loop_iters)
    assert y1.shape == y2.shape
    assert (y1 - y2).abs().max().item() < 1e-5  # 無損失 roundtrip


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
