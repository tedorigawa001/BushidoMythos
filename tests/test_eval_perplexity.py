"""
Tests for training/eval_perplexity.py — network-independent (mocked).

Covers:
  - _build_gpt2_tokenizer: skips a candidate that encodes to empty, returns a working one
  - compute_perplexity: raises on 0 counted tokens (no silent PPL=1.0)
  - compute_perplexity: --max_chunks caps the number of forward passes
"""

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import training.eval_perplexity as ev
from bushido_mythos import MythosConfig, BushidoMythos


def _tiny_cfg() -> MythosConfig:
    return MythosConfig(
        vocab_size=64, dim=16, n_heads=2, n_kv_heads=1, max_seq_len=64,
        max_loop_iters=2, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=2, n_shared_experts=1, n_experts_per_tok=1, expert_dim=8,
        lora_rank=2, kv_lora_rank=8, q_lora_rank=16, qk_rope_head_dim=4,
        qk_nope_head_dim=4, v_head_dim=4,
    )


class TestVerifiedTokenizer:
    def test_skips_empty_encoder(self):
        import transformers

        class EmptyTok:
            def encode(self, text, add_special_tokens=False):
                return []          # 壊れたトークナイザ（常に空）

        class GoodTok:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

        # 最初の候補(AutoTokenizer)は空を返す → スキップし、次(GPT2TokenizerFast)を採用
        with patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=EmptyTok()), \
             patch.object(transformers.GPT2TokenizerFast, "from_pretrained", return_value=GoodTok()):
            tok = ev._build_gpt2_tokenizer()
        assert tok.encode("Hello", add_special_tokens=False) == [1, 2, 3]

    def test_raises_when_all_empty(self):
        import transformers

        class EmptyTok:
            def encode(self, text, add_special_tokens=False):
                return []

        with patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=EmptyTok()), \
             patch.object(transformers.GPT2TokenizerFast, "from_pretrained", return_value=EmptyTok()):
            with pytest.raises(RuntimeError):
                ev._build_gpt2_tokenizer()


class TestComputePerplexityGuards:
    def test_zero_tokens_raises_not_ppl_one(self):
        # 0 トークン → silent な PPL=1.0 を返さず RuntimeError
        cfg = types.SimpleNamespace(vocab_size=64)
        with pytest.raises(RuntimeError):
            ev.compute_perplexity(None, cfg, torch.zeros(0, dtype=torch.long),
                                  torch.device("cpu"), seq_len=32, n_loops=2, max_chunks=10)

    def test_max_chunks_caps_forwards(self):
        cfg = _tiny_cfg()
        model = BushidoMythos(cfg)
        calls = []
        orig = model.forward

        def rec(x, **kw):
            calls.append(1)
            return orig(x, **kw)

        model.forward = rec
        ids = torch.randint(0, cfg.vocab_size, (5000,), dtype=torch.long)
        ev.compute_perplexity(model, cfg, ids, torch.device("cpu"),
                              seq_len=64, n_loops=2, max_chunks=3)
        assert len(calls) == 3  # 5000 tokens でも 3 チャンクで打ち切り


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
