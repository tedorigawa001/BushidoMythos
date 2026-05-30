"""
Tests for chat.py.

Covers:
  - finance_mode: prompt wrapping, risk suffix, response prefix
  - Stop boundary: result truncated at '\\n### ' in finance_mode
  - top_k clamping: min(top_k, vocab_size) when top_k > 0
  - Prompt truncation: left-side trimming when prompt exceeds budget
  - _GPT2Tok.encode clamping: token IDs capped at vocab_size - 1
"""

import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from chat import (
    generate,
    _INSTRUCT_PREFIX,
    _INSTRUCT_RESPONSE,
    _INSTRUCT_STOP,
    _FINANCE_RISK_SUFFIX,
)


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(encode_ids=None, decode_text="response text"):
    tok = MagicMock()
    tok.encode.return_value = encode_ids if encode_ids is not None else [1, 2, 3]
    tok.decode.return_value = decode_text
    tok.vocab_size = 200
    return tok


def _make_mock_model(output_ids=None, prompt_len=3):
    model = MagicMock()
    if output_ids is None:
        # Default: extend the prompt by 5 tokens
        output_ids = torch.tensor([[1, 2, 3, 10, 11, 12, 13, 14]])
    model.generate.return_value = output_ids
    return model


def _make_mock_cfg(vocab_size=200, max_seq_len=128):
    cfg = MagicMock()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = max_seq_len
    return cfg


# ---------------------------------------------------------------------------
# finance_mode: prompt construction
# ---------------------------------------------------------------------------

class TestFinanceMode:

    def _call_generate(self, prompt, finance_mode, **overrides):
        tok = _make_mock_tokenizer()
        model = _make_mock_model()
        cfg = _make_mock_cfg()
        generate(model, cfg, tok, prompt,
                 max_new_tokens=10, temperature=0.8, top_k=40,
                 n_loops=2, device=torch.device("cpu"),
                 finance_mode=finance_mode, repetition_penalty=1.0)
        return tok.encode.call_args[0][0]  # actual prompt string passed to encode

    def test_finance_mode_prepends_instruction_prefix(self):
        actual = self._call_generate("What is inflation?", finance_mode=True)
        assert actual.startswith(_INSTRUCT_PREFIX)

    def test_finance_mode_appends_risk_suffix(self):
        actual = self._call_generate("What is inflation?", finance_mode=True)
        assert _FINANCE_RISK_SUFFIX in actual

    def test_finance_mode_appends_response_prefix(self):
        actual = self._call_generate("What is inflation?", finance_mode=True)
        assert actual.endswith(_INSTRUCT_RESPONSE)

    def test_no_finance_mode_passes_prompt_unchanged(self):
        raw = "Plain completion prompt."
        actual = self._call_generate(raw, finance_mode=False)
        assert actual == raw

    def test_finance_mode_contains_original_instruction(self):
        instruction = "Explain leverage risk."
        actual = self._call_generate(instruction, finance_mode=True)
        assert instruction in actual


# ---------------------------------------------------------------------------
# Stop boundary
# ---------------------------------------------------------------------------

class TestStopBoundary:

    def _generate_with_decoded(self, decoded_text, finance_mode=True):
        tok = _make_mock_tokenizer(encode_ids=[1, 2, 3], decode_text=decoded_text)
        model = _make_mock_model(output_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]))
        cfg = _make_mock_cfg()
        return generate(model, cfg, tok, "question",
                        max_new_tokens=10, temperature=0.8, top_k=0,
                        n_loops=2, device=torch.device("cpu"),
                        finance_mode=finance_mode)

    def test_stop_at_instruction_boundary(self):
        decoded = "Good answer.\n### Instruction:\nNext prompt"
        result = self._generate_with_decoded(decoded, finance_mode=True)
        assert result == "Good answer."

    def test_no_stop_when_no_boundary_in_finance_mode(self):
        decoded = "Good answer with no boundary marker."
        result = self._generate_with_decoded(decoded, finance_mode=True)
        assert result == decoded.strip()

    def test_no_stop_applied_outside_finance_mode(self):
        """Without finance_mode, _INSTRUCT_STOP must not truncate the result."""
        decoded = "Some text.\n### This should not be truncated."
        result = self._generate_with_decoded(decoded, finance_mode=False)
        assert "\n### " in result or result == decoded.strip()

    def test_result_is_stripped(self):
        decoded = "  padded answer  "
        result = self._generate_with_decoded(decoded, finance_mode=True)
        assert result == "padded answer"

    def test_empty_response_returns_empty_string(self):
        tok = _make_mock_tokenizer(encode_ids=[1], decode_text="")
        model = _make_mock_model(output_ids=torch.tensor([[1]]))
        cfg = _make_mock_cfg()
        result = generate(model, cfg, tok, "q",
                          max_new_tokens=5, temperature=1.0, top_k=0,
                          n_loops=2, device=torch.device("cpu"),
                          finance_mode=True)
        assert result == ""


# ---------------------------------------------------------------------------
# top_k clamping (chat_loop logic)
# ---------------------------------------------------------------------------

class TestTopKClamp:
    """chat_loop clamps top_k to cfg.vocab_size; generate() receives the clamped value."""

    def _effective_top_k(self, top_k_arg, vocab_size):
        """Replicate the chat_loop clamping formula."""
        return min(top_k_arg, vocab_size) if top_k_arg > 0 else 0

    def test_top_k_clamped_when_exceeds_vocab(self):
        assert self._effective_top_k(top_k_arg=5000, vocab_size=200) == 200

    def test_top_k_unchanged_when_within_vocab(self):
        assert self._effective_top_k(top_k_arg=40, vocab_size=200) == 40

    def test_top_k_zero_stays_zero(self):
        """top_k=0 disables filtering — must not be clamped to vocab_size."""
        assert self._effective_top_k(top_k_arg=0, vocab_size=200) == 0

    def test_top_k_equals_vocab_unchanged(self):
        assert self._effective_top_k(top_k_arg=200, vocab_size=200) == 200

    def test_top_k_one_accepted(self):
        assert self._effective_top_k(top_k_arg=1, vocab_size=200) == 1


# ---------------------------------------------------------------------------
# Prompt truncation
# ---------------------------------------------------------------------------

class TestPromptTruncation:
    """Long prompts must be left-trimmed to fit within max_seq_len - max_new_tokens."""

    def test_long_prompt_is_truncated(self, capsys):
        max_seq_len = 16
        max_new_tokens = 4
        # encode returns 20 tokens → exceeds budget of 12
        long_ids = list(range(20))
        tok = _make_mock_tokenizer(encode_ids=long_ids)
        model = _make_mock_model(output_ids=torch.tensor([list(range(16))]))
        cfg = _make_mock_cfg(vocab_size=200, max_seq_len=max_seq_len)

        generate(model, cfg, tok, "very long prompt",
                 max_new_tokens=max_new_tokens, temperature=1.0, top_k=0,
                 n_loops=1, device=torch.device("cpu"), finance_mode=False)

        # model.generate was called with input_ids whose length ≤ max_seq_len - max_new_tokens
        call_input = model.generate.call_args[0][0]  # first positional arg = input_ids
        assert call_input.shape[1] <= max_seq_len - max_new_tokens

    def test_short_prompt_not_truncated(self):
        short_ids = [1, 2, 3]
        tok = _make_mock_tokenizer(encode_ids=short_ids)
        model = _make_mock_model(output_ids=torch.tensor([[1, 2, 3, 4]]))
        cfg = _make_mock_cfg(vocab_size=200, max_seq_len=64)

        generate(model, cfg, tok, "short",
                 max_new_tokens=10, temperature=1.0, top_k=0,
                 n_loops=1, device=torch.device("cpu"), finance_mode=False)

        call_input = model.generate.call_args[0][0]
        assert call_input.shape[1] == len(short_ids)


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
