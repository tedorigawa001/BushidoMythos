from unittest.mock import patch

import pytest

from bushido_mythos.tokenizer import MythosTokenizer


class FakeHFTokenizer:
    name_or_path = "fake/local-tokenizer"
    vocab_size = 1234

    def encode(self, text, add_special_tokens=False):
        if text == "":
            return []
        return [ord(ch) % 251 for ch in text]

    def decode(self, token_ids, skip_special_tokens=True):
        if not token_ids:
            return ""
        return "decoded:" + ",".join(str(i) for i in token_ids)


@pytest.fixture
def fake_auto_tokenizer():
    with patch("bushido_mythos.tokenizer.AutoTokenizer.from_pretrained") as mock:
        mock.return_value = FakeHFTokenizer()
        yield mock


@pytest.fixture
def tokenizer(fake_auto_tokenizer):
    return MythosTokenizer()


def test_loads(tokenizer):
    assert tokenizer is not None
    assert tokenizer.tokenizer.name_or_path == "fake/local-tokenizer"


def test_loads_local_only_by_default(fake_auto_tokenizer):
    MythosTokenizer()
    fake_auto_tokenizer.assert_called_once_with("openai/gpt-oss-20b", local_files_only=True)


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 1234


def test_encode_returns_list_of_ints(tokenizer):
    ids = tokenizer.encode("Hello, world!")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_encode_empty_string(tokenizer):
    assert tokenizer.encode("") == []


def test_decode_returns_string(tokenizer):
    text = tokenizer.decode([1, 2, 3])
    assert isinstance(text, str)
    assert text == "decoded:1,2,3"


def test_roundtrip_calls_underlying_methods(tokenizer):
    ids = tokenizer.encode("The quick brown fox.")
    recovered = tokenizer.decode(ids)
    assert recovered.startswith("decoded:")


def test_encode_long_text(tokenizer):
    text = "BushidoMythos is a recurrent depth transformer. " * 100
    ids = tokenizer.encode(text)
    assert len(ids) > 100


def test_custom_model_id(fake_auto_tokenizer):
    tok = MythosTokenizer(model_id="custom/local-tokenizer")
    assert tok.vocab_size == 1234
    fake_auto_tokenizer.assert_called_once_with("custom/local-tokenizer", local_files_only=True)


def test_vocab_size_consistent(tokenizer):
    assert tokenizer.vocab_size == tokenizer.tokenizer.vocab_size


def test_local_cache_miss_raises_without_download():
    with patch("bushido_mythos.tokenizer.AutoTokenizer.from_pretrained", side_effect=OSError("missing")):
        with pytest.raises(RuntimeError, match="allow_download=True"):
            MythosTokenizer()


def test_allow_download_retries_without_local_only():
    calls = []

    def fake_loader(model_id, **kwargs):
        calls.append((model_id, kwargs))
        if kwargs.get("local_files_only"):
            raise OSError("missing")
        return FakeHFTokenizer()

    with patch("bushido_mythos.tokenizer.AutoTokenizer.from_pretrained", side_effect=fake_loader):
        tok = MythosTokenizer(allow_download=True)

    assert tok.vocab_size == 1234
    assert calls == [
        ("openai/gpt-oss-20b", {"local_files_only": True}),
        ("openai/gpt-oss-20b", {}),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
