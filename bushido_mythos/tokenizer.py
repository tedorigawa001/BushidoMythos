from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for BushidoMythos.

    Args:
        model_id (str): The HuggingFace model ID or path to use with AutoTokenizer.
            Defaults to "openai/gpt-oss-20b".

    Attributes:
        tokenizer: An instance of HuggingFace's AutoTokenizer.

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> s = tok.decode(ids)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, allow_download: bool = False):
        """
        Initialize the MythosTokenizer.

        Args:
            model_id (str): HuggingFace model identifier or path to tokenizer files.
            allow_download (bool): If True, fall back to network download when the
                tokenizer is not cached locally. Defaults to False (local-only).
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception as first_err:
            if not allow_download:
                raise RuntimeError(
                    f"トークナイザー {model_id!r} がローカルキャッシュに見つかりませんでした: {first_err}\n"
                    "allow_download=True を渡すか、事前に tokenizer をダウンロードしてください。"
                ) from first_err
            import warnings
            warnings.warn(
                f"local_files_only=True が失敗したためネットワークからダウンロードします ({model_id!r}): {first_err}",
                stacklevel=2,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the tokenizer vocabulary.

        Returns:
            int: The number of unique tokens in the tokenizer vocabulary.
        """
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """
        Encode input text into a list of token IDs.

        Args:
            text (str): The input text string to tokenize.

        Returns:
            list[int]: List of integer token IDs representing the input text.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        Args:
            token_ids (list[int]): A list of integer token IDs to decode.

        Returns:
            str: Decoded string representation of the token IDs.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
