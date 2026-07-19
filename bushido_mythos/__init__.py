from bushido_mythos.main import (
    ACTHalting,
    Expert,
    GQAttention,
    LoRAAdapter,
    LTIInjection,
    MLAttention,
    MoEFFN,
    MythosConfig,
    BushidoMythos,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    chunked_linear_cross_entropy,
    loop_index_embedding,
    precompute_rope_freqs,
)
from bushido_mythos.variants import (
    mythos_tiny,
    mythos_1b,
    mythos_1t,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
)


def __getattr__(name: str):
    if name == "MythosTokenizer":
        from bushido_mythos.tokenizer import MythosTokenizer

        globals()[name] = MythosTokenizer
        return MythosTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MythosConfig",
    "RMSNorm",
    "GQAttention",
    "MLAttention",
    "Expert",
    "MoEFFN",
    "LoRAAdapter",
    "TransformerBlock",
    "LTIInjection",
    "ACTHalting",
    "RecurrentBlock",
    "BushidoMythos",
    "precompute_rope_freqs",
    "apply_rope",
    "chunked_linear_cross_entropy",
    "loop_index_embedding",
    "mythos_tiny",
    "mythos_1b",
    "mythos_3b",
    "mythos_10b",
    "mythos_50b",
    "mythos_100b",
    "mythos_500b",
    "mythos_1t",
    "load_tokenizer",
    "get_vocab_size",
    "MythosTokenizer",
]
