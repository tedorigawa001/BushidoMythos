import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

try:
    from flash_attn import flash_attn_func  # type: ignore

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute tied LM-head cross entropy without retaining full-vocabulary logits.

    Each token chunk is activation-checkpointed, so its logits are discarded after
    the forward pass and recomputed during backward. This trades one extra LM-head
    projection for peak memory proportional to ``chunk_size * vocab_size`` instead
    of ``batch * sequence * vocab_size``.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError(
            f"hidden/targets shape mismatch: {hidden.shape[:-1]} vs {targets.shape}"
        )
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError(
            f"weight shape {weight.shape} is incompatible with hidden dim {hidden.shape[-1]}"
        )
    if loss_mask is not None and loss_mask.shape != targets.shape:
        raise ValueError(
            f"loss_mask/targets shape mismatch: {loss_mask.shape} vs {targets.shape}"
        )

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1) if loss_mask is not None else None

    def unmasked_chunk_loss(
        hidden_chunk: torch.Tensor,
        head_weight: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        logits = F.linear(hidden_chunk, head_weight)
        return F.cross_entropy(logits, target_chunk, reduction="sum")

    def masked_chunk_loss(
        hidden_chunk: torch.Tensor,
        head_weight: torch.Tensor,
        target_chunk: torch.Tensor,
        mask_chunk: torch.Tensor,
    ) -> torch.Tensor:
        logits = F.linear(hidden_chunk, head_weight)
        per_token = F.cross_entropy(logits, target_chunk, reduction="none")
        return (per_token * mask_chunk.to(per_token.dtype)).sum()

    total = None
    checkpoint_chunks = torch.is_grad_enabled() and (
        flat_hidden.requires_grad or weight.requires_grad
    )
    for start in range(0, flat_targets.numel(), chunk_size):
        end = min(start + chunk_size, flat_targets.numel())
        hidden_chunk = flat_hidden[start:end]
        target_chunk = flat_targets[start:end]
        if flat_mask is None:
            args = (hidden_chunk, weight, target_chunk)
            chunk_loss = (
                _grad_ckpt(unmasked_chunk_loss, *args, use_reentrant=False)
                if checkpoint_chunks
                else unmasked_chunk_loss(*args)
            )
        else:
            mask_chunk = flat_mask[start:end]
            args = (hidden_chunk, weight, target_chunk, mask_chunk)
            chunk_loss = (
                _grad_ckpt(masked_chunk_loss, *args, use_reentrant=False)
                if checkpoint_chunks
                else masked_chunk_loss(*args)
            )
        total = chunk_loss if total is None else total + chunk_loss

    if total is None:
        raise ValueError("targets must contain at least one token")
    if flat_mask is None:
        denominator = flat_targets.numel()
    else:
        denominator = flat_mask.to(dtype=torch.float32).sum().clamp_min(1.0)
    return total / denominator


class _KVCache(dict):
    """Dictionary-compatible KV cache with an optional allocation capacity."""

    def __init__(self, capacity: Optional[int] = None):
        super().__init__()
        self.capacity = capacity


class _CacheEntry(dict):
    """Public cache views backed by reusable, over-allocated tensors."""

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.length = 0
        self.storage = {}


def _append_kv_cache(
    kv_cache: dict,
    cache_key: str,
    tensors: dict,
) -> dict:
    """Append detached cache values and return tensors for the current forward."""
    entry = kv_cache.get(cache_key)
    old_length = entry.length if isinstance(entry, _CacheEntry) else 0
    if entry is not None and not isinstance(entry, _CacheEntry):
        first = next(iter(tensors))
        old_length = entry[first].shape[1]

    append_length = next(iter(tensors.values())).shape[1]
    required = old_length + append_length
    requested_capacity = getattr(kv_cache, "capacity", None)
    current_capacity = entry.capacity if isinstance(entry, _CacheEntry) else 0
    capacity = max(required, requested_capacity or 0, max(1, current_capacity * 2))

    needs_allocation = not isinstance(entry, _CacheEntry) or required > entry.capacity
    if needs_allocation:
        new_entry = _CacheEntry(capacity)
        for name, value in tensors.items():
            shape = list(value.shape)
            shape[1] = capacity
            storage = torch.empty(shape, dtype=value.dtype, device=value.device)
            if entry is not None and old_length:
                old_value = entry[name]
                storage[:, :old_length].copy_(old_value)
            new_entry.storage[name] = storage
        entry = new_entry
        kv_cache[cache_key] = entry

    attention_tensors = {}
    for name, value in tensors.items():
        entry.storage[name][:, old_length:required].copy_(value.detach())
        entry[name] = entry.storage[name][:, :required]
        if torch.is_grad_enabled() and value.requires_grad:
            if old_length:
                history = entry.storage[name][:, :old_length]
                attention_tensors[name] = torch.cat([history, value], dim=1)
            else:
                attention_tensors[name] = value
        else:
            attention_tensors[name] = entry[name]
    entry.length = required
    return attention_tensors


@dataclass
class MythosConfig:
    """
    Hyperparameter configuration for BushidoMythos.

    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2
    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0
    # Hyper-connections: replace residual x+f(x) with alpha*x + beta*f(x) (learned mixing)
    use_hyper_connections: bool = False
    # ACT auxiliary loss weight; trains the model to use fewer loop steps (0 = disabled)
    act_aux_loss_weight: float = 0.0
    # Loop count curriculum: randomly sample n_loops in [1, max_loop_iters] during training
    loop_curriculum: bool = False
    # COCONUT continuous latent thought steps before discrete token generation (0 = disabled)
    coconut_steps: int = 0
    # Depth cross-attention: attend to K/V from previous loop iterations (MoDA-style)
    use_depth_attn: bool = False
    # Gradient checkpointing: recompute loop activations on backward instead of storing them.
    # Reduces VRAM usage proportionally to n_loops (~7x for 8 loops) at the cost of ~30-40%
    # extra compute. Only active during training when kv_cache is None.
    use_gradient_checkpointing: bool = False
    def __post_init__(self) -> None:
        if self.attn_type not in ("gqa", "mla"):
            raise ValueError(f"attn_type must be 'gqa' or 'mla', got {self.attn_type!r}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be positive, got {self.n_kv_heads}")
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        head_dim = self.dim // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim (dim/n_heads={head_dim}) must be even for RoPE")
        if self.attn_type == "mla":
            if self.qk_rope_head_dim % 2 != 0:
                raise ValueError(f"qk_rope_head_dim ({self.qk_rope_head_dim}) must be even for RoPE")
        if self.n_experts_per_tok > self.n_experts:
            raise ValueError(f"n_experts_per_tok ({self.n_experts_per_tok}) > n_experts ({self.n_experts})")


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. Used in place of
    LayerNorm throughout the model for stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim -- feature dimension to normalize over
            eps -- small constant added before sqrt for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor of the same shape, rescaled by self.weight
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """
    Precompute RoPE rotation matrices for positions 0..max_len-1.

    Stored as a real tensor of shape (max_len, dim//2, 2) where the last
    dimension is [cos, sin]. This avoids complex dtypes, which are unsupported
    on MPS (Apple Silicon) backends.

    Args:
        dim     -- head dimension (must be even); frequencies are computed for dim//2 pairs
        max_len -- maximum sequence length to precompute
        theta   -- RoPE base (higher = slower frequency decay; 500k is the LLaMA-3 default)

    Returns:
        float32 tensor of shape (max_len, dim//2, 2)  — [cos, sin] in last dim
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (max_len, dim//2, 2)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.

    Each adjacent pair of features (x_{2i}, x_{2i+1}) is rotated by the
    angle for frequency i at the corresponding position:
        y_{2i}   = x_{2i}   * cos - x_{2i+1} * sin
        y_{2i+1} = x_{2i}   * sin + x_{2i+1} * cos

    Args:
        x         -- tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis -- real [cos, sin] frequencies of shape (max_len, head_dim//2, 2)

    Returns:
        Rotated tensor of the same shape and dtype as x
    """
    T = x.shape[1]
    freqs_cis = freqs_cis[:T]                               # (T, head_dim//2, 2)
    cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(2)       # (1, T, 1, head_dim//2)
    sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(2)       # (1, T, 1, head_dim//2)
    xf = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)  # (B, T, H, head_dim//2, 2)
    x0, x1 = xf[..., 0], xf[..., 1]                        # (B, T, H, head_dim//2) each
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack([y0, y1], dim=-1).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023) with Flash Attention 2 (Dao et al., 2023).

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing the KV cache size
    by that factor while keeping full query expressiveness.

    When flash-attn is installed, uses flash_attn_func which handles GQA natively
    (no KV head expansion needed) and is IO-bound-optimal. Inputs are cast to
    bfloat16 for flash_attn and restored to the original dtype afterward.
    Falls back to manual scaled dot-product attention when flash-attn is absent.

    RoPE is applied to both Q and K. K and V are stored in kv_cache after
    RoPE application so that cached values are already positionally encoded and
    do not need to be re-rotated on retrieval.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, n_kv_heads
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            entry = _append_kv_cache(kv_cache, cache_key, {"k": k, "v": v})
            k, v = entry["k"], entry["v"]

        S = k.shape[1]
        # When KV cache has past tokens, mask is (1,1,T,T) but attn will be
        # (B,H,T,S) where S > T. Pad left with 0 so past tokens are all visible.
        if mask is not None and S > T:
            mask = F.pad(mask, (S - T, 0), value=0.0)

        # Skip FlashAttention when S > T (chunked KV-cache decode): flash-attn's
        # causal alignment is version-dependent for seqlen_q != seqlen_k and the
        # padded mask cannot be passed to flash_attn_func directly.
        _use_flash = _HAS_FLASH_ATTN and S == T
        if _use_flash:
            # flash_attn_func expects (B, T, H, head_dim) — GQA is handled natively
            # (n_kv_heads < n_heads is supported without repeat_interleave).
            # causal=True when mask is present (full-sequence prefill/training);
            # causal=False for single-token decode where T=1 and mask is None.
            orig_dtype = q.dtype
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            dropout_p = self.dropout_p if self.training else 0.0
            out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                causal=is_causal or mask is not None,
            )
            out = out.to(orig_dtype).contiguous().view(B, T, self.n_heads * self.head_dim)
        else:
            # PyTorch SDPA selects the best available fused backend and avoids
            # materializing the full (B, H, T, S) attention probability tensor.
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)

        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    The key insight: instead of caching full K and V tensors (each of size
    n_heads × head_dim per token), MLA compresses the KV path through a
    low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and
    V are reconstructed from c_kv at each decoding step, trading a cheap
    linear projection for dramatically smaller cache memory.

    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope)  per head

    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into c_kv (latent, cached) and k_rope_raw (shared across heads)
        k_rope = RoPE(expand(k_rope_raw))  — applied before caching
        c_kv → kv_norm → kv_up → [k_nope | v]  — reconstructed each step
        k = cat(k_nope, k_rope)  per head

    Cache stores: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim),
    versus full GQA cache: n_kv_heads × head_dim × 2.  At production scale this
    is roughly a 10–20× memory reduction.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, kv_lora_rank, q_lora_rank,
                   qk_rope_head_dim, qk_nope_head_dim, v_head_dim
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False
        )

        # KV compression: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = nn.Linear(
            cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies sized for qk_rope_head_dim, shape (T, rope_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"c_kv": ..., "k_rope": ...}
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # expand rope keys across heads and apply RoPE before caching so
        # retrieved keys are already positionally encoded
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        if kv_cache is not None:
            entry = _append_kv_cache(
                kv_cache, cache_key, {"c_kv": c_kv, "k_rope": k_rope}
            )
            c_kv, k_rope = entry["c_kv"], entry["k_rope"]

        S = c_kv.shape[1]  # full sequence length including cache

        # reconstruct K_nope and V from latent (not cached, recomputed each step)
        kv = self.kv_up(self.kv_norm(c_kv))  # (B, S, H*(nope+v))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]  # (B, S, H, nope)
        v = kv[..., self.qk_nope_dim :]  # (B, S, H, v_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, S, H, nope+rope)

        # attention
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        if mask is not None:
            # When KV cache has past tokens, S > T: pad left with 0 so past
            # tokens are fully visible (they're all in the causal past).
            if S > T:
                mask = F.pad(mask, (S - T, 0), value=0.0)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=is_causal,
            scale=self.q_head_dim**-0.5,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.v_dim)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


_NATIVE_GROUPED_MM = getattr(F, "grouped_mm", None)


def grouped_moe_runtime_status(
    device: torch.device, compute_dtype: torch.dtype
) -> tuple[bool, str]:
    """Return native grouped MoE availability and a stable diagnostic reason."""
    if _NATIVE_GROUPED_MM is None:
        return False, "api_unavailable"
    if device.type != "cuda":
        return False, "device_not_cuda"
    if compute_dtype != torch.bfloat16:
        return False, "dtype_not_bfloat16"
    if not torch.cuda.is_available():
        return False, "cuda_unavailable"
    if torch.cuda.get_device_capability(device)[0] < 8:
        return False, "compute_capability_lt_80"
    return True, "active"


def _native_grouped_linear(
    x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Grouped equivalent of F.linear(x_group, weight[group]) without bias."""
    if _NATIVE_GROUPED_MM is None:
        raise RuntimeError("torch.nn.functional.grouped_mm is unavailable")
    # grouped_mm consumes [group, in_features, out_features], whereas nn.Linear
    # stores weights as [out_features, in_features].
    return _NATIVE_GROUPED_MM(
        x, weight.transpose(-2, -1).contiguous(), offs=offsets
    )


class _GroupedLinear(torch.autograd.Function):
    """Differentiable wrapper around the non-differentiable grouped_mm primitive."""

    @staticmethod
    def forward(ctx, x, weight, offsets):
        ctx.save_for_backward(x, weight, offsets)
        return _native_grouped_linear(x, weight, offsets)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, offsets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = _native_grouped_linear(
            grad_output, weight.transpose(-2, -1).contiguous(), offsets
        )
        # The 2D x 2D weight-gradient kernel requires a 16-byte row stride.
        # Zero-pad the final group so arbitrary token counts satisfy that
        # constraint without changing any gradient values.
        row_alignment = max(1, 16 // grad_output.element_size())
        pad_rows = (-grad_output.shape[0]) % row_alignment
        if pad_rows:
            grad_output_for_weight = F.pad(grad_output, (0, 0, 0, pad_rows))
            x_for_weight = F.pad(x, (0, 0, 0, pad_rows))
            offsets_for_weight = torch.cat(
                (offsets[:-1], offsets[-1:] + pad_rows)
            )
        else:
            grad_output_for_weight = grad_output
            x_for_weight = x
            offsets_for_weight = offsets
        # In 2D x 2D mode grouped_mm partitions the contracting dimension by
        # offsets and returns one [out_features, in_features] gradient per group.
        grad_weight = _NATIVE_GROUPED_MM(
            grad_output_for_weight.transpose(0, 1).contiguous(),
            x_for_weight,
            offs=offsets_for_weight,
        )
        return grad_x, grad_weight, None


class Expert(nn.Module):
    """
    Single SwiGLU feed-forward expert.

    Implements the gated linear unit variant: output = down(silu(gate(x)) * up(x)).
    Used both as individual routed experts inside MoEFFN and as the standard dense
    FFN in prelude/coda blocks (where expert_dim = dim * 4 // 3).
    """

    def __init__(self, dim: int, expert_dim: int):
        """
        Args:
            dim        -- input and output feature dimension
            expert_dim -- inner (hidden) dimension of the expert
        """
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (..., dim)
        Returns:
            Tensor of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """
    Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024).

    Two classes of experts:
    - Routed experts: n_experts small FFNs; each token activates top-K of them
      via a learned router. A per-expert bias on router logits is updated during
      training to keep load balanced across experts without distorting the loss.
    - Shared experts: n_shared_experts larger FFNs always activated for every token,
      absorbing common cross-domain patterns (syntax, basic reasoning) that would
      otherwise be redundantly learned by many routed experts.

    Total activated parameters per token ≈ topk/n_experts of routed + all shared,
    keeping compute sparse while the total parameter count stays large.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses n_experts, n_shared_experts, n_experts_per_tok,
                   dim, expert_dim
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        self.use_grouped_moe = False

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # load-balancing bias adjusted externally during training; not a gradient param
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )
        self._last_expert_counts: Optional[torch.Tensor] = None
        self._accum_expert_counts: Optional[torch.Tensor] = None

    def _forward_routed_loop(
        self,
        flat: torch.Tensor,
        tok_rows_sorted: torch.Tensor,
        scores_sorted: torch.Tensor,
        counts_int: torch.Tensor,
    ) -> torch.Tensor:
        """Checkpoint-compatible fallback for runtimes without grouped_mm."""
        counts_cpu = counts_int.tolist()
        out = torch.zeros_like(flat)
        offset = 0
        for eid, cnt in enumerate(counts_cpu):
            if cnt > 0:
                tok_rows_e = tok_rows_sorted[offset : offset + cnt]
                scores_e = scores_sorted[offset : offset + cnt].unsqueeze(-1)
                expert_out = scores_e * self.routed_experts[eid](flat[tok_rows_e])
                out = out.index_add(0, tok_rows_e, expert_out)
            offset += cnt
        return out

    def _forward_routed_grouped(
        self,
        flat: torch.Tensor,
        tok_rows_sorted: torch.Tensor,
        scores_sorted: torch.Tensor,
        counts_int: torch.Tensor,
    ) -> torch.Tensor:
        routed = flat[tok_rows_sorted].to(dtype=torch.bfloat16)
        offsets = counts_int.cumsum(0).to(dtype=torch.int32)

        gate_weight = torch.stack(
            [expert.gate.weight for expert in self.routed_experts]
        ).to(dtype=torch.bfloat16)
        up_weight = torch.stack(
            [expert.up.weight for expert in self.routed_experts]
        ).to(dtype=torch.bfloat16)
        down_weight = torch.stack(
            [expert.down.weight for expert in self.routed_experts]
        ).to(dtype=torch.bfloat16)

        gate = _GroupedLinear.apply(routed, gate_weight, offsets)
        up = _GroupedLinear.apply(routed, up_weight, offsets)
        activated = F.silu(gate) * up
        routed_out = _GroupedLinear.apply(activated, down_weight, offsets)
        routed_out = routed_out * scores_sorted.unsqueeze(-1)
        routed_out = routed_out.to(dtype=flat.dtype)

        out = torch.zeros_like(flat)
        return out.index_add(0, tok_rows_sorted, routed_out)

    def _forward_impl(self, x: torch.Tensor, grouped: bool) -> torch.Tensor:
        """
        Args:
            x -- input of shape (B, T, dim)
        Returns:
            Tensor of shape (B, T, dim); shared expert outputs are summed on top
            of the weighted routed expert outputs
        """
        B, T, D = x.shape
        flat = x.reshape(B * T, D)  # reshape handles non-contiguous inputs

        # Aux-loss-free load balancing (DeepSeek-V3): the bias shifts only the
        # selection of which experts fire so underused experts are picked more,
        # but the gating weights come from unbiased softmax scores so the bias
        # never shows up in the gradient.
        logits = self.router(flat)  # (B*T, n_experts), unbiased
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # renorm

        # bincount as int for dispatch, float for load-balancing tracking.
        # One GPU→CPU transfer (counts_cpu) covers all expert boundaries,
        # avoiding the per-expert any()/nonzero() syncs of the previous approach.
        counts_int = torch.bincount(topk_idx.flatten(), minlength=self.n_experts)
        counts = counts_int.float().detach()
        self._last_expert_counts = counts
        if self._accum_expert_counts is None:
            self._accum_expert_counts = counts.clone()
        else:
            self._accum_expert_counts = self._accum_expert_counts + counts

        # Sort all (token, k-slot) pairs by expert index with a single argsort kernel.
        # Grouped mode keeps dispatch boundaries on-device. The fallback performs
        # one counts transfer and preserves compatibility with older runtimes.
        N = flat.shape[0]
        tok_rows_all = torch.arange(N, device=flat.device).unsqueeze(1).expand_as(topk_idx).reshape(-1)
        eid_all = topk_idx.reshape(-1)
        scores_all = topk_scores.reshape(-1)

        sort_order = eid_all.argsort(stable=True)      # single GPU kernel, no sync
        tok_rows_sorted = tok_rows_all[sort_order]
        scores_sorted = scores_all[sort_order]

        if grouped:
            out = self._forward_routed_grouped(
                flat, tok_rows_sorted, scores_sorted, counts_int
            )
        else:
            out = self._forward_routed_loop(
                flat, tok_rows_sorted, scores_sorted, counts_int
            )

        # shared experts always fire for every token
        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.reshape(B, T, D)

    @torch._dynamo.disable
    def _forward_legacy(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, grouped=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grouped_moe:
            return self._forward_impl(x, grouped=True)
        return self._forward_legacy(x)


# ---------------------------------------------------------------------------
# Loop-index RoPE (differentiates recurrent block across iterations)
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Inject a sinusoidal loop-index signal into the first loop_dim channels of h.

    Analogous to RoPE for sequence position, but applied over recurrence depth
    instead of token position. Without this, the shared recurrent block weights
    must handle both early-stage pattern-matching and late-stage refinement with
    no signal distinguishing which loop they are on. Adding the loop index lets
    the same parameters implement functionally distinct operations per iteration.

    Args:
        h        -- hidden state tensor of shape (B, T, dim)
        loop_t   -- current loop iteration index (0-based)
        loop_dim -- number of leading channels to receive the embedding (must be even)
        theta    -- sinusoidal base frequency

    Returns:
        h with a sinusoidal bias added to its first loop_dim channels; same shape
    """
    # Compute in fp32: in bf16 adjacent indices quantize to the same float and the
    # loop-index signal degenerates (adjacent k values become identical frequencies).
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=torch.float32) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    # Interleave sin/cos: [sin(θ₀), cos(θ₀), sin(θ₁), cos(θ₁), ...]
    # matches the standard sinusoidal positional encoding layout (Vaswani et al. 2017).
    # The previous block layout [sin..., cos...] left the sin half as all-zeros at t=0.
    emb = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten()[:loop_dim].to(h.dtype)
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter (per loop iteration)
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for the recurrent block (Bae et al., 2024).

    Pure weight-tying (identical weights every loop) limits expressiveness;
    fully distinct weights per loop eliminate parameter savings. This adapter
    sits in between: a shared low-rank down-projection and up-projection matrix B
    are shared across all loops, while a small per-loop scale vector shifts the
    effective transformation at each depth without adding significant parameters.

    delta(x, t) = (down(x) * scale[t]) @ B
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        Args:
            dim       -- model hidden dimension (input and output size)
            rank      -- low-rank bottleneck dimension
            max_loops -- maximum number of loop iterations (determines embedding table size)
        """
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # shared A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # shared B: rank → dim
        self.scale = nn.Embedding(max_loops, rank)  # per-loop element-wise scale
        nn.init.normal_(self.scale.weight, std=0.01)  # small unique scale per loop; much smaller than default Normal(0,1)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x      -- input tensor of shape (B, T, dim)
            loop_t -- current loop index used to look up the per-loop scale

        Returns:
            Delta tensor of shape (B, T, dim) to be added to the block output
        """
        # Clamp for depth extrapolation: at inference n_loops can exceed the
        # training max_loop_iters. Iterations beyond the trained range reuse
        # the last learned per-loop scale rather than indexing out of range.
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))  # (rank,)
        down = self.down(x) * s  # (B, T, rank)
        return down @ self.B  # (B, T, dim)


# ---------------------------------------------------------------------------
# Single Transformer Block (shared across recurrent loops)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with swappable attention and optional MoE FFN.

    Attention is selected by cfg.attn_type:
        "gqa" → GQAttention  (Grouped Query Attention, fewer KV heads)
        "mla" → MLAttention  (Multi-Latent Attention, compressed KV cache)

    FFN is selected by use_moe:
        True  → MoEFFN  (fine-grained routed + shared experts; used in RecurrentBlock)
        False → Expert  (dense SwiGLU FFN; used in Prelude and Coda)
    """

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        """
        Args:
            cfg     -- MythosConfig; attn_type selects the attention class
            use_moe -- if True, use MoEFFN; otherwise use a dense Expert FFN
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)
        if cfg.use_hyper_connections:
            # Learned per-channel mixing coefficients; initialized to 1 so the model
            # starts as a standard residual and can learn to mix more freely.
            self.alpha_attn = nn.Parameter(torch.ones(cfg.dim))
            self.beta_attn = nn.Parameter(torch.ones(cfg.dim))
            self.alpha_ffn = nn.Parameter(torch.ones(cfg.dim))
            self.beta_ffn = nn.Parameter(torch.ones(cfg.dim))

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- precomputed RoPE frequencies
            mask      -- additive causal mask or None
            kv_cache  -- cache dict mutated in-place by the attention layer
            cache_key -- key identifying this layer in the cache

        Returns:
            Output tensor of shape (B, T, dim)
        """
        attn_out = self.resid_drop(
            self.attn(
                self.attn_norm(x),
                freqs_cis,
                mask,
                kv_cache,
                cache_key,
                is_causal,
            )
        )
        if hasattr(self, "alpha_attn"):
            x = self.alpha_attn * x + self.beta_attn * attn_out
        else:
            x = x + attn_out
        ffn_out = self.resid_drop(self.ffn(self.ffn_norm(x)))
        if hasattr(self, "alpha_ffn"):
            x = self.alpha_ffn * x + self.beta_ffn * ffn_out
        else:
            x = x + ffn_out
        return x


# ---------------------------------------------------------------------------
# LTI-stable injection parameters  (spectral radius < 1 by construction)
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """
    Stable input injection for the recurrent update rule (Parcae, Prairie et al., 2026).

    The recurrent hidden state evolves as:
        h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

    where e is the encoded input injected at every loop step to prevent drift.
    Without constraints, A can develop spectral radius ≥ 1, causing the hidden
    state to explode across loop iterations and destabilize training.

    This class guarantees ρ(A) < 1 by construction via a ZOH discretization:
        A_continuous = Diag(-exp(log_A))       always negative diagonal
        A_discrete   = exp(Δt · A_continuous)  element-wise, values in (0, 1)

    where log_A and log_dt are learned parameters and exp ensures positivity.
    This makes looped model training robust to hyperparameter choices and stable
    even at high learning rates.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; one scalar per channel for A and B
        """
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))  # log of A_continuous magnitude
        self.log_dt = nn.Parameter(torch.zeros(1))  # log of discretization step Δt
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        """
        Compute the discretized diagonal state matrix A_discrete.

        Returns:
            1-D tensor of shape (dim,) with all values strictly in (0, 1),
            guaranteeing ρ(A) < 1 regardless of learned parameter values.
        """
        # Compute in log space to avoid 0 * inf = NaN when log_dt → -∞, log_A → +∞.
        # dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
        # Outer clamp keeps the exponent finite for any gradient step size.
        # Inner clamp(min=1e-5) ensures the negative exponent magnitude is at least 1e-5,
        # so A = exp(-x) ≤ exp(-1e-5) ≈ 0.99999 — strictly < 1 even in float32
        # (without this, exp(-2e-9) rounds to exactly 1.0 in float32 precision).
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)).clamp(min=1e-5))

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = A·h_t + B·e + transformer_out.

        Args:
            h               -- current hidden state (B, T, dim)
            e               -- encoded input from Prelude, frozen across loops (B, T, dim)
            transformer_out -- output of the recurrent TransformerBlock at this step (B, T, dim)

        Returns:
            Updated hidden state of shape (B, T, dim)
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# ACT halting (Adaptive Computation Time)
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting mechanism (Graves, 2016).

    Learns a per-position halting probability at each loop iteration. Positions
    where the hidden state has converged (high cumulative halting probability)
    stop accumulating updates, while positions still being refined continue.
    This lets easy tokens halt early and hard tokens receive more computation,
    all within the same batch. Also makes the model Turing-complete under
    certain assumptions about the expressiveness of the transformer block.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; input to the halting scalar predictor
        """
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-position halting probability from the current hidden state.

        Args:
            h -- hidden state of shape (B, T, dim)

        Returns:
            Halting probability tensor of shape (B, T), values in (0, 1)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Depth Cross-Attention (MoDA-style intra-layer self-recurrence)
# ---------------------------------------------------------------------------


class DepthCrossAttention(nn.Module):
    """Cross-loop depth attention for MoDA-style intra-layer self-recurrence.

    Each query at position i attends jointly to:
      - Sequence K/V from positions 0..i (causal, via mask).
      - Depth K/V at position i from all previous loop iterations (unconstrained).

    The depth K/V are written from the block output h via separate write projections
    and accumulated in a per-forward-call list. Later loops read this list to attend
    to what earlier loops computed at the same token position, enabling selective
    cross-depth information flow without parameter growth across loops.

    No RoPE is applied: positional information flows implicitly through h (already
    refined by the main attention's positional encoding). This avoids head-dim vs
    rope-dim mismatches between GQA and MLA modes.
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.scale = self.head_dim ** -0.5

        self.norm = RMSNorm(cfg.dim)
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        # Separate write projections: transform loop output h into depth cache K/V.
        self.wk_write = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv_write = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        depth_keys: list,
        depth_vals: list,
    ) -> torch.Tensor:
        """Attend to current sequence (causal) + previous loops' per-position depth K/V.

        Args:
            x          -- input hidden state (B, T, dim)
            mask       -- additive causal mask (1, 1, T, T) or None
            depth_keys -- list of L tensors (B, n_kv_heads, T, head_dim) from prior loops
            depth_vals -- matching depth value tensors

        Returns:
            Output delta tensor (B, T, dim) to be added as a residual to h
        """
        B, T, _ = x.shape
        H, Hk, d = self.n_heads, self.n_kv_heads, self.head_dim

        x_norm = self.norm(x)
        q = self.wq(x_norm).view(B, T, H, d).transpose(1, 2)      # (B, H, T, d)
        k_seq = self.wk(x_norm).view(B, T, Hk, d).transpose(1, 2)  # (B, Hk, T, d)
        v_seq = self.wv(x_norm).view(B, T, Hk, d).transpose(1, 2)  # (B, Hk, T, d)

        # GQA expansion: repeat KV heads to match query head count
        k_seq_e = k_seq.repeat_interleave(self.groups, dim=1)  # (B, H, T, d)
        v_seq_e = v_seq.repeat_interleave(self.groups, dim=1)  # (B, H, T, d)

        # Sequence logits with causal mask
        seq_logits = torch.matmul(q, k_seq_e.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        if mask is not None:
            seq_logits = seq_logits + mask

        L = len(depth_keys)
        if L == 0:
            weights = F.softmax(seq_logits, dim=-1)
            out = torch.matmul(weights, v_seq_e)
        else:
            # Stack depth K/V and rearrange to per-position layout.
            # Each position i sees depth_k[l][i] for l = 0..L-1 (same token, earlier loop).
            k_depth = torch.stack(depth_keys, dim=2)   # (B, Hk, L, T, d)
            v_depth = torch.stack(depth_vals, dim=2)   # (B, Hk, L, T, d)
            k_depth_e = k_depth.repeat_interleave(self.groups, dim=1)  # (B, H, L, T, d)
            v_depth_e = v_depth.repeat_interleave(self.groups, dim=1)  # (B, H, L, T, d)

            k_depth_pos = k_depth_e.permute(0, 1, 3, 2, 4)  # (B, H, T, L, d)
            v_depth_pos = v_depth_e.permute(0, 1, 3, 2, 4)  # (B, H, T, L, d)

            # Depth logits: q[b,h,i,:] · depth_k[b,h,i,l,:] for each (i, l)
            depth_logits = (q.unsqueeze(-2) * k_depth_pos).sum(-1) * self.scale  # (B, H, T, L)

            # Unified softmax over T sequence positions + L depth positions
            combined = torch.cat([seq_logits, depth_logits], dim=-1)  # (B, H, T, T+L)
            weights = F.softmax(combined, dim=-1)

            w_seq = weights[:, :, :, :T]    # (B, H, T, T)
            w_depth = weights[:, :, :, T:]  # (B, H, T, L)

            seq_contrib = torch.matmul(w_seq, v_seq_e)                     # (B, H, T, d)
            depth_contrib = (w_depth.unsqueeze(-1) * v_depth_pos).sum(-2)  # (B, H, T, d)
            out = seq_contrib + depth_contrib

        out = out.transpose(1, 2).contiguous().view(B, T, H * d)
        return self.wo(out)

    def write_cache(self, h: torch.Tensor) -> tuple:
        """Project loop output h into K/V for the depth cache.

        Args:
            h -- loop output hidden state (B, T, dim)

        Returns:
            (k, v) each of shape (B, n_kv_heads, T, head_dim)
        """
        B, T, _ = h.shape
        k = self.wk_write(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv_write(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return k, v


# ---------------------------------------------------------------------------
# Recurrent Block (one set of weights, looped T times)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """
    The core recurrent block of BushidoMythos — a single TransformerBlock looped T times.

    At each loop iteration t, the hidden state h is updated via:
        1. loop_index_embedding: inject sinusoidal loop-index signal into h
        2. TransformerBlock:     compute attention + MoE FFN on normalized (h + e)
        3. LoRAAdapter:          apply depth-wise LoRA delta to transformer output
        4. LTIInjection:         stable update h = A·h + B·e + transformer_out
        5. ACTHalting:           accumulate per-position halting probabilities;
                                  positions that have converged stop contributing

    The encoded input e (output of the Prelude) is injected at every step to keep
    the original input signal alive across arbitrary loop depth, preventing drift.
    The ACT mechanism produces a weighted sum of hidden states across iterations,
    where the weights reflect when each position converged.

    More loop iterations at inference = deeper reasoning chains, following the
    depth-extrapolation property of looped transformers (Saunshi et al., 2025).
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, lora_rank, max_loop_iters, act_threshold
        """
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.register_buffer(
            "_act_threshold",
            torch.tensor(float(cfg.act_threshold)),
            persistent=False,
        )
        self.loop_dim = (
            cfg.dim // 8
        )  # fraction of channels receiving loop-index embedding
        self._last_ponder_cost: torch.Tensor = torch.tensor(0.0)
        if cfg.use_depth_attn:
            self.depth_attn = DepthCrossAttention(cfg)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Run the recurrent loop for up to n_loops iterations with ACT early exit.

        Args:
            h        -- initial hidden state from the Prelude, shape (B, T, dim)
            e        -- encoded input frozen for injection each step, shape (B, T, dim)
            freqs_cis-- precomputed RoPE frequencies
            mask     -- additive causal mask or None
            n_loops  -- number of loop iterations; defaults to cfg.max_loop_iters.
                        When cfg.loop_curriculum is True during training, a random
                        value in [1, max_loop_iters] is sampled to encourage
                        inference-time depth extrapolation.
            kv_cache -- cache dict passed through to the inner TransformerBlock;
                        each loop iteration uses a separate cache key

        Returns:
            ACT-weighted sum of hidden states across iterations, shape (B, T, dim).
            Ponder cost (mean loop steps used) is stored in self._last_ponder_cost.
        """
        if n_loops is None:
            if self.training and self.cfg.loop_curriculum:
                # Sample random depth during training so the model learns to solve
                # problems with varying compute budgets (depth extrapolation).
                n_loops = random.randint(1, self.cfg.max_loop_iters)
            else:
                n_loops = self.cfg.max_loop_iters
        B, T, _ = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)
        ponder_steps = torch.zeros(B, T, device=h.device)
        # combined at the halt step, saved so loop embedding doesn't keep
        # changing K/V for halted positions in subsequent iterations.
        combined_frozen: Optional[torch.Tensor] = None
        depth_keys: list = []
        depth_vals: list = []

        for t in range(n_loops):
            # Freeze h for halted positions before building combined.
            # detach() stops gradient flow through halted positions in later loops.
            if halted.any():
                h_in = torch.where(halted.unsqueeze(-1), h.detach(), h)
            else:
                h_in = h

            h_loop = loop_index_embedding(h_in, t, self.loop_dim)
            combined_new = self.norm(h_loop + e)

            # Substitute frozen combined for halted positions so the loop-index
            # embedding doesn't alter their K/V representation each iteration.
            if combined_frozen is not None and halted.any():
                combined = torch.where(halted.unsqueeze(-1), combined_frozen, combined_new)
            else:
                combined = combined_new

            cache_key = f"recurrent_loop_{t}"
            if self.cfg.use_gradient_checkpointing and self.training and kv_cache is None:
                # Recompute this iteration's activations during backward instead of
                # storing them. kv_cache must NOT enter the checkpointed region:
                # it is a mutable dict modified in-place and checkpoint would
                # re-execute those mutations on the recomputation pass.
                _t, _key = t, cache_key
                def _ckpt_fn(c, t_=_t, k_=_key):
                    out = self.block(c, freqs_cis, mask, None, k_, is_causal)
                    return out + self.lora(out, t_)
                trans_out = _grad_ckpt(_ckpt_fn, combined, use_reentrant=False)
            else:
                trans_out = self.block(
                    combined, freqs_cis, mask, kv_cache, cache_key, is_causal
                )
                trans_out = trans_out + self.lora(trans_out, t)
            h_new = self.injection(h_in, e, trans_out)

            # Depth cross-attention: read from previous loop iterations' K/V.
            # Skipped on the first loop (no depth cache yet). Applied only to
            # non-halted positions; halted positions' delta is zeroed out.
            if self.cfg.use_depth_attn and hasattr(self, "depth_attn") and depth_keys:
                depth_delta = self.depth_attn(h_new, mask, depth_keys, depth_vals)
                if halted.any():
                    depth_delta = torch.where(
                        halted.unsqueeze(-1),
                        torch.zeros_like(depth_delta),
                        depth_delta,
                    )
                h_new = h_new + depth_delta

            # Only update still-running positions
            if halted.any():
                h = torch.where(halted.unsqueeze(-1), h, h_new)
            else:
                h = h_new

            p = self.act(h)  # (B, T)
            still_running = ~halted

            # Accumulate ponder steps for ACT auxiliary loss
            ponder_steps = ponder_steps + still_running.float()

            # ACT remainder trick: once cumulative_p + p crosses threshold,
            # assign the remaining probability mass as the final weight.
            # Gate by still_running so halted positions contribute exactly
            # once (on the halting step) and zero thereafter — otherwise
            # threshold<1 leaves a non-zero remainder that leaks every step.
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self._act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()

            # Capture combined_new for positions halting THIS step, before
            # updating halted, so future loops use their halt-step K/V representation.
            newly_halting = still_running & (cumulative_p >= self._act_threshold)
            if newly_halting.any():
                if combined_frozen is None:
                    combined_frozen = combined_new.detach().clone()
                else:
                    combined_frozen = torch.where(
                        newly_halting.unsqueeze(-1),
                        combined_new.detach(),
                        combined_frozen,
                    )

            halted = halted | newly_halting

            # Write current h to depth cache for the next loop's cross-attention.
            if self.cfg.use_depth_attn and hasattr(self, "depth_attn"):
                dk, dv = self.depth_attn.write_cache(h)
                depth_keys.append(dk)
                depth_vals.append(dv)

            # Only short-circuit when there is no KV cache to keep consistent.
            # With a cache, every loop depth must run on every forward pass so
            # later decode steps find populated keys at every cache_key.
            if halted.all() and kv_cache is None:
                break

        # Positions that exhausted n_loops without crossing the threshold have
        # cumulative_p < act_threshold, so their h_out weights sum to < 1.
        # Assign the remaining mass to the final h to restore the invariant.
        still_not_halted = ~halted
        if still_not_halted.any():
            final_remainder = (1.0 - cumulative_p).clamp(min=0)
            h_out = h_out + (final_remainder * still_not_halted.float()).unsqueeze(-1) * h

        # Store mean ponder cost; BushidoMythos.forward() scales by act_aux_loss_weight.
        self._last_ponder_cost = ponder_steps.mean()
        return h_out


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class BushidoMythos(nn.Module):
    """
    BushidoMythos — Recurrent-Depth Transformer language model.

    Implements the hypothesized Claude Mythos architecture as a Recurrent-Depth
    Transformer (RDT). The model divides computation into three functional blocks:

        Input tokens
             ↓
        [Prelude]          — prelude_layers standard transformer blocks, run once
             ↓
        [Recurrent Block]  — one transformer block looped T times with input injection
             ↑_______↓      h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
             ↓
        [Coda]             — coda_layers standard transformer blocks, run once
             ↓
        Output logits

    Key properties:
    - Same weights, more loops → deeper reasoning, no parameter growth
    - Depth extrapolation: train on N loops, test on N+k loops (emergent)
    - ACT halting: variable compute per position within a batch
    - MoE FFN in the recurrent block: breadth across domains
    - LTI-stable injection: spectral radius < 1 guaranteed by construction
    - Supports both GQA and MLA attention (set via cfg.attn_type)
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig specifying all architecture hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # GQA uses full head_dim for RoPE; MLA uses only qk_rope_head_dim (decoupled)
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()
        self.register_buffer(
            "_act_aux_loss_weight",
            torch.tensor(float(cfg.act_aux_loss_weight)),
            persistent=False,
        )
        self._last_hidden: Optional[torch.Tensor] = None
        self._last_aux_loss: torch.Tensor = torch.tensor(0.0)

    def _init_weights(self) -> None:
        """Initialize all linear and embedding weights with N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @torch.no_grad()
    def set_act_curriculum_values(self, threshold: float, ponder_weight: float) -> None:
        """Update compile-safe ACT scalars while keeping cfg values in sync."""
        self.recurrent._act_threshold.fill_(threshold)
        self._act_aux_loss_weight.fill_(ponder_weight)
        self.cfg.act_threshold = float(threshold)
        self.cfg.act_aux_loss_weight = float(ponder_weight)

    def set_grouped_moe(
        self, enabled: bool, compute_dtype: Optional[torch.dtype] = None
    ) -> bool:
        """Select native grouped GEMM without changing config or checkpoint schema."""
        active = False
        for module in self.modules():
            if isinstance(module, MoEFFN):
                runtime_dtype = compute_dtype or module.router.weight.dtype
                supported, _ = grouped_moe_runtime_status(
                    module.router.weight.device, runtime_dtype
                )
                module.use_grouped_moe = bool(enabled and supported)
                active = active or module.use_grouped_moe
        return active

    @torch.no_grad()
    def update_moe_router_bias(
        self, bias_lr: float = 1e-3, distributed: bool = False
    ) -> None:
        """Update MoE router_bias for aux-loss-free load balancing (DeepSeek-V3 style).

        Call after each optimizer.step(). Adjusts router_bias so that overloaded
        experts become less likely to be selected and underloaded experts more likely,
        without adding any gradient-based loss term.

        In DDP/FSDP training, set distributed=True to all_reduce expert counts
        across ranks before updating — this ensures all ranks apply the same bias
        delta and prevents rank-level routing drift over long training runs.

        Rule: bias_i -= bias_lr * sign(load_i - target_load)
        where target_load = total_tokens * topk / n_experts (perfect balance).
        """
        for module in self.modules():
            if isinstance(module, MoEFFN) and module._accum_expert_counts is not None:
                counts = module._accum_expert_counts
                if distributed:
                    import torch.distributed as dist
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                target = counts.sum() / module.n_experts
                overload = counts - target
                module.router_bias.data -= bias_lr * overload.sign()
                module._accum_expert_counts = None  # reset for next accumulation window

    @staticmethod
    def _causal_mask(
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build an additive causal mask: 0 on and below the diagonal, -inf above.

        Args:
            seq_len -- sequence length
            device  -- target device
            dtype   -- tensor dtype (must match activation dtype so the additive
                       mask doesn't upcast the attention logits in the fallback
                       attention path — e.g. bf16 weights with an fp32 mask
                       promotes attn to fp32 and then breaks the fp32-vs-bf16
                       matmul against V)

        Returns:
            Tensor of shape (1, 1, seq_len, seq_len) broadcastable over (B, H, T, S)
        """
        # Build the boolean upper-triangle mask first, then fill -inf into a
        # zero tensor. Doing torch.triu on a -inf-filled tensor produces NaN on
        # MPS (Apple Silicon) because triu zeros below-diagonal elements via
        # multiplication, and -inf * 0 = NaN on that backend.
        upper = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1)
        mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
        return mask.masked_fill(upper, float("-inf"))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
        inputs_embeds: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[int] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through Prelude → Recurrent Block → Coda.

        Args:
            input_ids     -- token indices of shape (B, T); mutually exclusive with inputs_embeds
            n_loops       -- recurrent loop depth; defaults to cfg.max_loop_iters.
                             Increase at inference to extrapolate to harder problems.
            kv_cache      -- dict mutated in-place for autoregressive KV caching;
                             pass an empty dict {} and reuse across decode steps
            start_pos     -- index of the first token in the full sequence; selects the
                             correct RoPE frequencies during incremental decoding
                             (0 for prefill, prompt_len for each decode step)
            inputs_embeds -- pre-computed embeddings (B, T, dim); bypasses the token
                             embedding lookup. Used by generate_coconut() to feed the
                             last hidden state directly as the next input (COCONUT).
            return_hidden -- return normalized hidden states before the tied LM head;
                             intended for memory-efficient training losses

        Returns:
            Logits of shape (B, T, vocab_size), or normalized hidden states of
            shape (B, T, dim) when return_hidden=True.
            Side effects: self._last_hidden stores the pre-lm-head hidden state;
            self._last_aux_loss stores the scaled ACT ponder cost.
        """
        if inputs_embeds is not None:
            x = inputs_embeds
            T = x.shape[1]
            device = x.device
        elif input_ids is not None:
            T = input_ids.shape[1]
            device = input_ids.device
            x = self.embed(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        if T == 0:
            raise ValueError(
                f"Input sequence length is 0 (input_ids.shape={getattr(input_ids, 'shape', None)}, "
                f"inputs_embeds.shape={getattr(inputs_embeds, 'shape', None)}). "
                "The tokenizer likely returned an empty token list for the given prompt."
            )

        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]
        # SDPA can construct the standard causal bias internally without an
        # explicit T x T mask. Chunked cache updates need the padded mask because
        # their query and key lengths differ; single-token decode needs no mask.
        is_causal = T > 1 and (kv_cache is None or start_pos == 0)
        mask = (
            self._causal_mask(T, device, x.dtype)
            if T > 1 and kv_cache is not None and start_pos > 0
            else None
        )

        for i, layer in enumerate(self.prelude):
            x = layer(
                x,
                freqs_cis,
                mask,
                kv_cache,
                cache_key=f"prelude_{i}",
                is_causal=is_causal,
            )

        e = x  # encoded input frozen for injection every loop
        x = self.recurrent(
            x, e, freqs_cis, mask, n_loops, kv_cache, is_causal
        )
        self._last_aux_loss = (
            self.recurrent._last_ponder_cost * self._act_aux_loss_weight
        )

        for i, layer in enumerate(self.coda):
            x = layer(
                x,
                freqs_cis,
                mask,
                kv_cache,
                cache_key=f"coda_{i}",
                is_causal=is_causal,
            )

        hidden = self.norm(x)
        self._last_hidden = hidden
        if return_hidden:
            if logits_to_keep is not None:
                raise ValueError("return_hidden cannot be combined with logits_to_keep")
            return hidden
        if logits_to_keep is not None:
            if logits_to_keep < 1:
                raise ValueError("logits_to_keep must be at least 1")
            hidden = hidden[:, -logits_to_keep:]
        return self.head(hidden)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with KV caching.

        On step 0 the full prompt is processed. On subsequent steps only the
        last generated token is passed, with all previous keys and values
        retrieved from kv_cache. This keeps decode cost proportional to one
        token per step rather than the full growing sequence.

        n_loops can be set higher than the training value to extrapolate to
        harder problems at inference time (depth extrapolation property).

        Args:
            input_ids          -- prompt token indices of shape (B, T)
            max_new_tokens     -- number of tokens to generate
            n_loops            -- recurrent loop depth for each decode step
            temperature        -- softmax temperature; lower = more greedy
            top_k              -- restrict sampling to top-K logits (0 = disabled)
            repetition_penalty -- > 1.0 penalises tokens already in the sequence;
                                   1.3 is a good starting value (1.0 = disabled)
            eos_token_id       -- if set, generation stops once every sequence in the
                                   batch has emitted this token (e.g. GPT-2 <|endoftext|>
                                   = 50256). None disables early stopping.

        Returns:
            Token indices of shape (B, T + n) where n <= max_new_tokens
            (n < max_new_tokens if all sequences hit eos_token_id early).
        """
        was_training = self.training
        self.eval()
        try:
            return self._generate_inner(
                input_ids, max_new_tokens, n_loops, temperature, top_k,
                repetition_penalty, eos_token_id
            )
        finally:
            self.train(was_training)

    def _generate_inner(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        n_loops: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        prompt_len = input_ids.shape[1]
        if prompt_len == 0:
            raise ValueError(
                f"input_ids is empty (shape {input_ids.shape}). "
                "The tokenizer returned no tokens for the given prompt."
            )
        kv_cache: dict = _KVCache(capacity=prompt_len + max_new_tokens)
        # Track which batch rows have emitted eos_token_id so we can stop early.
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
            logits = self.forward(
                cur_ids,
                n_loops=n_loops,
                kv_cache=kv_cache,
                start_pos=start_pos,
                logits_to_keep=1,
            )
            if logits.shape[1] == 0:
                raise RuntimeError(
                    f"forward() returned empty logits (shape {logits.shape}) at step={step}. "
                    f"cur_ids.shape={cur_ids.shape}, start_pos={start_pos}, prompt_len={prompt_len}. "
                    "This may indicate a KV cache inconsistency or model state issue."
                )
            logits = logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                # 既出トークンのロジットにペナルティ（HuggingFace 方式）
                for b in range(input_ids.shape[0]):
                    unique_ids = input_ids[b].unique()
                    score = logits[b, unique_ids]
                    logits[b, unique_ids] = torch.where(
                        score < 0, score * repetition_penalty, score / repetition_penalty
                    )
            if top_k > 0:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None:
                # 既に終了した行は EOS で埋める（他行の生成が続いても汚さない）
                if bool(finished.any()):
                    next_tok[finished] = eos_token_id
                input_ids = torch.cat([input_ids, next_tok], dim=1)
                finished |= next_tok.squeeze(1) == eos_token_id
                if bool(finished.all()):
                    break
            else:
                input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    @torch.no_grad()
    def generate_coconut(
        self,
        input_ids: torch.Tensor,
        coconut_steps: Optional[int] = None,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate tokens using COCONUT continuous latent thought steps.

        Before producing discrete tokens, runs coconut_steps iterations where the
        model's last hidden state is fed directly as the next input embedding,
        bypassing the vocabulary projection. This continuous latent reasoning
        preserves more information per step than discrete chain-of-thought tokens.

        Args:
            input_ids      -- prompt token indices of shape (B, T)
            coconut_steps  -- continuous thought steps; defaults to cfg.coconut_steps
            max_new_tokens -- discrete tokens to generate after thought steps
            n_loops        -- recurrent loop depth for each forward pass
            temperature    -- softmax temperature for sampling
            top_k          -- restrict sampling to top-K logits (0 = disabled)

        Returns:
            Token indices of shape (B, T + max_new_tokens)
        """
        was_training = self.training
        self.eval()
        try:
            steps = coconut_steps if coconut_steps is not None else self.cfg.coconut_steps
            kv_cache: dict = _KVCache(
                capacity=input_ids.shape[1] + steps + max_new_tokens
            )
            prompt_len = input_ids.shape[1]

            # Phase 1: process full prompt; KV cache is populated, _last_hidden captured
            last_logits = self.forward(
                input_ids,
                n_loops=n_loops,
                kv_cache=kv_cache,
                start_pos=0,
                logits_to_keep=1,
            )

            # Phase 2: continuous thought steps — feed last hidden as next input embedding
            for step in range(steps):
                if self._last_hidden is None:
                    raise RuntimeError("self._last_hidden is None but coconut_steps > 0")
                thought_emb = self._last_hidden[:, -1:, :]  # (B, 1, dim)
                last_logits = self.forward(
                    inputs_embeds=thought_emb,
                    n_loops=n_loops,
                    kv_cache=kv_cache,
                    start_pos=prompt_len + step,
                    logits_to_keep=1,
                )

            def _sample(logits_2d: torch.Tensor) -> torch.Tensor:
                logits_2d = logits_2d / temperature
                if top_k > 0:
                    v, _ = logits_2d.topk(min(top_k, logits_2d.size(-1)))
                    logits_2d[logits_2d < v[:, -1:]] = float("-inf")
                return torch.multinomial(F.softmax(logits_2d, dim=-1), num_samples=1)

            if max_new_tokens == 0:
                return input_ids

            # Phase 3: sample first discrete token from the last thought step's logits
            new_ids = _sample(last_logits[:, -1, :])  # (B, 1)

            # Autoregressively generate remaining tokens using the populated KV cache
            for step in range(1, max_new_tokens):
                start_pos = prompt_len + steps + step - 1
                logits = self.forward(
                    new_ids[:, -1:],
                    n_loops=n_loops,
                    kv_cache=kv_cache,
                    start_pos=start_pos,
                    logits_to_keep=1,
                )
                next_tok = _sample(logits[:, -1, :])
                new_ids = torch.cat([new_ids, next_tok], dim=1)

            return torch.cat([input_ids, new_ids], dim=1)
        finally:
            self.train(was_training)
