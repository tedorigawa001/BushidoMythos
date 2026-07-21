# `BushidoMythos` — Class Reference

**Module:** `bushido_mythos.main`  
**Base class:** `torch.nn.Module`

---

## Overview

`BushidoMythos` is the top-level model class implementing the Recurrent-Depth Transformer (RDT) architecture described in [the BushidoMythos README](../README.md). It assembles three functional stages — **Prelude**, **Recurrent Block**, and **Coda** — into a complete autoregressive language model for financial-trading research, market reasoning, risk review, and Bushido-inspired decision discipline.

```
Input token IDs  (B, T)
        ↓
   [Embedding]          token index → dim-dimensional vector
        ↓
   [Prelude]            prelude_layers × standard TransformerBlock  (run once)
        ↓
   [Recurrent Block]    one TransformerBlock looped T times
        ↑___________↓   h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
        ↓
   [Coda]               coda_layers × standard TransformerBlock  (run once)
        ↓
   [RMSNorm → LM head]
        ↓
Output logits  (B, T, vocab_size)
```

Every architectural choice in `BushidoMythos` can be configured through a single [`MythosConfig`](#mythosconfig) dataclass passed at construction.

---

## `MythosConfig`

```python
@dataclass
class MythosConfig
```

All hyperparameters for the model are stored in this single frozen-style dataclass. Pass an instance to `BushidoMythos.__init__`.

### Core fields

| Field | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int` | `32000` | Token vocabulary size; sets the embedding and LM head dimension |
| `dim` | `int` | `2048` | Model hidden dimension — the width of the residual stream throughout |
| `n_heads` | `int` | `16` | Number of query attention heads |
| `n_kv_heads` | `int` | `4` | Number of key/value heads (GQA only); `n_heads // n_kv_heads` Q heads share each KV pair |
| `max_seq_len` | `int` | `4096` | Maximum sequence length; RoPE frequencies are precomputed up to this length |
| `max_loop_iters` | `int` | `16` | Default recurrent loop depth T at inference. Can be overridden per call |
| `prelude_layers` | `int` | `2` | Number of standard transformer blocks run once before the recurrent loop |
| `coda_layers` | `int` | `2` | Number of standard transformer blocks run once after the recurrent loop |

### Advanced feature fields

All of these default to `False` / `0` for full backward compatibility with checkpoints trained without them.

| Field | Type | Default | Description |
|---|---|---|---|
| `use_hyper_connections` | `bool` | `False` | Replace residual `x + f(x)` with `α·x + β·f(x)` where α and β are learned per-channel vectors initialized to 1. Adds 4 × `dim` parameters per `TransformerBlock`. |
| `act_aux_loss_weight` | `float` | `0.0` | Scale factor for the ACT ponder-cost auxiliary loss. When > 0, `model._last_aux_loss` is non-zero after each forward and should be added to the CE loss during training: `loss = ce_loss + model._last_aux_loss`. |
| `loop_curriculum` | `bool` | `False` | During training (`model.train()` mode), randomly sample `n_loops ∈ [1, max_loop_iters]` each forward pass instead of always using `max_loop_iters`. Improves depth extrapolation at inference. Has no effect in eval mode. |
| `coconut_steps` | `int` | `0` | Default number of continuous latent thought steps in `generate_coconut()`. Can be overridden per call. |
| `use_depth_attn` | `bool` | `False` | Attach a `DepthCrossAttention` module to `RecurrentBlock`. Each loop step attends to the per-position K/V written by all previous loop iterations (MoDA-style cross-loop attention), enabling selective cross-depth information flow. |

### Attention fields

`attn_type` selects between two complete attention implementations. All other attention fields are implementation-specific.

| Field | Type | Default | Description |
|---|---|---|---|
| `attn_type` | `str` | `"mla"` | `"gqa"` for Grouped Query Attention; `"mla"` for Multi-Latent Attention |
| `kv_lora_rank` | `int` | `512` | **[MLA only]** Compressed KV latent rank stored in the cache instead of full K and V |
| `q_lora_rank` | `int` | `1536` | **[MLA only]** Compressed Q latent rank |
| `qk_rope_head_dim` | `int` | `64` | **[MLA only]** Per-head dimension receiving RoPE positional encoding |
| `qk_nope_head_dim` | `int` | `128` | **[MLA only]** Per-head dimension without positional encoding |
| `v_head_dim` | `int` | `128` | **[MLA only]** Per-head value dimension |

**GQA vs MLA:** GQA reduces KV cache by having fewer KV heads than Q heads (factor of `n_heads / n_kv_heads`). MLA achieves a much larger reduction by caching a low-rank KV latent (`kv_lora_rank`) and the RoPE keys (`n_heads × qk_rope_head_dim`), then reconstructing full K and V on the fly. At production scale MLA yields roughly 10–20× smaller KV cache than standard attention.

### MoE FFN fields

The Mixture-of-Experts FFN is used exclusively inside the Recurrent Block. Prelude and Coda use a dense SwiGLU FFN.

| Field | Type | Default | Description |
|---|---|---|---|
| `n_experts` | `int` | `64` | Total number of routed expert FFNs |
| `n_shared_experts` | `int` | `2` | Always-active shared experts; absorb common cross-domain patterns |
| `n_experts_per_tok` | `int` | `4` | Top-K routed experts selected per token by the router |
| `expert_dim` | `int` | `512` | Hidden dimension inside each fine-grained routed expert |

Approximately `n_experts_per_tok / n_experts = 6.25%` of routed expert parameters are activated per token, plus all shared expert parameters.

### Stability and adaptation fields

| Field | Type | Default | Description |
|---|---|---|---|
| `act_threshold` | `float` | `0.99` | ACT cumulative halting threshold; loop exits per-position once this is exceeded |
| `rope_theta` | `float` | `500000.0` | RoPE base frequency (LLaMA-3 default; higher = slower frequency decay over sequence positions) |
| `lora_rank` | `int` | `16` | Rank of the depth-wise LoRA adapter applied inside each loop iteration |

---

## Constructor

```python
BushidoMythos(cfg: MythosConfig)
```

Builds all sub-modules, precomputes RoPE frequency buffers, and runs weight initialization.

**What happens internally:**

1. `nn.Embedding(vocab_size, dim)` — token embedding table, weight-tied with the LM head.
2. RoPE buffers — `freqs_cis` (for GQA, dim = `dim // n_heads`) and `freqs_cis_mla` (for MLA, dim = `qk_rope_head_dim`) are precomputed once and registered as non-parameter buffers. The correct buffer is selected at forward time based on `cfg.attn_type`.
3. `prelude` — `nn.ModuleList` of `prelude_layers` `TransformerBlock` instances with dense SwiGLU FFN.
4. `recurrent` — a single `RecurrentBlock` containing one `TransformerBlock` (with MoE FFN), `LTIInjection`, `ACTHalting`, and `LoRAAdapter`.
5. `coda` — `nn.ModuleList` of `coda_layers` `TransformerBlock` instances with dense SwiGLU FFN.
6. `RMSNorm(dim)` applied before the LM head.
7. `nn.Linear(dim, vocab_size, bias=False)` LM head with weights tied to the embedding.
8. All `nn.Linear` and `nn.Embedding` weights initialized from N(0, 0.02).

**Example:**

```python
from bushido_mythos.main import BushidoMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=32000,
    dim=2048,
    n_heads=16,
    n_kv_heads=4,
    max_loop_iters=16,
    attn_type="mla",
)
model = BushidoMythos(cfg)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## `forward`

```python
def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    n_loops: Optional[int] = None,
    kv_cache: Optional[dict] = None,
    start_pos: int = 0,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

Single forward pass through the full Prelude → Recurrent Block → Coda pipeline.

Provide either `input_ids` or `inputs_embeds`. `inputs_embeds` is intended for COCONUT-style continuous latent thought steps and bypasses the token embedding lookup.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `input_ids` | `Tensor (B, T) \| None` | Batch of token index sequences. `B` = batch size, `T` = sequence length. Mutually exclusive with `inputs_embeds`. |
| `n_loops` | `int \| None` | Recurrent loop depth for this call. Defaults to `cfg.max_loop_iters`. Pass a higher value at inference to extrapolate to harder problems (depth extrapolation property). When `cfg.loop_curriculum=True` and the model is in training mode, this parameter is ignored and `n_loops` is sampled randomly from `[1, max_loop_iters]`. |
| `kv_cache` | `dict \| None` | If provided, keys and values are accumulated here for autoregressive decoding. Pass `{}` on the first decode step and reuse the same dict across steps. Pass `None` for training or full-context inference. |
| `start_pos` | `int` | Starting sequence position for KV cache offset in autoregressive decoding. Defaults to `0`. Used internally by `generate_coconut()`. |
| `inputs_embeds` | `Tensor (B, T, dim) \| None` | Pre-computed input embeddings. When provided, the embedding lookup is skipped. Used by `generate_coconut()` to feed continuous latent thought vectors directly into the model. |

### Side effects

After every `forward` call:
- `model._last_aux_loss` — scalar tensor with the ACT auxiliary loss (`ponder_steps.mean() × act_aux_loss_weight`). Zero when `act_aux_loss_weight=0.0`. Add to CE loss during training.
- `model._last_hidden` — final hidden states `(B, T, dim)` before the LM head projection. Used by `generate_coconut()` as continuous thought embeddings.

### Returns

`Tensor (B, T, vocab_size)` — raw (unnormalized) logits over the vocabulary for each position.

### Behavior walkthrough

```
1. Embed:     x = embedding(input_ids)              # (B, T, dim)
2. Select RoPE buffer:
     if attn_type == "mla": use freqs_cis_mla[:T]
     else:                   use freqs_cis[:T]
3. Build causal mask (upper-triangular -inf):
     if T > 1: mask = _causal_mask(T, device)
     else:     mask = None  (single-token decode step)
4. Prelude:
     for i, layer in prelude:
         x = layer(x, freqs_cis, mask, kv_cache, f"prelude_{i}")
5. Freeze encoded input:
     e = x                                          # (B, T, dim)
6. Recurrent loop:
     x = recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
7. Coda:
     for i, layer in coda:
         x = layer(x, freqs_cis, mask, kv_cache, f"coda_{i}")
8. Project:   logits = lm_head(norm(x))             # (B, T, vocab_size)
```

**Step 5 (freeze `e`)** is the key architectural invariant: the encoded input `e` is captured after the Prelude and injected at *every* loop iteration unchanged. This prevents the hidden state from drifting away from the original input signal regardless of loop depth.

### Training example

```python
import torch
from bushido_mythos.main import BushidoMythos, MythosConfig

cfg = MythosConfig(act_aux_loss_weight=0.01, loop_curriculum=True)
model = BushidoMythos(cfg).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

input_ids = torch.randint(0, 32000, (2, 512)).cuda()
labels    = torch.randint(0, 32000, (2, 512)).cuda()

logits   = model(input_ids)                  # (2, 512, 32000)
ce_loss  = torch.nn.functional.cross_entropy(
    logits.view(-1, 32000),
    labels.view(-1),
)
# Add ACT auxiliary loss to encourage early halting
loss = ce_loss + model._last_aux_loss
loss.backward()
optimizer.step()

# Update MoE router bias after each optimizer step to balance expert load
model.update_moe_router_bias(bias_lr=1e-3)
```

### Depth extrapolation at inference

A looped transformer trained on `N` loops can be evaluated on `N + k` loops and often achieves higher quality on hard multi-hop problems. Pass `n_loops` at inference time:

```python
# Trained with max_loop_iters=16 — try deeper reasoning at test time
logits_deep = model(input_ids, n_loops=32)
```

---

## `generate`

```python
@torch.no_grad()
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    n_loops: int = 8,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor
```

Autoregressive token generation with KV caching. Processes the full prompt on step 0, then decodes one token at a time using the accumulated cache.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_ids` | `Tensor (B, T)` | — | Prompt token indices |
| `max_new_tokens` | `int` | `64` | Number of new tokens to generate |
| `n_loops` | `int` | `8` | Recurrent loop depth per decode step. Can be higher than the training value for harder prompts (depth extrapolation) |
| `temperature` | `float` | `1.0` | Softmax temperature applied to logits before sampling. Values < 1 make the distribution more peaked (less random); values > 1 make it flatter |
| `top_k` | `int` | `50` | Restricts sampling to the top-K most probable tokens at each step. `0` disables filtering (full vocabulary sampling). Automatically clamped to `vocab_size` to prevent out-of-bounds errors |

### Returns

`Tensor (B, T + max_new_tokens)` — the original prompt concatenated with the generated token indices.

### KV caching mechanism

On step 0, the full prompt `(B, T)` is passed and all keys/values for every layer are populated in `kv_cache`. On steps 1…N only the single most recent token `(B, 1)` is passed; the attention layers read back all prior K/V from the cache. This makes decode cost proportional to a single token per step rather than the full growing sequence.

Each layer caches under a deterministic string key (`"prelude_0"`, `"recurrent_loop_3"`, `"coda_1"`, etc.), so caches from different layers never collide.

### Sampling strategy

```
logits = forward(cur_ids, n_loops, kv_cache)[:, -1, :] / temperature

if top_k > 0:
    k = min(top_k, logits.size(-1))   # clamp to vocab_size
    threshold = logits.topk(k).values[:, -1:]
    logits[logits < threshold] = -inf

probs    = softmax(logits)
next_tok = multinomial(probs, num_samples=1)
```

### Generation example

```python
import torch
from bushido_mythos.main import BushidoMythos, MythosConfig

model = BushidoMythos(MythosConfig()).eval()

# Tokenized prompt (use your tokenizer of choice)
prompt = torch.tensor([[1, 450, 3118, 310, 278]])   # (1, 5)

output = model.generate(
    prompt,
    max_new_tokens=128,
    n_loops=16,        # deeper reasoning
    temperature=0.8,
    top_k=40,
)
# output.shape == (1, 133)
```

---

## `generate_coconut`

```python
@torch.no_grad()
def generate_coconut(
    self,
    input_ids: torch.Tensor,
    coconut_steps: Optional[int] = None,
    max_new_tokens: int = 64,
    n_loops: int = 8,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor
```

Token generation using COCONUT continuous latent thought steps ([Hao et al., 2024](https://arxiv.org/abs/2412.06769)). Before generating discrete tokens, the model runs `coconut_steps` iterations where `model._last_hidden[:, -1:, :]` is fed back as the next input embedding, bypassing the vocabulary projection. This keeps reasoning in continuous latent space and preserves more information per step than discrete chain-of-thought tokens.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_ids` | `Tensor (B, T)` | — | Prompt token indices |
| `coconut_steps` | `int \| None` | `cfg.coconut_steps` | Number of continuous latent thought steps before discrete generation |
| `max_new_tokens` | `int` | `64` | Number of discrete tokens to generate after thought steps. Pass `0` to run thought steps only and return the original prompt unchanged. |
| `n_loops` | `int` | `8` | Recurrent loop depth per forward pass |
| `temperature` | `float` | `1.0` | Softmax temperature for discrete token sampling |
| `top_k` | `int` | `50` | Restrict sampling to top-K logits. `0` = full vocabulary |

### Returns

`Tensor (B, T + max_new_tokens)` — prompt concatenated with generated tokens. If `max_new_tokens=0`, returns `input_ids` unchanged.

### Three-phase pipeline

```
Phase 1: forward(input_ids)           → KV cache populated, _last_hidden stored
Phase 2: for i in range(coconut_steps):
              forward(inputs_embeds=_last_hidden[:, -1:, :])  → refine latent
Phase 3: sample first discrete token from last thought logits
         autoregressive decode for (max_new_tokens - 1) more tokens
```

The model is temporarily set to `eval()` for the duration of the call and restored to its original mode (`train`/`eval`) afterward.

### Example

```python
model = BushidoMythos(MythosConfig(coconut_steps=4)).eval()
prompt = torch.tensor([[1, 42, 117, 8]])  # (1, 4)

# 4 continuous thought steps, then generate 32 tokens
output = model.generate_coconut(prompt, coconut_steps=4, max_new_tokens=32)
# output.shape == (1, 36)

# Thought-only mode: run continuous reasoning without emitting tokens
refined = model.generate_coconut(prompt, coconut_steps=8, max_new_tokens=0)
# refined.shape == (1, 4)  ← same as prompt
```

---

## `update_moe_router_bias`

```python
@torch.no_grad()
def update_moe_router_bias(
    self,
    bias_lr: float = 1e-3,
    distributed: bool = False,
) -> None
```

Updates the `router_bias` buffer of every `MoEFFN` module in the model using the DeepSeek-V3 auxiliary-loss-free load balancing algorithm. Call this once per optimizer step (after `optimizer.step()`), outside of the gradient computation.

### Algorithm

For each `MoEFFN` that has accumulated forward-pass expert selection counts:

```
target = total_tokens_routed / n_experts          # uniform ideal load
overload = _accum_expert_counts - target
router_bias -= bias_lr × sign(overload)           # ±bias_lr per over/under-loaded expert
_accum_expert_counts = None                       # reset for next accumulation window
```

Expert counts accumulate across all microbatches (gradient accumulation steps) between optimizer steps. The sign update avoids distorting the router logit scale, unlike auxiliary loss penalties.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bias_lr` | `float` | `1e-3` | Step size applied per expert per optimizer step. Tune between `1e-4` and `1e-2`. |
| `distributed` | `bool` | `False` | When `True`, calls `dist.all_reduce(counts, op=SUM)` to aggregate expert counts across all DDP/FSDP ranks before updating. Required for multi-GPU training to prevent per-rank bias drift. |

### Example

```python
# Single-GPU training
optimizer.step()
model.update_moe_router_bias(bias_lr=1e-3)

# Multi-GPU DDP training
optimizer.step()
raw_model = model.module  # unwrap DDP
raw_model.update_moe_router_bias(bias_lr=1e-3, distributed=True)
```

---

## Internal Components

The following sub-modules are assembled inside `BushidoMythos`. They are not typically called directly but understanding them clarifies the model's behavior.

### `RecurrentBlock`

The heart of the architecture. A single `TransformerBlock` (with MoE FFN) is run in a loop for up to `n_loops` iterations, with the following per-iteration pipeline:

```
h_loop = loop_index_embedding(h, t, loop_dim)   # inject sinusoidal loop-index signal
combined = RMSNorm(h_loop + e)                   # add frozen encoded input
trans_out = TransformerBlock(combined, ...)       # attention + MoE FFN
trans_out = trans_out + LoRAAdapter(trans_out, t) # depth-wise LoRA delta
if use_depth_attn:
    h_new = h_new + DepthCrossAttention(h_new, mask, depth_keys, depth_vals)
h = LTIInjection(h, e, trans_out)               # stable update: A·h + B·e + trans_out
p = ACTHalting(h)                                # per-position halting probability
if use_depth_attn:
    dk, dv = depth_attn.write_cache(h)           # write current loop's K/V to depth cache
    depth_keys.append(dk); depth_vals.append(dv)
```

The loop exits early for positions whose cumulative halting probability exceeds `cfg.act_threshold`. If all positions have halted, the loop exits before `n_loops`. The final output is an ACT-weighted sum of `h` across iterations.

When `cfg.loop_curriculum=True` and the block is in training mode, `n_loops` is sampled uniformly from `[1, cfg.max_loop_iters]` each forward call, regardless of the `n_loops` argument.

After each forward call, `recurrent._last_ponder_cost` holds the mean ponder steps across the batch and sequence positions. `BushidoMythos` scales this by `cfg.act_aux_loss_weight` and stores the result in `model._last_aux_loss`.

### `DepthCrossAttention`

Optional sub-module of `RecurrentBlock`, activated when `cfg.use_depth_attn=True`. Implements MoDA-style (Mixture-of-Depths Attention) cross-loop depth attention: each token at loop step `t` attends jointly to:

- **Current-sequence K/V** at positions `0..i` (causal mask applied, same as the main attention).
- **Depth-cache K/V** at position `i` from loops `0..t-1` (no position mask — each loop can freely read from any prior loop at the same position).

K/V for the depth cache are computed via separate `wk_write`/`wv_write` projections from the post-loop hidden state `h`, not from the same projections as the sequence attention. This allows the cache to encode loop-level state without disturbing the main attention pathway. No RoPE is applied to avoid head-dim / rope-dim mismatches between GQA and MLA modes.

The depth cache is a local Python list that exists only within a single `forward()` call — it is not a persistent buffer and has no memory overhead between calls.

### `LTIInjection`

Implements the stable recurrent update rule `h_{t+1} = A·h_t + B·e + transformer_out`. The diagonal matrix `A` is parameterized as:

```
A_continuous = Diag(-exp(log_A))     # always negative diagonal
A_discrete   = exp(Δt · A_continuous) # ZOH discretization, values ∈ (0, 1)
```

This guarantees spectral radius `ρ(A) < 1` by construction, making the looped model unconditionally stable regardless of learning rate or batch noise. See [Parcae (Prairie et al., 2026)](https://arxiv.org/abs/2604.12946) for the theoretical foundation.

### `ACTHalting`

A single linear layer mapping `(B, T, dim) → (B, T)` followed by sigmoid. At each loop step, the scalar halting probability per position is accumulated. When the cumulative sum exceeds `cfg.act_threshold`, the ACT remainder trick assigns the remaining probability mass as the final weight and the position stops contributing. Implements Graves (2016) ACT.

### `LoRAAdapter`

A depth-wise low-rank adapter with three components:

- `down`: shared `Linear(dim, rank)` — down-projects the transformer output
- `B`: shared parameter matrix `(rank, dim)` — up-projects back to full dimension
- `scale`: `Embedding(max_loops, rank)` — per-loop element-wise scale

The delta per iteration is `(down(x) * scale[t]) @ B`. Bridges the expressiveness gap between pure weight-tying and fully distinct per-layer weights. Based on [Relaxed Recursive Transformers (Bae et al., 2024)](https://arxiv.org/pdf/2410.20672).

### `TransformerBlock`

Pre-norm transformer block with swappable attention and FFN:

- **Attention:** `MLAttention` (MLA) or `GQAttention` (GQA), selected by `cfg.attn_type`
- **FFN:** `MoEFFN` (when `use_moe=True`, inside `RecurrentBlock`) or dense `Expert` (Prelude, Coda)
- Pre-norm via `RMSNorm` applied to both the attention input and FFN input
- **Hyper-connections** (when `cfg.use_hyper_connections=True`): residuals become `α·x + β·f(x)` where `α` and `β` are per-channel learned `nn.Parameter` vectors initialized to `1` (standard residual at init). Adds four parameter vectors per block: `alpha_attn`, `beta_attn`, `alpha_ffn`, `beta_ffn`.

### `MLAttention`

Multi-Latent Attention ([DeepSeek-V2, 2024](https://arxiv.org/abs/2405.04434)). The cache stores only the compressed KV latent `c_kv` (rank `kv_lora_rank`) plus the RoPE-encoded keys. At each decode step, `K_nope` and `V` are cheaply reconstructed from `c_kv` via a shared up-projection, trading a fast linear multiply for dramatically smaller KV memory footprint.

Cache size per layer per token: `kv_lora_rank + n_heads × qk_rope_head_dim` vs. full GQA cache of `n_kv_heads × head_dim × 2`.

### `GQAttention`

Grouped Query Attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)). `n_kv_heads` KV pairs are shared across `n_heads // n_kv_heads` query heads each, reducing KV cache by that factor while preserving full query expressiveness.

### `MoEFFN`

Fine-grained Mixture-of-Experts FFN ([DeepSeekMoE, Dai et al., 2024](https://arxiv.org/abs/2401.06066)):

- **Routed experts:** `n_experts` small SwiGLU FFNs. Each token's router selects the top-`n_experts_per_tok` via softmax over learned logits. A per-expert bias `router_bias` (non-gradient, updated externally) keeps load balanced.
- **Shared experts:** `n_shared_experts` always-active FFNs with width `expert_dim × n_experts_per_tok`, absorbing cross-domain patterns.

Total activated parameters per token: `(n_experts_per_tok / n_experts)` of routed capacity + all shared capacity.

### `Expert`

Single SwiGLU feed-forward unit: `down(silu(gate(x)) * up(x))`. Used both as individual routed experts inside `MoEFFN` and as the dense FFN in Prelude/Coda blocks.

### `RMSNorm`

Root Mean Square Layer Normalization ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)). Normalizes by `x / rms(x)` with a learned per-channel rescaling weight. No bias, no mean subtraction. Used throughout in place of standard LayerNorm.

---

## Utility functions

### `precompute_rope_freqs(dim, max_len, theta)`

Precomputes real-valued RoPE rotation tables as a `(max_len, dim//2, 2)` tensor, where the last dimension stores `[cos, sin]`. Called once in `__init__` and stored as a buffer. This avoids complex tensors on MPS.

### `apply_rope(x, freqs_cis)`

Applies precomputed RoPE frequencies to a query or key tensor by rotating adjacent feature pairs with the cached `[cos, sin]` values.

### `loop_index_embedding(h, loop_t, loop_dim, theta)`

Injects a sinusoidal loop-index signal into the first `loop_dim` channels of the hidden state, analogous to RoPE but over recurrence depth rather than sequence position. Allows the shared recurrent block weights to behave differently at different loop iterations.

> **Implementation note:** Frequencies are computed in `float32` regardless of the model dtype. In `bfloat16` (7-bit mantissa), adjacent loop indices can quantize to the same frequency value, causing the loop-index signal to degenerate. The result is cast back to the model dtype before addition.

---

## Key design properties

| Property | Mechanism | Benefit |
|---|---|---|
| Depth extrapolation | Recurrent block with looped identical weights | Train on N loops, test on N+k — harder problems solved without retraining |
| Parameter efficiency | Weight sharing across all loop iterations | k-layer model achieves quality of kL-layer model; parameters ≈ k, compute ∝ L |
| Adaptive compute | ACT halting per position | Easy tokens exit early; hard tokens receive full loop depth — within the same batch |
| Stable training | LTI injection with ZOH-constrained A (ρ(A) < 1) | No residual explosion; robust to high learning rates |
| Domain breadth | MoE FFN in recurrent block | Different expert subsets can specialize by market regime, asset class, and task type |
| Loop differentiation | Loop-index sinusoidal embedding | Same weights implement functionally distinct phases per iteration |
| Efficient KV memory | MLA (default) or GQA | MLA: 10–20× smaller cache vs standard attention at production scale |
| Depth-wise adaptation | LoRA adapter per loop iteration | Expressiveness beyond pure weight-tying; minimal parameter overhead |
| Residual mixing | Hyper-connections (`use_hyper_connections`) | Learned α/β per channel replace fixed-1 residual scaling; initialized to standard residual |
| ACT regularization | `act_aux_loss_weight` | Auxiliary loss penalizes unnecessary ponder steps; encourages efficient halting |
| Loop depth curriculum | `loop_curriculum` | Training on random loop depths improves generalization to unseen inference depths |
| Cross-loop attention | `DepthCrossAttention` (`use_depth_attn`) | Each loop can selectively read from prior-loop K/V at the same position |
| Continuous latent reasoning | `generate_coconut()` | N thought steps in embedding space before discrete decoding; richer than token-level CoT |
| Expert load balancing | `update_moe_router_bias()` | Bias-based auxiliary-loss-free load balancing; accumulates across gradient accumulation steps |

---

## Full configuration reference

The default `MythosConfig()` targets a mid-scale research model. Below is a minimal configuration for quick experimentation:

```python
from bushido_mythos.main import BushidoMythos, MythosConfig

# Minimal config for fast iteration / unit testing
small_cfg = MythosConfig(
    vocab_size=8192,
    dim=256,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=512,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    attn_type="gqa",
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=4,
)
model = BushidoMythos(small_cfg)
```

And a production-oriented MLA configuration matching the default hyperparameters:

```python
# Default MLA config (matches MythosConfig() defaults)
prod_cfg = MythosConfig(
    vocab_size=32000,
    dim=2048,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=4096,
    max_loop_iters=16,
    prelude_layers=2,
    coda_layers=2,
    attn_type="mla",           # Multi-Latent Attention
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    n_experts=64,
    n_shared_experts=2,
    n_experts_per_tok=4,
    expert_dim=512,
    act_threshold=0.99,
    rope_theta=500000.0,
    lora_rank=16,
)
model = BushidoMythos(prod_cfg)
```

---

## Training Scripts

### `training/finance_pretrain.py`

A 5-phase staged training script for adapting any BushidoMythos checkpoint toward reasoning-heavy financial trading behavior. See the [README Training section](../README.md#finance-domain-pretraining) for full CLI usage.

#### Design overview

| Phase | Dataset | Purpose |
|---|---|---|
| Phase 1 | WikiText-103 (≈ 115M tokens) | General language grounding with GPT-2 tokenizer |
| Phase 2 | OpenWebMath + Orca Math + Dolly 15k | Quantitative reasoning, step decomposition, and prose explanation logic |
| Phase 3 | `financial-news-articles` + `finance-alpaca` | Finance vocabulary and instruction-format exposure |
| Phase 4 | FinGPT forecaster + sentiment | Trading methodology SFT |
| Phase 5 | Audited local Finance QA pilot | Risk-management and calculation SFT; disabled by default pending adoption |

Both phases share one scheduler, one optimizer, and one checkpoint stream. Resuming an interrupted run loads `scheduler.state_dict()` directly — no step replay needed. If a legacy checkpoint (without `scheduler_state`) is loaded, the scheduler is re-advanced by replaying N steps.

#### `TextDataset`

```python
TextDataset(rows, vocab_size, seq_len, batch_size, device, cache_path)
```

Tokenizes `rows` once and writes the result to `cache_path`. On subsequent runs the cache is loaded directly (skipping tokenization). Cache filenames include `_CACHE_VERSION` (currently `v1`) to prevent stale-cache reads when tokenization logic changes.

`__iter__` raises `ValueError` if the token count is too small for the given `seq_len` (catches misconfigured or missing datasets early). Each batch yields `(x, y)` where `y` is `x` shifted left by one token.

#### Memory-efficient linear cross entropy

`chunked_linear_cross_entropy(hidden, weight, targets, chunk_size, loss_mask=None)` computes tied LM-head cross entropy without retaining full `(B, T, vocab_size)` logits. Each token chunk is activation-checkpointed and its LM-head projection is recomputed during backward. `BushidoMythos.forward(..., return_hidden=True)` supplies the normalized pre-head hidden states; the default forward API remains unchanged and returns logits.

#### `save_checkpoint` / `load_checkpoint`

```python
save_checkpoint(path, step, model, optimizer, scheduler, cfg,
                tag="", phase1_steps=0, phase2_steps=0,
                phase3_steps=0, phase4_steps=0, phase5_steps=0)

load_checkpoint(path, model, optimizer, scheduler) -> int  # returns step
```

`save_checkpoint` always writes:

| Key | Content |
|---|---|
| `step` | Current global step |
| `model_state` | `model.state_dict()` |
| `optimizer_state` | `optimizer.state_dict()` |
| `scheduler_state` | `scheduler.state_dict()` — exact LR curve preserved |
| `cfg` | `cfg.__dict__` for config reconstruction |
| `tag` | Phase label, e.g. `"Phase1-WikiText103"` |
| `phase1_steps` | Phase 1 step total at save time |
| `phase2_steps` | Phase 2 step total at save time |
| `phase3_steps` | Phase 3 step total at save time |
| `phase4_steps` | Phase 4 step total at save time |
| `phase5_steps` | Phase 5 step total at save time |

When loading, all `phase*_steps` values are printed along with a note to pass matching flags on resume.

#### Phase 2 skip condition

`phase2_end` is always anchored to `phase1_steps + phase2_steps`, never to the current step. This prevents off-by-N errors when resuming mid-phase:

```python
phase2_end = p1_total + args.phase2_steps  # always correct regardless of current step
if args.phase in (0, 2) and step < phase2_end:
    # build dataset and run Phase 2
```

Phase 2 is silently skipped when `step >= phase2_end`, so re-running with `--auto_resume` after a completed training is safe.

#### Base checkpoint loading

Shape-mismatched keys (e.g. `freqs_cis` / `freqs_cis_mla` RoPE buffers when `max_seq_len` differs between base and target configs) are filtered before `load_state_dict`:

```python
filtered = {k: v for k, v in ckpt_sd.items()
            if k in model_sd and v.shape == model_sd[k].shape}
model.load_state_dict(filtered, strict=False)
```

Skipped keys are re-initialized from scratch and printed at load time.

#### Test coverage

[`tests/test_finance_pretrain.py`](../tests/test_finance_pretrain.py) — 38 tests, runs in < 10 seconds on CPU:

| Class | Tests |
|---|---|
| `TestTextDataset` | Cache roundtrip, correct shapes, y=x+1 shift, `ValueError` on insufficient tokens, cache key version |
| `TestLrLambda` | Linear warmup, cosine decay midpoint, min-lr floor, custom `min_lr_ratio` |
| `TestCheckpoint` | `scheduler_state` saved, `phase_steps` saved, `tag` saved, step restored, `load_state_dict` used (not replay), legacy fallback replays N steps |
| `TestPhase2SkipCondition` | End anchoring math, skip when `step >= phase2_end`, run when `step < phase2_end`, `build_reasoning_mix` not called when complete |

---

## `chat.py` — Interactive Inference

`chat.py` provides a REPL-style chat loop over any checkpoint. See the [README Chat section](../README.md#chat-inference) for quick-start examples.

### Checkpoint discovery

`find_latest_ckpt(ckpt_dir)` searches in priority order:

1. `phase5_final.pt`
2. `phase4_final.pt`
3. `phase3_final.pt`
4. `final.pt`
5. `phase2_final.pt`
6. `phase1_final.pt`
7. Latest `step_*.pt` by filename sort

### Tokenizer selection

`build_tokenizer(vocab_size, mode)` supports three modes:

| Mode | Behavior |
|---|---|
| `auto` | Uses GPT-2 tokenizer when `vocab_size == 50257`; otherwise tries `MythosTokenizer`, falls back to GPT-2 on failure |
| `gpt2` | Uses GPT-2 tokenizer; loads from local cache first, falls back to network download if not cached |
| `mythos` | Forces `MythosTokenizer` regardless of vocab size |

### Prompt truncation

If the encoded prompt is longer than `max_seq_len - max_new_tokens`, the oldest tokens are discarded and a warning is printed. If `max_new_tokens >= max_seq_len`, `max_new_tokens` is clamped to `max_seq_len // 2` before truncation.

### Finance mode

`--finance_mode` wraps every prompt in the instruction-response format used during Phase 3–5 SFT,
and appends a risk-context suffix so the model generates responses that include uncertainty and
risk acknowledgement:

```
### Instruction:
{your prompt}

Please acknowledge uncertainty where relevant, include risk considerations,
and note that outputs should be verified from authoritative sources before any trading action.

### Response:
```

Generation is truncated at the next `\n### ` boundary to prevent the model from generating
additional instruction blocks. This mode is **recommended** for any checkpoint trained through
Phase 3 or later; without it the model receives a raw text prompt and produces free-form
completion rather than a structured response.

### CLI argument reference

| Flag | Default | Constraint | Description |
|---|---|---|---|
| `--ckpt` | `None` | — | Explicit checkpoint path; overrides `--ckpt_dir` |
| `--ckpt_dir` | `checkpoints/finance_a100_v2` | — | Directory searched when `--ckpt` is not set |
| `--tokenizer` | `auto` | `auto`/`gpt2`/`mythos` | Tokenizer selection mode |
| `--temp` | `0.8` | `> 0` | Sampling temperature |
| `--top_k` | `50` | `>= 0` | Top-K filtering (`0` = disabled); clamped to `vocab_size` |
| `--max_tokens` | `64` | `>= 1` | Maximum new tokens per response |
| `--loops` | `4` | `>= 1` | Recurrent loop depth at inference |
| `--finance_mode` | off | — | Enable finance disclaimer prefix |

---

## References

| Component | Paper |
|---|---|
| Recurrent-Depth Transformer | [Loop, Think, & Generalize (2025)](https://arxiv.org/pdf/2604.07822) |
| LTI-stable injection (Parcae) | [Scaling Laws for Stable Looped Language Models (Prairie et al., 2026)](https://arxiv.org/abs/2604.12946) |
| Looped transformer reasoning | [Reasoning with Latent Thoughts (Saunshi et al., 2025)](https://arxiv.org/abs/2502.17416) |
| Multi-Latent Attention | [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) |
| Grouped Query Attention | [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245) |
| Mixture-of-Experts FFN | [DeepSeekMoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066) |
| MoE load balancing (router bias) | [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) |
| Adaptive Computation Time | [Graves, 2016](https://arxiv.org/abs/1603.08983) |
| Depth-wise LoRA | [Relaxed Recursive Transformers (Bae et al., 2024)](https://arxiv.org/pdf/2410.20672) |
| Depth cross-attention (MoDA) | [Mixture-of-Depths Attention (2025)](https://arxiv.org/abs/2603.15619) |
| Continuous latent reasoning (COCONUT) | [Training LLMs to Reason in Continuous Latent Space (Hao et al., 2024)](https://arxiv.org/abs/2412.06769) |
| RMSNorm | [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467) |
| RoPE | [Su et al., 2021](https://arxiv.org/abs/2104.09864) |
| Universal Transformer (ACT basis) | [Dehghani et al., 2018](https://arxiv.org/pdf/1807.03819) |
