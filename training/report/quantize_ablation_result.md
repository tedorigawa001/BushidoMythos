# Quantization ablation — where does INT8 break the finance model?

phase5_final (finance-specialized), CPU, INT8 dynamic, finance PPL
(financial_news, max_chunks=30). fp32 baseline finance PPL = 43.56;
full INT8 = 63.85 (+46.6%). "INT8 except G" keeps group G in fp32.

## Module-level (training/exp_quantize_ablation.py)

| config | finance PPL | Δ vs fp32 |
|---|---|---|
| INT8 all | 63.85 | +46.6% |
| except head | 63.16 | +45.0% |
| except experts | 57.90 | +32.9% |
| **except attn** | **49.71** | **+14.1%** |
| except ffn_dense | 60.15 | +38.1% |
| except router | 63.51 | +45.8% |

→ Attention (only 2.1M params) is the main culprit; keeping it fp32 costs ~6MB.
(Note: "except head" is *smaller* — head is weight-tied to the input embedding,
so quantizing head un-ties it and duplicates ~38MB.)

## Inside attention (training/exp_attn_ablation.py)

| config | finance PPL | Δ vs fp32 |
|---|---|---|
| INT8 all | 63.85 | +46.6% |
| except attn.wo | 63.45 | +45.7% |
| except attn.q | 63.14 | +44.9% |
| except attn.q_up_rope (RoPE) | 63.53 | +45.8% |
| **except attn.kv (kv_down/up)** | **51.07** | **+17.2%** |
| **except recurrent attn (loop)** | **50.80** | **+16.6%** |
| except prelude+coda attn | 62.69 | +43.9% |
| except ALL attn (ref) | 49.71 | +14.1% |

Two findings, both confirming the hypotheses:
1. The MLA **KV latent projections (kv_down/kv_up)** are the sensitive part —
   q / out_proj / RoPE projections barely matter. MLA compresses K/V into a
   low-rank latent; that dense bottleneck is precision-critical.
2. The **recurrent (looped) attention** dominates: protecting only the
   recurrent-block attention (run 8x) recovers nearly as much as all attention,
   while prelude/coda attention (run once) barely helps -> loop amplification.

Implication: a mixed-precision recipe that keeps the recurrent-block KV
projections (a handful of tiny layers) in fp32 should recover most quality at
near-zero size cost. INT8 the rest.

## Minimal mixed precision (training/exp_mixed_precision.py)

phase5_final, finance PPL. "回復率" = (INT8 - config) / (INT8 - fp32).

| keep fp32 | #layers | fp32 params | size | finance PPL | recovery |
|---|---|---|---|---|---|
| fp32 | 115 | 98.6M | 395M | 43.56 | 100% |
| INT8 all | 0 | 0 | 254M | 63.85 | 0% |
| all attn | 18 | 2.14M | 260M | 49.71 | 70% |
| recurrent attn | 6 | 0.71M | 256M | 50.80 | 64% |
| recurrent KV (2 layers) | 2 | 0.12M | 254M | 51.70 | 60% |
| **recurrent kv_down only** | **1** | **0.07M** | **254M** | **51.69** | **60%** |
| recurrent kv_up only | 1 | 0.05M | 254M | 63.15 | 3% |

Decisive finding: a SINGLE layer — `recurrent.block.attn.kv_down` (the MLA K/V
down-projection / compression into the latent, 0.07M params = 0.07% of the
model) — recovers 60% of the INT8 quality loss at zero size cost (254MB, same
as full INT8). kv_up (the expansion) is irrelevant (3%). The compression step
is precision-critical; the expansion tolerates INT8; the recurrent loop (8x)
amplifies the error.

Recipe: keep `recurrent.block.attn.kv_down` (or all attn for 70%) in fp32,
INT8 the rest -> -36% size, finance PPL +46.6% -> +18.7% (or +14.1% for all-attn).

## WikiText confirmation (also_wikitext)

Same minimal configs, measured on WikiText (general) too. fp32: finance 43.56 /
WikiText 338.94. Full INT8: 63.85 (+46.6%) / 444.29 (+31.1%).

| keep fp32 | params | finance recovery | WikiText recovery |
|---|---|---|---|
| all attn | 2.14M | 70% | 39% |
| recurrent KV (2 layers) | 0.12M | 60% | 34% |
| **recurrent kv_down only** | **0.07M** | **60%** | **35%** |
| recurrent kv_up only | 0.05M | 3% | -4% |

The SAME layer (kv_down) recovers both finance and general PPL, and kv_up is
irrelevant in both -> the sensitivity is layer-specific / domain-independent
(a structural property of MLA + recurrent), not a finance artifact.

Caveats: state_dict size != runtime memory/speed (mixed precision mixes
quantized/unquantized kernels). Productionizing the kv_down-fp32 recipe needs
an explicit mixed-precision export/load path. n=1, partial eval.
