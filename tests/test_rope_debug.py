"""
Standalone RoPE debug test — logs tensor outputs and intermediate calculations
so you can visually verify correctness of precompute_rope_freqs and apply_rope.

freqs_cis is now stored as a real float32 tensor of shape (max_len, dim//2, 2)
where freqs_cis[t, k, 0] = cos(t * θ_k) and freqs_cis[t, k, 1] = sin(t * θ_k).
"""

import torch
from bushido_mythos.main import apply_rope, precompute_rope_freqs

DIM = 8
MAX_LEN = 6
THETA = 500000.0


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def log(label: str, value) -> None:
    print(f"\n[{label}]")
    print(value)


# ---------------------------------------------------------------------------
# 1. precompute_rope_freqs — raw frequency table
# ---------------------------------------------------------------------------

section("1. precompute_rope_freqs")

freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)
log("freqs shape", freqs.shape)  # (MAX_LEN, DIM//2, 2)
assert freqs.shape == (MAX_LEN, DIM // 2, 2), f"Expected ({MAX_LEN}, {DIM // 2}, 2), got {freqs.shape}"

freqs_cos = freqs[..., 0]  # (MAX_LEN, DIM//2)
freqs_sin = freqs[..., 1]  # (MAX_LEN, DIM//2)
log("freqs cos (position x freq)", freqs_cos)
log("freqs sin (position x freq)", freqs_sin)

magnitudes = (freqs_cos ** 2 + freqs_sin ** 2).sqrt()
log("freqs magnitude (should be all 1.0)", magnitudes)
print(f"  All magnitudes = 1.0: {torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)}")

angles = freqs_sin.atan2(freqs_cos)
log("freqs angle (radians)", angles)

print("\nExpected: base frequencies (1 per dim pair)")
base = 1.0 / (THETA ** (torch.arange(0, DIM, 2, dtype=torch.float32) / DIM))
log("base freqs", base)

print("\nExpected: freqs[t, k].angle == t * base[k]")
for t in range(MAX_LEN):
    expected_angles = t * base
    actual_angles = angles[t]
    match = torch.allclose(actual_angles, expected_angles, atol=1e-5)
    print(f"  t={t}: angles match = {match}")

# ---------------------------------------------------------------------------
# 2. Position 0 is identity (cos=1, sin=0)
# ---------------------------------------------------------------------------

section("2. freqs[0] is identity (cos=1, sin=0)")
log("freqs[0] cos", freqs_cos[0])
log("freqs[0] sin", freqs_sin[0])
print(f"  All cos[0]=1: {torch.allclose(freqs_cos[0], torch.ones(DIM // 2), atol=1e-5)}")
print(f"  All sin[0]=0: {torch.allclose(freqs_sin[0], torch.zeros(DIM // 2), atol=1e-5)}")

# ---------------------------------------------------------------------------
# 3. apply_rope — shape and dtype
# ---------------------------------------------------------------------------

section("3. apply_rope — shape and dtype")

B, T, H = 2, MAX_LEN, 3
x = torch.randn(B, T, H, DIM)
log("input x shape", x.shape)

out = apply_rope(x, freqs)
log("output shape", out.shape)
print(f"  Shape preserved: {out.shape == x.shape}")

# dtype float16
x_half = x.half()
out_half = apply_rope(x_half, freqs)
print(f"  float16 dtype preserved: {out_half.dtype == torch.float16}")

# ---------------------------------------------------------------------------
# 4. apply_rope — isometry (norm preservation)
# ---------------------------------------------------------------------------

section("4. apply_rope — norm preservation (isometry)")

norms_in = x.norm(dim=-1)
norms_out = out.norm(dim=-1)
log("input norms (first batch item)", norms_in[0])
log("output norms (first batch item)", norms_out[0])
print(f"  Max absolute norm difference: {(norms_in - norms_out).abs().max().item():.2e}")
print(f"  Norms preserved (atol=1e-5): {torch.allclose(norms_in, norms_out, atol=1e-5)}")

# ---------------------------------------------------------------------------
# 5. Position 0 is the identity transformation
# ---------------------------------------------------------------------------

section("5. Position 0 is identity transformation")

x1 = torch.randn(1, 1, 2, DIM)
out1 = apply_rope(x1, freqs[:1])
log("input  x[:,0]", x1[0, 0])
log("output x[:,0]", out1[0, 0])
log("diff (should be ~0)", (x1 - out1).abs())
print(f"  Identity at pos 0: {torch.allclose(x1, out1, atol=1e-6)}")

# ---------------------------------------------------------------------------
# 6. Different positions produce different rotations
# ---------------------------------------------------------------------------

section("6. Different positions produce different rotations")

x2 = torch.ones(1, MAX_LEN, 1, DIM)
out2 = apply_rope(x2, freqs)
print("Per-position outputs (all input=1.0):")
for t in range(MAX_LEN):
    print(f"  pos={t}: {out2[0, t, 0].tolist()}")

# ---------------------------------------------------------------------------
# 7. Inverse rotation recovers original
# ---------------------------------------------------------------------------

section("7. Inverse rotation recovers original")

x3 = torch.randn(2, T, H, DIM)
rotated = apply_rope(x3, freqs)

# Inverse: negate sin → equivalent of complex conjugate
inv_freqs = freqs.clone()
inv_freqs[..., 1] = -inv_freqs[..., 1]  # cos stays, sin negated
recovered = apply_rope(rotated, inv_freqs)

diff = (x3 - recovered).abs()
log("max recovery error", diff.max().item())
print(f"  Recovery succeeded (atol=1e-5): {torch.allclose(x3, recovered, atol=1e-5)}")

# ---------------------------------------------------------------------------
# 8. start_pos correctness — the core generation bug fix
# ---------------------------------------------------------------------------

section("8. start_pos correctness (generation bug)")

print(
    "Simulates: prompt of length 4, then decode step receives token at position 4.\n"
    "Old buggy code: freqs[:1] → position 0 always.\n"
    "Fixed code: freqs[4:5] → position 4."
)

prompt_len = 4
decode_token = torch.randn(1, 1, 1, DIM)

out_buggy = apply_rope(decode_token, freqs[:1])                    # wrong: always pos 0
out_fixed = apply_rope(decode_token, freqs[prompt_len : prompt_len + 1])  # correct: pos 4

log("freqs[0] cos/sin", freqs[0])
log(f"freqs[{prompt_len}] cos/sin", freqs[prompt_len])
log("buggy output (pos 0 encoding)", out_buggy[0, 0, 0])
log("fixed output (pos 4 encoding)", out_fixed[0, 0, 0])
print(f"  Outputs differ (they should): {not torch.allclose(out_buggy, out_fixed)}")

# ---------------------------------------------------------------------------
# 9. Relative position property
# ---------------------------------------------------------------------------

section("9. Relative position property: <RoPE(q,m), RoPE(k,n)> depends only on (n-m)")

dim = 16
max_len = 32
freqs_big = precompute_rope_freqs(dim=dim, max_len=max_len, theta=THETA)
torch.manual_seed(42)
q = torch.randn(1, 1, 1, dim)
k = torch.randn(1, 1, 1, dim)


def rope_at(tensor, pos):
    seq = torch.zeros(1, pos + 1, 1, dim)
    seq[0, pos] = tensor[0, 0]
    return apply_rope(seq, freqs_big[: pos + 1])[:, pos : pos + 1]


dot_3_9 = (rope_at(q, 3) * rope_at(k, 9)).sum()
dot_1_7 = (rope_at(q, 1) * rope_at(k, 7)).sum()
dot_0_6 = (rope_at(q, 0) * rope_at(k, 6)).sum()
print(f"  <RoPE(q,3), RoPE(k,9)>: {dot_3_9.item():.6f}")
print(f"  <RoPE(q,1), RoPE(k,7)>: {dot_1_7.item():.6f}")
print(f"  <RoPE(q,0), RoPE(k,6)>: {dot_0_6.item():.6f}")
print(
    f"  All equal (offset=6): {torch.allclose(dot_3_9, dot_1_7, atol=1e-5) and torch.allclose(dot_1_7, dot_0_6, atol=1e-5)}"
)

section("DONE — all checks complete")
