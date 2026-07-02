# 8-bit optimizer comparison — result

Setup: same model (98.6M, phase1_final), random batches, same seed, fixed
n_loops (deterministic forward), 50 steps, batch=4, seq_len=256, A100.
Run: `python training/exp_optim8bit.py --device cuda --steps 50 --batch_size 4 --seq_len 256`

| optimizer | final loss | peak MB | time s |
|---|---|---|---|
| fp32 AdamW | 11.2494 | 3236 | 15.5 |
| 8-bit AdamW (bnb) | 11.2470 | 2636 | 14.0 |

- Near-lossless: max per-step loss diff = 0.00379 (both converge to 11.25).
- Peak memory: -600MB (-19%). Matches theory: optimizer states fp32 789MB →
  8-bit ~197MB (-592MB), i.e. the savings are entirely optimizer state.
- Slightly faster (14.0 vs 15.5s).

Savings scale with params: 520M ≈ -3.1GB, 1B ≈ -6GB. With bf16 training the
optimizer-state fraction is larger, so the relative benefit grows.
