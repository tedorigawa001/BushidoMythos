# Memory-replay pilot — result

Setup: from `phase1_final` (general-language model), train finance SFT
(`trading_qa`) for 400 steps at flat LR 1e-4 (seq_len=256, batch=1, CPU),
varying `--replay_ratio`. Measure WikiText-103 validation PPL afterwards
(partial eval, `--eval_max_chunks 40`). Each run reloads `phase1_final` and
shares the torch seed, so the only difference is replay on/off.

| condition | WikiText PPL | Δ vs baseline |
|---|---|---|
| baseline (phase1_final, before finance SFT) | 59.28 | — |
| replay_ratio = 0.0 | 751.87 | **+692.59** |
| replay_ratio = 0.2 | 135.47 | **+76.19** |

- Without replay, 400 steps of finance SFT degrade general-language PPL ~12.7×
  (59 → 752) — strong catastrophic forgetting.
- With 20% general-language (WikiText) replay, the forgetting Δ shrinks ~89%
  (692 → 76); PPL stays at 135.
- Finance learning still proceeds in both runs (training loss decreases
  similarly); replay does not block specialization.

Reproduce:
```
python training/exp_replay_pilot.py --steps 400 --lr 1e-4 --seq_len 256 \
  --replay_ratios 0.0 0.2 --eval_max_chunks 40 --device cpu
```

Caveats (pilot):
- Partial eval (40 chunks) → absolute PPL differs from full WikiText-103 PPL
  (baseline here 59.28 vs full ≈ 54.86); relative comparison is what matters.
- Flat LR + 400 steps is aggressive (amplifies forgetting); the real 5-phase
  run uses LR decay, so absolute magnitudes differ.
- n=1 per condition, single seed. The effect size (692 vs 76) is large enough
  to be convincing as a directional result; A100 gives production numbers.
