# Phase 5 Step-Dose Ablation Analysis

## Decision

Do not adopt any 10, 50, or 200-step Phase 5 checkpoint, and do not run a
finer step search on `finance_qa_curated_v1`.

Evaluation used the original `finance_qa_v1` suite revision with SHA-256
`e524cc1b83ce854736d6dc5997ea6244042734d8da9cf24d9df4d87b849ab894`.
The generated Markdown report supplied for review has SHA-256
`27e350eed7de3278c1d5513eb409b3e10eebd8dabae731af867200d2d9e8f54c`.

| checkpoint | score | pass rate | concept recall | numeric accuracy | largest exact response |
|---|---:|---:|---:|---:|---:|
| Phase 3 control | 0.270 | 0.0% | 3.5% | 5.6% | 2.8% |
| Historical Phase 5 | 0.272 | 0.0% | 3.7% | 5.6% | 2.8% |
| Aligned 10-step | 0.320 | 2.8% | 10.4% | 0.0% | 2.8% |
| Aligned 50-step | 0.278 | 0.0% | 4.2% | 11.1% | 2.8% |
| Aligned 200-step | 0.282 | 0.0% | 4.4% | 16.7% | 5.6% |

The 10-step result captures a short-lived concept-keyword signal but cannot perform
the calculation cases. Increasing the dose raises numeric-token matches while concept
recall falls. Manual output review shows copied or spliced calculation patterns with
incorrect variables, so the numeric increase is not reliable arithmetic skill.

The 10-step checkpoint's only pass was `earnings_event` for one seed. That response
never mentioned earnings. It passed by containing generic terms corresponding to
expectations, volatility, and hedging. This is a deterministic rubric false positive,
not a model-quality gain.

## Evaluation Correction

Finance QA cases now define topic anchors. A response must mention at least one
case-specific anchor as well as satisfying the existing concept, numeric, safety, and
non-degeneracy checks. Reports also include `topic_relevance_rate`. The revised suite
is versioned separately as `finance_qa_v2`; the original v1 file is unchanged for
historical reproduction. The v2 content SHA-256 is
`0c7dfdd169d9ee5628eb665e576389fde0bc4f10b808d19c757936e0729d79ec`.

## V2 Re-evaluation

The five existing checkpoints were re-scored on A100 BF16 with loops 8, 128
generated tokens, temperature 0.7, top-k 40, repetition penalty 1.3, and seeds
0/1/2. No training was repeated. The supplied machine-readable result has SHA-256
`b54fc73e6772818712048523eddc95cf0bfdfc36092d9512b7305e0889269d4a`.

| checkpoint | score | pass rate | topic relevance | concept recall | numeric accuracy | largest exact response |
|---|---:|---:|---:|---:|---:|---:|
| Phase 3 control | 0.270 | 0.0% | 2.8% | 3.5% | 5.6% | 2.8% |
| Historical Phase 5 | 0.272 | 0.0% | 5.6% | 3.7% | 5.6% | 2.8% |
| Aligned 10-step | 0.320 | 0.0% | 8.3% | 10.4% | 0.0% | 2.8% |
| Aligned 50-step | 0.278 | 0.0% | 11.1% | 4.2% | 11.1% | 2.8% |
| Aligned 200-step | 0.282 | 0.0% | 11.1% | 4.4% | 16.7% | 5.6% |

The topic-anchor correction removed the 10-step false positive: its pass rate fell
from 2.8% under v1 to 0% under v2. Scores and the other aggregate metrics remain
unchanged because topic relevance is an eligibility gate rather than a score term.
All five checkpoints fail the absolute adoption gate, and the evaluator recommends
no checkpoint. The low topic relevance rates (2.8%-11.1%) also show that these models
usually do not answer the requested financial scenario even when isolated concept or
number matches increase.

## Next Experiment

1. Expand curated training data beyond the 29-example pilot, with verified arithmetic
   generation and broader paraphrase/scenario coverage.
2. Create a separate in-distribution validation split for update-dose selection and
   early stopping.
3. Keep the final 12-case Finance QA suite excluded from training and validation; use
   it only for final checkpoint adoption.
4. Track answer-template or calculation-fragment reuse in addition to exact-response
   concentration.
