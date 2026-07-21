# Finance QA Curated V2 Corpus

## Decision

Replace the 29-example dose-search dataset for future Phase 5 experiments with a
deterministic 640-example training corpus and a separate 160-example validation
corpus. Keep `finance_qa_v2` unchanged as the 12-case final adoption suite.

| split | examples | categories | scenario families | calculation examples |
|---|---:|---:|---:|---:|
| train | 640 | 8 x 80 | 16 | 80 |
| validation | 160 | 8 x 20 | 16 | 20 |
| final held-out | 12 | 8 | 12 | rubric-defined |

No scenario family appears in more than one split. Training examples are written to
`training/train_data/finance_qa_curated_v2_train.json`; validation examples are in
`training/eval_data/finance_qa_curated_v2_validation.json`. The final held-out suite
remains `training/eval_data/finance_qa_v2.json` with SHA-256
`0c7dfdd169d9ee5628eb665e576389fde0bc4f10b808d19c757936e0729d79ec`.

## Reproducibility And Audit

`training/build_finance_qa_curated_v2.py` uses fixed project-authored templates and
deterministic parameter schedules. It validates unique IDs, instructions, and
responses; disjoint scenario families; and the final held-out family registry.
Calculation records retain formula names, inputs, and outputs. All 100 calculation
answers are recomputed with `Decimal`, and every expected value must appear in the
rendered response.

The Phase 5 loader independently repeats split checks before tokenization. It rejects
duplicate IDs, shared scenario families, exact train/validation text, and lexical
similarity at or above 0.80 against the final held-out questions or references.
Observed maximum instruction/response similarities are 0.488/0.261 for training and
0.413/0.304 for validation.

## Checkpoint Selection

`training/eval_finance_validation.py` evaluates periodic checkpoints using response
teacher-forced NLL and response-token accuracy. Prompt binding compares each correct
response with a counterfactual response taken from another example in the same
scenario family; calculation binding reports the same comparison for the 20 numeric
validation examples.

A candidate must improve NLL by at least 5% over Phase 3, preserve every category
within 10% of its baseline NLL, and reach at least 60% general and calculation binding
without falling below the baseline. `training/run_phase5_validation.py` selects the
lowest-NLL passing checkpoint. It does not invoke `finance_qa_v2` when no candidate
passes; otherwise it evaluates the selected checkpoint exactly once against Phase 3.

## A100 Outcome

The 50-step checkpoint passed validation with NLL 4.905 versus 5.663 for Phase 3.
The one-time final evaluation rejected it: both checkpoints scored 0.270 with 0% pass
rate, 3.5% concept recall, and 5.6% numeric accuracy. Prompt binding was already 96.2%
on Phase 3 and calculation binding was 100%, so these teacher-forced metrics were
saturated and did not predict free-generation quality. See
`training/report/finance_qa_curated_v2_result.md`; no checkpoint is promoted.
