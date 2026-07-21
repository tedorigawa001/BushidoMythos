# Finance QA Phase 2-5 Analysis

## Decision

No v3 checkpoint is eligible for finance QA deployment.

Evaluation conditions: A100 BF16, loops 8, 128 generated tokens, temperature 0.7,
top-k 40, repetition penalty 1.3, seeds 0/1/2, 12 held-out cases (36 responses per
checkpoint). The suite SHA-256 is
`e524cc1b83ce854736d6dc5997ea6244042734d8da9cf24d9df4d87b849ab894`.

| checkpoint | score | pass rate | concept recall | numeric accuracy | non-degenerate |
|---|---:|---:|---:|---:|---:|
| Phase 2 | 0.252 | 0.0% | 0.7% | 11.1% | 100% |
| Phase 3 | 0.270 | 0.0% | 3.5% | 5.6% | 100% |
| Phase 4 | 0.142 | 0.0% | 0.0% | 0.0% | 0% |
| Phase 5 | 0.272 | 0.0% | 3.7% | 5.6% | 100% |

Phase 5 has the highest relative score by only 0.002 over Phase 3. This is not a
meaningful win: both have 0% pass rate and less than 4% required-concept recall.
The 0% unsafe-claim rate is also not evidence of safety because the responses mostly
avoid the requested concepts rather than answering safely.

## Phase 4 Collapse

Phase 4 generated exactly `neutral` for all 36 question/seed combinations. The
training log provides a concrete data imbalance signal:

- FinGPT forecaster: 1,230 examples
- FinGPT sentiment: 76,772 examples
- Combined Phase 4: 78,002 examples
- Phase 4 loss: 2.50 at step 46,100 to 0.73 at step 49,000

The sentiment dataset represents 98.4% of examples and uses short classification
labels. The observed all-`neutral` behavior is therefore consistent with response-label
collapse caused by using a classification corpus as dominant generative SFT data.
The controlled ablation confirmed that simple balancing does not repair the phase.
Starting from the same Phase 3 checkpoint, forecaster-only scored 0.256 versus the
0.270 control and reduced concept recall from 3.5% to 0.7%. Balanced 1:1 scored
0.142; all 36 responses were sentiment labels and its dominant exact response share
was 27.8%, above the 20% collapse gate. This points to task-format interference,
not only raw class imbalance.

## Phase 5 Limitation

Phase 5 restores long outputs after the Phase 4 collapse but does not restore correct
finance QA. Its concept recall is 3.7%, numeric accuracy is 5.6%, and every case fails.
The FinGPT FIQA corpus is therefore insufficient for the target rubric in the current
schedule. More steps on the same data are not justified until dataset-target alignment
is audited.

## Phase 4 Decision

Exclude Phase 4 from the broad Finance QA production path. Its default step count and
sentiment ratio are now zero, Phase 5 resumes directly from Phase 3, and automatic chat
checkpoint selection prefers Phase 3 over Phase 4. Phase 4 remains available only for
explicit task-specific experiments.

The next experiment is implemented as `training/run_phase5_pilot.py`. It trains a
500-step curated Phase 5 directly from Phase 3 and compares the control, optional
historical checkpoints, and the pilot under the same three-seed evaluation. The local
training set contains 29 auditable examples across eight categories. Before
tokenization, every instruction and response is compared with all held-out questions
and reference answers; lexical similarity at or above 0.80 fails the run. The current
data reaches maximum similarities of 0.617 for instructions and 0.374 for responses.
The 12 evaluation prompts and their reference answers remain excluded from training.

The first curated 500-step pilot was not adopted. It reached score 0.310 and concept
recall 8.8%, but pass rate remained 0%, numeric accuracy remained 5.6%, and largest
exact-response share rose to 22.2%. Generated outputs replayed complete curated answers
for unrelated questions. Root-cause tracing found that the SFT loader flattened every
pair into one token stream and sampled arbitrary offsets, so a response could be trained
without its instruction in context. `SFTDataset` now preserves one example per padded
row, rejects legacy flat caches, and keys caches by sequence length. The next rerun keeps
the original 500-step pilot settings fixed to isolate this layout correction.

The aligned rerun reduced largest exact-response share from 22.2% to 5.6% and
increased numeric accuracy from 5.6% to 16.7%, validating the sequence-layout fix.
It still failed adoption: score 0.289, concept recall 5.3%, pass rate 0%. Outputs
continued to splice memorized training responses.

The 10/50/200-step dose ablation also found no adoptable range. The 10-step variant
had the best relative score (0.320) and concept recall (10.4%) but numeric accuracy
was 0%. The 50-step and 200-step variants scored 0.278 and 0.282; concept recall fell
to 4.2% and 4.4% while numeric accuracy rose only to 11.1% and 16.7%. This is the
signature of calculation-fragment memorization without reliable prompt/value binding,
not useful skill acquisition. Finer dose search is therefore stopped.

The only passing response in the dose report was one `earnings_event` seed at 10
steps. Manual review rejected it: the response never mentioned earnings and matched
generic aliases such as expectations, volatility, and hedge. The held-out rubric now
requires a case-specific topic anchor in addition to concept, numeric, safety, and
non-degeneracy checks, and reports topic relevance explicitly. Historical reports
remain attributable by suite SHA-256. The next quality experiment must expand the
curated data, keep a separate in-distribution validation split for dose selection,
and retain the final 12-case suite only for adoption evaluation.

The controlled experiment is implemented in `training/run_phase4_ablation.py`. It
created isolated `forecaster_only` and `balanced_1to1` checkpoint directories, trained
each for 500 steps from the same Phase 3 checkpoint, and ran the same three-seed QA
comparison. Phase 4 dataset construction still reports its response distribution and
fails before training when one exact response exceeds 20% of examples.

## Evidence

- Full generated report: `checkpoints/finance_a100_v3_full/finance_qa_phase2_5.md`
- Machine-readable result: `checkpoints/finance_a100_v3_full/finance_qa_phase2_5.json`
- Training log: `checkpoints/finance_a100_v3_full/train.log`
