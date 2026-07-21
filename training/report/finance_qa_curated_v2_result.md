# Finance QA Curated V2 Validation Result

## Decision

Do not adopt any Phase 5 curated v2 checkpoint. Validation selected the 50-step
checkpoint, but the one-time final held-out evaluation showed no improvement over
Phase 3 in score, pass rate, concept recall, or numeric accuracy.

Evaluation artifacts:

- validation JSON SHA-256: `1597ad1c259b88d0b4798dd70ae35e07308984ab98991e004b501d34a32906e6`
- final JSON SHA-256: `366a38e6568adbcd0df674005466ad718a304ff0cd2424543d5b4c6303367ac2`
- final suite SHA-256: `0c7dfdd169d9ee5628eb665e576389fde0bc4f10b808d19c757936e0729d79ec`

## Validation Selection

| checkpoint | response NLL | token accuracy | prompt binding | calculation binding | gate |
|---|---:|---:|---:|---:|---|
| Phase 3 | 5.663 | 19.4% | 96.2% | 100.0% | baseline |
| 50-step | 4.905 | 36.5% | 99.4% | 100.0% | pass |
| 100-step | 5.408 | 36.9% | 100.0% | 100.0% | fail |
| 150-step | 5.484 | 37.2% | 100.0% | 100.0% | fail |
| 500-step | 5.775 | 37.1% | 98.1% | 95.0% | fail |

The 50-step checkpoint reduced validation NLL by 13.4% and nearly doubled response
token accuracy. NLL then worsened with additional updates, which is a clear early
overfitting signal. At effective batch 128, 50 steps expose the model to roughly ten
training presentations per example.

Prompt-binding accuracy was already 96.2% on the Phase 3 baseline and calculation
binding was 100%. These metrics were saturated before Phase 5 and did not measure
factual generation. Teacher forcing made a same-family correct response easier to
score than a distractor even though free generation could not produce the answer.

## Final Held-Out

| metric | Phase 3 | selected 50-step |
|---|---:|---:|
| overall score | 0.270 | 0.270 |
| pass rate | 0.0% | 0.0% |
| topic relevance | 2.8% | 11.1% |
| concept recall | 3.5% | 3.5% |
| numeric accuracy | 5.6% | 5.6% |
| unsafe-claim rate | 0.0% | 0.0% |

Topic relevance rose, but generated answers still spliced template fragments and news
text, failed both calculations, and did not improve required-concept coverage. The
absolute adoption gate rejected the selected checkpoint and recommended no model.

## Consequences

1. Teacher-forced NLL and pairwise response preference remain diagnostic metrics, not
   checkpoint-adoption gates.
2. The next validation suite needs free-generation scoring with scenario-specific topic
   anchors, required concepts, and exact recomputation of numeric outputs.
3. Training templates need more structural and linguistic diversity; parameter changes
   alone are insufficient for held-out transfer at this model scale.
4. `finance_qa_v2` has now been consumed for this experiment. A future corpus iteration
   must use a newly sealed final suite rather than tune against these 12 observed cases.
