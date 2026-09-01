# Phase 0 diagnostic — first attempt

## Task 1: pre-execution contract

- B/D/C: 7, 4, 5
- X: (7,4)
- W: (4,5)
- b: (7)  ← **incorrect, preserved as evidence.** Bias is per output class: `(C,) = (5,)`. `(7)` is the batch size.
- y: (7)
- logits: (7,5)
- loss: ()
- dW: (4,5)
- db: (7)  ← **incorrect, preserved as evidence.** `db` mirrors `b`, so `(C,) = (5,)`.

- Why the log-normalizer reduces over this axis: to execute across batches
- Expected one-step SGD behaviour: loss reduction

> Annotation added 2026-09-01 during curriculum review. The two bias shapes above were wrong in the committed pre-execution contract and were never corrected in these notes, although the Variant 2 repair fixed the same confusion in code. Because this file *is* the evidence record, the errors are annotated rather than overwritten. Both are the same miss as Variant 2: batch size substituted for class count. Under the code-discrimination rules this is `R`, not `P` — the batch/class distinction was not immediately available, it was not merely mis-typed.

## Task 2: diagnoses before execution

Attempt-error codes are single letters: `K` never encoded · `R` encoded but unavailable · `M` confident wrong model · `D` correct procedure from a neighbouring case · `P` knowledge present, hands wrong · `F` accurate but slow · `T` failed to transfer · `C` process or care slip. Test for `P`: asked right now, could you state the correct procedure without looking it up? If not, it is `R` or `K`.

### Variant 1 — global logsumexp reduction

- Observable symptom: reduces beyond just the desired axis
- Violated invariant: batches are independent
- Smallest repair: add 'axis=1' and 'keepdims=True'
- Tentative attempt-error code: ~~shapes incorrect~~ — prose, not a code. Reclassified 2026-09-01 as `R`: the reduction axis was not available on demand.

### Variant 2 — parameter dimensions derived incorrectly

- Observable symptom: mismatches when executing dot products/summation
- Violated invariant: shapes must match
- Smallest repair: review and fix parameter shape
- Tentative attempt-error code:

### Variant 3 — data-dependent unique inside jit

- Observable symptom:
- Violated invariant:
- Smallest repair:
- Tentative attempt-error code:

## API lookup prediction

### jax.value_and_grad

- Which argument should receive gradients?
  params
- What inputs should the transformed function be called with?
  x,y
- What two values/structures do you expect it to return?
  output value and a tree of gradients
- What should the gradient pytree mirror?
  the params

## Variant 3 — second attempt, pre-run

- Predicted symptom: runtime failure
- Which operation derives a size from runtime data: the jnp.unique
- JIT/static-shape invariant: n_classes
- Smallest repair: pass the n_classes as a parameter
- Tentative attempt-error code: ~~n_classes = jnp.unique(y).shape[0]~~ — an expression, not a code. The post-commit section records the correct primary `K` and secondary `C`.

## Post-commit repair outcomes

- Task 1: the first repair computed a log-probability expression from `y` rather than logits, broadcasting to `(B, B)` and causing an out-of-bounds gather/`NaN` on the extreme-logit test. Repair: normalise logits per class, gather from the resulting `(B, C)` log-probabilities, and return their mean negative value. Attempt error: ~~`P`~~ → reclassified 2026-09-01 as `M`. Substituting labels for logits is a wrong model of what the loss normalises over, not a slip of the hand; the correct procedure could not be stated on request.
- Variant 1: the repair makes `logsumexp` reduce on the class axis with the class dimension retained. This preserves a separate normaliser for each batch row. Attempt error: ~~`P`~~ → reclassified 2026-09-01 as `R`. The pre-execution diagnosis named the symptom but not the invariant, so the reduction axis was not available on demand.
- Variant 2: parameters now use feature/class dimensions for `W` and one class bias per output class, rather than batch-derived dimensions. Attempt error: ~~`P`~~ → reclassified 2026-09-01 as `R`, matching the same batch/class substitution in the Task 1 bias contract above. This variant needed direct shape feedback to repair, which a genuine `P` does not.
- Variant 3: `jnp.unique(y)` under `jit` raised `ConcretizationTypeError` because the class count depended on traced data values. Repair: obtain the class count from the static parameter-shape contract. Attempt errors: primary `K` (JIT static-shape constraint), secondary `C` (the required pre-execution diagnosis/taxonomy was incomplete).
- Verification: `uv run pytest -q test_phase_0_diagnostic.py` → `5 passed in 1.58s`.
- API-reference lookup: no consulted URL was recorded; add one later only if a lookup actually occurred.
- Elapsed time: pending learner report.

## Reclassification note — 2026-09-01

Three misses originally filed as `P` are reclassified above as `M`/`R` under the code-discrimination rules added to `robot-learning-curriculum.md`. This matters because the remedies differ: `P` routes to faded skeletons and focused unit tests, `R` routes to closed-resource retrieval, and `M` routes to contrast cases and a corrected re-attempt. The revised distribution for this diagnostic is `R`×3, `M`×1, `K`×1, `C`×1 — a retrieval problem, not a carelessness problem.
