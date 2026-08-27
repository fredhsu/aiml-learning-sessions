# Phase 0 diagnostic — first attempt

## Task 1: pre-execution contract

- B/D/C: 7, 4, 5
- X: (7,4)
- W: (4,5)
- b: (7)
- y: (7)
- logits: (7,5)
- loss: ()
- dW: (4,5)
- db: (7)
- Why the log-normalizer reduces over this axis: to execute across batches
- Expected one-step SGD behaviour: loss reduction

## Task 2: diagnoses before execution

### Variant 1 — global logsumexp reduction

- Observable symptom: reduces beyond just the desired axis
- Violated invariant: batches are independent
- Smallest repair: add 'axis=1' and 'keepdims=True'
- Tentative attempt-error code: shapes incorrect

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
- Tentative attempt-error code: n_classes = jnp.unique(y).shape[0]

## Post-commit repair outcomes

- Task 1: the first repair computed a log-probability expression from `y` rather than logits, broadcasting to `(B, B)` and causing an out-of-bounds gather/`NaN` on the extreme-logit test. Repair: normalise logits per class, gather from the resulting `(B, C)` log-probabilities, and return their mean negative value. Attempt error: `P`.
- Variant 1: the repair makes `logsumexp` reduce on the class axis with the class dimension retained. This preserves a separate normaliser for each batch row. Attempt error: `P`.
- Variant 2: parameters now use feature/class dimensions for `W` and one class bias per output class, rather than batch-derived dimensions. Attempt error: `P`.
- Variant 3: `jnp.unique(y)` under `jit` raised `ConcretizationTypeError` because the class count depended on traced data values. Repair: obtain the class count from the static parameter-shape contract. Attempt errors: primary `K` (JIT static-shape constraint), secondary `C` (the required pre-execution diagnosis/taxonomy was incomplete).
- Verification: `uv run pytest -q test_phase_0_diagnostic.py` → `5 passed in 1.58s`.
- API-reference lookup: no consulted URL was recorded; add one later only if a lookup actually occurred.
- Elapsed time: pending learner report.
