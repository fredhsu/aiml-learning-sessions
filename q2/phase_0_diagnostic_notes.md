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
