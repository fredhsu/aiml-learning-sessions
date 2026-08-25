# Phase 0 remaining diagnostic

**Design stage:** confirmed assessment design  
**Learning phase:** Phase 0 — ML/JAX experimental foundations  
**Purpose:** sample the two entry-diagnostic gaps that remain open—closed-resource JAX debugging and a small authentic experiment—without restarting intake or repeating established diagnostics.

This diagnostic is pending until the learner produces the named artifacts and the verification evidence is logged. Its results update node states; this document alone is not evidence of learner performance.

## Session card

- **Bounded objective:** reconstruct stable cross-entropy, diagnose three seeded failure variants, and build a leakage-safe held-out baseline vertical slice.
- **Prerequisites:** the Session 1 shape contract and scaffolded evidence for F1, F4, J1, J2, and J3.
- **Theory output:** before execution, state the required reduction axis, parameter shapes, split unit, metric, and expected failure signature for each bug.
- **Implementation output:** `phase_0_diagnostic_attempt.py` plus a short experiment record in `phase_0_diagnostic_notes.md`.
- **Verification:** tutor-supplied tests after the attempt is committed, one independent numerical reference, and a fixed-seed held-out result.
- **Prediction:** learner records score out of 4, elapsed time, and confidence before opening Session 1 implementation or notes.

## Conditions

1. Use this file as the task statement. Keep `session_01_linear_classifier.py` and `session_01_notes.md` closed until Tasks 1 and 2 are committed.
2. Documentation lookup for exact JAX API spelling is allowed only after the learner writes the intended operation and shapes.
3. Record predictions before running code. Preserve the first committed attempt so diagnosis can distinguish retrieval, procedure, and careless errors.
4. The tutor supplies tests or reference values only after the learner commits an answer.

## Task 1 — changed-surface retrieval

In a new file, implement stable mean multiclass cross-entropy for:

- `x: (B, D)`
- `W: (D, C)`
- `b: (C,)`
- `y: (B,)`

Use a different batch size, feature count, and class count from Session 1. Implement the loss from its contract, then use `jax.value_and_grad` to perform one pytree SGD update. Before execution, write the expected shapes of logits, loss, `dW`, and `db`.

## Task 2 — seeded debugging

For each variant, commit a diagnosis before running it. State the observable symptom, violated invariant, smallest repair, and likely attempt-error code.

1. `logsumexp` reduces over the entire logits array instead of the class axis.
2. parameters are initialised from batch size and feature count rather than feature count and class count.
3. a jitted training step derives `n_classes` with a data-dependent `unique` operation inside the compiled path.

After committing the diagnoses, repair the variants and run focused tests.

## Task 3 — authentic experiment vertical slice

Generate or load a small labelled tabular dataset and produce a fixed-seed experiment with:

1. a split performed before any fitted preprocessing;
2. one trivial baseline and one learned linear baseline;
3. a metric chosen and defended before results are computed;
4. preprocessing statistics fitted on training data only;
5. a held-out result plus at least one error slice;
6. one command that reproduces the result.

This is a narrow T0 slice, not the Phase 0 exit project. Prefer a small complete experiment over a larger model.

## Scoring and routing

| Point | Evidence required |
|---|---|
| 1 | Stable loss and gradient shapes pass changed-surface checks without consulting the prior implementation. |
| 1 | All three bugs are diagnosed from invariants before execution and repaired with focused tests. |
| 1 | Split, preprocessing, baseline, and metric choices are leakage-safe and defended before evaluation. |
| 1 | Fixed-seed held-out result, error slice, and reproduction command are recorded. |

- A score records task evidence, not node mastery.
- Record scaffold level for each point: `scaffolded`, `independent`, or `transfer`.
- Classify every substantive miss with an attempt error before choosing a remedy.
- Any leakage or invalid held-out evaluation is a critical failure and blocks credit for Tasks 3–4 regardless of aggregate score.

## Completion record

Append the result to `curriculum-progress.md` with artifact paths, commands and output, actual score/time, attempt errors, node-state transitions, calibration gap, and the next due delayed check.
