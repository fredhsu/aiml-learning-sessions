# Curriculum progress

## Current control state — 2026-08-27

- Design stage: approved working curriculum, version 0.2.
- Learning phase: Phase 0 — ML/JAX experimental foundations.
- Active frontier: retrieve F1, F4, J1, J2, and J3 from scaffolded evidence; sample E1–E3 and T0 through the remaining diagnostic.
- Current node evidence: F1/F4/J2/J3 are `scaffolded`; F2 is `encoded`; J1 is `scaffolded` for arrays/pytrees/PRNG but `not-encoded` for `jit`/`vmap`; J4 is `scaffolded` for tests and seeds with remaining surface `not-assessed`; E1 is `not-encoded`.
- Weekly hours actual: not yet tracked; log actual hours here each week against the 3–6h budget so the sustainability and parallel-load revision triggers have real data to fire on.
- Last learner evidence: delayed-retrieval Task 1 and seeded-variant repairs verified on 2026-08-27; first attempt `7ba644ef499a01598686ff5e3bc5b76898428a09`, repair commit `3f2299f630f7ce21c7d4beb4796334951836d943`.
- Last whole-task evidence: none; authentic mini-task remains unsampled.
- Due check: 2026-08-26 delayed retrieval: Task 1 and all three seeded-variant tests now pass after feedback (`5 passed in 1.58s`); its assessment score/time remain pending. Task 3 has not started.
- Next whole-task block: complete Task 3 from the remaining diagnostic within the same study week; this is the first narrow T0 vertical slice.
- Open commitments: record actual time and task-local score; complete Task 3 with a predeclared metric and leakage-safe data protocol. Schedule the resulting 7–14 day delayed check only after the diagnostic and T0 slice yield qualifying evidence.

## 2026-08-24 — Phase 0 / J1, F1, F4, J2 → J3
- Session card: Implement a deterministic pure-JAX full-batch three-class linear classifier: synthetic data, small-normal initialisation, stable cross-entropy, `value_and_grad` + pytree SGD update, and fixed-seed training.
- Prediction: score 3/4; time 75 min; confidence 65%
- Evidence: `uv run pytest -q` → `4 passed in 4.38s`; fixed-seed run: `initial_loss=1.089214`, `final_loss=0.011020`, `accuracy=1.000`; artifacts: `session_01_linear_classifier.py`, `test_session_01_linear_classifier.py`, `session_01_notes.md`.
- Actual: score 4/4; time ~60 min
- Assistance: `scaffolded` — the attempt used a traced shape contract, coaching, TODO skeleton, and supplied tests; the 4/4 task score is not independent or transfer evidence.
- Attempt errors: `K` — stable cross-entropy procedure initially unavailable; resolved through shape trace, per-class `logsumexp`, and correct-class gather. `P` — swapped PRNG-key/centre tuple unpacking; used batch count as feature/class dimensions; initially omitted the per-class reduction axis. Resolved with local traces and targeted tests.
- Node-state transitions: F1, F4, J1, J2, and J3 → `scaffolded`; J4 → `scaffolded` for deterministic tests/seeds only. These transitions were assigned during the 2026-08-25 evidence-model update and do not retroactively claim independent competence.
- Calibration gap: score +1/4 (predicted 75%, actual 100%); time 15 min faster than predicted. Confidence was moderately conservative.
- Due checks / whole-task status: changed-surface retrieval due 2026-08-26; no authentic whole-task evidence yet.
- Decision / next smallest action: after ~2 days, perform a closed-resource 20-minute retrieval: reconstruct `cross_entropy` from its shape contract, then diagnose two seeded variants (global `logsumexp` reduction; incorrect parameter dimensions) before rerunning tests. No new model type until this succeeds.
- Graph or curriculum change: none

## 2026-08-25 — Curriculum control update / evidence model and Phase 0 diagnostic

- Session card: no learner session; repository design update separating design stage, learning phase, node state, attempt error, prerequisite edges, sequence constraints, and integration requirements.
- Prediction: score not applicable; time not used as learner evidence; confidence not applicable.
- Evidence: `CONTEXT.md`; revised `robot-learning-curriculum.md`; revised `robot-learning-dependency-graph.md`; `phase-0-remaining-diagnostic.md`; document/test validation commands recorded in the implementation handoff.
- Actual: no learner-performance claim. Session 1 evidence was reclassified as scaffolded node evidence without changing its historical 4/4 task result.
- Attempt errors: none classified; no learner attempt occurred.
- Calibration gap: not applicable.
- Decision / next smallest action: on 2026-08-26, commit score/time/confidence predictions and complete the changed-surface cross-entropy reconstruction plus seeded diagnoses before consulting Session 1 artifacts.
- Graph or curriculum change: version 0.2 separates edge/state semantics, removes R1 as a BC prerequisite while retaining it as a sequence constraint, splits T3 into T3A/T3B, and adds binary phase exit scorecards. Material change confirmed by the learner's instruction to execute all high-priority improvements.

## 2026-08-25 — Curriculum control update / F2 evidence correction and process addition

- Session card: no learner session; correcting an uncredited evidence gap found during a curriculum review, plus a process addition for sustainability tracking.
- Prediction: not applicable; no learner attempt.
- Evidence: `session_01_notes.md:1-24` records a hand-derived chain-rule gradient (`dL/dZ`, `dL/dW`, `dL/db`) from the cross-entropy loss definition, produced during Session 1. `session_01_linear_classifier.py:65-74` shows `update()` computes gradients via `jax.value_and_grad`, so the hand derivation was never independently coded or cross-checked against autodiff.
- Actual: no learner-performance claim; this reclassifies existing evidence, it does not record a new attempt.
- Attempt errors: none classified; not an attempt.
- Node-state transitions: F2 `not-assessed → encoded` — the derivation is accurate and matches the correct chain-rule mechanism, satisfying `encoded` (accurately derived/explained) per `CONTEXT.md`. It does not meet `scaffolded`, since executable performance (an independent gradient implementation checked against `jax.value_and_grad`) was never produced.
- Calibration gap: not applicable.
- Due checks / whole-task status: unchanged; 2026-08-26 changed-surface retrieval and seeded diagnoses from `phase-0-remaining-diagnostic.md` remain due.
- Decision / next smallest action: F2 can advance to `scaffolded` opportunistically the next time a gradient is implemented and checked against autodiff (e.g. as an optional cross-check inside Task 1 of the remaining diagnostic) — no dedicated session is required solely for this.
- Graph or curriculum change: `robot-learning-dependency-graph.md` F2 row updated from `not-assessed` to `encoded`. Added a "Weekly hours actual" field to this file's current-control-state block so the sustainability and parallel-load revision triggers (`robot-learning-curriculum.md` revision-triggers table) have real data to evaluate against, rather than being assumed satisfied.

## 2026-08-27 — Phase 0 / delayed retrieval diagnostic (in progress)
- Session card: closed-resource reconstruction of stable multiclass loss and one JAX pytree update on `B=7, D=4, C=5`; diagnose and repair three seeded variants before the T0 vertical slice.
- Prediction: score 4/4; time 30 min; confidence 70%
- Evidence: first attempt committed before tests as `7ba644ef499a01598686ff5e3bc5b76898428a09` (`phase_0_diagnostic_attempt.py`, `phase_0_diagnostic_notes.md`); repair commit `3f2299f630f7ce21c7d4beb4796334951836d943`. `uv run pytest -q test_phase_0_diagnostic.py` → `5 passed in 1.58s`. `phase_0_diagnostic_notes.md` records the repairs and the Variant 3 `ConcretizationTypeError`.
- Actual: in progress; score and elapsed time not yet recorded
- Assistance: Task 1 final correction `scaffolded` (post-commit test harness, API-reference lookup, and shape/loss feedback). Variant 1 repair was based on its pre-execution axis prediction but the full debugging point is unscored; Variant 2 required direct shape feedback; Variant 3 remains open.
- Attempt errors: `P` — Task 1 used labels rather than logits in log-probability construction, producing `(B,B)` broadcasting and an out-of-bounds gather/NaN; Variant 2 initially used batch size for bias shape. `K` — Variant 3 static-shape invariant under `jit` was unavailable; `jnp.unique(y)` requires data values while tracing. `C` — Variant 3 diagnosis and taxonomy were absent from the committed first attempt; its later error-code field contained an expression rather than `K/R/M/D/P/F/T/C`.
- Node-state transitions: J1 `scaffolded → scaffolded` for arrays/pytrees/PRNG with `jit`/`vmap` surface recorded `not-encoded`; no advancement. F1/F4/J2/J3 remain `scaffolded`; F2 remains `encoded`.
- Calibration gap: pending completion
- Due checks / whole-task status: delayed retrieval remains open; no authentic T0 evidence. Task 3 is still required this study week.
- Decision / next smallest action: record actual elapsed time, then start Task 3 with a pre-run metric and data-protocol prediction; no Phase 0 node advances from the scaffolded retrieval repair alone.
- Graph or curriculum change: `robot-learning-dependency-graph.md` J1 evidence state refined to distinguish scaffolded arrays/pytrees/PRNG from unencoded `jit`/`vmap`; this is an evidence-state correction, not a prerequisite or architecture change.
