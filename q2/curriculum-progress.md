# Curriculum progress

## Current control state — 2026-08-25

- Design stage: approved working curriculum, version 0.2.
- Learning phase: Phase 0 — ML/JAX experimental foundations.
- Active frontier: retrieve F1, F4, J1, J2, and J3 from scaffolded evidence; sample E1–E3 and T0 through the remaining diagnostic.
- Current node evidence: F1/F4/J1/J2/J3 are `scaffolded`; F2 is `encoded`; J4 is `scaffolded` for tests and seeds with remaining surface `not-assessed`; E1 is `not-encoded`.
- Weekly hours actual: not yet tracked; log actual hours here each week against the 3–6h budget so the sustainability and parallel-load revision triggers have real data to fire on.
- Last learner evidence: Session 1 passing tests and fixed-seed full-batch training on 2026-08-24.
- Last whole-task evidence: none; authentic mini-task remains unsampled.
- Due check: 2026-08-26 — closed-resource Task 1 and seeded diagnoses from [`phase-0-remaining-diagnostic.md`](phase-0-remaining-diagnostic.md).
- Next whole-task block: complete Task 3 from the remaining diagnostic within the same study week; this is the first narrow T0 vertical slice.
- Open commitments: record assistance per assessed point; preserve the first committed diagnostic attempt; add tutor-supplied tests only after commitment; schedule the resulting 7–14 day delayed check.

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
