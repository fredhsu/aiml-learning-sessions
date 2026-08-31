# Curriculum progress

## Current control state — 2026-08-27

- Design stage: approved working curriculum, version 0.2.
- Learning phase: Phase 0 — ML/JAX experimental foundations.
- Active frontier: encode J1 `jit`/`vmap` constraints, then fade the scaffold on F1/F4/J2/J3 and E1–E3/T0 through independent changed-surface implementation and diagnosis.
- Current node evidence: F1/F4/J2/J3 are `scaffolded`; F2 is `encoded`; J1 is `scaffolded` for arrays/pytrees/PRNG but `not-encoded` for `jit`/`vmap`; J4 is `scaffolded` for tests/seeds and one-command reproduction; E1/E3 and the narrow T0 slice are `scaffolded`; E2 is `scaffolded` for imbalanced binary classification.
- Weekly hours actual: ~4h recorded this week (Session 1 ~60 min; Tasks 1–2 diagnostic ~60 min; Task 3 ~120 min piecemeal), within the 3–6h budget.
- Last learner evidence: Task 3 leakage-safe tabular vertical slice verified on 2026-08-27; first attempt `a2c1a2d`, tutor-supplied checks `4 passed in 3.55s`, and fixed-seed reproduction recorded in `task3_experiment_record.md`.
- Last whole-task evidence: narrow T0 vertical slice on 2026-08-27 — split-before-fit standardisation, majority and learned baselines, predeclared balanced accuracy, held-out class recalls, and fixed-seed reproduction; all produced with scaffolded assistance.
- Due check: no phase-gate delayed check is scheduled yet because no qualifying independent/transfer attempt exists. The next independent changed-surface attempt must precede a 7–14 day delayed check.
- Next whole-task block: an independent changed-surface T0 attempt after bounded JIT/static-shape encoding and focused `P` remediation.
- Open commitments: independently diagnose the prior three bug classes from invariants; reconstruct loss/update under a changed surface; apply split-before-fit and metric selection with faded guidance; clean generated files accidentally included in `a2c1a2d` without deleting intentional environment/skill files.

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
- Actual: final diagnostic score 3/4; total time ~180 min. Task 1 earns 1/1 with scaffolded assistance; Task 2 earns 0/1 because the committed pre-execution diagnosis condition was not met for all variants; Tasks 3–4 earn 2/2 with scaffolded assistance and reproducible held-out evidence.
- Assistance: Task 1 final correction `scaffolded` (post-commit test harness, API-reference lookup, and shape/loss feedback). Variant 1 repair used its pre-execution axis prediction, Variant 2 required direct shape feedback, and Variant 3 required static-shape instruction; the full debugging point is unscored. Tasks 3–4 are `scaffolded` due repeated contract reviews, adapted prior classifier code, and tutor-supplied tests.
- Attempt errors: `P` — Task 1 used labels rather than logits in log-probability construction, producing `(B,B)` broadcasting and an out-of-bounds gather/NaN; Variant 2 initially used batch size for bias shape. `K` — Variant 3 static-shape invariant under `jit` was unavailable; `jnp.unique(y)` requires data values while tracing. `C` — Variant 3 diagnosis and taxonomy were absent from the committed first attempt; its later error-code field contained an expression rather than `K/R/M/D/P/F/T/C`.
- Node-state transitions: J1 `scaffolded → scaffolded` for arrays/pytrees/PRNG with `jit`/`vmap` surface recorded `not-encoded`; no advancement. F1/F4/J2/J3 remain `scaffolded`; F2 remains `encoded`.
- Calibration gap: initial diagnostic prediction was 4/4 in 30 min; actual was 3/4 in ~180 min (score −1/4; time +150 min). Task 3 separately predicted 2/2 in 45 min at 70% confidence and achieved 2/2 in ~120 min (time +75 min).
- Due checks / whole-task status: remaining entry diagnostic complete; narrow T0 evidence now exists but is scaffolded, so it does not start the phase-gate delayed-check clock.
- Decision / next smallest action: apply `K` remediation to the JIT static-shape model, then a focused faded-skeleton `P` drill before an independent changed-surface loss/debug/data-protocol attempt.
- Graph or curriculum change: evidence-state updates only: J1 distinguishes scaffolded arrays/pytrees/PRNG from unencoded `jit`/`vmap`; E1/E3 and narrow T0 `not-assessed/not-encoded → scaffolded`; E2 `not-assessed → scaffolded` for imbalanced binary classification. No prerequisite or architecture change.

## 2026-08-27 — Phase 0 / Task 3 narrow T0 vertical slice
- Session card: build a fixed-seed, leakage-safe binary tabular experiment with row-level IID split, train-only standardisation, majority and JAX linear baselines, predeclared balanced accuracy, held-out class-recall slices, and one-command reproduction.
- Prediction: score 2/2; time 45 min; confidence 70%
- Evidence: first attempt `a2c1a2d`; `uv run pytest -q test_task3_tabular_experiment.py` → `4 passed in 3.55s`; `uv run python task3_tabular_experiment.py` → train/test 168/72, test counts 52/20, scaler fit rows 168, majority balanced accuracy 0.5, learned balanced accuracy 1.0, recalls 1.0/1.0, target met. Artifact: `task3_experiment_record.md`.
- Actual: score 2/2; time ~120 min, completed piecemeal
- Assistance: `scaffolded` for both points — reused/adapted prior classifier code, received repeated data/metric/model contract reviews, and ran tutor-supplied post-commit tests.
- Attempt errors: `P` — initial balanced-accuracy/class-recall implementation, model-bias omission, and incomplete pipeline wiring required focused corrections. `C` — first-attempt commit included unrelated/generated files.
- Node-state transitions: E1 `not-encoded → scaffolded`; E2 `not-assessed → scaffolded` for imbalanced binary classification; E3 `not-assessed → scaffolded`; T0 `not-assessed → scaffolded` narrow vertical slice. J4 remains `scaffolded`, now with one-command fixed-seed reproduction evidence.
- Calibration gap: score exact; time +75 min versus prediction.
- Due checks / whole-task status: first narrow T0 whole-task evidence recorded. No phase-gate delayed check scheduled because assistance prevents an independent/transfer claim.
- Decision / next smallest action: encode JIT static-shape constraints, perform focused `P` remediation, then repeat a changed-surface mechanism/debug/data-protocol attempt with faded guidance.
- Graph or curriculum change: evidence-state updates listed above; no edge or curriculum change.
