# Curriculum progress

## Current control state — 2026-08-31

- **Design stage:** approved working curriculum, version 0.1. Design stages 1–5 of `ai-curriculum-builder-prompt.md` are complete; stage 6 (iteration) is open.
- **Learning phase:** Phase 0 — measurement and inference arithmetic.
- **Active frontier:** Guided A2/A3 encoding followed by an alternate-form, closed-resource A1 re-attempt. W0 has supplied the first measured H1 constant; its 949× prediction miss must be explained before it is used as a planning constant.
- **Current node evidence:** A2/A3 arithmetic has been correct only with explicit formulas and corrections (`scaffolded`). A1 remains unattempted closed-resource. W0 hardware evidence exists, but the learner has not independently constructed or reproduced its protocol.
- **Weekly hours actual:** not yet recorded. Budget is 2–3 h/week as a secondary track; primary track is `../q2`.
- **Last learner evidence:** none.
- **Last whole-task evidence:** none.
- **Due check:** none scheduled. The first delayed check becomes due 7–14 days after the first qualifying independent or transfer attempt.
- **Next whole-task block:** Block C of the entry diagnostic (`C1` — predict, measure, explain the gap), which is the first narrow T0 slice.
- **Environment status:** `uv sync` completed on 2026-08-31, producing `.venv` and `uv.lock`: Python 3.12.7, PyTorch 2.13.0+cu130, CUDA 13.0. RTX 4090 confirmed present (24564 MiB); GB10 not yet exercised from this repository.
- **Open commitments:**
  1. Run entry diagnostic Block A (arithmetic + confound discrimination), closed-resource, predictions committed first.
  2. Run Block B (from-memory KV cache and cached attention; five seeded faults diagnosed before execution).
  3. Run Block C (predict → measure → explain), producing the first `bench/workload-contract.md` entry.
  4. Explain the W0 bandwidth prediction miss without revising the committed prediction; then measure matmul throughput under its own predeclared contract.
  5. Confirm or amend the model family choice for the bounded scope before Phase 1.
  6. Write `bench/test_harness_catches.py` against the T0 acceptance specification, declaring each check's threshold in the contract before the test is written. This is now a Phase 0 exit gate row.

## Curriculum changes — 2026-09-01

Design review of the working curriculum against the first two sessions of contact. Four material changes, recorded in the design log of `inference-curriculum.md`:

1. **Entry diagnostic Item A1** decomposed into seven independently scored sub-items with per-sub-item definitional unlocks and an error-carried-forward rule. Conditions 5 and 6 added. The 2026-08-31 halt is the evidence that motivated it; the re-attempt runs on an alternate form.
2. **T0 acceptance specification** added, including the six-case confound self-test. New Phase 0 exit gate row: *Harness self-test*.
3. **Phase 2 exit gate** gained a *Correctness under concurrency* row: per-request greedy parity between engine v1 under concurrency and engine v0 at batch 1, gating every frontier number.
4. **Session prediction instruments separated.** Metric prediction remains mandatory per session; score prediction moves to macrocycle checkpoints and multi-item assessments; the per-session confidence percentage is removed in favour of per-item binary bets. Calibration is now recorded and reviewed in log space.

Not yet acted on, raised in the same review and still open: banning bare `guess` as a prediction basis in favour of a named bound plus efficiency fraction; promoting the H1 constants run to a supplied Session 0; adding a host-side (tokenise/detokenise/serialise) node so the graph teaches all seven bottleneck classes; moving a recognition-level X1 claim analysis into Phase 0 to get external feedback before month ten; committing the repository to git, which currently tracks none of it; and the stale environment status in `README.md`.

## Prediction ledger

The primary calibration instrument for this curriculum. Every measurement appends a row, whether or not the prediction was met.

| Date | Node | Metric | Predicted (tolerance) | Basis | Actual | log₁₀(actual/predicted) | In tolerance? | Explained? |
|---|---|---|---|---|---|---:|---|---|
| 2026-08-31 | H1 | Effective streaming source-read bandwidth, W0 p50 | 1 GB/s (±10%) | guess | 948.648 GB/s | +2.977 | no | no — causal model pending |

Basis is `arithmetic`, `prior measurement`, or `guess`. Three consecutive `guess` rows on the same node route to `K` regardless of whether the predictions were met.

Gaps are recorded in log space so that a large miss and a small one are not averaged together. The macrocycle review statistic is the **median absolute log ratio** plus the **tolerance hit-rate**; see the revision triggers in `inference-curriculum.md`. A hit-rate at or near 100% means the tolerances are too wide to falsify anything, not that calibration is good.

## Session log

## 2026-08-31 — Phase 0 / entry diagnostic Block A
- Session card: Closed-resource A1 inference arithmetic plus A2 confound discrimination; target 2 points in about 40 minutes.
- Prediction: score 2/2; time 30 min; confidence 80%.
- Metric prediction: not applicable — no benchmark, profile, or optimisation was run.
- Evidence: learner committed the prediction in conversation before Block A was revealed; learner then reported that an accurate A1 computation would be guessing. No resources consulted and no commands or measurements run.
- Actual: score incomplete (A1 not attempted); time not recorded; measured metric not applicable.
- Prediction gap: not yet quantifiable; the 2/2 prediction is currently unsupported because the A1 procedure could not be initiated.
- Assistance: no assessed point; diagnostic halted before tutoring could leak the mechanism.
- Attempt errors: `K` primary — unable to derive A1 rather than produce a structured but incorrect derivation; routes to guided A2/A3 encoding. `C` not evidenced.
- Bottleneck class (if a performance shortfall): not applicable.
- Node-state transitions: A2 `not-assessed → not-encoded`; A3 `not-assessed → not-encoded`, based on the blocked cold derivation. Other nodes unchanged.
- Calibration gap: open; resolve only after a completed alternate-form A1 attempt.
- Due checks / whole-task status: no delayed check due; T0 not started.
- Decision / next smallest action: stop the diagnostic rather than guess. Complete one guided parameter/KV/decode-byte accounting example, then re-attempt A1 on a near variant closed-resource.
- Graph or curriculum change: none.

## 2026-08-31 — Phase 0 / H1 W0 bandwidth constant
- Session card: establish the Python/CUDA environment; declare a source-read bandwidth workload contract; predict then measure the first 4090 hardware constant.
- Prediction: metric only — 1 GB/s ±10%, confidence not separately recorded; basis `guess`.
- Metric prediction: W0 p50 effective streaming source-read bandwidth = 1 GB/s (tolerance ±10%); committed and entered in the prediction ledger before the run.
- Evidence: `uv sync`; contract `W0` in `bench/workload-contract.md`; `uv run python -m py_compile bench/bandwidth_microbench.py`; `uv run python bench/bandwidth_microbench.py --tensor-bytes 1073741824 --warmup 10 --repetitions 30`; result `bench/results/W0-2026-08-31.json`; clock snapshots `bench/results/W0-2026-08-31-{pre,post}.csv`; constants table `bench/constants.md`.
- Actual: formal score not assessed; time not recorded; W0 p50 = 948.648 GB/s, mean = 919.987 GB/s, n = 30.
- Prediction gap: 948.648× higher than predicted. Learner's committed explanation: no physical scale model; do not revise the prediction. The p99 latency (5.743 ms) is 5.07× p50 (1.132 ms), so the slow tail must not be erased by the mean. Why a real decode path falls below its bandwidth-only upper bound remains hypotheses, not a diagnosis.
- Assistance: scaffolded protocol — tutor created the contract and implementation after the learner's committed prediction. Tutor prematurely requested profiler-level discriminating evidence before M3 was encoded; that prompt was withdrawn rather than scored.
- Attempt errors: `M` secondary in guided explanation — initially treated weight loading as an alternative to matrix multiplication rather than its required input; corrected to a 2 FLOP / 2 B = 1 FLOP/B account. `K` remains the primary diagnostic error on the unassisted A1 derivation. No learner error is assigned for the withdrawn profiler-evidence prompt.
- Bottleneck class (if a performance shortfall): not applicable; W0 is a hardware microbenchmark, and the source of its slow tail is unconfirmed.
- Node-state transitions: A2 `not-encoded → scaffolded`; A3 `not-encoded → scaffolded` through guided parameter/KV/decode-byte/FLOP accounting. H1 unchanged (`not-assessed`): a measured artifact is not independent learner performance.
- Calibration gap: 948.648× on W0; do not revise the prediction retrospectively. The next task is to derive the model that would have predicted the observed scale.
- Due checks / whole-task status: no delayed check due; T0 remains a narrow in-progress slice. W0 has a valid declared contract but is not an inference baseline.
- Decision / next smallest action: encode M1/M2 with one worked benchmark-confound contrast before requesting trace evidence; then re-attempt A1 on an alternate form closed-resource. No second measurement until its contract and prediction are committed.
- Graph or curriculum change: none.
