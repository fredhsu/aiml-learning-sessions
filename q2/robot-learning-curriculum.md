# Robot-Learning Curriculum

**Version:** 0.2  
**Design stage:** approved working curriculum; evidence-gated and revisable  
**Learning phase:** Phase 0 — ML/JAX experimental foundations  
**Dependency graph:** [`robot-learning-dependency-graph.md`](robot-learning-dependency-graph.md)

## North-star performance

Given a bounded tabletop manipulation task for an SO-101, independently build, evaluate, and publicly document a reproducible robot-learning system: specify success, safety, and generalisation metrics; establish scripted/control and learned baselines; validate a simulation environment; train a learning-based policy; diagnose data, perception, optimisation, control, and sim-to-real failures; and deploy/evaluate on the physical arm where feasible.

For each major learning component, reconstruct its core mathematical/algorithmic mechanism in JAX, critically analyse papers/references, and publish code, experiment records, ablations, and an explanation of load-bearing design choices.

- **Primary criterion:** end-to-end system performance.
- **Supporting criterion:** theory and reproduction work must improve design, diagnosis, or justification of that system.
- **Retention target:** usable performance at 1, 3, and 12 months.
- **Physical deployment:** desired evidence, not a hard completion gate.

## Entry diagnostic snapshot — historical

This table records the evidence that selected the initial frontier. It is not the current node-state store; current state and open evidence are maintained in the dependency graph and `curriculum-progress.md`.

| Area | Entry evidence | Entry attempt diagnosis |
|---|---|---|
| VJP / reverse-mode rationale | Correct explanation | usable conceptual base |
| Softmax and parameter-gradient shapes | Material errors | `P` / possible `R` |
| Numerically stable softmax | Stability mechanism unavailable | `K/R` |
| Data splitting, leakage, metrics | No reliable procedure | `K` |
| Error taxonomy | Correctly classified sample cases | usable declarative distinction |
| JAX training-loop implementation | Explicitly unavailable | `K` |
| JAX debugging | Not sampled | unknown |
| Whole-task experiment | Not sampled | unknown |
| Robot learning, control, sim-to-real | Not yet encoded | `K` |

The unsampled JAX-debugging and whole-task rows are completed by [`phase-0-remaining-diagnostic.md`](phase-0-remaining-diagnostic.md). Do not restart intake or repeat the evidence already collected.

## Operating settings

| Setting | Design |
|---|---|
| Domain type | Type 1 theory/prerequisites + Type 5 engineering; Type 3 debugging |
| Framework | JAX primary |
| Theory → implementation | Worked derivation → traced reference → completed skeleton → independent implementation |
| Worked examples | High initially, explicitly faded **[A]** |
| Retrieval | Low-volume and implementation/debugging-shaped; mostly embedded in projects **[A]** |
| Interleaving | Begin only after isolated procedures work; used for discrimination **[B]** |
| Whole tasks | Begin in Phase 0 and continue throughout |
| Advancement | Exit evidence, never time elapsed or resources consumed |

## State and error model

Use the vocabulary in [`CONTEXT.md`](CONTEXT.md):

- **Design stage** describes the maturity of the curriculum design.
- **Learning phase** identifies the active body of learner work.
- **Node state** records the strongest evidence held: `not-assessed → not-encoded → encoded → scaffolded → independent → transfer → delayed-secure`.
- **Attempt error** is a `K/R/M/D/P/F/T/C` diagnosis for one miss. It routes a remedy and is never used as a persistent node state.
- **Prerequisite**, **sequence**, and **integration** edges remain separate. Preferred order must not be represented as a capability dependency.

## Theory-to-code loop

For every new mechanism:

1. Derive/explain its model, assumptions, and expected behaviour.
2. Trace a known-correct reference implementation, including shapes and invariants.
3. Complete a faded skeleton and predict test outcomes before execution.
4. Implement or modify it independently in the active project.
5. Reproduce, debug, or apply it later under changed conditions.

Do not use open-ended discovery before step 3 works. This uses worked examples and fading deliberately **[A]**.

# Phase sequence

| Phase | Frontier / theory | Whole task | Scaffolding fade | Exit milestone |
|---|---|---|---|---|
| 0. ML/JAX experimental foundations | F1–F5, J1–J4, E1–E3, L1 | Reproducible tabular baseline | Reference → skeleton → from-memory linear classifier | Public repo with leakage-safe pipeline, metric defence, held-out result, and error analysis |
| 1. Robot and control foundations | C1–C4, S1 | State-based simulated reaching with scripted controller | Given environment → modified environment → bounded constructed environment | Controller and simulator with fixed-seed evaluation and failure report |
| 2. Early RL in simulation | L4, RL theory and optimisation | State-based RL reaching task, compared with scripted baseline | Canonical algorithm → partial implementation → independent ablation | Reproducible RL result on locked evaluation seeds, with instability/failure explanation |
| 3. Demonstrations and imitation | S2, L2, L5 | Validate simulation demonstrations; behavioural-cloning policy | Supplied schema → learner audit → self-designed data protocol | BC policy, rollout evaluation, covariate-shift analysis, and RL comparison |
| 4. Policy transfer and paper reproduction | L3 as needed, L5, X1–X2, T3A–T3B | Shifted-condition policy transfer plus paper reproduction/ablation; one combined project may satisfy both evidence sets | Paper annotation → partial reproduction → independent implementation, transfer, and ablation | Public transfer/reproduction report including one negative or discrepant result and both T3A/T3B gates |
| 5. SO-101 deployment | S3, T4 | Safety-gated tabletop reaching/grasping deployment where feasible | Checklist → learner-authored checklist → independent preflight | Sim-to-real report; physical result or explicit evidenced deployment blocker |

Phases are evidence-gated. At 3–6 hours/week, a phase may occupy one or more four-week macrocycles; this is not a deadline.

# Per-phase control design

| Phase | Encoding resources and outputs | Retrieval / interleaving | Deliberate-practice target | Feedback and milestone |
|---|---|---|---|---|
| 0 | Official JAX introductory/autodiff documentation; canonical linear-classifier implementation; selected supervised-learning, optimisation, and validation theory. Output: shape traces, stable-softmax derivation, JAX loss/update tests. | Prompts: implement stable CE; predict gradient shapes; locate leakage; choose metric. Mix only after isolated competence. | Shape tracing; stable loss; PRNG discipline; `value_and_grad`; split-before-fit. | JAX property tests; numerical parity with independent reference; public code-review request. Milestone: tabular project. |
| 1 | Frames/kinematics material, introductory feedback-control material, MuJoCo/MJX docs. Output: coordinate/error derivation, modified and then minimal constructed simulation task. | Define state/action/goal from memory; explain controller failure; identify frame mismatch. Mix frame, dynamics, and log cases. | Coordinate transforms; discrete rollout reasoning; PID/controller tuning under fixed seeds. | MuJoCo/MJX engine behaviour; locked-seed controller evaluation; external task-definition review. |
| 2 | Sutton & Barto MDP, return, value/policy-method material; one minimal JAX RL reference. Output: Bellman/policy-gradient derivations tied to code. | Rebuild loss/update fragments; predict learning-curve failures; discriminate reward, exploration, optimiser, and environment errors. | Returns/advantages; rollout batching; RNG handling; fixed evaluation; learning-curve diagnosis. | Held-out simulator seeds and benchmark-style evaluation; paper/reference metric comparison where applicable. |
| 3 | Behavioural-cloning, covariate-shift, and robot-data documentation; LeRobot data conventions as relevant. Output: data-lineage diagram and action-observation alignment tests. | Diagnose whether bad rollouts arise from data, temporal offset, objective, or control. Mix BC and RL failure logs. | Dataset split unit; temporal alignment; masking; rollout versus per-step metrics. | Data-integrity tests; scripted/RL baseline comparison; LeRobot or robotics-community review request. |
| 4 | One selected architecture paper plus a primary reference implementation; vision/temporal-policy theory only when the T3A route needs it. Output: claim/assumption map before code. | From-memory core mechanism; discriminate architecture, data, optimiser, and evaluation explanations of discrepancy. | Paper-to-test translation; ablation design; policy transfer; image/action tensor contracts when applicable. | T3A shifted-condition evidence and T3B reproduction/ablation evidence; they may share one artifact. Compare against the paper or public benchmark and request external review. |
| 5 | SO-101 documentation, calibration/safety procedure, targeted sim-to-real material. Output: physical preflight and risk log. | Preflight from memory; predict mismatch; diagnose run traces. | Calibration; latency measurement; workspace/safety constraints; deployment logging. | Physical arm feedback when available; otherwise record exact blocker and validate a changed simulator-side condition. |

Resources are tools, not completion metrics. Each has an attached output.

# Weekly operating system

## Default allocation

| Phase | Retrieval | Encoding | Targeted practice | Whole task | Feedback / planning |
|---|---:|---:|---:|---:|---:|
| 0–1 | 15% | 35% | 30% | 15% | 5% |
| 2–3 | 15% | 25% | 25% | 25% | 10% |
| 4–5 | 10% | 20% | 20% | 40% | 10% |

At the 3-hour floor, retain one 45-minute whole-task block; cut new material first. At 6 hours, add experiment/evaluation time rather than passive consumption.

## Session template: 75–90 minutes

1. **10–15 min:** closed-resource retrieval from prior work.
2. **25–30 min:** theory/worked example or active-project continuation.
3. **25–35 min:** targeted implementation/problem practice.
4. **10 min:** run a test, fixed-seed evaluation, or improve the whole task.
5. **5 min:** log evidence, error code, calibration, and next action.

## Four-week macrocycle

| Week | Function |
|---|---|
| 1 | Encode one bounded prerequisite; worked and faded examples |
| 2 | Independent implementation; delayed retrieval of Week 1 |
| 3 | Mixed/debugging practice; external feedback or locked evaluation |
| 4 | Closed-resource cumulative assessment, transfer task, error-log review, plan adjustment |

Prior nodes reappear after roughly 2 days, 1 week, 3–4 weeks, then in later projects and 12-week checks. Distributed retrieval is **[A]**; precise intervals are adaptive planning heuristics.

# Error-routing rules

Classify every substantive miss before altering the plan.

| Dominant pattern over two sessions | Remedy | Do not do |
|---|---|---|
| `K` | Pause dependent task; guided derivation and worked example; retry near variant | Schedule retrieval for unencoded material |
| `R` | Raise retrieval to 25%; add closed-resource code/derivation next session and next week | Add more explanation |
| `M` | Contrast cases and corrected re-attempt; predict before rerun | Repeat the same solution |
| `D` | Mix confusable loss/metric/data/control/debug cases after isolated competence | Return only to isolated study |
| `P` | Faded code skeleton and focused unit tests | Re-teach broad theory |
| `F` | Timed shape/notation/debug-trace drill after accuracy | Introduce algorithms |
| `T` | Changed-environment whole task and assumption debrief | Add flashcards |
| `C` | Checklist, commit-before-run rule, pacing break | Treat as content deficit |

**Dominant** means at least three instances or one-third of substantive errors across two sessions. New material labelled `K` is the active frontier, not a personal failure.

# Assessment stack

| Measure | Cadence | Evidence |
|---|---|---|
| Closed-resource retrieval | Most sessions | Derivation, shape trace, code fragment, or diagnosis |
| Implementation check | Weekly | Passing tests plus predicted failure mode before execution |
| Cumulative checkpoint | Every macrocycle | Mixed prior/current problems and debugging task |
| Transfer measure | Every macrocycle | New dataset, seed distribution, task surface, or changed requirement |
| Phase-gate delayed check | 7–14 days after the qualifying independent/transfer attempt | Alternate-form implementation or diagnosis before phase advancement |
| Maintenance delayed measure | 4–12 weeks after a node exits active study | Alternate-form implementation or diagnosis; regression changes the node state and reopens remediation |
| Long retention | At 6 and 12 months | Rebuild/modify a representative subsystem and run novel diagnostic experiment |
| Calibration | Every checkpoint | Predict score/time/confidence before committing work |
| Explanation defence | Each phase exit | Record technical defence of metrics, assumptions, and design choices |

## Phase exit rule

Advance only with all of:

1. accurate independent performance;
2. justified choice among confusable alternatives;
3. a changed-surface transfer result;
4. one delayed recheck;
5. a reproducible experiment record.

Scores are task-local and require an explicit point rubric. A score never implies a node state by itself. Every assessed point also records assistance as `scaffolded`, `independent`, or `transfer`. Critical failures named in a gate override aggregate scores.

## Phase exit scorecards

Every row is a binary gate. Performance thresholds that depend on a task or environment must be declared in its experiment contract before results are run; they may not be chosen after observing results.

### Phase 0 — ML/JAX experimental foundations

| Gate | Required evidence |
|---|---|
| Independent mechanism | Closed-resource reconstruction of stable multiclass loss and a functional JAX update on unseen `B/D/C` shapes; executable shape, finite-value, and gradient checks pass. |
| Debugging and discrimination | Diagnose seeded global-reduction, parameter-dimension, PRNG/compilation, leakage, and metric-selection cases from invariants before execution; no unresolved critical case. |
| Whole task / transfer | T0 runs on a fresh tabular dataset or materially changed data contract with split-before-fit preprocessing, a trivial and learned baseline, pre-declared metric, held-out result, and error slices. Any leakage invalidates the gate. |
| Delayed | After 7–14 days, rebuild or repair an alternate loss/update/data-pipeline variant without the prior implementation. Required nodes reach `delayed-secure`. |
| Reproducibility | One clean command recreates the environment and fixed-seed result from tracked code; the record includes configuration, dependency lock, result, and explanation defence. |

### Phase 1 — robot and control foundations

| Gate | Required evidence |
|---|---|
| Independent mechanism | From a task contract, derive frames/error/action semantics and implement the scripted controller and reset/step loop without a completed reference. |
| Debugging and discrimination | Correctly distinguish at least one frame, discretisation, controller, actuator-limit, and simulator-interface failure using traces or targeted tests. |
| Whole task / transfer | T1 meets its pre-registered success and safety thresholds on locked seeds, then survives a changed goal distribution and one changed dynamics/discretisation condition. |
| Delayed | After 7–14 days, modify or reconstruct the controller/environment interface under a changed coordinate or action convention. |
| Reproducibility | Fixed-seed evaluation table, simulator/config version, controller parameters, failure report, and reproduction command. |

### Phase 2 — early RL in simulation

| Gate | Required evidence |
|---|---|
| Independent mechanism | Reconstruct the selected return/advantage and policy/value-loss path, connect it to the JAX update, and pass numerical/shape checks. |
| Debugging and discrimination | From committed predictions and logs, distinguish reward, exploration, optimiser, RNG/batching, and environment causes; confirm at least one diagnosis with a controlled intervention. |
| Whole task / transfer | R1 is evaluated on locked seeds across at least three declared training seeds, compared with the scripted baseline, and retested on a changed goal or dynamics distribution. |
| Delayed | After 7–14 days, repair or ablate an alternate-form RL training path without the completed implementation. |
| Reproducibility | Seed-level results, aggregate uncertainty, learning curves, configuration, checkpoint/evaluation separation, and one-command evaluation. |

### Phase 3 — demonstrations and imitation

| Gate | Required evidence |
|---|---|
| Independent mechanism | Audit and implement data lineage, split unit, temporal alignment, masking, BC loss, and rollout evaluation without a supplied completed pipeline. |
| Debugging and discrimination | Correctly distinguish data corruption/leakage, temporal offset, objective, optimisation, covariate shift, and control failures with targeted evidence. |
| Whole task / transfer | T2 passes data-integrity tests and pre-registered rollout criteria on held-out initial conditions; compare per-step and rollout metrics with R1 and test one shifted condition. |
| Delayed | After 7–14 days, diagnose and repair a changed demonstration schema or alignment fault closed-resource. |
| Reproducibility | Dataset lineage/version, split manifest, fixed rollout seeds, policy configuration, comparison table, and failure analysis. |

### Phase 4 — policy transfer and paper reproduction

| Gate | Required evidence |
|---|---|
| Independent mechanism | Produce a claim/assumption map, implement the selected paper mechanism in JAX, and defend which choices are load-bearing. |
| T3A transfer | An R1- or T2-derived policy is evaluated under a pre-declared shifted condition; a vision route additionally demonstrates L3 tensor/temporal contracts. |
| T3B reproduction | Reproduce a bounded reported result or document a quantified discrepancy, then run at least one pre-declared ablation under changed conditions. |
| Delayed | After 7–14 days, reconstruct or modify the core mechanism and diagnose an alternate discrepancy without the completed code. |
| Reproducibility | Public-quality report, primary-reference comparison, seed/config record, negative or discrepant result, and reproduction command. T3A and T3B may share one project. |

### Phase 5 — SO-101 deployment

| Gate | Required evidence |
|---|---|
| Independent preflight | Learner-authored checklist passes calibration, workspace, action/limit, latency, stop-condition, logging, and rollback checks before any physical policy run. Any safety anomaly blocks deployment. |
| Debugging and discrimination | Diagnose representative calibration, latency, frame, actuation, contact, perception, and policy failures from preflight or run traces. |
| Whole task / transfer | T4 reports a safety-gated physical evaluation under pre-registered metrics, or an exact evidenced blocker plus a changed simulator-side validation that exercises the blocked assumption. |
| Delayed | After 7–14 days, perform the preflight and discrepancy diagnosis from memory with checklist verification; checklist lookup remains permitted for safety-critical execution. |
| Reproducibility | Hardware/software versions, calibration and risk log, run identifiers, metrics, stop events, video/log evidence when available, and a sim-to-real discrepancy report. |

# Feedback channels not controlled by the learner

| Channel | Used in | Role |
|---|---|---|
| MuJoCo/MJX dynamics and locked-seed evaluation | 1–4 | Tests behavioural claims against an environment |
| Reference-paper results and benchmark protocols | 2, 4 | External target for reproduction/discrepancy analysis |
| SO-101 physical behaviour | 5 | Reveals latency, calibration, contact, and sim-to-real failures |
| Requested public review: LeRobot community, robotics forums, or GitHub review | 0, 3–5 | Critique outside self-authored tests |

If human review does not arrive within a week, do not interpret silence as approval. Preserve benchmark, engine, and hardware feedback.

# Motivation and operations

## Implementation intention

> If it is my planned study window on Monday, Wednesday, or Saturday, then I open the active repository, run the retrieval prompt first, and work until the next committed artifact or test result.

Adapt the days, not the cue-action structure.

## Fallback session: 20 minutes

1. Complete one old retrieval or debugging prompt.
2. Run one existing test or fixed-seed evaluation.
3. Log one error code and the next smallest action.

No new material in fallback mode.

## Parallel-load rule

The 3–6 hour budget includes parallel robotics work. Do not run two unrelated theory frontiers simultaneously. Parallel work counts here only when it produces a logged artifact, retrieval event, or diagnostic result.

# Revision triggers

| Trigger | Mandatory response |
|---|---|
| Two failed exit attempts | Audit graph dependencies; do not simply repeat phase |
| `K`/`M` dominates | Reduce scope to prerequisite nodes; restore worked examples |
| `R` dominates | Increase delayed closed-resource retrieval; reduce new material |
| Calibration misses by >20 points on three assessments | Require score/time prediction every session for one macrocycle |
| Whole-task work absent for two weeks | Reserve next session for integration before more theory |
| No delayed measure in a macrocycle | Block advancement until completed |
| Two missed weeks | Resume with fallback sessions and shrink active task; never catch up |
| Physical safety/calibration anomaly | Stop deployment; reproduce/diagnose in logs or simulation before retry |

# Design log

| Decision | Rationale |
|---|---|
| Systems outcome primary; theory/reproduction substantial supporting loop | Builds practical competence while retaining deep theoretical understanding |
| Theory leads bounded implementation cycles | Prevents unguided construction without deferring integration indefinitely |
| Simulation-first; physical deployment desired | Faster verifiable feedback while retaining real-world criterion |
| RL before behavioural cloning is a sequence constraint, not a prerequisite | Preserves learner preference without making RL a false capability dependency of BC |
| Tabular baseline first | Cheaply exposes leakage, metric, baseline, and reproducibility gaps |
| No large SRS deck | Future performance is implementation/debugging/prediction, not verbal recall |
| Separate node state from attempt errors | Evidence maturity persists across attempts; `K/R/M/D/P/F/T/C` diagnoses only the miss that routes a remedy |
| Split Phase 4 into T3A policy transfer and T3B paper reproduction | Makes the two evidence obligations inspectable while allowing one combined project to satisfy both |
| Binary phase scorecards with declared thresholds | Prevents elapsed time, resource completion, or an unqualified aggregate score from being mistaken for advancement evidence |

# Self-critique

| Criterion | Score / 2 |
|---|---:|
| Outcome specificity | 2 |
| Domain-type fit | 2 |
| Prerequisite mapping | 2 |
| Diagnostic placement | 1 |
| Encoding and fading | 2 |
| Retrieval layer | 2 |
| Spacing | 2 |
| Discrimination | 2 |
| Whole-task integration | 2 |
| Feedback integrity | 2 |
| Measurement | 2 |
| Sustainability | 1 |
| **Total** | **22 / 24** |

Weakest areas: the remaining JAX-debugging and authentic-mini-task diagnostic is designed but not yet performed, and exact study-window/energy constraints are not yet known. External human review is requested but cannot be guaranteed; engine, benchmark, and hardware feedback remain independent channels.
