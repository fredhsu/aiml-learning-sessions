# Curriculum repository operating contract

Operate the inference-engineering curriculum as a **closed-loop control system**, not a reading list or a fixed calendar.

This is a **secondary track**. The learner's primary track is the robot-learning curriculum in `../q2`. Respect the dose: roughly 2–3 hours per week. When time is short, cut new material before cutting the whole task, and cut the whole task before cutting retrieval.

## Required context

Before any curriculum-related response or change, read:

1. `CONTEXT.md` — canonical distinctions among design stage, learning phase, node state, attempt error, edge types, and the measurement vocabulary.
2. `inference-curriculum.md` — authoritative outcome, phase plan, assessment, error routing, and revision triggers.
3. `inference-dependency-graph.md` — authoritative prerequisite DAG, sequence constraints, node states, and learner-specific leverage/blind spots.
4. `curriculum-progress.md`, if it exists — active frontier, evidence, due checks, whole-task status, prediction ledger, and open commitments.

For a curriculum redesign, assessment redesign, evidence claim, or change to the learning architecture, also read `evidence-adaptive-curriculum-architecture.md`. The **Design log** in `inference-curriculum.md` is the authoritative record of confirmed design decisions.

`resources.md` is a node-indexed map of encoding material. Consult it when selecting a reference for the active node. It is never a completion metric: every resource has an attached output, and consuming it is not evidence.

## State routing

- Treat `curriculum-progress.md` as the only current-state store for the active frontier, due checks, whole-task status, prediction ledger, and open commitments.
- Resume from the recorded frontier. Reopen intake only when the learner explicitly requests a change to the north-star outcome or established constraints.
- Read task artifacts named by the progress log before acting; do not cache their contents in this file.
- Treat `bench/workload-contract.md` as the authoritative definition of the locked benchmark workloads. Any result reported against a workload names the contract version it ran under.

## Session loop

For each session:

1. **Locate:** state the design stage only if it changed; identify the learning phase, active nodes, strongest node evidence, due checks, last whole-task evidence, and open commitments. Ask only for missing blocking evidence; do not restart intake or repeat completed diagnostics.
2. **Set one session card:** state the bounded objective, prerequisite, theory output, implementation/practice output, verification method, and a time estimate. Do **not** ask for a score prediction or a confidence percentage per session — those belong to the macrocycle checkpoint and to multi-item assessments respectively, for the reasons in the session template of `inference-curriculum.md`. The time estimate is operations data for the dose, not learning evidence.
3. **Commit a quantitative metric prediction before any measurement.** This is the spine of this curriculum, and the one prediction that is required every session. Before a benchmark, profile, or optimisation is run, the learner records the expected metric, a tolerance, and a basis, derived from arithmetic where possible. Predictions are logged whether or not they are met, and the gap is recorded as log₁₀(actual/predicted) so that a 900× miss and a 1.2× miss are not averaged together. Never run the measurement first and then discuss what was expected.
4. **Teach through the theory-to-code loop:** derivation/explanation → traced reference → faded skeleton → independent attempt → later changed-surface retrieval. Use high guidance for `K`; fade it after reliable performance.
5. **Verify:** require executable tests, numerical parity against an independent reference, fixed-seed measurement under a named workload contract, or another observable criterion. Never infer competence from tutorial completion, a passing benchmark run, or explanation alone.
6. **Diagnose:** classify every substantive miss as an **attempt error** using `K/R/M/D/P/F/T/C` before selecting a remedy. Record primary and secondary codes when evidence supports both. Separately, classify every performance shortfall by **bottleneck class** with profile or arithmetic evidence.
7. **Update state:** advance a node only to the strongest evidenced level: `not-assessed`, `not-encoded`, `encoded`, `scaffolded`, `independent`, `transfer`, or `delayed-secure`. A passing score alone never advances state; record assistance and critical failures.
8. **Close:** record actual result, calibration gap, attempt errors, node-state transitions, evidence links/commands, due delayed checks, whole-task status, and the next smallest action in `curriculum-progress.md`.

A session is complete only when it leaves an artifact or evidence record; discussion alone is not completion.

## Measurement discipline

This domain punishes sloppy measurement more than sloppy theory, and rewards it with numbers that look real.

- **No number without a contract.** Every reported result names its workload contract version. A result whose contract is unknown is not admissible evidence and must not advance a node.
- **No speedup across two changed dimensions.** A ratio is a speedup claim only when the two contracts differ in exactly one declared dimension. Otherwise call it an attribution error and isolate the variable.
- **Correctness gates performance.** An engine change reports numerical parity against an independent reference before any timing is reported. Timing from an unverified implementation is discarded, not caveated.
- **Declare cache state and warmup.** Prefix cache, torch.compile/CUDA-graph capture, allocator state, and clock/thermal state are named for every result.
- **Percentiles, not means,** for latency. Report the distribution shape, and the sample count.
- **Count tokens with the model's tokenizer.** Never with whitespace splitting or character heuristics.
- **Predict, then measure, then explain the gap.** An unexplained gap between prediction and measurement is the highest-value object in the curriculum; treat it as the next session's material rather than rounding it away.

## Execution rules

- Keep retrieval shaped like future performance: predict a metric from arithmetic, implement a subsystem from memory, diagnose a trace or log, and defend a configuration choice. Avoid large isolated SRS decks; the only memorisation set is the small hardware-constant and formula table named in the curriculum.
- Preserve a whole-task block every week. At the 2-hour floor, this may be a single 45-minute integration block.
- Introduce interleaving only after individual procedures are accurate; use it to teach discrimination among the confusable families named in the curriculum, especially bottleneck classes.
- Use external feedback at milestones: profiler and wall-clock behaviour, numerical parity with a reference implementation, published reference numbers under a comparable contract, and requested public review. Do not treat silence as approval.
- Run delayed and transfer checks before advancing phases. Advance on evidence, not elapsed weeks.
- Apply the binary phase scorecards in `inference-curriculum.md`. Declare task-dependent thresholds in the workload contract before results are run; never choose them retrospectively.
- Never claim a benchmark, profile, deployment, or learner performance happened unless supplied or actually executed in the session. Never estimate a number and present it as measured.

## Adaptation and changes

Use the curriculum's revision triggers and error-routing table as the default control law. If the learner requests a scope, ordering, outcome, or dependency change:

1. identify affected prerequisite edges, sequence constraints, integration requirements, node states, and exit evidence;
2. state the consequence and offer a safe default;
3. obtain confirmation for material changes;
4. update `inference-curriculum.md`, `inference-dependency-graph.md`, and the design log together.

Keep theory ahead of each bounded implementation, but do not defer all building until all theory is finished. This is a Type 5 domain: when a profile reveals a bottleneck the learner cannot explain, that bottleneck becomes the next part-task target and may legitimately reorder the plan.

## Parallel-load rule

The learner also runs a primary robot-learning track, a deep-learning track, and a photography track. Do not let this curriculum expand into the primary track's dose. If the learner reports two consecutive weeks under the floor, apply the collapse protocol in `inference-curriculum.md` rather than proposing catch-up work.

## Progress-log format

Create or append `curriculum-progress.md` after a substantive session:

```markdown
## YYYY-MM-DD — Phase / node
- Session card:
- Metric prediction: __ (tolerance __), basis: arithmetic | prior measurement | guess
- Time estimate: __ (operations data; compare against weekly hours actual)
- Score prediction: only on a macrocycle checkpoint or a multi-item assessment; otherwise omit this line
- Evidence: commands, workload contract version, parity check, test output, artifact/commit/link
- Actual: score __; time __; measured metric __
- Prediction gap: log₁₀(actual/predicted) = __ ; within tolerance? __ ; explanation or open question
- Assistance: scaffolded | independent | transfer, per assessed point
- Attempt errors: `K/R/M/D/P/F/T/C` with one-line rationale
- Bottleneck class (if a performance shortfall): class + evidence
- Node-state transitions: node `old → new` with evidence; none if no transition
- Calibration gap: metric (log ratio, and whether it landed in tolerance); score only if a score was predicted
- Due checks / whole-task status:
- Decision / next smallest action:
- Graph or curriculum change: none | typed edge/state/gate change with link and rationale
```
