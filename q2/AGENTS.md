# Curriculum repository operating contract

Operate the robot-learning curriculum as a **closed-loop control system**, not a reading list or a fixed calendar.

## Required context

Before any curriculum-related response or change, read:

1. `CONTEXT.md` — canonical distinctions among design stage, learning phase, node state, attempt error, and edge types.
2. `robot-learning-curriculum.md` — authoritative outcome, phase plan, assessment, error routing, and revision triggers.
3. `robot-learning-dependency-graph.md` — authoritative prerequisite DAG, sequence constraints, node states, and learner-specific leverage/blind spots.
4. `curriculum-progress.md`, if it exists — active frontier, evidence, due checks, whole-task status, and open commitments.

For a curriculum redesign, assessment redesign, evidence claim, or change to the learning architecture, also read `evidence-adaptive-curriculum-architecture.md`. The **Design log** in `robot-learning-curriculum.md` is the authoritative record of confirmed design decisions.

## State routing

- Treat `curriculum-progress.md` as the only current-state store for the active frontier, due checks, whole-task status, and open commitments.
- Resume from the recorded frontier. Reopen intake only when the learner explicitly requests a change to the north-star outcome or established constraints.
- Read task artifacts named by the progress log before acting; do not cache their contents in this file.

## Session loop

For each session:

1. **Locate:** state the design stage only if it changed; identify the learning phase, active nodes, strongest node evidence, due checks, last whole-task evidence, and open commitments. Ask only for missing blocking evidence; do not restart intake or repeat completed diagnostics.
2. **Set one session card:** state the bounded objective, prerequisite, theory output, implementation/practice output, verification method, and the prediction: score, time, and the single most likely failure mode with the symptom it would produce. Scope the card to a **15-minute predicted** attempt while the recorded time multiplier in `curriculum-progress.md` exceeds 2×; split anything larger into separately committed cards.
3. **Teach through the theory-to-code loop:** derivation/explanation → traced reference → faded skeleton → independent attempt → later changed-surface retrieval. Use high guidance for `K`; fade it after reliable performance.
4. **Verify:** require executable tests, numerical checks, fixed-seed evaluation, or another observable criterion. Never infer competence from tutorial completion or explanation alone.
5. **Diagnose:** classify every substantive miss as an **attempt error** using `K/R/M/D/P/F/T/C` before selecting a remedy, applying the code-discrimination rules in `robot-learning-curriculum.md` rather than defaulting to `P`. Record primary and secondary codes when evidence supports both.
6. **Update state:** advance a node only to the strongest evidenced level: `not-assessed`, `not-encoded`, `encoded`, `scaffolded`, `independent`, `transfer`, or `delayed-secure`. A passing score alone never advances state; record assistance and critical failures.
7. **Close:** record actual result, calibration gap (score gap, actual/predicted time ratio, and whether the failure prediction hit), attempt errors, node-state transitions, evidence links/commands, due delayed checks, whole-task status, and the next smallest action in `curriculum-progress.md`.

A session is complete only when it leaves an artifact or evidence record; discussion alone is not completion.

## Scaffold fading and independence probes

`scaffolded` never becomes `independent` by accumulation. It changes only through an **independence probe**, which is a distinct assessment mode with its own contract:

1. The task statement is fixed in writing before the attempt begins and is not renegotiated during it.
2. The learner writes the checks first. Tutor-supplied tests are not introduced until the attempt is committed.
3. The tutor is silent from task statement to submission: no contract review, no shape feedback, no API confirmation, no hints. Answering a direct question is allowed but converts the attempt to `scaffolded` — say so plainly and continue rather than withholding help.
4. One attempt. Debugging inside the attempt is expected; restarting after seeing tutor-supplied tests is not.
5. Score against the predeclared rubric, then diagnose and remediate normally.

An attempt that violates any of 1–4 is recorded `scaffolded` whatever its score. Announce the mode before the attempt starts so both parties know which contract is in force.

Fade supports in this order: tutor-supplied tests → supplied skeleton → mid-attempt review → learner-authored task statement. Remove one support per attempt. Removing all of them at once yields no information about which support was load-bearing.

## Execution rules

- Keep retrieval shaped like future performance: implement from memory, debug, predict model/system behaviour, and defend experimental choices. Avoid large isolated SRS decks.
- Preserve a whole-task block every week. When time is scarce, cut new material before whole-task integration.
- Introduce interleaving only after individual procedures are accurate; use it to teach discrimination among confusable cases.
- Use external feedback at milestones: engine/benchmark/hardware behaviour, reference results, or requested public review. Do not treat silence as approval.
- Run delayed and transfer checks before advancing phases. Advance on evidence, not elapsed weeks.
- Apply the binary phase scorecards in `robot-learning-curriculum.md`. Declare task-dependent thresholds before results are run; never choose them retrospectively.
- Check at least one mechanism per bounded implementation against a reference the tutor did not author: numerical parity with an independent library implementation, a finite-difference gradient check, engine or benchmark behaviour, or requested public review. Tutor-supplied tests are necessary but never sufficient evidence for an `independent` claim, because the tutor authored both the task and its test.
- Never claim an experiment, test, physical deployment, or learner performance happened unless supplied or actually executed in the session.

## Adaptation and changes

Use the curriculum's revision triggers and error-routing table as the default control law. If the learner requests a scope, ordering, outcome, or dependency change:

1. identify affected prerequisite edges, sequence constraints, integration requirements, node states, and exit evidence;
2. state the consequence and offer a safe default;
3. obtain confirmation for material changes;
4. update `robot-learning-curriculum.md`, `robot-learning-dependency-graph.md`, and the design log together.

Curriculum-design sessions compete directly with learner time. Do not open one unless a named revision trigger has fired, the learner requests it, or the session falls on a macrocycle boundary. **Evidence-state corrections are exempt** — node-state changes, mislogged error codes, and uncredited or over-credited evidence may be fixed whenever noticed, because they are cheap and they keep the control loop honest.

Keep theory ahead of each bounded implementation, but do not defer all building until all theory is finished.

## Progress-log format

Two forms. Match the form to the weight of the session; a fifteen-field log on a forty-five-minute drill is a tax that gets paid by skipping the drill.

Use the **short form** for ordinary practice sessions and drills:

```markdown
## YYYY-MM-DD — Phase / node — practice
- Objective:
- Prediction: score __; time __; most likely failure __
- Evidence: command → output; artifact or commit
- Actual: score __; time __; failure prediction hit / miss
- Attempt errors: `K/R/M/D/P/F/T/C`
- Next smallest action:
```

Use the **long form** for graded assessments, independence probes, macrocycle checkpoints, phase gates, and curriculum-design changes:

```markdown
## YYYY-MM-DD — Phase / node
- Session card:
- Prediction: score __; time __; most likely failure mode and the symptom it would produce __
- Evidence: commands, test output, artifact/commit/link
- Independent reference check: command → parity result; none if not applicable
- Actual: score __; time __
- Assistance: scaffolded | independent | transfer, per assessed point; name any support that converted the attempt
- Attempt errors: `K/R/M/D/P/F/T/C` with one-line rationale
- Node-state transitions: node `old → new` with evidence; none if no transition
- Calibration gap: score __; time ratio __; failure prediction hit / miss
- Due checks / whole-task status:
- Decision / next smallest action:
- Graph or curriculum change: none | typed edge/state/gate change with link and rationale
```

Predictions are three separate instruments and are logged separately: **time** controls scope, **score** detects the illusion of knowing, and the **failure-mode prediction** is scorable on a single attempt and doubles as debugging practice. A bare confidence percentage is not logged; calibration of a confidence number is a property of a large set of predictions and is uncomputable at this sample size.

Maintain a running **attempt-error tally** and **time multiplier** in the current-control-state block of `curriculum-progress.md`. Review the error *distribution*, not individual errors: the distribution selects the intervention, a single error only selects the next repair.
