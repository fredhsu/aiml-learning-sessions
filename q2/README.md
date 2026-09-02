# Evidence-Adaptive Robot-Learning Curriculum

This repository is an executable, evidence-gated curriculum for developing robot-learning competence through a persistent one-to-one AI professor. It combines machine-learning and JAX foundations, control and simulation, reinforcement learning, imitation learning, experiment design, paper reproduction, and eventual SO-101 deployment where feasible.

The curriculum is a closed-loop system:

> choose a bounded performance → produce observable evidence → diagnose the miss → apply the matching remedy → remeasure → update the learning state

It is not a fixed calendar or a collection of resources. Advancement depends on independent, transfer, delayed, and reproducible evidence defined in [`robot-learning-curriculum.md`](robot-learning-curriculum.md).

## Quick start

Prerequisites:

- [Pi](https://github.com/earendil-works/pi) available as `pi`
- [`uv`](https://docs.astral.sh/uv/) for the Python environment

Set up and verify the repository:

```bash
uv sync
uv run pytest -q
```

Launch the persistent tutor:

```bash
./pi-tutor.sh
```

The launcher starts Pi from this directory, trusts the project-local resources for that run, and resumes the `robot-learning-tutor` session. Override the session ID when a separate thread is useful:

```bash
PI_TUTOR_SESSION_ID=robot-learning-experiment ./pi-tutor.sh
```

For the current learning phase, frontier, due checks, and next action, read [`curriculum-progress.md`](curriculum-progress.md). Current state is intentionally not copied into this README or the prompt files.

## How the Pi tutor is assembled

Pi combines several instruction layers, each with one job:

1. Pi supplies its default coding-agent system prompt and tools.
2. [`.pi/APPEND_SYSTEM.md`](.pi/APPEND_SYSTEM.md) adds the personal-professor identity and teaching posture without replacing Pi's defaults.
3. [`AGENTS.md`](AGENTS.md) supplies the repository operating contract and points to the canonical curriculum documents.
4. The canonical documents supply the curriculum design, graph, current evidence, and active work.
5. Session artifacts and executable checks provide the evidence used to update state.

[`pi-tutor.sh`](pi-tutor.sh) only launches this composition. It does not contain a second curriculum prompt.

## Responsibility allocation

The main maintenance rule is: edit the file that owns the meaning instead of copying the same instruction into several layers.

### Runtime and agent instructions

| Location | Owns | Does not own |
|---|---|---|
| [`.pi/APPEND_SYSTEM.md`](.pi/APPEND_SYSTEM.md) | Tutor identity, teaching relationship, intellectual standards, instructional-versus-assessment posture | Session procedure, current phase, node states, curriculum gates, learner evidence |
| [`AGENTS.md`](AGENTS.md) | Repository workflow, required-context routing, session loop, verification discipline, change control, progress-record fields | Tutor personality, volatile current state, detailed curriculum content |
| [`CLAUDE.md`](CLAUDE.md) | Compatibility pointer that imports `AGENTS.md` for Claude-style harnesses | Independent instructions that could diverge from `AGENTS.md` |
| [`pi-tutor.sh`](pi-tutor.sh) | Working directory, Pi trust flag, persistent session ID, launcher checks | Tutor behavior or curriculum policy |
| [`README.md`](README.md) | Human entry point, setup, navigation, and this responsibility map | Authoritative curriculum rules or live learner state |

### Curriculum sources of truth

| Location | Owns | Update when |
|---|---|---|
| [`CONTEXT.md`](CONTEXT.md) | Canonical distinctions among design stage, learning phase, node state, attempt error, edge types, and exit gates | A core term or its boundary changes |
| [`robot-learning-curriculum.md`](robot-learning-curriculum.md) | North-star outcome, phase design, assessment stack, scorecards, error routing, operating constraints, revision triggers, and design log | Outcome, scope, phase gates, assessment architecture, or control law changes |
| [`robot-learning-dependency-graph.md`](robot-learning-dependency-graph.md) | Capability DAG, prerequisite edges, sequence constraints, integration requirements, node specifications, and strongest node evidence | A dependency, teaching order, capability definition, required level, or evidenced node state changes |
| [`curriculum-progress.md`](curriculum-progress.md) | Active frontier, latest evidence, due checks, whole-task status, calibration, assistance, open commitments, and next action | A substantive session or evidence review completes |
| [`evidence-adaptive-curriculum-architecture.md`](evidence-adaptive-curriculum-architecture.md) | General evidence base and architecture used to evaluate material curriculum or assessment redesigns | The research synthesis or general design framework changes |
| [`phase-0-remaining-diagnostic.md`](phase-0-remaining-diagnostic.md) | The bounded task contract, conditions, rubric, and completion record for the Phase 0 entry diagnostic, completed 2026-08-27 | That assessment is corrected or deliberately redesigned |
| [`ai-curriculum-builder-prompt.md`](ai-curriculum-builder-prompt.md) | Standalone bootstrap prompt for designing a new curriculum from intake | The reusable curriculum-construction process changes |

`ai-curriculum-builder-prompt.md` is a design asset, not part of the live Pi tutor startup. The established tutor resumes from `curriculum-progress.md`; it does not restart the builder prompt's intake sequence.

### Environment and evidence artifacts

| Location | Owns |
|---|---|
| [`pyproject.toml`](pyproject.toml) and `uv.lock` | Python version, declared dependencies, and reproducible environment resolution |
| `session_*.py`, `test_session_*.py`, and `session_*_notes.md` | Dated learner implementations, executable checks, derivations, and session notes |
| [`evidence.sh`](evidence.sh) | Reproduction command for the Session 1 fixed-seed result |
| `research/` | Focused supporting investigations that inform implementation or design decisions |
| `__marimo__/` and notebook-side metadata | Tool-generated interactive notebook state rather than curriculum policy |

## Where a change belongs

| Intended change | Primary location | Coordinated updates |
|---|---|---|
| Change the professor's voice or teaching relationship | `.pi/APPEND_SYSTEM.md` | None unless the operating procedure also changes |
| Change how every repository session is run or recorded | `AGENTS.md` | Update the progress schema or curriculum only if their contracts change |
| Change terminology | `CONTEXT.md` | Update every canonical file that uses the changed term |
| Change the outcome, phase structure, assessment gates, or revision triggers | `robot-learning-curriculum.md` | Update the dependency graph and curriculum design log together; update progress if current work is affected |
| Add, remove, or retype a dependency or teaching-order edge | `robot-learning-dependency-graph.md` | Update curriculum gates and design log when the change is material |
| Record performance or advance a node state | `curriculum-progress.md` and the graph | Link the artifact and verification; record assistance and attempt errors |
| Change a bounded diagnostic task | Its diagnostic Markdown file | Update the curriculum, graph, and design log if the change alters exit evidence or learning architecture |
| Change dependencies or commands | `pyproject.toml` and `uv.lock` | Update this README only when the human setup workflow changes |

Material curriculum changes follow the change-control procedure in [`AGENTS.md`](AGENTS.md): identify affected prerequisite edges, sequence constraints, integration requirements, node states, and exit evidence; explain the consequence; obtain confirmation; then update the curriculum, graph, and design log consistently.

## Evidence and session artifacts

A curriculum session is complete only when it leaves an artifact or evidence record. Depending on the task, acceptable evidence includes:

- executable unit or property tests;
- numerical parity with an independent reference;
- fixed-seed experiment output;
- a committed derivation or diagnosis;
- a benchmark, simulator, or physical-system observation;
- a reproducible command and experiment record.

Run the current automated checks with:

```bash
uv run pytest -q
```

Reproduce the Session 1 training result with:

```bash
bash evidence.sh
```

Passing output is evidence for the bounded task only. Node advancement also depends on assistance level, task conditions, transfer, delay, and any critical failures named by the applicable scorecard.

## Working conventions

- Begin with the active frontier in `curriculum-progress.md`; do not restart intake.
- Use `CONTEXT.md` vocabulary when describing state, errors, or graph relationships.
- Keep live state out of `.pi/APPEND_SYSTEM.md`, `AGENTS.md`, and this README.
- Preserve learner attempts before adding tutor-supplied tests or completed solutions when independence is being assessed.
- Declare task-dependent thresholds before running results.
- Record actual evidence; never infer performance from resource completion or discussion alone.
- Treat unrelated working-tree changes as learner-owned and leave them untouched.

## Repository map

```text
.
├── .pi/APPEND_SYSTEM.md                 # personal-professor persona
├── AGENTS.md                            # repository operating contract
├── CONTEXT.md                           # canonical vocabulary
├── robot-learning-curriculum.md         # outcome, phases, gates, control law
├── robot-learning-dependency-graph.md   # DAG, edge types, node evidence
├── curriculum-progress.md               # live frontier and evidence history
├── phase-0-remaining-diagnostic.md      # Phase 0 entry diagnostic (complete)
├── evidence-adaptive-curriculum-architecture.md
├── ai-curriculum-builder-prompt.md      # standalone design bootstrap
├── pi-tutor.sh                          # Pi launcher
├── pyproject.toml / uv.lock             # Python environment
└── session and test artifacts           # executable learning evidence
```
