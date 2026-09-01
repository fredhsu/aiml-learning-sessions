# Evidence-Adaptive Inference-Engineering Curriculum

This repository is an executable, evidence-gated curriculum for developing **LLM inference-engineering** competence through a persistent one-to-one AI professor. It covers transformer inference arithmetic, measurement discipline, KV-cache and scheduler design, GPU performance diagnosis, quantisation, speculative decoding, and single-GPU serving — with a from-scratch inference engine built along the way as the mechanism-encoding route.

The curriculum is a closed-loop system:

> **predict a number → produce evidence under a declared contract → diagnose the gap → apply the matching remedy → remeasure → update the learning state**

It is not a fixed calendar or a collection of resources. Advancement depends on independent, transfer, delayed, and reproducible evidence defined in [`inference-curriculum.md`](inference-curriculum.md).

**Track role:** secondary, ~2–3 h/week. The primary track is the robot-learning curriculum in [`../q2`](../q2). This repository reuses that curriculum's architecture and vocabulary, with knob settings deliberately re-derived for a different domain type.

## The two rules that define this repository

Everything else is machinery around these.

1. **Predict before you measure.** Every benchmark, profile, and optimisation is preceded by a committed quantitative prediction with a tolerance. An unexplained prediction gap is the most valuable object in the curriculum, not an embarrassment.
2. **No number without its contract.** A result whose workload contract is unknown is not weak evidence; it is not evidence. See [`bench/workload-contract.md`](bench/workload-contract.md).

## Quick start

Prerequisites:

- [Pi](https://github.com/earendil-works/pi) available as `pi`
- [`uv`](https://docs.astral.sh/uv/) for the Python environment
- An NVIDIA GPU. The primary bench is an RTX 4090; the DGX Spark GB10 is deliberately reserved as the Phase 5 transfer surface.

Set up the environment (**not yet run — this is the first session's work**, and it pulls a large PyTorch download):

```bash
uv sync
uv run pytest -q
```

Launch the persistent tutor:

```bash
./pi-tutor.sh
```

The launcher starts Pi from this directory, trusts the project-local resources for that run, and resumes the `inference-engineering-tutor` session. Override the session ID when a separate thread is useful:

```bash
PI_TUTOR_SESSION_ID=inference-experiment ./pi-tutor.sh
```

**Start here:** [`phase-0-entry-diagnostic.md`](phase-0-entry-diagnostic.md). Nothing in the dependency graph is evidenced until it runs, and the plan should not be trusted to have located the frontier correctly before then.

For the current learning phase, frontier, due checks, prediction ledger, and next action, read [`curriculum-progress.md`](curriculum-progress.md). Current state is intentionally not copied into this README or the prompt files.

## How the Pi tutor is assembled

Pi combines several instruction layers, each with one job:

1. Pi supplies its default coding-agent system prompt and tools.
2. [`.pi/APPEND_SYSTEM.md`](.pi/APPEND_SYSTEM.md) adds the professor identity, the prediction rule, and the measurement-integrity posture without replacing Pi's defaults.
3. [`AGENTS.md`](AGENTS.md) supplies the repository operating contract and points to the canonical curriculum documents.
4. The canonical documents supply the curriculum design, graph, current evidence, and active work.
5. Session artifacts, parity tests, contracts, and measurements provide the evidence used to update state.

[`pi-tutor.sh`](pi-tutor.sh) only launches this composition. It does not contain a second curriculum prompt.

`CLAUDE.md` imports `AGENTS.md`, so Claude-style harnesses operate under the same contract.

## Responsibility allocation

The main maintenance rule is: edit the file that owns the meaning instead of copying the same instruction into several layers.

### Runtime and agent instructions

| Location | Owns | Does not own |
|---|---|---|
| [`.pi/APPEND_SYSTEM.md`](.pi/APPEND_SYSTEM.md) | Tutor identity, teaching relationship, prediction rule, measurement-integrity posture, instructional-versus-assessment stance | Session procedure, current phase, node states, curriculum gates, learner evidence |
| [`AGENTS.md`](AGENTS.md) | Repository workflow, required-context routing, session loop, measurement discipline, change control, progress-record fields | Tutor personality, volatile current state, detailed curriculum content |
| [`CLAUDE.md`](CLAUDE.md) | Compatibility pointer that imports `AGENTS.md` | Independent instructions that could diverge from `AGENTS.md` |
| [`pi-tutor.sh`](pi-tutor.sh) | Working directory, Pi trust flag, persistent session ID, launcher checks | Tutor behaviour or curriculum policy |
| [`README.md`](README.md) | Human entry point, setup, navigation, and this responsibility map | Authoritative curriculum rules or live learner state |

### Curriculum sources of truth

| Location | Owns | Update when |
|---|---|---|
| [`CONTEXT.md`](CONTEXT.md) | Canonical distinctions among design stage, learning phase, node state, attempt error, edge types, exit gates — plus the measurement vocabulary (contract, prediction, speedup claim, bottleneck class, parity) | A core term or its boundary changes |
| [`inference-curriculum.md`](inference-curriculum.md) | North-star outcome, domain typing and knob settings, phase design, fluency set, confusable families, misconception bank, assessment stack, scorecards, error and bottleneck routing, revision triggers, design log | Outcome, scope, phase gates, assessment architecture, or control law changes |
| [`inference-dependency-graph.md`](inference-dependency-graph.md) | Capability DAG, prerequisite edges, sequence constraints, integration requirements, node specifications, recognition-level leaves, learner leverage and blind spots, strongest node evidence | A dependency, teaching order, capability definition, required level, or evidenced node state changes |
| [`resources.md`](resources.md) | Node-indexed encoding material with an attached output for each, and the exclusions | A resource is added, retired, or reassigned to a node |
| [`phase-0-entry-diagnostic.md`](phase-0-entry-diagnostic.md) | The bounded entry assessment: conditions, items, scoring, routing, and expected-shape statement | That assessment is corrected or deliberately redesigned |
| [`bench/workload-contract.md`](bench/workload-contract.md) | Contract template, contract rules, and the locked workload definitions | A workload is declared, versioned, or superseded |
| [`curriculum-progress.md`](curriculum-progress.md) | Active frontier, latest evidence, prediction ledger, due checks, whole-task status, calibration, assistance, open commitments, next action | A substantive session or evidence review completes |
| [`evidence-adaptive-curriculum-architecture.md`](evidence-adaptive-curriculum-architecture.md) | General evidence base and architecture used to evaluate material curriculum or assessment redesigns | The research synthesis or general design framework changes |
| [`ai-curriculum-builder-prompt.md`](ai-curriculum-builder-prompt.md) | Standalone bootstrap prompt for designing a curriculum from intake | The reusable curriculum-construction process changes |

`ai-curriculum-builder-prompt.md` is a design asset, not part of the live Pi tutor startup. The established tutor resumes from `curriculum-progress.md`; it does not restart the builder prompt's intake sequence.

### Environment and evidence artifacts

| Location | Owns |
|---|---|
| [`pyproject.toml`](pyproject.toml) and `uv.lock` | Python version, declared dependencies, and reproducible environment resolution. Resolved versions belong in every workload contract. |
| `bench/` | Workload contracts and measurement records. Raw traces and profile dumps are gitignored; contracts and summaries are tracked. |
| `engine/`, `session_*.py`, `test_*.py` | Dated learner implementations, parity tests, and executable checks |
| `*_notes.md`, `*_record.md` | Derivations, prediction commitments, experiment records, and session notes |

## Where a change belongs

| Intended change | Primary location | Coordinated updates |
|---|---|---|
| Change the professor's voice, prediction rule, or measurement posture | `.pi/APPEND_SYSTEM.md` | None unless the operating procedure also changes |
| Change how every session is run or recorded | `AGENTS.md` | Update the progress schema or curriculum only if their contracts change |
| Change terminology | `CONTEXT.md` | Update every canonical file that uses the changed term |
| Change the outcome, phase structure, assessment gates, or revision triggers | `inference-curriculum.md` | Update the dependency graph and design log together; update progress if current work is affected |
| Add, remove, or retype a dependency or teaching-order edge | `inference-dependency-graph.md` | Update curriculum gates and design log when the change is material |
| Add or retire a learning resource | `resources.md` | Name the node and the attached output; a resource without an output does not belong |
| Declare or version a benchmark workload | `bench/workload-contract.md` | Supersede rather than edit any contract with results attached |
| Record performance or advance a node state | `curriculum-progress.md` and the graph | Link the artifact, contract version, and verification; record assistance and attempt errors |
| Change the entry diagnostic | `phase-0-entry-diagnostic.md` | Update the curriculum, graph, and design log if it alters exit evidence or learning architecture |

Material curriculum changes follow the change-control procedure in [`AGENTS.md`](AGENTS.md): identify affected prerequisite edges, sequence constraints, integration requirements, node states, and exit evidence; explain the consequence; obtain confirmation; then update the curriculum, graph, and design log consistently.

## Evidence and session artifacts

A curriculum session is complete only when it leaves an artifact or evidence record. Depending on the task, acceptable evidence includes:

- a committed prediction with its tolerance and basis, recorded before the run;
- numerical parity with an independent reference at a declared tolerance;
- a fixed-seed measurement under a named workload contract version;
- a profiler trace with the dominant kernel and gap structure identified;
- an accuracy evaluation accompanying any quantisation result;
- a committed diagnosis of a seeded fault, made before execution;
- a reproducible command and experiment record.

Passing output is evidence for the bounded task only. Node advancement also depends on assistance level, task conditions, transfer, delay, and any critical failures named by the applicable scorecard.

## Working conventions

- Begin with the active frontier in `curriculum-progress.md`; do not restart intake.
- Use `CONTEXT.md` vocabulary when describing state, errors, graph relationships, or measurements.
- Keep live state out of `.pi/APPEND_SYSTEM.md`, `AGENTS.md`, and this README.
- Commit the prediction before the measurement, always, in writing.
- Preserve learner attempts before adding tutor-supplied tests or completed solutions when independence is being assessed.
- Declare thresholds and tolerances in the contract before results are run.
- Record actual evidence; never infer performance from resource completion or discussion alone.
- A gap of a week or two is spacing, not neglect. After a gap over three weeks, re-diagnose rather than resume.
- Treat unrelated working-tree changes as learner-owned and leave them untouched.

## Repository map

```text
.
├── .pi/APPEND_SYSTEM.md                 # professor persona, prediction rule, measurement posture
├── AGENTS.md                            # repository operating contract
├── CLAUDE.md                            # compatibility pointer to AGENTS.md
├── CONTEXT.md                           # canonical vocabulary, incl. measurement terms
├── inference-curriculum.md              # outcome, phases, gates, control law, design log
├── inference-dependency-graph.md        # DAG, edge types, node evidence, blind spots
├── resources.md                         # node-indexed videos, blogs, papers, with attached outputs
├── phase-0-entry-diagnostic.md          # the pending entry assessment — start here
├── curriculum-progress.md               # live frontier, prediction ledger, evidence history
├── bench/workload-contract.md           # contract template, rules, locked workloads
├── evidence-adaptive-curriculum-architecture.md
├── ai-curriculum-builder-prompt.md      # standalone design bootstrap
├── pi-tutor.sh                          # Pi launcher
└── pyproject.toml                       # Python environment (not yet synced)
```
