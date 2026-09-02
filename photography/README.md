# Evidence-Adaptive Photography Curriculum

An executable, evidence-gated curriculum for learning **travel photography** through a persistent one-to-one AI tutor. It covers optics and exposure, camera fluency, one-kit field readiness, rapid orientation, ethical people photography, place/story coverage, data stewardship, raw development in darktable, and a travel-story capstone.

The curriculum is a closed-loop system:

> commit an intent → make the frames → verify against the file → diagnose the miss → apply the matching remedy → remeasure → update the learning state

It is not a shooting-prompt calendar and not a resource list. Advancement depends on independent, transfer, delayed, and reproducible evidence defined in [`photography-curriculum.md`](photography-curriculum.md). Frames made, gear owned, and hours logged are never advancement evidence.

## Quick start

Prerequisites:

- [Pi](https://github.com/earendil-works/pi) available as `pi`
- Python 3.11+ with Pillow — used by the verification harness
- `exiftool` for raw support (`sudo pacman -S perl-image-exiftool`). Optional while shooting on a phone; required once you shoot raw.
- `darktable` for Phase 4 onward — already installed here

Launch the tutor:

```bash
./pi-tutor.sh
```

Send frames for a fixed-rubric critique:

```bash
./critique.sh shoots/2026-08-30-riverside/keepers/*.jpg
```

Verify an outing against its recorded intent:

```bash
python3 tools/verify_shot.py shoots/2026-08-30-riverside/intent.toml
```

For the current phase, frontier, due checks, and next action, read [`curriculum-progress.md`](curriculum-progress.md). Current state is deliberately not copied into this README or the prompt files.

## Start here, in order

1. Read [`photography-curriculum.md`](photography-curriculum.md) — the north star and how advancement works.
2. Commit the four predictions at the top of [`phase-0-entry-diagnostic.md`](phase-0-entry-diagnostic.md), then run it. It needs no camera.
3. Complete the phone-based `T0` micro-trip, then open the `G0` travel-camera time box in [`reference/gear-decision-rubric.md`](reference/gear-decision-rubric.md) — three hours, hard stop, ending in a purchase and return-window acceptance test.

The entry diagnostic and `T0` run on a phone so the camera decision follows one real baseline. Phase 1 is blocked on buying and accepting the travel camera; ownership itself is never learning evidence.

## The three ideas doing the most work

**Shot intent is committed before the shutter.** Subject, why it is a picture, intended depth and motion rendering, exposure placement, and predicted settings, recorded before release. A frame with no recorded intent is exposure, not evidence — it can be discussed, but never culled, scored, or used to advance a node. Without a prior intent, keeper rate is meaningless and every critique becomes post-hoc rationalisation.

**Technical questions are settled by the file; perceptual ones are not.** Whether the frame matched its settings, whether depth of field covered what it needed to, whether the shutter was adequate, whether highlights clipped — all arithmetic and inspection, and `tools/verify_shot.py` decides them. Whether the picture *works* has no ground truth here, and the system is built so that the two are never confused.

**The calibration gap matters more than the score — but not every gap is worth the same.** You predict your rubric scores and keeper rate before submitting anything, and the prediction, once committed, cannot be revised. Keeper-rate calibration is the strong one: its actual comes from the file, so the gap measures your model of your own execution against ground truth. Rubric-score calibration is the weak one: it measures agreement with a known-unreliable evaluator, and closing it is consistent both with learning to see and with learning to predict the tutor. The bands, the signed tracking, and the discrimination check that keep this honest are declared in [`photography-curriculum.md`](photography-curriculum.md).

## Known weakness, stated plainly

**There is no human feedback channel in this plan.** The architecture treats at least one channel the learner does not control as non-negotiable, and this design does not fully satisfy it.

What mitigates it: technical and travel-operational dimensions have real machine or recorded ground truth, so optics, camera operation, field readiness, data stewardship, and technical development are unaffected; reference-matching supplies a fixed external target for part of the perceptual work; delayed blind re-culls and re-sequences are partially independent; and both critique rubrics are engineered against agreeable drift with fixed dimensions, numeric anchors, blind scoring, and mandatory failure identification.

What does not go away: open perceptual, ethical-judgement, and travel-story nodes are **capped at `independent`** and cannot claim `transfer`, and a capstone assessed only by this system is recorded as unvalidated on those dimensions. The trigger for adding a culturally informed human critique channel fires at the Phase 2 exit, before the first actual-trip set, before publishing identifiable or sensitive work, or sooner if rubric scores rise for three cycles while the calibration gap fails to narrow. That pattern is a drifting evaluator and cannot be repaired from inside the system.

## How the Pi tutor is assembled

Pi combines several instruction layers, each with one job:

1. Pi supplies its default agent system prompt and tools.
2. [`.pi/APPEND_SYSTEM.md`](.pi/APPEND_SYSTEM.md) adds the tutor identity, the Type 2 teaching posture, and the anti-agreeableness discipline.
3. [`AGENTS.md`](AGENTS.md) supplies the repository operating contract and routes to the canonical documents.
4. The canonical documents supply the design, graph, current evidence, and active work.
5. Frames, EXIF, harness output, and rubric scores provide the evidence used to update state.

[`pi-tutor.sh`](pi-tutor.sh) only launches this composition. It contains no second curriculum prompt.

## Responsibility allocation

The maintenance rule: edit the file that owns the meaning instead of copying the same instruction into several layers.

### Runtime and agent instructions

| Location | Owns | Does not own |
|---|---|---|
| [`.pi/APPEND_SYSTEM.md`](.pi/APPEND_SYSTEM.md) | Tutor identity, Type 2 teaching posture, critique discipline, instruction-versus-assessment stance | Session procedure, current phase, node states, curriculum gates, learner evidence |
| [`AGENTS.md`](AGENTS.md) | Repository workflow, required-context routing, session loop, verification discipline, change control, progress-record fields | Tutor personality, volatile current state, detailed curriculum content |
| [`CLAUDE.md`](CLAUDE.md) | Compatibility pointer importing `AGENTS.md` for Claude-style harnesses | Independent instructions that could diverge |
| [`pi-tutor.sh`](pi-tutor.sh), [`critique.sh`](critique.sh) | Working directory, trust flag, session IDs, launcher checks | Tutor behaviour or curriculum policy |
| [`README.md`](README.md) | Human entry point, setup, navigation, this map | Authoritative curriculum rules or live learner state |

### Curriculum sources of truth

| Location | Owns | Update when |
|---|---|---|
| [`CONTEXT.md`](CONTEXT.md) | Canonical distinctions among design stage, learning phase, node state, attempt error, edge types, and the photography-specific terms | A core term or its boundary changes |
| [`photography-curriculum.md`](photography-curriculum.md) | North star, knob settings, phase design, assessment stack, exit scorecards, error routing, operating constraints, revision triggers, design log | Outcome, scope, gates, assessment architecture, or the control law changes |
| [`photography-dependency-graph.md`](photography-dependency-graph.md) | Capability DAG, prerequisite and attention-limited edges, sequence constraints, node specifications, evidenced node states, feedback-integrity ceiling | A dependency, teaching order, capability definition, required level, or evidenced state changes |
| [`curriculum-progress.md`](curriculum-progress.md) | Active frontier, latest evidence, due checks, whole-task status, calibration, assistance, weekly hours, open commitments, next action | A substantive session or evidence review completes |
| [`phase-0-entry-diagnostic.md`](phase-0-entry-diagnostic.md) | The bounded task contract, conditions, rubric, and completion record for the entry diagnostic | That assessment is corrected or deliberately redesigned |
| [`rubrics/image-critique-rubric.md`](rubrics/image-critique-rubric.md) | The fixed critique dimensions, anchors, protocol, mandatory fields, and prohibited behaviours | A dimension, anchor, or anti-drift countermeasure changes |
| [`rubrics/travel-story-rubric.md`](rubrics/travel-story-rubric.md) | Ordered-set critique: place specificity, story question, coverage, sequence, coherence, edit discipline, ethics, and technical continuity | A travel-story dimension, anchor, critical failure, or sequence protocol changes |
| [`rubrics/reference-matching-protocol.md`](rubrics/reference-matching-protocol.md) | The integrity properties, task classes, measured tolerances, and advancement rules for reference-matching — the only mechanism lifting a perceptual node above the feedback-integrity ceiling | A task class, metric, tolerance, or the `RM-R` tooling exception changes |
| [`reference/vocabulary.md`](reference/vocabulary.md) | The capped declarative corpus and what is deliberately excluded from it | An item earns its place, or fails to be used and is deleted |
| [`reference/gear-decision-rubric.md`](reference/gear-decision-rubric.md) | The `G0` travel contract, musts, time box, purchase acceptance test, and decision record | The purchase is made, accepted/returned, or later revisited from repeated coverage evidence |
| [`reference/travel-field-checklist.md`](reference/travel-field-checklist.md) | One-kit preflight, access/safety plan, on-location orientation, and two-copy data procedure | Travel readiness or data-stewardship protocol changes |
| [`evidence-adaptive-curriculum-architecture.md`](evidence-adaptive-curriculum-architecture.md) | The general evidence base used to evaluate material redesigns | The research synthesis or general framework changes |
| [`photography-builder-prompt.md`](photography-builder-prompt.md) | Standalone bootstrap prompt for designing a photography curriculum from intake | The reusable construction process changes |

`photography-builder-prompt.md` is a design asset, not part of the live tutor startup. The established tutor resumes from `curriculum-progress.md`; it does not restart the builder's intake.

### Evidence artifacts

| Location | Owns |
|---|---|
| [`tools/verify_shot.py`](tools/verify_shot.py) | Intent-versus-EXIF verification, depth-of-field and motion arithmetic, clipping measurement, keeper rate, candidate error codes |
| [`tools/intent-template.toml`](tools/intent-template.toml) | The shot-intent schema and its field reference |
| `shoots/<date>-<place>/` | One outing: `intent.toml` plus the frames and any derived output |

## Design summary

| Setting | Value |
|---|---|
| North star | Unfamiliar destination, one carryable kit, limited time/available light: orient, cover a story ethically, execute, protect files, sequence, develop, deliver, defend |
| Domain type | Type 2 dominant; Type 3 light-reading; small Type 1 optics chain; bounded Type 4 vocabulary |
| Dose | 2–4 h/week, third and lightest track, not in peak acquisition |
| Whole task | Every week, from Phase 0. Never deferred |
| SRS | Capped at ~50 items; exceeding the cap is a revision trigger |
| Blocking | Long, before any interleaving |
| Technical feedback | Machine ground truth via the harness |
| Perceptual feedback | Fixed rubric + committed prediction + reference-matching + delayed blind re-cull |
| Self-critique score | 23/24, weakest on feedback integrity — see above |
