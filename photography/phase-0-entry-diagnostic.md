# Phase 0 entry diagnostic

**Status:** in progress — Task 1 scored 2026-08-31; Tasks 2–5 outstanding
**Status values:** `designed, not yet performed` → `in progress` → `performed`, the last set only when the completion record below is filled. The diagnostic spans at least two days by design, so `in progress` is a real state and not a formality.
**Purpose:** locate the actual frontier empirically before teaching; set every sampled node from evidence and explicitly mark every unsampled node with its reason.
**Runs on:** a phone. Camera purchase is a confirmed Phase 0 outcome, but the diagnostic happens first so `G0` is informed by one real unfamiliar-place baseline rather than by specifications alone.
**Total time:** about 100 minutes, and it should be split across at least two days.

This is an assessment, not a lesson. The tutor withholds explanation, hints, and confirmation until each task is committed. If a task is genuinely unfamiliar, the correct action is to say so and record `K` — that is a *result*, not a failure, and guessing wastes the measurement.

## Before starting

Commit these predictions in writing. They are the first calibration measurement and cannot be revised afterwards.

| Prediction | Your commitment |
|---|---|
| Total score, out of 14 | |
| Which task you will score worst on | |
| Your dominant error code across the diagnostic | |

Session duration is deliberately not predicted — it routes no remedy. The worst-task prediction is the discrimination instrument here: getting the total right while naming the wrong weakest task is a real miss, and one the total alone would hide.

## Task 1 — Optics and exposure, closed-resource (4 points)

No dedicated camera, no reference, no calculator. Written answers, committed before any checking.

| # | Item | Tests | Point |
|---|---|---|---|
| 1.1 | A frame is correctly exposed at f/8, 1/250, ISO 200. You need three stops more depth of field and cannot change ISO. State the new aperture and shutter. | `O1`, `O2` | 1 for both correct |
| 1.2 | You are photographing a person 3 m away at 50 mm, f/2. Roughly how much will be sharp — centimetres, tens of centimetres, or metres? Then: what happens to that if you step back to 6 m and keep everything else fixed? | `O4` | 1 for the right order of magnitude *and* the right direction of change |
| 1.3 | Two frames of the same runner are both blurred. In one the background is sharp; in the other the whole frame is smeared. Name the cause of each and the control that fixes each. | `O5`, `G3` | 1 for both causes and both controls |
| 1.4 | You stand in one spot and swap a 24 mm for an 85 mm. What changes about the perspective — the relationship between near and far objects? Answer precisely. | `O7` | 1 only if the answer is that perspective does **not** change, because it is set by position |

**What failures mean.** A miss on 1.1 is `K` and blocks nearly everything. A confident wrong answer on 1.4 is `M` — the most common misconception in the domain — and needs a contrast pair, not an explanation. A miss on 1.2 or 1.3 with a correct *direction* but wrong magnitude is `K` on the relation, not `M`.

## Task 2 — Perceptual baseline: reading other people's frames (3 points)

### Sourcing the frames

The tutor cannot supply photographs — it has none to give. The frames must be sourced before the task, and *how* they are sourced determines whether the task measures anything.

The requirement is that you have **not analysed** them, which is weaker than not having seen them and is the strongest thing achievable without a second person. Procedure:

1. Bulk-collect a pool **much larger than six** from a public archive, without studying it — no browsing for interesting frames, no selecting for light condition.
2. Drop the pool into a directory and have the tutor choose six spanning at least four light conditions, revealing them **one at a time** during the assessment.
3. Record the pool source, its size, and the selection method in the completion record.

If a second person is available to pick the six, use them instead; that is strictly better and removes the weakness below.

**Residual weakness, stated rather than hidden:** you assembled the pool, so you have seen thumbnails of the frames you will be scored on. This inflates Task 2 by an unknown amount. It does not invalidate the task — light direction, hardness, and camera position are not things a glance at a thumbnail teaches you — but a strong Task 2 score obtained this way is weaker evidence than the same score from a set chosen by someone else. Note it beside the score.

The frames used for this run are in [`diagnostic-task-2/`](diagnostic-task-2/).

### The task

For each of the six, commit before any checking:

- the light condition — direction, hardness, rough contrast ratio, colour
- roughly where the camera was, and roughly what focal length range
- what the picture is about, in one clause
- what role it could play in a travel story: orientation, context, human, detail, transition, closure, or standalone

| Point | Awarded for |
|---|---|
| 1 | Light direction and hardness correct on at least 4 of 6 |
| 1 | Camera position described in terms of height and distance, not just "in front", on at least 4 of 6 |
| 1 | Subject clause is a single clause and the proposed story role is defensible on at least 4 of 6 |

**What failures mean.** This samples `L1`, `L2`, `V1`, and `V2` cheaply and without equipment. It is the highest-information item per minute in the whole diagnostic. Failure here with success on Task 1 is the expected pattern for a technically-minded beginner and sets the phase weighting.

## Task 3 — Authentic mini-task: the baseline shoot (4 points)

**This is the item most likely to be skipped, and it is the one that matters most.** Everything above measures talk; this measures performance.

Run a **local micro-trip**: go somewhere nearby that you have not photographed, set one story question, spend **45 minutes**, and make **at least 20 frames** on a phone. Treat it as travel: bound scouting time, note access/safety constraints, and work toward a small sequence rather than isolated keepers.

Rules:
- Commit a `[[frame]]` block from `tools/intent-template.toml` before each release or valid tight group. A group covers one subject/relationship, light, position/strategy, and stop condition; start a new commitment when any changes. A timestamped voice memo or pocket card may be transcribed later without alteration. Frames without recorded intent do not count toward the 20.
- Phone cameras will not honour most predicted settings; predict anyway and record what you *wanted*. The mismatch is itself the measurement.
- No editing, no deleting in the field. Delete nothing until the delayed cull.
- Commit a predicted keeper rate before you start.
- Record `coverage_role` and `ethical_access` for every frame or tight group. Record at least one deliberate non-capture for a visual, safety, access, or ethical reason.

| Point | Awarded for |
|---|---|
| 1 | 20+ frames with intent recorded before release |
| 1 | `python3 tools/verify_shot.py` runs clean over the set and produces per-frame verdicts |
| 1 | Delayed cull performed **at least 24 hours later**, selecting against recorded intent |
| 1 | A delayed 5–7 frame ordered micro-trip edit with at least four non-redundant coverage roles, scored with the travel-story rubric against committed predictions |

**What failures mean.** Failing the intent-recording point is `C` or an operations failure and must be repaired before Phase 1, because every later measurement depends on it. A pile of individually plausible frames that cannot form a short sequence is evidence at the `TR3`/`TR6` frontier, not a reason to buy a wider lens.

## Task 4 — Diagnosis from the file (2 points)

The original form of this task asked the tutor for "four frames, each with one dominant technical fault". It cannot supply them, and anyone who sources frames *selected for a known fault* also knows the answers — which destroys the measurement. The fix is to let **the file settle the fault**, exactly as it does everywhere else in this curriculum.

### Sourcing

Four frames from your own camera roll, **predating this curriculum**, with EXIF intact, chosen by a rule that cannot select for a fault: take every *n*th frame across a date range, or the first frame of each of four different months. Do not browse for bad pictures — a frame you picked because it looks soft is a frame you have already diagnosed.

Because the sample is unselected, it may not contain one of each fault, and may contain frames with no dominant fault at all. That is a valid result and the scoring below accounts for it.

### The task

For each frame, commit a diagnosis from the **image alone** — dominant fault, or "no dominant fault" — before opening EXIF, before inspecting at 100%, and before measuring clipping. Then check against the file:

| Candidate fault | Settled by |
|---|---|
| Camera shake | Shutter speed against equivalent focal length in EXIF, plus the direction of smear being uniform across the frame |
| Subject motion | Shutter speed in EXIF, plus smear confined to the moving subject while static detail holds |
| Missed focus plane | Inspection at 100%: something else in the frame is sharp |
| Clipped highlights | Measured clipping percentage, not the rendered appearance |

| Point | Awarded for |
|---|---|
| 1 | Diagnosis agrees with the file evidence on three or more of four, counting a correct "no dominant fault" as agreement |
| 1 | For each frame carrying a fault, naming the control that would have prevented it **and the cost of using it** |

**What failures mean.** This is the `L5`/`O3`/`O8` discrimination sample. Confusing shake with subject motion is `D`. Not knowing what to look for is `K`. Diagnosing from the rendered image when the file was available is `C`, and it is the same error the curriculum guards against everywhere else.

**Residual weakness:** these are your own frames, so you may recall the circumstances of a shot. Choose the oldest material available to minimise it, and say so if you recognise a frame. The fuller version of this task — one of each fault, guaranteed — runs at the **Phase 1 entry**, where your own missed frames from `T0` and the first camera outings supply the material naturally, with real EXIF and causes you did not choose.

## Task 5 — Harness and protocol (1 point)

| Point | Awarded for |
|---|---|
| 1 | `verify_shot.py` runs over your own set, and you deliberately introduce one intent mismatch and confirm the harness catches it |

This is the Phase 0 harness gate. It is scored here so the gate has evidence rather than an assumption.

## Deliberately not sampled

Record these as unsampled with a reason rather than assuming a state:

| Node | Why not sampled now |
|---|---|
| `D3`–`D7` development | No raw files exist yet. Sampled at the Phase 1 exit, once real raw material is available. |
| `G1`, `G2` camera fluency | No camera. Blocked on `G0`; sampled at the Phase 1 entry. |
| `V7` previsualisation | Depends on `L2` and `O3`; not meaningfully assessable at entry. |
| `L3`–`L5` | Require a library and hard cases that cannot exist yet. |
| `TR1`, `TR5` field readiness and data stewardship | Require the purchased camera, travel carry kit, cards, and two real backup destinations. Sampled during the purchase acceptance test and Phase 1 entry. |
| `TR2`–`TR4`, `TR6` beyond scaffolded baseline | The micro-trip samples story question, roles, and ethical decisions with heavy scaffolding. Independent state requires camera outings under time/access constraints and blind sequence review. |

## Scoring and what happens next

| Total | Reading |
|---|---|
| 0–4 | Beginner frontier confirmed. Phase 0 proceeds as written. |
| 5–9 | Mixed. Expect Task 1 strong and Task 2 weak; weight Phases 2–3 more heavily and compress Phase 1's encoding. |
| 10–12 | Technical nodes may enter Phase 1 at `encoded` or `scaffolded`. Do **not** skip the fluency drills — they are motor, and Task 1 does not sample them. |
| 13–14 | Re-examine whether the entry assumption of "beginner" is right, and re-diagnose against Phase 2 criteria instead. |

**The score is not the output.** The outputs are: an error-code distribution, a node state for every sampled node, and a calibration gap. Record all three in `curriculum-progress.md` and set the Phase 0 frontier from them.

### Reading the calibration gap

The total-score table above routes on the score alone, which throws away the most informative number the diagnostic produces. The gap is **signed**: predicted minus actual, out of 14.

| Signed gap | Reading | Response |
|---|---|---|
| within ±2 | Calibrated at entry | Proceed as the score table directs |
| +3 to +4 | Mild overconfidence, the expected direction for someone who can discuss the domain | Proceed, and watch the first three cycles for the same sign |
| **beyond +4** | Substantial overconfidence — you did not know what you did not know | For the first full macrocycle, predict **per item and per frame**, not per session |
| **beyond −4** | Substantial underconfidence, which corrupts gates just as badly by making every threshold easy to clear | Same remedy: per-item prediction for one macrocycle |

This is the standing three-cycle revision trigger in `photography-curriculum.md`, applied once at entry — the trigger cannot fire until three cycles have passed, and the signal is available now.

The magnitude matters more here than anywhere later, because the diagnostic is the one measurement taken before any teaching has had a chance to move the result.

A high Task 1 score with a low Task 2 or Task 3 score is the **predicted** result for this learner and is not a problem — it is the whole reason the curriculum weights seeing and light over optics. The result that would genuinely change the plan is the reverse.

## Completion record

Fill in when performed. Until then, this file's status stays `designed, not yet performed`.

| Field | Value |
|---|---|
| Date performed | |
| Predicted score / actual score | |
| Signed calibration gap, and the response it triggers | |
| Predicted worst task / actual worst task | |
| Error-code distribution | |
| Node states set | |
| Frontier decision | |
