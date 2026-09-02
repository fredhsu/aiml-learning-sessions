# Travel Photography Curriculum

**Version:** 0.6 — reference-matching protocol, declared bands, runnable diagnostic, trimmed calibration workflow
**Design stage:** approved working curriculum; travel scope confirmed, evidence-gated and revisable
**Dependency graph:** [`photography-dependency-graph.md`](photography-dependency-graph.md)
**Current learning phase, frontier, and node evidence:** [`curriculum-progress.md`](curriculum-progress.md) — the only current-state store. This document owns the design and does not restate live state.

This file is the sole owner of the curriculum version number. No other document restates it.

## North-star performance

Arrive at an unfamiliar destination with one carryable camera kit, limited time, and whatever light exists, and independently:

1. **Read the light** — name the condition, predict how it will render the subject, and decide whether it can be made into a picture at all.
2. **Find the travel story** — state what this place or journey is about, then build non-redundant coverage through orientation, context, human presence, detail, transition, and closure as the story requires.
3. **Execute deliberately** — set exposure and focus to serve that decision, without conscious attention to the controls, before the moment passes.
4. **Operate responsibly** — respect safety, access, consent, culture, weather, power, media, and a declared carry envelope; record deliberate non-captures as decisions.
5. **Protect the work** — ingest and verify two independent copies before card reuse, with originals and metadata intact.
6. **Cull and sequence honestly** — select against recorded intent and story function rather than effort, attachment, or a one-of-each checklist.
7. **Develop the raw files** into a coherent interpretation, and stop when the story is realised.
8. **Deliver and defend a concise travel edit** — why each frame is present, why the order works, why alternatives were rejected, and what would change on a return visit.

- **Primary criterion:** the whole performance above, in unfamiliar conditions and under a declared time/carry/access constraint, unaided.
- **Supporting criterion:** technical and theoretical work earns its place only by improving that performance or its diagnosis.
- **Retention target:** usable performance at 1, 3, and 12 months.
- **Explicitly not the criterion:** a portfolio of pleasing images. A polished portfolio with no visible revision loop is evidence of nothing. Frames made, gear owned, tutorials completed, and hours logged are process metrics and are never advancement evidence.

### Travel constraints

- **One-kit default.** A body and one everyday lens are the normal travel configuration. Extra equipment must solve a repeated, evidenced coverage failure.
- **Finite opportunity is explicit.** Every travel task declares its shooting window, access limits, and whether a return is possible. Planning a bounded first pass is competence; "no plan" is not.
- **Local simulations count.** A nearby unfamiliar neighborhood, transit route, market, park, or day trip can reproduce limited time, uncertain light, public-space ethics, carrying, and story coverage without waiting for travel.
- **Safety and ethics are gates.** Unsafe, prohibited, exploitative, or non-consensual capture is a critical failure, not a tradeoff against image quality.
- **Story beats checklist.** Coverage roles expose omissions during capture; the final edit includes only frames the story needs.
- **Operational success is separate from photographic success.** A charged battery and two backups prevent failure but do not advance seeing or story nodes.

### Evidence classes

| Class | What must be shown |
|---|---|
| **Recall** | Produce the stop relations, depth-of-field and motion relations, and the light vocabulary without reference, in the field, at speed. |
| **Discrimination** | Given confusable situations, select the right control mode and the binding constraint — and say why the alternative is wrong. |
| **Performance** | Frames match their recorded intent on machine-checked dimensions, at a pre-declared rate, under time pressure where the subject imposes it. |
| **Transfer** | The whole workflow survives an unfamiliar destination, story question, light/flow, time/access constraint, and delivery deadline. |
| **Retention** | It holds after 7–14 days, after a macrocycle, and at 6 and 12 months. |

## Domain typing and operating settings

Photography is **Type 2 (perceptual-motor and aesthetic) dominant**. Its knobs are close to the opposite of the Type 1 settings used in the parallel ML curriculum, and the most likely way this plan fails is by drifting toward those familiar settings.

| Setting | Design | Rationale |
|---|---|---|
| Graph density | Flat and entangled, except the `O` optics chain | Type 2. Most edges are attention-limited, not logical **[B]** |
| Explicit SRS weight | **Near zero, hard-capped** | You cannot flashcard a confident frame. Bounded to the `L1`/`O1` declarative substrate only |
| Worked examples | **Demonstration, not worked examples** | Watch the *process* — contact sheets, outtakes, sequences — not the finished artifact. Use targeted YouTube video when motion or sequence is the load-bearing information **[A]** |
| Block → interleave | **Block long, then vary** | Wulf & Shea: interleaving before the coordination pattern stabilises overwhelms rather than helps **[B]** |
| Feedback — technical | Automated ground truth: intent versus EXIF, computed depth of field, clipping, focus at 100% | Cheap, real, and fully verifiable |
| Feedback — perceptual | Fixed rubric, committed prior prediction, delayed re-cull, reference-matching against fixed targets | The known weak point; see *Feedback integrity* below |
| Discovery permission | Encouraged once camera fluency exists | Type 2; harmful only before the controls are automatic |
| Whole-task cadence | **From day one, every week, non-negotiable** | Type 2 pushes whole-task share far higher from the start **[A]** |
| Fluency emphasis | **High** — camera fluency is this domain's line confidence | Slow-but-correct operation consumes the attention that seeing requires |
| Sleep | Treated as a scheduled input, not a lifestyle note | Motor and perceptual consolidation is disproportionately overnight **[A]** |

### Per-subgraph knob overrides

The graph is a mixture, and averaging the knobs would serve none of it.

| Subgraph | Type | Override |
|---|---|---|
| `O` optics and exposure | Type 1 | Worked examples and derivations are appropriate here, faded deliberately. This is the only place discovery learning is switched off. |
| `G` camera operation | Type 2 motor | Blocked timed drills to automaticity. Measured in seconds, not correctness. |
| `V` seeing | Type 2 core | Demonstration, blocked constraints, long blocking, perceptual feedback, whole tasks throughout. |
| `L` light | Type 3 pattern | High-volume exposure to curated, personally annotated conditions with rapid predict-then-verify. Abstract principle-learning underperforms here. |
| `D` development | Type 2 with a Type 5 tooling layer | Reference-matching supplies ground truth for the interpretive half; the tooling half is just-in-time only. |
| `TR1`, `TR5` travel operations | Type 5 operational | Rehearse the real packed-to-ready and card-to-two-copies workflows; verify timing, weight, counts, and recovery rather than discuss readiness. |
| `TR2`–`TR4`, `TR6` travel judgement/story | Type 2 + Type 3 | Demonstrate contact sheets and sequences, practise local simulations, interleave access/coverage cases only after each is reliable, and keep open judgement under the human-feedback ceiling. |
| `L1`, `O1` vocabulary | Type 4, bounded | The *only* legitimate SRS. Capped at roughly 50 items. Growth beyond the cap is a design failure, not progress. |

## State and error model

Use the vocabulary in [`CONTEXT.md`](CONTEXT.md).

- **Design stage** describes the maturity of the curriculum design.
- **Learning phase** identifies the active body of learner work.
- **Node state** records the strongest evidence held: `not-assessed → not-encoded → encoded → scaffolded → independent → transfer → delayed-secure`.
- **Attempt error** is a `K/R/M/D/P/F/T/C` diagnosis for one miss. It routes a remedy and is never a persistent node state.
- **Technical verification** and **perceptual judgement** are separate evidence with different ceilings. A rubric score never advances a technical node, and harness output never advances a perceptual one.
- Prerequisite, attention-limited, sequence, and integration edges remain distinct. Preferred order must not be recorded as a capability dependency.

## The demonstration-to-independence loop

The Type 2 analogue of a theory-to-code loop. For every new capability:

1. **Demonstrate the process, not the product.** Watch or read through how the decision is made — the sequence, what was rejected, what the outtakes looked like. A finished image teaches almost nothing about how it was arrived at. When motion, timing, physical controls, changing light, position changes, contact-sheet reasoning, or development changes carry the lesson, use one verified YouTube video or timestamped segment. Give one observation or prediction prompt before viewing and require an immediate diagnosis, blocked drill, or shooting output afterward. Video viewing counts as encoding and proves exposure only; it cannot advance a node or replace the outing.
2. **Predict, then verify on someone else's work.** Given a frame, state the settings, the light condition, and the camera position before checking. Cheap reps, no equipment needed, and it directly builds the `L2` pattern library.
3. **Blocked drill under one fixed constraint.** One control, one focal length, one condition. Repeat until execution is unattended. Do not vary anything.
4. **Committed intent, then independent frames.** Record the shot intent before the shutter; shoot; verify against the file.
5. **Delayed cull and rubric cycle**, at least a day later, with predicted scores committed first.
6. **Vary one surface** — a new subject, a new condition, a new distance — and repeat.

Do not introduce variation before step 3 is stable. Do not defer step 4 until the theory is finished; every week contains a real outing regardless of where the theory stands.

# Phase sequence

Phases are **evidence-gated, not calendar-gated**. At 2–4 hours per week, a phase may occupy one or several four-week macrocycles. That is not a deadline and not a delay.

**Phase 0 is the phase most likely to violate the weekly-outing rule, so it is stated explicitly here.** Phase 0's named artifacts — the entry diagnostic, the harness gate, and the `G0` time box — are all desk work, and at this dose the phase realistically spans three to five weeks. `T0` alone would leave two or more weeks with no outing, firing the "no outing for two weeks" revision trigger inside a phase whose own gates never asked for one. So the weekly outing in Phase 0 is a **repeat phone micro-trip**: same form as `T0`, new place, one recorded constraint. This costs nothing extra — the Phase 0 scorecard already requires the shot-intent protocol executed unprompted on a second outing, and these are the outings that supply it.

| Phase | Frontier | Whole task | Scaffolding fade | Exit milestone |
|---|---|---|---|---|
| 0. Baseline, camera purchase, and harness | X1–X2, V1, O1, O7, L1, G0; scaffolded TR3–TR4 baseline | **T0** — phone-based local micro-trip: unfamiliar place, story question, intent and coverage role recorded — then **a repeat phone micro-trip every week until `G0` closes** | Guided intent template → prompted intent → unprompted intent | Working harness, scored baseline, diagnostic complete, and travel-camera purchase/acceptance contract completed |
| 1. Camera fluency and field readiness | O2–O6, O8–O9, G1, G3, X3, D1–D2, TR1; scaffolded TR5 | **T1** — one-kit blocked-constraint outing from packed bag to verified ingest | Full field checklist → faded preflight → unaided/timed readiness | Timed controls and first-frame target passed; intent-match threshold met; preventable readiness failures absent twice |
| 2. Seeing, rapid orientation, and coverage | V2–V5, X4, G2, TR2–TR4 | **T2** — 6–8 frame place study made in an unfamiliar location under a 60-minute field limit | Supplied story question/roles → chosen roles → self-set question and ethical boundary | Reference match plus a coherent place edit; delayed re-cull; no ethical critical failure |
| 3. Light, people, and changing conditions | L2–L5, V6–V7, TR2–TR4 | **T3** — one place across materially different light/flow conditions, including a wait, return, or recorded rejection | Named condition → predicted rendering → previsualised story coverage | Light predictions verified; pattern library built; ethical/access decisions and one deliberate non-capture recorded |
| 4. Travel field workflow and interpretation | D3–D7, TR5–TR6 | **T4** — capture, two-copy verified ingest, edit, develop, and sequence within 48 hours | Guided ingest/edit → reference match → independent field-to-story workflow | Backup drill passed; reference-matched rendering; rescue detection and travel-story rubric at threshold |
| 5. Integrated travel story | Whole graph | **T5** — 10–12 frame story from an unfamiliar destination, one carryable kit, stated question and delivery deadline | Supplied brief → self-set brief → self-edited story with defence | Story defended against unseen questions, never-practised conditions handled, and delayed re-edit/re-shoot completed |

# Per-phase control design

| Phase | Encoding — demonstration and reference | Retrieval and interleaving | Deliberate-practice target | Feedback and milestone |
|---|---|---|---|---|
| 0 | Derive stops and equivalent focal length; demonstrate a short unfamiliar-place contact sheet and its rejected coverage; complete the travel contract and camera acceptance test. | Predict settings, light, camera position, and frame role from travel photographs before checking. No interleaving. | One-clause subject plus coverage role before release; phone `T0`; harness unaided. | Harness output, image rubric, and travel-story rubric on the baseline; camera purchase unblocks Phase 1. |
| 1 | Demonstrate camera controls, packed-to-ready preflight, quiet/public-space operation, raw ingest, and field backup. | Predict aperture/shutter/ISO/histogram and choose the correct control mode for described travel scenes. Block one control at a time. | Timed controls; packed-to-first-frame drill; focus placement; one-kit carry; two-copy ingest rehearsal. | Harness, timing, readiness log, and diagnosis from files. `T1` repeats in two conditions. |
| 2 | Demonstrate rapid orientation, position-finding, coverage-role decisions, people/consent choices, and set editing from contact sheets. | Delayed blind re-cull; interleave subject/edge/separation and coverage omissions only after each is reliable. | Three positions; bounded scouting; context/detail/human role discrimination; explicit consent drill with no capture required. | Image rubric plus travel-story rubric against predictions. Milestone: **T2**. |
| 3 | Build the learner's own light/flow library from places across times and weather; show sequences around moments, waits, and non-captures. | Predict rendering and subject flow; interleave confusable light and access cases once individually secure. | Read light/flow quickly; decide shoot/wait/return/reject; maintain story coverage as conditions change. | File-verified light predictions, ethical-access log, and story edit. Milestone: **T3**. |
| 4 | Demonstrate end-to-end field ingest, verified two-copy backup, cull, sequence, and coherent development with rejected edit directions shown. | Re-sequence and re-develop an old trip to a new declared question without consulting the prior edit. | Compress to necessary frames; match a target rendering; finish a travel edit within 48 hours. | Backup record, clipping/geometry checks, reference match, rescue detection, travel-story rubric. Milestone: **T4**. |
| 5 | Gap-driven study only. Output: travel-story question, route/access plan, contact sheet, final sequence, captions where needed, and defence. | Unannounced local simulations and full-graph mixed diagnosis. | The capstone's binding constraint; the project generates remediation. | Unseen-question defence, external-channel decision, delayed re-edit and re-shoot. Milestone: **T5**. |

Resources are tools, not completion metrics. Each has an attached output.

# Weekly operating system

## Default allocation

Photography's shares differ sharply from a Type 1 curriculum's: whole-task work is high from the start, explicit retrieval is minimal, and structured review is deliberately large because in a domain with weak external feedback, review *is* where most learning is extracted.

| Phase | Retrieval | Encoding | Blocked practice | Whole task (outing) | Review, verification, calibration |
|---|---:|---:|---:|---:|---:|
| 0–1 | 5% | 20% | 30% | 35% | 10% |
| 2–3 | 5% | 15% | 20% | 45% | 15% |
| 4–5 | 5% | 10% | 15% | 50% | 20% |

**At the 2-hour floor, the outing survives and everything else is cut.** New material is cut first, then blocked practice, then encoding. Review is never cut below 15 minutes, because an unreviewed outing produces no evidence and therefore no learning state change.

**Tooling cap:** building or improving the verification harness counts against *encoding*, not against whole task, and may not exceed encoding's share in any macrocycle. This is a named guard against substituting a tractable Type 1 activity for the intractable Type 2 one.

**One declared exception:** `tools/compare_render.py`, the `RM-R` rendering-comparison tool required by [`reference-matching-protocol.md`](rubrics/reference-matching-protocol.md), is budgeted at a single encoding block in Phase 1 or Phase 2 and is exempt from the cap for that block only. It is the only tooling that lifts a perceptual node above the feedback-integrity ceiling, which is why it earns an exception and nothing else does. Record the block in `curriculum-progress.md` when it is spent. Unspent by the Phase 4 gate, the correct response is to cap `D4`/`D5` at `independent` — not to extend the exception.

## Two session types

Unlike the parallel ML curriculum, this one cannot be run entirely at a desk.

**Outing — 45–90 minutes, away from the desk.**
1. **Before departure:** complete the travel-readiness gate at the current scaffold level; state the story question, opportunity window, whether return is possible, one blocked constraint, carry limit, access/safety fallback, and backup plan.
2. **5 min:** commit keeper rate, expected dominant error code, first-frame time, and—when producing a set—the needed coverage roles.
3. **35–75 min:** shoot. Record intent, coverage role, and ethical-access decision before each frame or valid tight group. A tight group covers one subject/relationship, light condition, position/technical strategy, and explicit stop condition; it ends when any of those changes. Frames with no recorded intent are not evidence and are not culled.
4. **5 min:** before leaving, note deliberate non-captures and rejected scenes, then check whether one genuinely necessary story role is missing.
5. **After return:** ingest, verify two copies, and optionally make a coarse safety index before card reuse. These may occur the same day; the evidence-bearing cull and sequence review may not.

The field commitment may be a timestamped voice memo, pocket card, or other safe low-friction record, then transcribed without changing its substance. Recording friction that repeatedly costs awareness or moments triggers an interface simplification, never abandonment of pre-shutter commitment.

**Desk — 45–60 minutes, at least one day later.**
1. **10 min:** closed-resource retrieval from prior work — a relation, a prediction, or a diagnosis. Before anything else.
2. **10 min:** run the verification harness; read the intent-versus-actual verdicts.
3. **10 min:** cull against intent. Commit the predicted rubric totals — one per frame, one for the set, plus the frontier dimension if one is active — *before* any critique.
4. **15 min:** critique individual frames against the fixed image rubric; from Phase 2, also score the ordered set with the travel-story rubric or do development work.
5. **10 min:** log evidence, story/sequence decisions, ethics/readiness/backup outcomes, error codes, calibration gap, node-state transitions, and the next smallest action.

**Never review the same day you shoot.** A delayed cull is a materially better instrument: the effort and the intent have faded, and what remains is closer to what a viewer sees. This is a design heuristic consistent with the delayed-testing evidence, not a research finding in its own right.

## Weekly shape

| Slot | At the 2 h floor | At 4 h |
|---|---|---|
| Outing | 1 × 60 min | 2 × 60–75 min |
| Desk | 1 × 45 min | 2 × 45–60 min |
| Micro | 1 × 15 min: retrieval plus light-library review | 2 × 15 min |

Distribute drills across days rather than massing them. Sleep between blocked-drill sessions is part of the mechanism, not a gap in it **[A]**.

## Four-week macrocycle

| Week | Function |
|---|---|
| 1 | Encode one bounded node; demonstration and prediction reps; outing under a single fixed constraint |
| 2 | Same constraint, unaided; delayed retrieval and delayed re-cull of Week 1 |
| 3 | Vary one surface; begin interleaving confusables that are already individually reliable; reference-matching |
| 4 | Cumulative: unfamiliar-conditions transfer outing, full rubric cycle, error-distribution review, calibration review, plan adjustment |

Prior material reappears after roughly 2 days, 1 week, 3–4 weeks, then inside later whole tasks and at the 3- and 12-month checks. Distributed retrieval is **[A]**; the specific intervals are adaptive planning heuristics, not empirical optima.

# Error-routing rules

Classify every substantive miss before changing anything. The photographic signature column is what makes this usable in the field.

| Code | Photographic signature | Remedy | Do not |
|---|---|---|---|
| `K` | The control, condition, or procedure was simply never available | Demonstration, then a blocked drill on that single element | Schedule retrieval for something not yet encoded |
| `R` | Executed it correctly before; could not produce it in the field this time | Distributed retrieval of the procedure; re-shoot the same situation within two days | Re-explain it |
| `M` | Confident and consistently wrong — e.g. believes a wider aperture will fix a motion problem, or that 1/60 freezes a running subject | **Contrast pair:** shoot the same scene both ways, predict first, compare the files side by side | Repeat the shot the same way; repetition entrenches it |
| `D` | Both procedures available, the wrong one selected — aperture priority chosen when motion was the binding constraint | Interleaved confusable scenario drills, **only after** both procedures are independently reliable | Return to isolated practice of the one you got wrong |
| `P` | Right decision, broken execution — meant f/2.8 and dialled f/8; focus point on the near eye | Faded checklist plus a timed part-task drill on that one control | Re-teach the concept |
| `F` | Correct but too slow — the moment passed, or setup consumed the attention that seeing needed | Timed unaided camera-fluency drills, measured in seconds | Introduce new material |
| `T` | Reliable under the blocked constraint, falls apart in unfamiliar light or with an unfamiliar subject | Varied whole tasks in genuinely unfamiliar conditions; assumption debrief | Add more drills or more vocabulary |
| `C` | Knew better — ISO left at 6400, an edge intrusion missed, battery/card state unchecked, or card reused before backup | Preflight/closeout checklist, explicit edge check, and pacing | Any content intervention |

**Dominant** means at least three instances, or one third of substantive errors, across two sessions. New material classified `K` is the active frontier, not a personal failure.

# Assessment stack

| Measure | Cadence | Evidence |
|---|---|---|
| Shot-intent match | Every frame | Harness verdict: predicted versus actual settings, depth, motion, placement |
| Keeper rate against intent | Every outing | Frames meeting their own recorded intent ÷ frames made |
| Closed-resource retrieval | Every desk session | A relation, a settings prediction, or a soft-frame diagnosis, unaided |
| Timed fluency drill | Weekly in Phases 1–2, then monthly | Seconds to correct declared settings, unaided, and handheld sharpness rate |
| Delayed blind re-cull | Weekly from Phase 2 | Agreement rate with the original cull on a set at least 7 days old, or an explanation per disagreement |
| Fixed-rubric critique | Weekly from Phase 2 | Scored per dimension by the tutor, against a committed prior self-prediction of the frame and set **totals** |
| Reference-matching deviation | Each macrocycle from Phase 2 | Measured deviation from a specified target framing, depth, motion, or rendering, per [`reference-matching-protocol.md`](rubrics/reference-matching-protocol.md) |
| Field-readiness drill | Every outing in Phases 1–2, then every travel simulation | Packed-to-first-intentional-frame time, carry-envelope compliance, preventable readiness failures, and recorded safety/access fallback |
| Travel coverage and story edit | Every outing from Phase 2 | Declared story question, coverage-role record, ordered edit, and [`travel-story-rubric.md`](rubrics/travel-story-rubric.md) against committed predictions |
| Ethical-access decision | Every outing with identifiable people or sensitive contexts | Capture/consent/public-use decision committed in the intent record; critical failures block advancement |
| Data-stewardship check | Every camera outing from Phase 1 | Two verified copies and recorded paths before card reuse; recovery drill once per macrocycle in Phase 4 |
| Cumulative transfer outing | Each macrocycle | Unfamiliar location and light, bounded reconnaissance, declared opportunity/access window, and no assumption of return |
| Phase-gate delayed check | 7–14 days after the qualifying attempt | Alternate-form performance before advancing |
| Maintenance delayed measure | 4–12 weeks after a node leaves active study | Regression changes the node state and reopens remediation |
| Long retention | 6 and 12 months | Re-shoot a comparable brief unaided and re-develop from raw |
| Calibration | Every outing and every critique | Predicted keeper rate, rubric totals per frame and per set, and dominant error code, committed before results. Session duration is **not** predicted; declared time targets are measured directly instead |

## The declared calibration band

Two Phase 2 gates and one revision trigger depend on a calibration gap being "inside the declared band". This is that declaration — without it those gates are discretionary rather than binary. A band chosen after seeing a result is not a band.

| Measure | Band | Notes |
|---|---|---|
| Keeper rate against intent | Signed gap within **±15 percentage points** | Actual is harness ground truth, so this band is the strongest of the three |
| Rubric total, per frame and per set | Signed gap within **±0.5 points per scored dimension** — divide the total gap by the number of dimensions actually scored | Normalised per dimension because dimensions 5, 7, and 8 are conditionally scored, so a raw total has a moving denominator |
| Discrimination | The frames predicted to score highest and lowest in a set are, in fact, the highest- and lowest-scoring frames | Checked per set from Phase 2 |
| Session duration | **Not measured.** Predicting how long a session takes routes no remedy in this domain | Time enters this curriculum as a declared performance threshold — first-intentional-frame time, seconds-to-correct-settings, the `T2` field limit — and those are measured against their target directly, not as a prediction gap |

**The gap is tracked signed, never absolute.** A learner who is consistently 3 points optimistic and one who alternates ±3 have the same absolute gap and completely different problems. Only the signed series distinguishes them.

**Discrimination is checked separately from the gap, because a flat prediction can pass the gap and mean nothing.** Predicting the same score for every frame in a set will produce a small mean gap while carrying no information about any individual frame. Per-frame predicted totals must vary across a set, and the top-and-bottom check above is the cheap test of whether that variation is real.

**Rubric-score calibration is agreement with a known-unreliable evaluator.** The prediction is high-integrity; the thing it is compared against is not. Closing this gap over months is consistent with learning to see, and equally consistent with learning to predict the tutor — this system cannot distinguish them. Keeper-rate calibration does not have this problem, because its actual comes from the file. Weight them accordingly, and do not let a numeric gap acquire the authority of a measurement it has not earned.

## The floor under pre-declared score bands

Distinct from the calibration band above. Two gates require an ordered edit to clear "its pre-declared band" — a score threshold declared before the outing. Since the learner declares the prediction *and* the threshold it must clear, an unfloored band makes the gate self-approving: declare a low enough target and any edit passes. The floor, stated in the travel-story rubric's own anchors rather than as invented numbers:

| Gate | Floor on the travel-story rubric |
|---|---|
| Phase 2 — `T2` | Mean **≥ 1.5** across scored dimensions. Below that, the dominant failure is still the first thing a viewer notices on more dimensions than not. |
| Phase 4 — `T4` story edit | Mean **≥ 2.0** across scored dimensions — every dimension "sound" on average, which is what clearing the story threshold means in the edit-discipline anchor. |
| Phase 5 — `T5` | Mean **≥ 2.0**, and the defence must hold on the dimensions scored 2 rather than 3. |

The mean is taken across **scored** dimensions, since context/caption accuracy and technical continuity are not always applicable. Ethical integrity is never inside the mean: a 0 fails outright at any phase, and from Phase 4 a 1 also fails, because "a defensible concern left unresolved" is not an acceptable state for work made at a real destination.

A learner may declare a band above the floor and should when the material warrants it. Declaring one below it is not a band.

## Phase exit rule

Advance only with **all** of:

1. accurate independent performance at the required level, including speed where the node requires fluency;
2. a justified choice among confusable alternatives;
3. a changed-surface transfer result — different subject, different light, or different constraint;
4. one delayed recheck at 7–14 days;
5. a reproducible record: intents, EXIF, harness output, cull decisions, rubric scores, and error codes.

Scores are task-local and require an explicit rubric declared **before** results are seen. A score never implies a node state by itself. Every assessed point records assistance as `scaffolded`, `independent`, or `transfer`. Critical failures named in a gate override aggregate scores.

## Phase exit scorecards

Every row is a binary gate. Thresholds that depend on a subject, location, or condition must be declared in the outing's intent record **before** the frames are made; they may never be chosen after seeing the results.

### Phase 0 — baseline, travel-camera purchase, and harness

| Gate | Required evidence |
|---|---|
| Harness | One command runs `tools/verify_shot.py` over a real set and produces per-frame intent-versus-actual verdicts, including at least one deliberate mismatch it correctly catches. |
| Baseline whole task | **T0**: at least 20 phone frames in an unfamiliar local place, each with shot intent and coverage role recorded; produce a 5–7 frame micro-trip edit at least one day later; score both frames and sequence against committed predictions. |
| Entry diagnostic | [`phase-0-entry-diagnostic.md`](phase-0-entry-diagnostic.md) completed. Every graph node has moved off `not-assessed` or is explicitly recorded as deliberately unsampled with a reason. |
| Travel-camera acquisition | A completed `G0` travel contract and decision record, camera ordered/received, and every purchase acceptance test passed within the return window. There is no open-ended deferral because camera purchase is now an explicit constraint. |
| Protocol | The shot-intent protocol executed unprompted on a second outing, without the template being handed to you. |

### Phase 1 — camera fluency and field readiness

| Gate | Required evidence |
|---|---|
| Independent mechanism | Closed-resource: given a described scene and a stated intent, predict aperture, shutter, ISO, and the rough histogram; then shoot it and match the prediction within a pre-declared tolerance. |
| Fluency | Timed unaided drill: camera-up to correct declared settings within the declared time, on at least 8 of 10 trials, **without looking at the controls**. Plus `G3`: handheld sharpness rate at the declared shutter speed meets its threshold. A slow pass is a fail. |
| Field readiness | From the packed one-kit state, pass the declared first-intentional-frame time, carry the complete kit for two hours, execute preflight without prompts, and complete a verified two-copy ingest. Repeat without a preventable readiness failure in a materially different condition. |
| Debugging and discrimination | From the file alone, correctly attribute soft frames to camera shake, subject motion, a missed focus plane, or an AF-mode error, across at least five cases including at least one of each. Diagnosis committed before checking. |
| Whole task / transfer | **T1** on one blocked constraint meets its pre-declared intent-match rate from packed bag through verified ingest; then repeats in a materially different light or weather condition. |
| Delayed | After 7–14 days with no drilling in between, repeat the timed fluency drill and the prediction task at threshold. |
| Reproducibility | Outing record with intents, EXIF, harness output, cull decisions, rubric scores, and error codes; one command regenerates the verification report. |

### Phase 2 — seeing, orientation, and coverage

| Gate | Required evidence |
|---|---|
| Independent mechanism | For an unfamiliar subject, produce at least three materially different camera positions, then state which is strongest and why — committed before culling. |
| Rapid orientation | At an unfamiliar place, identify light direction, subject flow, access boundary, safety fallback, and one story question within a pre-declared scouting limit, then make the first intentional frame within its target time. |
| Debugging and discrimination | Given your own frame pairs, correctly identify which fails on subject clarity, which on edges, and which on separation; committed before the rubric is applied. |
| Whole task / transfer | **T2**: a 6–8 frame place study in an unfamiliar location under a 60-minute field limit, plus an `RM-F`/`RM-D`/`RM-M` subset passing every declared dimension per [`reference-matching-protocol.md`](rubrics/reference-matching-protocol.md). The travel-story rubric must clear its pre-declared score band, subject to the floor declared above, with no ethical critical failure. |
| Calibration | Predicted rubric scores fall within the declared calibration band above, **and** the set passes the discrimination check. A widening signed gap fails the gate regardless of the scores themselves. |
| Delayed | After 7–14 days, a blind re-cull of an earlier set agrees with the original above the declared rate, or every disagreement is explained. |
| Ceiling | This gate certifies `independent` for open perceptual work, **not** `transfer`. Transfer on open perceptual work requires an external channel and is not claimable here. |
| Ethical access | Every identifiable-person or sensitive-context frame has a recorded access/consent decision; the learner also records at least one defensible non-capture. Any ethical-integrity score of 0 fails the gate. |

### Phase 3 — light, people, and changing conditions

| Gate | Required evidence |
|---|---|
| Independent mechanism | Name a condition and predict how it will render a stated subject — direction, contrast ratio, colour cast, and where it will clip — before the frame. Verified against the file across at least five conditions. |
| Pattern library | `L3` holds at least 20 annotated entries built from your own culled frames, each recording condition, subject, prediction, actual, and one line on what it taught. Entries copied from other people's work do not count. |
| Debugging and discrimination | On hard cases, correctly identify whether contrast, level, or colour is the binding constraint, and choose the matching response. |
| Whole task / transfer | **T3**: one place and story question across at least three materially different light or subject-flow conditions, including a deliberate response—reposition, adapt, wait, return, or reject—and at least one recorded safety, access, ethical, or visual rejection. Include one single-visit form with no assumed return. |
| Delayed | After 7–14 days, predict and verify a condition not in the library. |
| Reproducibility | Library entries link to files, intents, and harness output. |
| Coverage under change | The final sequence retains necessary orientation/context/detail/human/moment roles as conditions change without padding the edit to complete a checklist. |

### Phase 4 — travel field workflow and interpretation

| Gate | Required evidence |
|---|---|
| Independent mechanism | From a raw file and a declared intent, develop independently. The technical half is machine-verified: no unintended clipping, geometry corrected, declared neutral references neutral. |
| Reference-matching | Pass an `RM-R` rendering match on tone and colour per [`reference-matching-protocol.md`](rubrics/reference-matching-protocol.md), every declared patch within tolerance. If the `RM-R` tooling exception was never spent, this row cannot be satisfied and `D4`/`D5` are capped at `independent` rather than passed by eye. |
| Data stewardship | From a fresh card, ingest originals, verify two independent copies, record paths/counts/check time, and demonstrate recovery from one unavailable destination before card reuse. |
| Restraint (`D7`) | Across a set, the committed pre-edit intent is compared with the edit actually applied. The tutor, working blind to your intent, identifies which frames look like rescues; you must have rejected those in the cull. Any rescued frame that the cull should have killed fails the gate. |
| Whole task / transfer | **T4**: complete capture-to-two-copy-ingest-to-6–10-frame-story within 48 hours, then repeat the sequencing/development treatment on a materially different place. |
| Delayed | After 7–14 days, re-develop an older raw to a newly declared intent without consulting the previous edit. |
| Reproducibility | darktable XMP or exported edit history tracked; one command regenerates the outputs. |
| Story edit | Ordered sequence clears its pre-declared travel-story-rubric band; every frame has a necessary role, and one plausible but redundant frame is explicitly cut. |

### Phase 5 — integrated travel story

| Gate | Required evidence |
|---|---|
| Whole task | **T5**: 10–12 frames answering a stated travel-story question, made at an unfamiliar destination with one carryable kit under declared time/access/return constraints, delivered by a declared deadline with factual caption/context notes where needed. |
| Defence | Justify every load-bearing capture and sequence choice—position, focal length, moment, exposure, ethical access, inclusion, order, and rendering—and answer unseen questions about rejected alternatives. |
| Transfer | One set made under never-practised travel conditions such as unfamiliar transit, weather, crowd flow, language/access constraint, or severe time limit. |
| Delayed | At 4–12 weeks, re-edit the original contact sheet blind to the first sequence, then re-shoot a comparable local travel brief unaided and compare both decisions. |
| Operations | No preventable readiness, ethical-access, or data-loss critical failure; final work exists in two verified independent copies. |
| External channel | The capstone triggers the external-feedback decision below. A capstone assessed only by this system is recorded as **unvalidated on perceptual dimensions**, and the record must say so. |

# Feedback channels

## Channels that are genuine ground truth

| Channel | Used in | Role |
|---|---|---|
| Intent-versus-EXIF verification (`tools/verify_shot.py`) | 0–5 | Machine-checked agreement between what you decided and what the camera recorded. Not a matter of opinion. |
| Computed depth of field, motion adequacy, and clipping analysis | 1–5 | Turns "is this sharp enough / fast enough / blown" into arithmetic against declared intent |
| Focus inspection at 100% | 1–5 | Objectively settles focus-plane and AF errors |
| Reference-matching deviation | 2–5 | Converts part of perceptual practice into measurable deviation from a fixed external target |
| Delayed blind re-cull | 2–5 | Partially independent: your later self does not remember the intent or the effort |
| darktable clipping, geometry, and neutrality checks | 4–5 | Verifiable half of development |
| Field-readiness and data-stewardship checks | 1–5 | Timed packed-to-first-frame readiness, carry weight, two-copy verification, and recovery are operational ground truth—not evidence that the pictures work |

## The known structural weakness

**You have selected no human feedback channel.** The architecture treats at least one channel the learner does not control as non-negotiable, and this plan does not fully satisfy it.

The mitigation is real but partial:

- Technical and operational dimensions are covered by machine/recorded ground truth, so the `O`, `G`, `D3`, `X1`–`X2`, `TR1`, and `TR5` subgraphs are unaffected.
- Reference-matching supplies an external fixed target for part of the perceptual work, which is why `V5`, `V7`, `D4`, and `D5` can still reach `transfer`.
- Fixed-rubric critique with **committed prior self-prediction** makes the calibration gap the primary signal rather than the score. A score you predicted correctly carries information even if the absolute score is generous.
- Open perceptual, ethical-judgement, and travel-story nodes are **capped at `independent`**. The graph records this cap explicitly rather than letting it be forgotten.

**AI rubric critique is not ground truth.** Unconstrained critique drifts toward agreeableness, and agreeable critique is worse than none because it is mistaken for signal. Countermeasures are built into `rubrics/image-critique-rubric.md`: fixed dimensions, numeric anchors, mandatory identification of the single strongest failure, blind scoring before the stated intent is revealed, and a required calibration comparison.

**Trigger for adding a human channel:** at the Phase 2 exit, before the first actual-trip set, before publishing identifiable/sensitive travel work, or when rubric scores rise for three consecutive cycles while the calibration gap widens or stays flat—whichever comes first. That pattern is the signature of a drifting evaluator and cannot be repaired from inside the system. Prefer a reviewer who can judge both picture editing and the represented place/culture; fallback options are a critique community, local camera club, one-off paid edit, print review, or juried submission. Check privacy before using a public channel.

# Motivation and operations

## Implementation intention

> **Cue:** If it is a scheduled outing window,
> **Action:** then I take the camera out with one written constraint and one committed prediction, and I make frames before I read, plan, or adjust any tooling.
> **Fallback:** If a full outing is impossible, I make ten frames within five minutes of home against a single recorded intent, verify them, and log one error code.

Adapt the days, not the cue–action–fallback structure.

## Fallback session — 20 minutes

1. Ten frames, one recorded intent, anywhere reachable.
2. Run the harness on them.
3. Log one error code and the next smallest action.

No new material, no tooling work, no gear research in fallback mode.

## Parallel-load rule

Photography is the **third and lightest track**, alongside the primary ML curriculum and the robotics track. It runs at secondary-to-maintenance dose and is **not** in peak acquisition.

- Photography is Type 2 and the other two are Type 1 and Type 5, which is the pairing the architecture favours — dissimilar cognitive demands coexist far better than two symbolic tracks **[B]**.
- Do not schedule an outing and a dense theory session back to back if the theory session comes first; the outing is the one that degrades.
- **Gaps are spacing, not neglect.** A week with one outing and one desk session is a fully successful week at this dose. The instinct that daily contact is required comes from cramming.
- Sleep is the shared resource all three tracks draw on and is protected before any of them.

## Collapse protocol

1. Cut new material and tooling work first; protect the outing.
2. Never cut review before cutting acquisition — an unreviewed outing generates no evidence.
3. Fall back to the 20-minute session rather than to zero.
4. After any gap longer than three weeks, re-diagnose rather than resume. Camera fluency decays quickly; perceptual judgement decays slowly. You will be in a different place in both directions.

# Revision triggers

Set in advance so that plan changes are evidence-driven rather than mood-driven.

| Trigger | Mandatory response |
|---|---|
| No outing for two weeks | The next session is an outing. No desk work, no reading, no tooling until frames exist. |
| A real trip is within four weeks and `TR1` or `TR5` is below `independent` | Freeze new photographic material. Run local packed-bag, power/media, two-copy backup, and 60-minute story simulations until both pass. |
| Repeatedly missing a coverage role across two reviewed outings | Diagnose `TR2`/`TR3` before buying a lens. Change equipment only if position/time/access cannot make the frame with the current everyday lens. |
| Ethical-access uncertainty remains after capture | Exclude the frame from public/review output until resolved; route the next session to `TR4` scenario discrimination, not image critique. |
| Card reuse occurs before two verified copies | Critical `TR5` failure. Stop shooting workflow changes and rehearse ingest/recovery before the next outing. |
| Intent recording repeatedly costs awareness or fleeting moments | Simplify to a timestamped scene/group commitment with a clear stop condition; preserve the evidence invariant rather than blaming the learner or dropping intent. |
| Desk or tooling time exceeds shooting time across a macrocycle | Type 1 substitution is occurring. Cap tooling at zero for one macrocycle. |
| Keeper rate against intent above ~90% for two weeks | Under-loading. Tighten the constraint, shorten the time budget per frame, or move to harder light. |
| Keeper rate against intent below ~30% for two weeks | Too many free variables. Return to a single blocked constraint. |
| `F` dominates | Timed fluency drills only; no new material until the drill threshold is met. |
| `T` dominates | More varied whole tasks in unfamiliar conditions. Do **not** add drills or vocabulary. |
| `M` dominates | Contrast-pair shooting with committed predictions. Do not repeat the same shot the same way. |
| `O`/`G` nodes at `independent` while `V`/`L` nodes sit at `encoded` | The false summit. Freeze optics and gear work entirely; move the full budget to seeing and light. |
| Signed calibration gap outside the declared band, or the discrimination check failed, on three consecutive cycles | Predict every frame's outcome, not just the session's, for one full macrocycle. |
| Rubric scores rising for three cycles while the calibration gap does not narrow | The evaluator is drifting agreeable. Re-anchor with blind scoring and reference-matching, and add a human channel. |
| No delayed measure in a macrocycle | Block advancement until one is completed. |
| SRS or vocabulary list exceeds ~50 items | Over-atomisation. Suspend the excess; shift to embedded retrieval in outings. |
| Two missed weeks | Resume with fallback sessions and a shrunken constraint. Never attempt to catch up. |
| Two failed exit attempts on the same gate | Audit the graph for a missing prerequisite. Do not simply repeat the phase. |

# Design log

| Decision | Rationale |
|---|---|
| Type 2 dominant, with per-subgraph knob overrides | Averaging knobs across a mixed domain serves none of it. Optics genuinely is Type 1; seeing genuinely is not. |
| Whole task from Phase 0, every week, non-negotiable | Type 2 whole-task timing is *early*, unlike the parallel ML curriculum. Deferring shooting until theory is ready is the characteristic Type 1 error transplanted into the wrong domain. |
| Verification harness built in Phase 0, before any technique work | Appendix A: build the domain with the cheapest ground truth first, so the routing logic can be debugged while the stakes are low. |
| Shot intent is mandatory and pre-shutter | Without a committed prior intent nothing is assessable, keeper rate is meaningless, and every critique becomes post-hoc rationalisation. |
| SRS hard-capped at ~50 items | Type 2 SRS is close to a trap. The legitimate role is the declarative substrate only, and a cap is the only thing that reliably keeps it there. |
| Long blocking before any interleaving | Wulf & Shea: contextual interference principles from simple tasks do not transfer to complex motor skills, where early interleaving overwhelms. |
| Review is never cut below 15 minutes | An unreviewed outing produces no evidence and cannot change any node state, so it is indistinguishable from not shooting. |
| Delayed cull, never same-day | Effort and remembered intent bias same-day selection toward the frames that were hardest to make rather than the ones that work. |
| Open perceptual nodes capped at `independent` | Honest consequence of having no human feedback channel. Recording the cap is better than quietly claiming transfer. |
| Gear decision time-boxed and made after `O7`/`V1` | Gear is the most analysable and least valuable problem in the domain, and the standard beginner sink. |
| Tooling work counted against encoding and capped | The learner's strongest instinct is to build a system; that instinct must not consume the hours that only shooting can supply. |
| Reference-matching used as a perceptual ground-truth substitute | Deviation from a fixed external target is measurable, which recovers real feedback integrity for part of the perceptual work. |
| Reference-matching given an explicit protocol with measured tolerances | It was cited as load-bearing in seven places while specifying no target format, metric, or tolerance, and no harness check measured it. The only mechanism lifting `V5`/`V7`/`D4`/`D5` above the ceiling cannot itself be an assertion. |
| Calibration gap treated as the primary perceptual metric, with its limit stated | With a weak evaluator the score is unreliable but the committed prediction is not, so the gap carries the signal. But rubric-score calibration measures agreement with that same weak evaluator, and cannot distinguish learning to see from learning to predict the tutor. Keeper-rate calibration has machine ground truth and does not share the defect; the band table weights them accordingly. |
| Calibration tracked signed, with a separate discrimination check | An absolute gap hides directional bias, and a flat prediction across a set produces an excellent mean gap while carrying no information about any frame. Neither failure is visible without both instruments. |
| Pre-declared score bands given a floor | The learner declares both the prediction and the threshold it must clear, so an unfloored band is a gate that approves itself. The floor is stated in the rubric's own anchors so it is not an invented number. |
| Phase 0's weekly outing named explicitly | Phase 0's own artifacts are all desk work, so the phase would otherwise fire the two-week no-outing trigger while every gate it declares is satisfied. The strongest commitment in the design cannot be left implicit in the one phase most likely to break it. |
| Diagnostic faults settled by the file, not by whoever sourced the frames | Any frame selected *because* it shows a known fault comes with its answer attached, and the tutor has no frames to supply. Reading the fault from EXIF, 100% inspection, and measured clipping is the same principle the whole curriculum runs on, applied to its own diagnostic. |
| Rubric prediction collapsed to frame and set totals; session-duration prediction dropped | Predicting eight dimensions per frame is ceremony that does not survive a 2–4 h/week third track, and when it is abandoned it takes the high-signal predictions with it. The total carries nearly all the information; a frontier dimension is predicted separately when one is active. Session duration routed no remedy — time enters this domain as a declared threshold, measured directly. |
| Tool-supplied thresholds marked as defaults in harness output | The harness silently substituted its own depth and clipping thresholds when the intent file omitted them, producing a verdict against a target the learner never committed to. A ground-truth instrument must not quietly author the standard it measures against. |
| Travel photography is the outcome, not a late specialization | Limited time, one-kit carry, public-space ethics, coverage, sequencing, and data stewardship change the graph from Phase 0 onward. |
| Camera purchase is mandatory but tightly bounded | The phone remains useful for the entry baseline, then a travel-contract purchase and return-window acceptance test unblock motor fluency and raw workflow without turning shopping into the curriculum. |
| Coverage roles are prompts, not a final-edit checklist | They expose missing observation in the field; sequence quality and story necessity decide the final inclusion. |
| Safety, ethical access, and data stewardship are critical gates | Aesthetic quality cannot compensate for harm, prohibited capture, or preventable loss of irreplaceable travel files. |
| Field-safe group intent is valid evidence | A timestamped commitment for one subject/light/strategy with a stop condition preserves calibration integrity without training an unsafe or unusable per-frame writing ritual. |
| YouTube is used for targeted process demonstrations | Video can reveal timing, motor operation, changing conditions, and decision sequences that prose or finished frames hide. Each recommendation is verified and bounded by a pre-view prompt and immediate practice; viewing remains exposure, not advancement evidence. |

# Self-critique

Scored against the plan-quality rubric in `evidence-adaptive-curriculum-architecture.md` §V.2.

| Criterion | Score / 2 |
|---|---:|
| 1. Outcome specificity | 2 |
| 2. Domain-type fit | 2 |
| 3. Prerequisite mapping | 2 |
| 4. Diagnostic placement | 2 |
| 5. Encoding quality and fading | 2 |
| 6. Retrieval layer | 2 |
| 7. Spacing | 2 |
| 8. Discrimination | 2 |
| 9. Whole-task integration | 2 |
| 10. Feedback integrity | **1** |
| 11. Measurement | 2 |
| 12. Sustainability | 2 |
| **Total** | **23 / 24** |

**Weakest area, criterion 10 — feedback integrity.** There is no channel the learner does not control for open perceptual judgement, ethical nuance, or whether an ordered travel story works. Machine verification, reference-matching, operational checks, delayed re-culls/re-sequences, and calibration tracking cover technical nodes and mitigate the rest, but they do not replace an external human editor or culturally informed reviewer. Open perceptual/story nodes remain capped at `independent`; the Phase 5 capstone remains unvalidated on those dimensions absent a human channel. Criterion 10 returns to 2 when a suitable external channel is added.

**Second-weakest, criterion 12 — sustainability.** Scored 2 on design (2-hour floor, fallback session, collapse protocol, parallel-load rule, explicit tooling cap), but the learner's actual energy pattern and the availability of daylight outing windows are unknown until several macrocycles have run. The weekly-hours field in `curriculum-progress.md` exists to supply that data rather than assume it.

**Checked failure signatures.** Not over-atomised: criteria 9–11 are not weak relative to 3–8, whole tasks run from Phase 0, and the SRS is capped. Not premature authenticity: the whole task is present from day one, which is correct for Type 2, but it is scaffolded by a single blocked constraint and a supplied intent template, with fading defined per phase.
