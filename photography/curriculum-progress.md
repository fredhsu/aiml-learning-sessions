# Curriculum progress

## Current control state — 2026-08-31

- **Design stage:** approved travel-photography curriculum; the version number is owned by [`photography-curriculum.md`](photography-curriculum.md) and is not restated here.
- **Learning phase:** Phase 0 — baseline, travel-camera purchase, and verification harness.
- **Active frontier:** the entry diagnostic is in progress. Task 1 is scored; Tasks 2–5 are outstanding, so every node remains `not-assessed` until the diagnostic completes and the frontier must not be assumed before then.
- **Current node evidence:** Task 1 of the entry diagnostic only — a closed-resource optics/exposure sample scoring 0/4, recorded in the 2026-08-31 session below. No frames, no EXIF, no critique, no drills. No node state has been set.
- **Weekly hours actual:** not yet recorded. Budget is 2–4 h/week as the third and lightest track, alongside the primary ML curriculum and the robotics track. Photography is **not** in peak acquisition.
- **Last learner evidence:** none.
- **Last whole-task evidence:** none. `T0` unattempted.
- **Days since last outing:** n/a — no outing has occurred.
- **Due checks:** none yet. The first delayed check is the 24-hour delayed cull inside Task 3 of the entry diagnostic.
- **Next whole-task block:** `T0` local micro-trip, phone, 45 minutes, unfamiliar nearby place, 20+ frames with intent/coverage/ethical-access fields and a delayed 5–7 frame sequence. This is Task 3 of the entry diagnostic and is the item most likely to be skipped.
- **Open commitments:**
  1. Run the entry diagnostic, committing the four predictions in writing first.
  2. Confirm `tools/verify_shot.py` catches a deliberately introduced intent mismatch on your own set (Phase 0 harness gate; verified working on synthetic frames, not yet on yours).
  3. After `T0`, open the `G0` travel-camera time box, buy the cheapest camera that clears the declared travel contract, and complete its return-window acceptance test.
  4. Decide whether to install `exiftool` — required for raw verification, unnecessary while shooting on a phone.
  5. Spend the declared `RM-R` tooling exception — one encoding block in Phase 1 or Phase 2 to build `tools/compare_render.py`. Unspent by the Phase 4 gate, `D4`/`D5` cap at `independent`.

## Setup record — 2026-08-30 — curriculum construction

- **Session card:** no learner session. Repository construction: curriculum design, dependency graph, vocabulary, operating contract, tutor prompt, critique rubric, verification harness, and entry diagnostic.
- **Prediction:** not applicable; no learner attempt.
- **Evidence:** `CONTEXT.md`, `photography-curriculum.md`, `photography-dependency-graph.md`, `phase-0-entry-diagnostic.md`, `rubrics/image-critique-rubric.md`, `reference/vocabulary.md`, `reference/gear-decision-rubric.md`, `tools/verify_shot.py`, `tools/intent-template.toml`, `AGENTS.md`, `.pi/APPEND_SYSTEM.md`. Harness verified end to end on two synthetic frames: a matching control frame and a deliberate multi-dimension mismatch, with all eight checks firing correctly and depth-of-field arithmetic confirmed by hand.
- **Actual:** no learner-performance claim. All node states remain `not-assessed`.
- **Assistance:** not applicable.
- **Attempt errors:** none classified; no learner attempt occurred.
- **Node-state transitions:** none.
- **Calibration gap:** not applicable.
- **Due checks / whole-task status:** entry diagnostic outstanding; `T0` unattempted.
- **Decision / next smallest action:** commit the four entry-diagnostic predictions in writing, then complete Task 1 (closed-resource optics, about 15 minutes, no equipment). Task 3's baseline shoot follows on a separate day.
- **Graph or curriculum change:** initial version 0.1. Design decisions and their rationale are recorded in the design log in `photography-curriculum.md`; the known feedback-integrity weakness and its trigger are recorded there and in the graph's feedback-integrity ceiling table.

## Design revision — 2026-08-30 — travel-photography specialization

- **Evidence:** explicit learner request to specialize the curriculum for travel photography and confirmation that a camera will be purchased. No performance evidence was added; all node states remain `not-assessed`.
- **Changed outcome:** a concise travel story made at an unfamiliar destination with one carryable kit under time, light, access, safety, ethical, power/media, and delivery constraints.
- **Graph change:** added `TR1` field readiness, `TR2` rapid orientation, `TR3` travel coverage, `TR4` ethical access, `TR5` data stewardship, and `TR6` sequence editing; rewrote `T0`–`T5` as progressively less-scaffolded travel simulations.
- **Assessment change:** added travel-story critique, packed-to-first-frame timing, ethical critical failures, and two-copy recovery evidence while preserving separate technical and perceptual judgement.
- **Gear change:** `G0` now ends in a purchase and acceptance test, with a one-kit carry envelope and compact everyday lens prioritized; the phone remains only for the baseline diagnostic.
- **Audit refinement:** every travel task now declares opportunity duration and whether return is possible; field-safe timestamped group intent has a strict scope/stop condition; weather transitions and sampled restore checks are operational gates; context/caption accuracy is scored at set level; culturally informed human review now triggers by Phase 2 exit or before the first real-trip set.
- **Next smallest action:** commit the four entry-diagnostic predictions, then complete Task 1. Task 3 is a local micro-trip and must occur before the `G0` purchase time box opens.

## 2026-08-31 — Phase 0 entry diagnostic — desk (in progress)
- Session card: locate the frontier through the closed-resource entry diagnostic; first block is Task 1 optics/exposure.
- Blocked constraint in force: no references, calculator, camera, or tutor hints.
- Prediction: total score 10/14; weakest task 4; time 30 min; expected dominant error code `K`.
- Evidence: learner's written pre-task prediction; Task 1 responses; Task 2 Frame 1 response; Task 2 Frame 3 response, initially labelled Frame 2 and transparently corrected as an interface-label error before feedback.
- Actual: Task 1 scored 0/4; Task 2 scored 2/3 (light 0/1; camera-position description 1/1; subject-clause/story-role 1/1); Tasks 3–5 pending.
- Technical verification: Task 1.1 no settings supplied; 1.2 wrong depth-of-field magnitude (but correct direction after stepping back); 1.3 identified subject motion and camera shake but prescribed a slower shutter for shake; 1.4 incorrectly attributed changed perspective to focal length.
- Perceptual judgement: Task 2 light readings were unreliable: all six omitted the requested contrast ratio; Frame 5 omitted hardness; Frame 6 called visibly hard direct sun soft. Camera height/distance was expressed on at least four frames. All subjects were one-clause and each assigned role was defensible.
- Assistance: assessment mode; no task hints given before the committed answers; correction given afterward.
- Attempt errors: `K` — stops/settings and depth-of-field magnitude unavailable; `K` — camera-shake correction unavailable; `M` — focal length treated as changing perspective; `K` — contrast-ratio and hard/soft discrimination unavailable in Task 2.
- Node-state transitions: O1, O2, O4, O5, G3, O7 `not-assessed → not-encoded`; L1, L2 `not-assessed → not-encoded`; V1, V2 `not-assessed → scaffolded`, all from the diagnostic evidence above.
- Calibration gap: pending diagnostic result.
- Days since last outing / weekly hours: n/a / pending.
- Due checks / whole-task status: Task 3 delayed cull due at least 24 h after its future outing; `T0` unattempted.
- Decision / next smallest action: Task 2 perceptual-baseline frames must be supplied before scoring; retain Task 1 correction for later blocked practice.
- Graph or curriculum change: none.

## Design revision — 2026-09-01 — reference-matching protocol, calibration band, state deduplication

- **Evidence:** curriculum review. No learner performance evidence was added; no node state changed.
- **Reference-matching specified.** It was cited as load-bearing in seven places across the curriculum, graph, and `AGENTS.md` while defining no target format, metric, or tolerance, and no harness check measured it — yet it was the sole mechanism lifting `V5`/`V7`/`D4`/`D5` above the feedback-integrity ceiling. Added [`rubrics/reference-matching-protocol.md`](rubrics/reference-matching-protocol.md): three integrity properties, four task classes (`RM-F`/`RM-D`/`RM-M` runnable now, `RM-R` pending tooling), measured tolerances, and pass/advancement rules. The graph's ceiling table now conditions the `transfer` claim on that protocol.
- **Tooling exception declared.** `tools/compare_render.py` is exempt from the tooling cap for one encoding block in Phase 1 or 2. Unspent by the Phase 4 gate, `D4`/`D5` cap at `independent` rather than being passed by eye.
- **Calibration band declared.** "The declared band" was referenced by two Phase 2 gates and one revision trigger and defined nowhere, making those gates discretionary rather than binary. Now defined per measure, tracked **signed**, with a separate **discrimination check** so a flat prediction cannot pass on mean gap alone. The rubric-score band carries an explicit statement that it measures agreement with a known-unreliable evaluator; the keeper-rate band does not share that defect.
- **State deduplication.** `photography-curriculum.md` and `photography-dependency-graph.md` both restated learning phase and frontier, contradicting `AGENTS.md` and `README.md`'s own responsibility table; the version number had already drifted (0.3 here, 0.2 in this log). Both headers now point at this file; the curriculum is the sole version owner, at 0.4. The graph keeps per-node evidenced state, which it owns.
- **Not changed:** no node state, no gate threshold other than those newly declared, no phase ordering, no north star.
- **Decision / next smallest action:** unchanged — supply the Task 2 frames and continue the entry diagnostic.

## Design revision — 2026-09-01 — runnable diagnostic, floored score bands, Phase 0 outing cadence

Second review pass. No learner performance evidence was added; no node state changed. Curriculum now at version 0.5.

- **Entry diagnostic made runnable.** Tasks 2 and 4 both instructed the tutor to "supply" photographs it has no way to supply. Task 2 now specifies a sourcing procedure — bulk pool collected without study, tutor selects six, revealed one at a time — and states the residual weakness plainly, since the learner assembling the pool has seen thumbnails. Task 4 is rebuilt around **frames whose fault is settled by the file**: unselected frames from the pre-curriculum camera roll, diagnosed from the image alone, then checked against EXIF, 100% inspection, and measured clipping. Selecting frames for a known fault hands over the answer with the material; the guaranteed one-of-each version moves to the Phase 1 entry, where real miss-frames exist.
- **Diagnostic calibration gap now routes a response.** The scoring table read the total only and discarded the signed gap. Gaps beyond ±4 out of 14 now trigger per-item prediction for the first macrocycle — the standing three-cycle revision trigger applied once at entry, since that trigger cannot fire for three cycles and the signal exists now.
- **Pre-declared score bands floored.** Distinct from the calibration band declared in the previous pass, and initially conflated with it in the `T2` gate — corrected. The `T2` and `T4` gates require an edit to clear a band the learner declares, which without a floor is self-approving. Floors are now stated in the travel-story rubric's own anchors: mean ≥ 1.5 per scored dimension at Phase 2, ≥ 2.0 at Phases 4 and 5, with ethical integrity held outside the mean.
- **Phase 0 outing cadence made explicit.** Every Phase 0 artifact is desk work, so the phase would run three to five weeks against a single `T0` and fire the two-week no-outing trigger while satisfying all its own gates. The weekly outing is now a named repeat phone micro-trip, which also supplies the second unprompted-protocol outing the Phase 0 scorecard already required.
- **Harness marks tool-supplied thresholds.** `verify_shot.py` silently defaulted `max_depth_m`, `max_clipped_highlight_pct`, `max_clipped_shadow_pct`, and the full-range limit when the intent omitted them, then reported a verdict against a target never committed to. Defaults still apply but now print `[tool default; declare …]`. Verified on a synthetic EXIF frame; depth arithmetic and shake convention re-checked and correct.
- **Diagnostic status** gained an explicit `in progress` state, since the assessment spans at least two days by design.
- **Decision / next smallest action:** unchanged — source the Task 2 pool and continue the entry diagnostic.

## Design revision — 2026-09-01 — calibration workflow trimmed

Third pass, approved in review. No learner performance evidence; no node state changed. Curriculum now at version 0.6.

- **Rubric prediction collapsed to totals.** The learner now commits one predicted total per frame and one per set, not eight dimensions per frame. **The tutor still scores every applicable dimension** — per-dimension scoring is what routes a remedy to a node and is unchanged. Only the prediction collapses. One exception preserved: when a single dimension is the active frontier, it is predicted separately, because targeted calibration on the thing being trained earns its cost where a blanket sweep does not.
- **Session-duration prediction dropped** from the diagnostic, the session card, the assessment stack, and the progress-log format. It routed no remedy. Time targets that *are* performance thresholds — first-intentional-frame, seconds-to-correct-settings, the `T2` field limit — are untouched and still committed in advance; `AGENTS.md` now states that distinction so the tutor does not re-introduce the prediction.
- **Discrimination promoted to a first-class field** in both rubrics, since a per-frame total makes the top-and-bottom check natural. In the travel-story rubric the predicted weakest transition and predicted mandatory cut are now named as the discrimination instrument, and matter more than the sequence total.
- **The rubric gap's limit is now stated wherever the gap is defined** — `CONTEXT.md`, both rubrics, `AGENTS.md`, `README.md`, and the curriculum. Keeper-rate calibration is the strong measure; rubric calibration measures agreement with this system's own evaluator. The tutor is instructed not to present it with the authority of the harness.
- **Rationale:** eight predictions per frame does not survive a third track at 2–4 h/week, and the failure mode is abandoning the whole practice rather than trimming it — which would have taken keeper-rate calibration and the error-code hypothesis down with the ceremony.
- **Decision / next smallest action:** unchanged — source the Task 2 pool and continue the entry diagnostic.
