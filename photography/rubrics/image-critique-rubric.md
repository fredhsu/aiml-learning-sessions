# Fixed image-critique rubric

This rubric is the only permitted format for perceptual critique in this curriculum. It exists because unconstrained critique drifts toward agreeableness, and agreeable critique is worse than no critique — it gets mistaken for signal. Fixed dimensions with numeric anchors make critiques **comparable across months** rather than being a mood reading.

Applies from Phase 0 onward. Dimensions 7 and 8 are scored only when the relevant evidence exists. For an ordered travel edit, score its individual frames here first, then score the sequence separately with [`travel-story-rubric.md`](travel-story-rubric.md); strong-frame averages do not establish story quality.

## Protocol — order matters

1. **Learner commits predictions first.** **One predicted total per frame, and one predicted total for the set.** Written down before the frames are sent. **The tutor refuses to critique until the prediction exists.**

   Not per dimension. Predicting eight dimensions on every frame is ceremony that gets abandoned wholesale at 2–4 hours a week, and it takes the parts that carry real signal down with it. The gap on a total carries nearly all of the information at a fraction of the cost.

   **One exception:** when a single dimension is the active frontier — working `V3` means predicting dimension 2 — predict that dimension as well. Targeted calibration on the thing actually being trained is worth its cost; a blanket sweep is not.

   **The tutor still scores every applicable dimension.** Only the learner's prediction collapses. Per-dimension scoring is what routes a remedy to a node, and it does not change.
2. **Tutor scores blind where possible** — frames first, shot intent withheld. The tutor states whether it scored blind.
3. **Intent revealed.** The tutor then reports whether the frame achieved *its own stated intent*, separately from the blind scores. A frame can score well and fail its intent, or the reverse; both are informative.
4. **Calibration gap computed** on the frame totals and the set total — signed, never absolute — plus the discrimination check, and logged to `curriculum-progress.md`. Bands are declared in [`photography-curriculum.md`](../photography-curriculum.md).
5. **One strongest change** named per frame, and one **recurring pattern** named per set.

The calibration gap is the primary number here. A score you predicted accurately carries real information even when the absolute score is generous; a score you did not predict carries almost none.

But it is not ground truth, and the curriculum says so plainly: this gap measures your agreement with a **known-unreliable evaluator**, and narrowing it is consistent both with learning to see and with learning to predict the tutor. Keeper-rate calibration, whose actual comes from the file, does not share that defect. Weight them accordingly.

## Scoring anchors

Every dimension is scored **0–3**. Anchors are behavioural, not adjectival.

| Score | Meaning |
|---:|---|
| 0 | Absent or actively working against the picture |
| 1 | Attempted, but the failure is the first thing a viewer notices |
| 2 | Sound; no distracting failure, but nothing is doing extra work |
| 3 | Deliberate and load-bearing; removing it would collapse the picture |

A 3 requires that the tutor can state what would break if the dimension were changed. If it cannot, the score is 2.

## Dimensions

### 1. Subject clarity
Can the picture be said in one clause, and does the frame say it? Does the eye arrive where it should, and stay?
- **0** — no identifiable subject, or two subjects competing with no resolution
- **1** — a subject exists but the frame does not commit to it
- **2** — the subject is unambiguous
- **3** — the frame is organised so that the subject reads before anything else, and would not survive rearrangement

### 2. Frame and edges
What is included and excluded, and what the edges do. Check every edge individually.
- **0** — unintended inclusions dominate; limbs, poles, or bright shapes cut at the edge
- **1** — one distracting edge intrusion or an awkward crop of a significant element
- **2** — edges are clean and nothing intrudes accidentally
- **3** — edges are doing work: containing, implying continuation, or creating tension deliberately

### 3. Camera position and perspective
Was the camera in the right place? Perspective is set by position, not by lens.
- **0** — eye-level default from wherever the photographer happened to stand
- **1** — a position was chosen but a better one was clearly available and unexplored
- **2** — the position serves the subject
- **3** — the position is the picture; from anywhere else the relationships collapse

### 4. Use of light
Direction, hardness, contrast ratio, and colour, and whether they were *used* rather than merely tolerated.
- **0** — light works against the subject and was not accounted for
- **1** — light is neutral; the picture would be unchanged in any other condition
- **2** — the light suits the subject and was chosen or waited for
- **3** — the light is a load-bearing element; the picture exists because of this condition

### 5. Moment and timing
Gesture, relation, and the instant of release. Score n/a for static subjects and say so.
- **0** — released without regard to the moment
- **1** — the moment is nearly there; a beat early or late
- **2** — the moment is correct
- **3** — the moment is unrepeatable and the picture depends on it

### 6. Tonal and value structure
**Assess at thumbnail size and in greyscale.** Errors in value mass and balance are invisible at working scale and obvious at 5% size.
- **0** — no readable structure at thumbnail; the image dissolves
- **1** — reads at thumbnail but the value masses fight each other
- **2** — clear tonal separation; the subject holds at thumbnail
- **3** — the value structure alone carries the picture without colour or detail

### 7. Technical execution
Scored **only** against the harness output and 100% inspection, never from the rendered image alone. This dimension has ground truth; the others do not.
- **0** — a technical failure destroys the frame: missed focus plane, unintended motion, unrecoverable clipping
- **1** — a technical compromise is visible and was not intended
- **2** — technically correct and matches the recorded intent
- **3** — the technical choices are the interpretation: depth, motion, and placement all deliberately serve the subject

### 8. Development and interpretation
Scored from Phase 4, or earlier if the frame was developed.
- **0** — the edit fights the frame, or is a rescue attempt on a frame the cull should have killed
- **1** — the edit is applied but generic; it would suit any image
- **2** — the edit realises the stated intent and stops there
- **3** — the interpretation adds a reading the raw file did not have, and every move is defensible

## Mandatory fields per frame

The tutor must complete all of these. None may be skipped, including for high-scoring frames.

| Field | Requirement |
|---|---|
| **Scored blind?** | Yes or no. If no, say why the intent could not be withheld. |
| **Strongest failure** | One specifically located failure. If none is found, state exactly what was checked and not found — never leave it empty. |
| **Located evidence** | Every claim names a region: "the right edge cuts the standing figure at the wrist", not "the composition feels unbalanced". |
| **Thumbnail test** | Does the frame hold at thumbnail and in greyscale? Yes or no. |
| **Intent achieved?** | Separately from the score: did the frame do what its recorded intent said? |
| **Single strongest change** | The one change that would most improve this frame, stated as an action at capture time, not as an edit. |

## Mandatory fields per set

| Field | Requirement |
|---|---|
| **Recurring pattern** | The failure appearing in the most frames, with the count. |
| **Attempt-error codes** | `K/R/M/D/P/F/T/C` for the recurring pattern, with rationale. |
| **Calibration gap** | Signed predicted minus actual on each frame total and on the set total, plus any frontier dimension predicted separately. |
| **Discrimination check** | Were the frames predicted highest and lowest actually the highest- and lowest-scoring? A set whose per-frame predictions are all the same number fails this regardless of its gap. |
| **Evaluator drift check** | Compare this set's mean scores with the previous three sets. If scores are rising while the calibration gap is not narrowing, say so and name it as drift, not progress. |

## Prohibited behaviours

The tutor must not:

- Open with praise, or lead any frame's critique with what works before what fails.
- Grade on effort, difficulty, conditions endured, or improvement over past work. A hard-won frame is not thereby a better frame.
- Use adjectival critique without located evidence — "striking", "moody", "well-balanced", "nice light" are all uninformative alone.
- Award a 3 without stating what would break if the dimension were changed.
- Score a technical dimension from a rendered image when the file and harness output are available.
- Soften a score because the learner has defended the frame. Hold the position, or change it for a stated reason — never to end the disagreement.
- Critique before the learner's committed predictions exist.

## Why this rubric and not a longer one

Eight dimensions is small enough to apply to every frame sustainably and large enough to route a remedy. A longer rubric produces more scoring and less looking. Dimensions are deliberately mapped to graph nodes so that a low score routes directly to a capability:

| Dimension | Routes to |
|---|---|
| 1. Subject clarity | `V1` |
| 2. Frame and edges | `V3` |
| 3. Position and perspective | `V2`, `O7` |
| 4. Use of light | `L2`, `L4` |
| 5. Moment and timing | `V6` |
| 6. Tonal and value structure | `V4`, `V5` |
| 7. Technical execution | `O3`–`O8`, `G1`, `G3` |
| 8. Development and interpretation | `D4`, `D5`, `D7` |
