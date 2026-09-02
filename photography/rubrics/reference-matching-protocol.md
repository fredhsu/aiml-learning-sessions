# Reference-matching protocol

Reproducing a specified target rendering, where deviation from the target is **measured rather than judged**. This is the mechanism that converts part of perceptual practice into verifiable ground truth, and it is the only reason `V5`, `V7`, `D4`, and `D5` can claim `transfer` while every other open perceptual node is capped at `independent`.

That claim rests entirely on the properties below. If a task does not satisfy all three, it is a shooting exercise, not a reference-matching task, and it advances nothing above the cap.

## The three properties that make the target ground truth

1. **The target is external.** Neither the learner nor the tutor authors it at scoring time. Legitimate sources: a published photograph the learner did not make; a frame from the learner's own archive at least three months old, nominated by its file name before it is re-opened; or a numeric target spec written down before the outing. A target the tutor invents while scoring is not external, and a target the learner adjusts after seeing the attempt is not fixed.
2. **The target is fixed before the attempt.** Every declared dimension and every tolerance is written into the outing's intent record before the frames are made. Tolerances chosen after seeing the result destroy the measurement, exactly as they do everywhere else in this curriculum.
3. **The deviation is measured, not scored.** Each dimension resolves to a number and a tolerance. If a dimension cannot be measured, it does not belong in a reference-matching task — put it in the image-critique rubric instead, where it stays capped.

## What reference-matching does and does not certify

It certifies **execution of a specified rendering**. Matching a framing proves you can put the camera where a framing requires; it does not prove you can choose that framing. Matching a target's tone curve proves you can drive the raw to a stated destination; it does not prove the destination was worth reaching.

This is precisely why it can carry `transfer` where open work cannot: the question it answers has an answer that exists outside the system. Selection, story, and interpretation remain open perceptual judgement and remain capped at `independent`. Never let a passed reference match advance an open node.

## Task classes

Four classes. Each names what is measured, how, and its default tolerance. Defaults may be tightened or loosened per task, but the value in force is declared before the attempt.

### RM-F — Framing match

Reproduce a specified framing of an available subject. Available from Phase 2; no new tooling required.

| Measured | How | Default tolerance |
|---|---|---|
| Subject size in frame | Subject's longest dimension as a percentage of the frame's corresponding dimension, measured on the exported frame | ±8 percentage points |
| Camera height | Tape or declared body landmark, in centimetres from ground | ±15 cm |
| Subject distance | Tape, pace count, or focus distance from EXIF | ±15% |
| Horizon or principal edge placement | Position as a percentage of frame height or width | ±5 percentage points |
| Subject centroid | Position as a percentage of frame width and height | ±8 percentage points |

Declare between three and five of these per task. Declaring all five at once is a Phase 3 form.

### RM-D — Depth match

Reproduce a specified depth rendering. Available from Phase 2; measured by the existing harness.

| Measured | How | Default tolerance |
|---|---|---|
| Near limit of acceptable sharpness | `tools/verify_shot.py` depth-of-field arithmetic from EXIF, sensor format, and subject distance | ±20% of the target near limit |
| Far limit | Same | ±20%, or both "beyond hyperfocal" |
| Total depth | Same | ±25% |

Where the target is a published photograph rather than a numeric spec, the target's own near and far limits must be derivable from its stated or visible settings. If they are not, the target is unusable for `RM-D` — do not estimate them by eye.

### RM-M — Motion match

Reproduce a specified motion rendering. Available from Phase 2; partly measured by the existing harness.

| Measured | How | Default tolerance |
|---|---|---|
| Shutter speed | EXIF against the target's shutter | ±⅓ stop |
| Declared motion outcome | Harness `motion.subject` verdict against the declared `freeze` or `blur` intent | Binary; must match |
| Blur extent, where the target specifies one | Length of the blur trail as a percentage of frame width, measured on the export | ±30% relative |

Blur extent is the loosest tolerance here on purpose: subject speed is not under your control, and a tighter band would be measuring the subject rather than the photographer.

### RM-R — Rendering match (tone and colour)

Reproduce a specified tonal and colour rendering from your own raw file. Phase 4 onward. **This class requires tooling that does not exist yet** — see *Tooling budget* below.

| Measured | How | Default tolerance |
|---|---|---|
| Patch luminance | L\* of each declared sample patch, sRGB export at a fixed size | ΔL\* ≤ 4 per patch |
| Patch colour | ΔE76 in CIE L\*a\*b\* per declared patch | ΔE ≤ 6 per patch |
| Black point | L\* of the darkest declared patch | ΔL\* ≤ 3 |
| White point | L\* of the brightest declared patch | ΔL\* ≤ 3 |
| Declared neutral | a\* and b\* of a patch declared neutral | within ±3 of zero on both |

Declare five patches before developing: a shadow, a midtone, a highlight, the subject's key tone, and a neutral. Patch locations are recorded as frame coordinates in the intent record and may not be moved after the comparison is run.

`RM-R` compares your development of your own raw against a target rendering. It is not an instruction to reproduce someone else's photograph — the subject is yours, the destination is theirs.

## Protocol

1. **Declare the target and the tolerances** in the outing's intent record, alongside the shot intent. Name the target source and the specific dimensions in force.
2. **Predict the deviation** on each declared dimension before the attempt. This feeds the calibration record like any other committed prediction.
3. **Attempt.** Reposition, re-shoot, and re-develop as much as you like — iterating toward a fixed target is the exercise, not cheating.
4. **Nominate one frame** for measurement, before any measurement is run. A nominated frame cannot be swapped after its numbers come back.
5. **Measure.** Run the harness for `RM-D` and `RM-M`; measure `RM-F` on the export; run the comparison for `RM-R` once it exists.
6. **Record** predicted deviation, actual deviation, and pass or fail per dimension in `curriculum-progress.md`.

## Pass and advancement rules

- A task **passes** when every declared dimension is within its declared tolerance on the nominated frame. Dimensions are not averaged; a miss on one is a fail on the task.
- A pass on a familiar target and subject evidences `independent` on the matched node.
- **`transfer` requires a pass on a materially different target and subject from the one practised** — a different framing problem, a different depth or motion regime, a different tonal destination — with no rehearsal on that target.
- `delayed-secure` requires a further pass at 7–14 days on a target not seen during the interval.
- The nodes reachable this way are `V5`, `V7`, `D4`, and `D5`, and only on the dimension actually matched. No other node advances from a reference match.

## Cadence

Once per macrocycle from Phase 2, per the assessment stack in [`photography-curriculum.md`](../photography-curriculum.md). More often is tooling substitution wearing a rubric.

## Tooling budget

`RM-F`, `RM-D`, and `RM-M` are runnable now: they need the existing `tools/verify_shot.py`, a tape measure, and measurement on an export.

`RM-R` needs a comparison tool — provisionally `tools/compare_render.py`, taking a target image, a candidate export, and declared patch coordinates, and reporting ΔL\* and ΔE76 per patch. **It does not exist.** Building it is a **declared one-time exception to the tooling cap**, budgeted at a single encoding block in Phase 1 or Phase 2, and recorded as such in `curriculum-progress.md` when it is spent. The exception exists because this tool is the only thing standing between the `transfer` claim on `D4` and `D5` and an unsupported assertion.

If the exception is not spent before the Phase 4 gate, the correct response is to **drop the `transfer` claim for `D4` and `D5` and cap them at `independent`** — not to substitute a visual comparison. A rendering match judged by eye is an image critique, and image critique cannot lift a node above the cap.
