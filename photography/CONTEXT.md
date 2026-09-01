# Photography Curriculum

This repository operates an evidence-adaptive curriculum for photography. Its language separates curriculum design, learner progression, and evidence from individual attempts, so that each can change without being mistaken for another.

Photography adds one distinction the symbolic-domain curricula do not need: **technical execution** and **perceptual judgement** have different ground truth, different failure signatures, and different feedback integrity. Conflating them is the characteristic failure of self-taught photography, and most of this vocabulary exists to keep them apart.

## Language

**Design stage**:
The maturity of the curriculum design itself, such as draft, confirmed, or under revision.
_Avoid_: Phase, when referring to curriculum design

**Learning phase**:
One of the learner-facing curriculum phases 0–5. It describes the current body of work, not demonstrated competence.
_Avoid_: Design phase, current level

**Node state**:
The strongest evidence currently held for a bounded capability: `not-assessed`, `not-encoded`, `encoded`, `scaffolded`, `independent`, `transfer`, or `delayed-secure`.
_Avoid_: Error code, percentage, "I've done that"

**Attempt error**:
A `K/R/M/D/P/F/T/C` diagnosis attached to a particular miss. It selects the next remedy but is not a persistent node state.
_Avoid_: Node status, learner trait, "bad photo"

**Prerequisite edge**:
A capability dependency: the target cannot yet be attempted responsibly without the source capability.
_Avoid_: Preferred order, phase ordering

**Sequence constraint**:
A deliberate teaching order that is not a capability dependency. It may be changed without claiming that one capability logically requires the other.
_Avoid_: Prerequisite

**Integration requirement**:
Evidence that multiple prior capabilities can be combined in a whole task. It belongs to a milestone or exit gate rather than to the prerequisite graph.
_Avoid_: Prerequisite edge

**Exit gate**:
A binary, evidence-backed requirement for advancing a learning phase. Every gate names its artifact, verification, scaffold level, transfer condition, and delay.
_Avoid_: Weeks elapsed, frames shot, tutorials watched

## Photography-specific distinctions

**Shot intent**:
A pre-shutter commitment recorded before the frame is made: subject, why it is a picture, intended depth rendering, intended motion rendering, intended exposure placement, and the predicted settings. A timestamped group intent may cover an unfolding action only while subject/relationship, light, position/technical strategy, and its declared stop condition remain unchanged. Intent is what makes a frame assessable. A frame with no recorded intent is exposure, not evidence.
_Avoid_: Caption, description, post-hoc explanation

**Technical verification**:
Machine-checkable agreement between shot intent and the recorded file: EXIF settings versus predicted settings, computed depth of field versus intended depth rendering, shutter speed versus motion intent and focal length, clipping and exposure placement versus intended placement. This is ground truth and is **not** a matter of opinion.
_Avoid_: Critique, feedback, "the tutor said"

**Perceptual judgement**:
Whether the picture works — subject clarity, frame edges, camera position, use of light, moment, tonal structure, and interpretation. This has no automated ground truth in this system. It is assessed only through the fixed rubric in `rubrics/image-critique-rubric.md`, always against a committed prior prediction.
_Avoid_: Technical verification, score, "good photo"

**Calibration gap**:
Predicted keeper rate, or a predicted rubric total, minus the actual result. Always **signed** — a consistently optimistic learner and an alternating one share an absolute gap and have different problems. Always paired with a **discrimination check**, because a flat prediction repeated across a set produces an excellent mean gap while carrying no information about any individual frame.

The two kinds are not equally trustworthy. **Keeper-rate calibration** compares a committed prediction against machine ground truth and is the strong one. **Rubric-score calibration** compares it against this system's own critique, so it measures agreement with a known-unreliable evaluator: narrowing it is consistent with learning to see and equally consistent with learning to predict the tutor, and nothing here can separate those. Bands and floors are declared in `photography-curriculum.md`.
_Avoid_: Accuracy, score, treating the rubric gap as ground truth

**Keeper rate**:
Frames that meet their own recorded shot intent, divided by frames made. Judged against intent, never against a general standard of quality. A low keeper rate is a diagnosis, not a verdict.
_Avoid_: Hit rate for "good" images, portfolio rate

**Blocked constraint**:
A single fixed variable — one focal length, one aperture, one subject class, one distance, one light condition — held constant across an entire outing so that a coordination pattern can stabilise before variation is introduced.
_Avoid_: Assignment, theme, project

**Reference-matching task**:
Reproducing a specified target rendering — a given framing, depth rendering, motion rendering, or tonal interpretation — where deviation from the target is measurable. This is the mechanism that converts part of perceptual practice into verifiable ground truth.
_Avoid_: Copying, imitation exercise

**Light condition**:
A named, curated entry in the light pattern library: direction, hardness, colour, contrast ratio, and how it renders a subject. The library is the stored-pattern component of this domain and is built from the learner's own annotated frames.
_Avoid_: Lighting setup, weather

**Camera fluency**:
Executing an intended exposure and focus decision without conscious attention to the controls. This is the photographic analogue of line confidence in drawing: it is not knowledge and cannot be tested by explanation, only by timed unaided execution.
_Avoid_: Knowing the exposure triangle, understanding the controls

**Travel story**:
A deliberately edited set that conveys a specific place or journey through complementary frame roles rather than a pile of individually strong images. Coverage may include orientation, context, people, detail, transition, and closure, but the final sequence is judged by what the story needs rather than by checklist completion.
_Avoid_: Portfolio, trip dump, one-of-each shot list

**Coverage role**:
The job a candidate frame performs inside a travel story: orienting the viewer, establishing context, revealing human presence, isolating a telling detail, carrying transition or movement, or closing the sequence. A role is committed before capture when practical and may be revised during editing with the revision recorded.
_Avoid_: Genre label, composition rule

**Field readiness**:
The ability to leave with a compact working kit, arrive with power/media/time/access constraints understood, and make the first intentional frame without preventable setup failure. It includes carry discipline, preflight, weather protection, personal and gear security, and a declared fallback.
_Avoid_: Gear ownership, packing enthusiasm

**Ethical access**:
A pre-capture decision that local access and photography restrictions have been checked to the degree available, the photograph is safe for photographer and subject, culturally defensible, and consented where the subject or context requires it. Recheck destination-specific facts rather than generalising from home. A technically or aesthetically strong frame cannot compensate for an ethical-access failure.
_Avoid_: Legal advice, blanket permission to photograph strangers

**Data stewardship**:
The field-to-home procedure that preserves travel work: verify ingest, create two independent copies before card reuse, retain originals and metadata, and record backup status. A copied file is not backed up until a second copy has been verified.
_Avoid_: Editing workflow, cloud-sync assumption
