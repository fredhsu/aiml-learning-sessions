# Robot-Learning Curriculum

This repository operates an evidence-adaptive curriculum. Its language separates curriculum design, learner progression, and evidence from individual attempts so that each can change without being mistaken for another.

## Language

**Design stage**:
The maturity of the curriculum design itself, such as draft, confirmed, or under revision.
_Avoid_: Phase, when referring to curriculum design

**Learning phase**:
One of the learner-facing curriculum phases 0–5. It describes the current body of work, not demonstrated competence.
_Avoid_: Design phase, current level

**Node state**:
The strongest evidence currently held for a bounded capability: `not-assessed`, `not-encoded`, `encoded`, `scaffolded`, `independent`, `transfer`, or `delayed-secure`.
_Avoid_: Error code, mastery percentage

**Attempt error**:
A `K/R/M/D/P/F/T/C` diagnosis attached to a particular miss. It selects the next remedy but is not a persistent node state.
_Avoid_: Node status, learner trait

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
_Avoid_: Week completed, resources consumed
