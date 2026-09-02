# Inference-Engineering Curriculum

This repository operates an evidence-adaptive curriculum. Its language separates curriculum design, learner progression, and evidence from individual attempts so that each can change without being mistaken for another.

The vocabulary below is shared with the robot-learning curriculum in `../q2`. The measurement terms at the end are specific to this domain, because in inference engineering the difference between a real result and a measurement artifact is the whole discipline.

## Curriculum language

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

## Measurement language

These distinctions exist because most published inference numbers are not comparable, and most self-measured speedups are confounded. The curriculum treats a number without its contract as having no evidential value.

**Workload contract**:
The complete, committed specification that makes a measurement meaningful: model and revision, dtype and quantization, hardware and driver, engine and version, input/output length distribution, request arrival pattern, concurrency, sampling parameters, cache state, warmup policy, seeds, and the tokenizer used for counting. Stored in `bench/workload-contract.md`.
_Avoid_: Benchmark, when only a script is meant

**Prediction**:
A quantitative, committed estimate of a metric with a stated tolerance, recorded **before** the measurement is run. Every measurement session commits one.
_Avoid_: Expectation, intuition, guess after the fact

**Measured result**:
A number produced under a named workload contract, at steady state, with the cache state declared. A result whose contract is unknown is an anecdote.
_Avoid_: Benchmark result, speedup

**Speedup claim**:
A ratio between two measured results whose contracts differ in exactly one declared dimension. A ratio between contracts differing in more than one dimension is an attribution error, not a speedup.
_Avoid_: Improvement, faster

**Bottleneck class**:
The diagnosed reason a workload is not faster, drawn from a fixed discriminative set: memory-bandwidth-bound, compute-bound, launch/CPU-overhead-bound, synchronisation-bound, queueing-bound, KV-capacity-bound, or host-side (tokenise/detokenise/serialise). Naming a bottleneck class without profile or arithmetic evidence is speculation.
_Avoid_: Slow, unoptimised

**Numerical parity**:
Agreement with an independent reference implementation on the same inputs, to a declared tolerance, under declared determinism settings. Parity is the correctness ground truth for every engine change; a performance result from an unverified engine is worthless.
_Avoid_: It works, output looks fine
