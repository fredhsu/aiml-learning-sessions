# Phase 0 entry diagnostic

**Design stage:** confirmed assessment design
**Learning phase:** Phase 0 — measurement and inference arithmetic
**Status:** **pending.** No node state in [`inference-dependency-graph.md`](inference-dependency-graph.md) is evidenced until this runs. This document is an assessment design, not a record of performance.

## Purpose

Locate the actual frontier. The learner has roughly eight months of deep-learning study, an implemented micrograd, and JAX autodiff experience. **None of that establishes inference competence, and some of it actively misleads** — training intuitions invert at inference time (see the misconception bank in [`inference-curriculum.md`](inference-curriculum.md)).

This diagnostic is built to distinguish four things that a topic-level self-assessment cannot:

- **covered vs. secure** — the transformer is familiar; the *cost model* of running one may not be;
- **conceptual vs. procedural** — knowing what a KV cache is, versus writing one with correct offsets;
- **knowledge vs. misconception** — an absent model produces `K`; an inverted training-derived model produces `M`, and the two need opposite remedies;
- **framework transfer** — JAX fluency does not carry to PyTorch's in-place, mutable inference idiom.

It runs in three blocks across two or three sessions at the 2–3 h/week dose. Do not compress it into one sitting; the fatigue would confound the result.

## Conditions

1. **Commit predictions before each block:** the score out of the block's points and an elapsed-time estimate, written down before reading the block's tasks in detail. This is a multi-item assessment, which is where score prediction is informative; per-session score prediction is not collected (see the session template in [`inference-curriculum.md`](inference-curriculum.md)). Instead of one confidence percentage for the block, record a **binary bet per item** — would you stake the point on it, yes or no — so that a hit rate can be computed per answer rather than read as a mood.
2. **Closed resource** for Blocks A and B, except as explicitly permitted. Documentation lookup for exact PyTorch API spelling is allowed **only after** writing the intended operation, its shapes, and its dtypes.
3. **Commit each answer before running anything.** For Block B, the diagnosis must be committed to the repository before the code is executed. Preserve the first attempt so that `R`, `P`, and `C` can be told apart afterwards.
4. **The tutor supplies checks and reference values only after the learner commits.** During this diagnostic the tutor is assessing, not teaching: no hints that leak a mechanism, and no correcting an in-progress attempt.
5. **Definitional unlocks are permitted and recorded.** Item A1 names, per sub-item, a convention that cannot be derived and was never taught. Releasing one on request is not a hint: it supplies a definition, never a derivation, an intermediate value, or a direction of reasoning. Each release is recorded as `scaffolded` for that sub-item only. A block that halts because the learner had no way to begin produces one bit of information; a block that proceeds with unlocks recorded produces seven.
6. **Measured hardware constants are supplied, not assessed.** Achievable bandwidth and matmul throughput come from `bench/constants.md` under their contract versions. H1 is established by measurement, not by recall, so an A1 sub-item is never scored on whether the learner remembers a number that lives in a table.
7. **No number without a contract**, from the first measurement onwards. Block C is invalid if its result is reported without the contract it ran under.

---

## Block A — arithmetic and measurement discrimination

*Closed resource. Target ~40 minutes. 2 points.*

### Item A1 — napkin math, link by link

For this configuration, in `bfloat16`, on the RTX 4090:

```
layers            = 16
d_model           = 2048
attention heads   = 32      (head_dim = 64)
KV heads          = 8       (grouped-query)
FFN intermediate  = 8192    (gated: gate, up, down)
vocab             = 128256  (tied embedding / output)
```

Compute the following, showing the derivation rather than only the result. **Each of the seven is scored independently.** They are chained, so the rule below protects the chain:

> **Error carried forward is not penalised twice.** Commit an answer to a sub-item, then ask for the reference value before starting the next one. Work the next sub-item from the reference value, not from your own. A wrong link ends that sub-item, not the block.

Each sub-item also names a **definitional unlock**: a convention you cannot derive and were never taught. Ask for it if you need it. Requesting an unlock is not a failure and does not zero the sub-item — it is recorded as `scaffolded` for that sub-item, which is exactly the distinction this diagnostic exists to draw. Guessing in place of asking destroys that information.

| # | Compute | Definitional unlock, on request |
|---|---|---|
| A1.1 | Total parameter count and parameter bytes | Which matrices exist per layer, and that a tied embedding is counted once |
| A1.2 | KV cache bytes **per token**, and total KV bytes for 4096 tokens of context | The KV cache stores one key vector and one value vector per KV head, per layer, per token |
| A1.3 | Bytes moved per decode step at batch 1 with 2048 tokens of context | A decode step reads every weight once and reads the whole KV cache for the attended prefix; it writes one token's KV back |
| A1.4 | Arithmetic intensity of that decode step, in FLOPs per byte | A matrix–vector product of an `m × n` weight costs `2mn` FLOPs and moves `mn · bytes_per_elem` bytes |
| A1.5 | An upper bound on decode tokens/s at batch 1, naming the hardware number that bounds it and where the number came from | The bound is `achievable_bandwidth / bytes_per_step`; the achievable number is not the spec-sheet number |
| A1.6 | The batch size at which decode stops being memory-bandwidth-bound | The crossover is where arithmetic intensity meets the device's FLOP-per-byte ratio; weights are read once per step regardless of batch, KV is not |
| A1.7 | The maximum concurrent sequences at 4096 context that fit in 24 GB alongside the weights | Nothing withheld — this is A1.1 and A1.2 combined against a capacity budget |

**Tests:** A1, A2, A3, H1, H3.

**What a failure means.** Read the *first broken link*, not the aggregate — that index is the routing signal.

- A stall at A1.1–A1.3 with no unlock requested and no structure attempted → `K` on A2/A3; this is the frontier and Phase 0 starts here.
- A sub-item that succeeds immediately after its unlock → `K` on the convention only, not on the arithmetic. This is the cheapest gap in the block to close and should not be confused with the one above.
- Structure correct but reaching for FLOPs where bytes are needed, most visibly at A1.3 or A1.5 → `M` (misconception 1 or 2), which needs contrast cases, not more explanation.
- Correct but slow and effortful → `F`, which needs timed drilling after accuracy is established.
- A1.5 answered from a spec-sheet number without knowing whether it is the achievable one → note it, and see the H1 note in Conditions; it is not scored as a miss here.

### Item A2 — confound identification

For each benchmark description below, name (a) the confound, (b) the **direction** of the resulting error, and (c) the smallest change that would fix it.

1. A script loads the model, immediately times 20 generations of 128 tokens each, and reports mean tokens/s.
2. Two engine configurations are compared by sending the same 200-prompt set to each, in the same order, on a server with prefix caching enabled.
3. Throughput is reported as output tokens per second, where output tokens are counted by splitting the returned string on whitespace.
4. A kernel change is timed with `t0 = time.time(); out = model(x); t1 = time.time()`, and reported as a 40× speedup.
5. A latency SLO is validated by reporting mean end-to-end latency across a workload mixing 50-token and 2000-token generations.
6. An optimisation is validated by running the new configuration immediately after a 20-minute sweep of the old one, on an air-cooled GPU.

**Tests:** M1, M2, M3 (item 4 specifically).
**What a failure means.** Fewer than four correct → `K` on M1/M2, and Phase 0's measurement block is the frontier. Correct identification but wrong error direction → `M`. Correct on all but slow → fine; these become fluent through use.

---

## Block B — implementation and debugging

*Closed resource, except for API spelling after the operation is written. Target ~50 minutes. 2 points.*

### Item B1 — from-memory implementation

In PyTorch, from this contract alone and without consulting a reference implementation, write:

- a `KVCache` that preallocates for a fixed maximum batch and context, exposes an append for one decode step, and maintains an explicit length invariant;
- a single-layer cached attention forward using the Block A config's GQA shape (32 query heads, 8 KV heads, head_dim 64), correct causal masking with a cache present, and correct position handling for a decode step at arbitrary offset.

Before running anything, write down the shape and dtype of every intermediate tensor, and the invariant that must hold between the cache length and the position index.

Verify by comparing against full-recompute attention over the whole prefix, at a tolerance you declare before running.

**Tests:** E1, E2, and PyTorch idiom transfer.
**What a failure means.** No implementation → `K`, expected and unremarkable; Phase 1 is the frontier. Functional-style code that returns a new cache where an in-place buffer write is required → `P`, plus the JAX-idiom blind spot in the graph; the remedy is a faded skeleton, not theory. Correct shapes but wrong position/offset handling → `P` on the specific invariant. Declining to declare a tolerance → `C`, and a signal that parity discipline needs establishing explicitly.

### Item B2 — seeded debugging, diagnosed before execution

The tutor seeds the following faults, one at a time, into a working decode loop. **For each, commit before running:** the observable symptom, the invariant violated, the smallest repair, and the likely attempt-error code.

1. The position index passed to RoPE is derived from the current input length rather than the cache offset, so every decode step is treated as position 0.
2. The KV cache is written at the wrong offset for a left-padded batch, so shorter sequences read a neighbour's keys.
3. The GQA head mapping repeats KV heads with the wrong stride — `repeat` where `repeat_interleave` is required.
4. `torch.cuda.synchronize()` is absent from the timing path, so the measured decode time is implausibly small.
5. A "regression" in output text between two runs is actually sampling nondeterminism, not a numerical bug.

**Tests:** E1, E2, M3, and Family 6 (correctness illusions) discrimination.
**What a failure means.** Diagnosing 1–3 correctly but only after running → `R` or `P`, not `K`. Missing 4 → `M3` gap with direct consequences for every subsequent measurement; high priority. Treating 5 as a numerical bug → `M` (misconception 12's mirror image), and worth an explicit contrast pair, since the opposite error — dismissing a real bug as nondeterminism — is more expensive.

---

## Block C — authentic mini-task

*Open resource. Target ~45 minutes. 2 points.*

### Item C1 — predict, then measure, then explain the gap

1. Choose a small decoder-only open-weights model that fits comfortably on the 4090.
2. **Before running anything**, predict its batch-1 decode tokens/s using the Item A1 method, with a stated tolerance and the basis recorded as `arithmetic`.
3. Write a minimal but honest measurement: warmup, steady state, declared cache state, tokenizer-counted tokens, percentile-reported latency, fixed seed, at least 5 repetitions.
4. Record the workload contract in `bench/workload-contract.md`.
5. Measure.
6. Report the prediction gap and give **three ranked hypotheses** for it, each with the observation that would confirm or refute it.

**Expect the measurement to fall well short of the prediction.** That gap is the intended outcome of this item, not a failure — it is the curriculum's founding observation, and the ranked hypotheses are the real assessed output.

**Tests:** T0 vertical slice, M1, M2, A2, and calibration.
**What a failure means.** Measuring before predicting → the item scores zero regardless of the quality of the measurement; the prediction rule is the point. Hypotheses that are a list of techniques ("use vLLM", "quantise") rather than causes → `M`/`D`: the learner is reaching for remedies before diagnosis, which is the single most common failure mode in this field and the habit Phase 0 exists to break.

---

## Scoring and routing

| Point | Evidence required |
|---|---|
| 1 | Item A1: at least 5 of 7 sub-items correct **without an unlock**, with the reasoning visible. Record per sub-item: correct/incorrect, unlock requested or not, and the index of the first broken link. |
| 1 | Item A2: at least 4 of 6 confounds identified with the correct direction of error. |
| 1 | Item B1: cache and cached attention pass the declared-tolerance parity check against full recompute, written without a reference. |
| 1 | Item B2: at least 4 of 5 faults diagnosed **from invariants, before execution**, with a correct minimal repair. |
| 1 | Item C1: prediction committed before measurement, with contract, warmup, tokenizer-counted tokens, and percentile reporting all present. |
| 1 | Item C1: prediction gap reported with three ranked causal hypotheses, each naming a discriminating observation. |

- A score records task evidence, not node mastery. Record assistance per point as `scaffolded`, `independent`, or `transfer`.
- Classify every substantive miss with an attempt error before choosing a remedy. Record primary and secondary codes when both are supported.
- **Critical failures, which block their point regardless of aggregate score:** measuring before predicting in C1; reporting any number without its contract; claiming a parity result without declaring a tolerance.

## Expected shape of the result

Stated in advance so that the outcome is interpreted rather than rationalised. A plausible profile for this learner is: Block A correct in structure with gaps in the bytes-versus-FLOPs reasoning (`M` more likely than `K`), and unlocks requested at A1.2 and A1.3; Block B strong on tensor manipulation and weak on the in-place cache idiom and offset bookkeeping (`P`); Block C producing a large, initially unexplained gap.

The first attempt at Block A on 2026-08-31 did not match this shape: it halted before A1.1 with no derivation attempted. That outcome is what the sub-item decomposition and the definitional unlocks in Conditions 5–6 exist to convert into routable information; the re-attempt is on an alternate form.

If Block A comes out clean and fast, the Phase 0 encoding block shrinks to measurement only and the frontier moves to Phase 1 immediately. If Block B comes out clean, Phase 1 compresses to its whole task. **Do not run phases that the diagnostic shows are already secure** — but equally, do not skip a phase on the strength of a confident explanation without an executable artifact.

## Completion record

Append the result to `curriculum-progress.md` with artifact paths, commands and output, the workload contract version, actual score and time, prediction gaps, attempt errors, node-state transitions, calibration gap, and the next due delayed check.
