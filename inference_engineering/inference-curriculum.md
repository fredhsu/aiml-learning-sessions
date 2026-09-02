# Inference-Engineering Curriculum

**Version:** 0.1
**Design stage:** approved working curriculum; evidence-gated and revisable
**Learning phase:** Phase 0 — measurement and inference arithmetic (entry diagnostic pending)
**Dependency graph:** [`inference-dependency-graph.md`](inference-dependency-graph.md)
**Resource map:** [`resources.md`](resources.md)
**Track role:** secondary, 2–3 h/week. Primary track is `../q2` (robot learning).

## North-star performance

Given an open-weights decoder-only model, a single-GPU budget, and a declared workload and SLO, independently stand up a serving deployment that meets pre-registered latency and throughput targets on locked workloads; **predict its performance from first principles before measuring it**; diagnose regressions and shortfalls to a named bottleneck class from profiles and arithmetic rather than guesswork; and defend every optimisation choice with measured evidence, including at least one negative result.

Mechanism understanding is evidenced separately, by building an inference engine from scratch — KV cache, continuous batching, paged KV, quantised execution, speculative decoding — and comparing it against a production engine on an identical workload contract, explaining every gap.

- **Primary criterion:** a deployment that meets a declared SLO, and a correct causal account of why it does.
- **Supporting criterion:** the from-scratch engine exists to make that account possible. It is a teaching instrument, not a product; it is not required to beat vLLM, only to explain the difference.
- **Public criterion:** at least one write-up and one upstream contribution that a stranger reviews.
- **Retention target:** usable performance at 1, 3, and 12 months.

### The five evidence classes

| Class | What it requires |
|---|---|
| **Recall** | From memory: the decode-step byte/FLOP accounting, KV bytes per token for a given config, the acceptance-rate speedup model, the scheduler's states, the bottleneck-class list. |
| **Discrimination** | Given a shortfall, correctly select the bottleneck class and the matching remedy from the confusable set — and correctly reject the remedies that do not apply. |
| **Performance** | A deployment meeting a pre-registered SLO on a locked workload; engine subsystems passing parity and performance tests. |
| **Transfer** | The same competence exercised on a materially changed surface: the GB10's different memory hierarchy, a different model shape (GQA ratio, MoE, different depth/width tradeoff), or a changed workload mix (prefill-heavy vs decode-heavy). |
| **Retention** | 7–14 day delayed reconstruction of a subsystem or diagnosis; 4–12 week maintenance checks; a 6- and 12-month rebuild of a representative subsystem. |

## Domain typing and knob settings

Inference engineering is **Type 5 (integrative engineering) dominant**, on a **Type 1 (hierarchical-cumulative symbolic)** arithmetic foundation, with a substantial and often-underrated **Type 3 (pattern-recognition and search)** component in performance diagnosis.

This differs from the robot-learning curriculum's typing, and the knobs are set differently as a result.

| Knob | Setting | Why |
|---|---|---|
| Graph density | **Moderate** — deep in the arithmetic chain (A/H), broad and parallel in the systems layer (S/K/Q/D) | The systems capabilities are largely siblings, not a chain. Encoding them as a chain would invent false prerequisites. |
| Explicit SRS weight | **Very low** — one small table only | The single memorisation set is the hardware-constant and formula table in the Fluency section, and it is *measured on the learner's own bench*, not copied. Everything else is retrieved through implementation and prediction. |
| Worked-example weight | **High for the arithmetic, faded fast; high for kernels, faded slowly** | Arithmetic worked examples are cheap and expertise reversal arrives quickly. Kernel work stays example-heavy far longer; that is correct, not a failure. |
| Block → interleave | Short block per subsystem, then interleave **bottleneck diagnosis** aggressively | Diagnosis is a Type 3 discrimination skill and is the highest-value interleaving target in this domain. **[B]** |
| Feedback channel | **Automated and brutal** — wall clock, profiler, numerical parity, published reference numbers | The strongest feedback environment of any of the learner's tracks. Exploit it rather than substituting self-assessment. |
| Discovery permission | **Off** for arithmetic; **on** for profiling exploration once M3 is `scaffolded` | Reading traces rewards exploration; deriving the roofline does not. |
| Whole-task timing | **Immediately.** Phase 0's whole task is the harness | Type 5. Deferring integration here produces someone who can recite the KV formula and cannot tell a real speedup from a warm cache. |
| Fluency emphasis | **High** — napkin math, byte accounting, trace reading | Napkin math must be automatic, because its whole purpose is to be run *before* a measurement, in the seconds when it is cheap. |

## Operating settings

| Setting | Design |
|---|---|
| Stack | PyTorch + Triton. Not JAX — see the design log. |
| Bench | RTX 4090 primary; DGX Spark GB10 reserved as the transfer surface |
| Scope bound | One small decoder-only model family, single GPU, one quantisation scheme to depth |
| Theory → implementation | Derivation → traced reference → faded skeleton → independent implementation → changed-surface retrieval |
| Worked examples | High initially, explicitly faded **[A]** |
| Retrieval | Low-volume, prediction- and diagnosis-shaped; mostly embedded in projects **[A]** |
| Interleaving | Begins once isolated procedures work; targets bottleneck-class discrimination **[B]** |
| Whole tasks | Begin in Phase 0 and continue throughout |
| Advancement | Exit evidence, never time elapsed or resources consumed |

## The prediction rule

**Every measurement is preceded by a committed quantitative prediction with a stated tolerance.**

This is the spine of the curriculum, and it is the one rule that must not be relaxed. It is doing four jobs simultaneously:

- **Retrieval**, in the format future performance takes — you will never be paid to recite the roofline, but you will be paid to say "that should be about 40 tokens/s, so 6 is wrong" before anyone runs anything.
- **Calibration** **[B]** — free and continuous, because the ground truth arrives minutes later.
- **Discrimination** — a wrong prediction localises which part of the model is wrong.
- **Protection against measurement theatre** — a number you predicted is a number you have to explain; a number you merely observed is a number you can rationalise.

A prediction that is met teaches little. **An unexplained prediction gap is the highest-value object in this curriculum** and becomes the next session's material. Record the basis of each prediction as `arithmetic`, `prior measurement`, or `guess`; a `guess` is honest and useful, but three consecutive guesses on the same node means the arithmetic is not encoded.

## Theory-to-code loop

For every new mechanism:

1. Derive its cost model — bytes moved, FLOPs, where the time must go — and state the expected behaviour.
2. Trace a known-correct reference implementation, including shapes, dtypes, and buffer invariants.
3. Complete a faded skeleton; predict the parity-test outcome and the timing before execution.
4. Implement or modify it independently in the active engine.
5. Reproduce, debug, or apply it later under a changed model shape, workload, or device.

Do not use open-ended discovery before step 3 works. This uses worked examples and fading deliberately **[A]**.

## The fluency set

The only material held to automaticity. It is deliberately small, and the constants are **measured on the learner's own hardware**, not copied from a spec sheet — a measured bandwidth is both more accurate and better encoded than a quoted one.

| Item | Form | How it is established |
|---|---|---|
| Achievable memory bandwidth, 4090 and GB10 | GB/s | Measured with a bandwidth microbenchmark in Phase 0; recorded in the constants table |
| Achievable dense matmul throughput at fp16/bf16 and int8/fp8 | TFLOP/s | Measured, per dtype, in Phase 0 |
| Kernel launch overhead and a rough per-step Python/dispatch floor | µs | Measured in Phase 1 |
| KV bytes per token | `2 · layers · kv_heads · head_dim · bytes_per_elem` | Derived, then verified against observed memory |
| Decode-step bytes moved | weights + KV read per step | Derived, then verified against measured bandwidth utilisation |
| Decode arithmetic intensity and the batch size where it becomes compute-bound | ratio; crossover batch | Derived, then verified by a batch sweep |
| Expected speedup from acceptance rate α and draft length k | closed form | Derived in Phase 4, verified against measured acceptance |

If any of these takes more than about a minute to produce, it is a fluency bottleneck (`F`), not a knowledge gap, and it gets timed drilling rather than more explanation.

## The confusable families

Discrimination is the Type 3 core of this domain. These are the interleaving sets. They are introduced only after the individual procedures are accurate **[B]**.

**Family 1 — bottleneck class.** Memory-bandwidth-bound · compute-bound · launch/CPU-overhead-bound · synchronisation-bound · queueing-bound · KV-capacity-bound · host-side (tokenise/detokenise/serialise). *The central discrimination of the curriculum.*

**Family 2 — latency metrics.** TTFT vs queue delay vs prefill compute time · TPOT vs ITL vs e2e-per-token · throughput vs goodput · per-request vs per-system · mean vs p50 vs p99.

**Family 3 — batching.** Static vs dynamic vs continuous batching · `max_num_seqs` vs `max_num_batched_tokens` vs KV-block budget · what raises throughput vs what raises per-request latency.

**Family 4 — memory.** Weights vs KV vs activations vs fragmentation vs allocator reserve vs CUDA context · what paging fixes and what it does not.

**Family 5 — quantisation.** Weight-only vs weight+activation vs KV-cache quantisation · effect on bytes moved vs on capacity vs on accuracy · which one helps *this* bottleneck.

**Family 6 — correctness illusions.** Sampling nondeterminism vs a real numerical bug · batch-size-dependent output differences · reduction-order nondeterminism · fp16 vs bf16 range/precision failure modes.

**Family 7 — speedup attribution.** The change itself vs warmup vs a warm prefix cache vs compile/graph capture vs a different output length vs a different tokenisation vs thermal/clock state.

### The misconception bank

These are the `M` errors this domain reliably produces, several of them manufactured by a *training* background. Each is remediated by contrast cases and a corrected re-attempt, never by repetition **[A]**.

| # | Misconception | The correction |
|---|---|---|
| 1 | Decode is compute-bound, like training | At batch 1 it is memory-bandwidth-bound: the entire weight set is read per token to do a trivial amount of arithmetic |
| 2 | Fewer FLOPs means faster | For decode, fewer *bytes* means faster; FLOP reduction that does not reduce bytes moved buys nothing |
| 3 | Bigger batch is always better | Throughput rises, per-request latency and TTFT degrade; the optimum is defined by the SLO, not by the hardware |
| 4 | Quantisation helps because it reduces computation | It mainly helps by reducing bytes moved and KV capacity pressure; with unfused dequantisation it can be *slower* |
| 5 | TTFT is prefill time | TTFT includes queueing, scheduling, tokenisation, and prefill; under load the queue often dominates |
| 6 | MFU is the utilisation metric | For decode, model-bandwidth utilisation is the meaningful one; a "low MFU" decode phase may be running at hardware limit |
| 7 | PagedAttention makes attention faster | It makes memory *management* better, which raises achievable concurrency; the kernel itself is not the win |
| 8 | Continuous batching reduces latency | It raises throughput and cuts *queueing* delay; an individual request's per-token latency typically worsens as concurrency rises |
| 9 | `torch.compile` always helps | It helps overhead-bound regimes most; in bandwidth-bound regimes it can be neutral, and recompiles can make it negative |
| 10 | Speculative decoding always helps | It spends compute to buy latency; in a compute-bound high-batch regime it loses, and its benefit collapses as acceptance rate falls |
| 11 | A measured speedup is a speedup | Not until the two contracts differ in exactly one dimension and the result survives a cold-cache re-run |
| 12 | A faster engine that produces different text has a tolerance problem | It has a correctness problem until parity is demonstrated; performance from an unverified implementation is discarded |

# Phase sequence

Phases are evidence-gated. At 2–3 h/week a phase may occupy several four-week macrocycles; this is not a deadline. **Minimum viable completion is the end of Phase 2 plus one public artifact** — that point alone is a real, defensible competence, and the plan is designed so that stopping there is a success rather than an abandonment.

| Phase | Frontier / theory | Whole task | Scaffolding fade | Exit milestone |
|---|---|---|---|---|
| 0. Measurement and inference arithmetic | A1–A4, H1, H3, M1–M3 | T0: trustworthy benchmark harness + defended baseline | Worked napkin math → guided prediction → unassisted prediction on an unseen model shape | Public harness, measured constants table, a documented measurement bug you found in your own work, and a first-principles prediction met within tolerance |
| 1. The engine core | E1–E4, H2, M3 deepened | T1: engine v0 — KV-cached decode with parity | Traced reference → faded skeleton → from-memory KV cache and cached attention | Engine v0 passes parity against a reference implementation and its speedup over the baseline is explained arithmetically |
| 2. Scheduling and memory | S1–S4, A4 completed | T2: engine v1 — continuous batching + paged KV, frontier curve vs a production engine | Given scheduler contract → modified policy → self-designed admission policy | Reproducible throughput–latency frontier on locked workloads, compared against a production engine, with every gap attributed |
| 3. Making it fast | K1–K3 (K4 on demand), Q1–Q3 | T3: optimisation campaign with a pre-declared negative result | Guided profile reading → independent attribution → independent optimisation choice | Two optimisations with predicted-then-measured speedups, an accuracy evaluation for the quantised path, and one honest negative result |
| 4. Speculation and the production stack | D1–D2, V1–V2 | T4: in-engine speculation + a production engine tuned to a pre-registered SLO | Read the reference implementation → partial implementation → independent tuning and defence | Distribution-preservation test passes; a deployment meets its pre-registered SLO with a defended configuration |
| 5. Transfer and public artifact | X1–X2, V3 at recognition | T5: changed-surface transfer + public write-up + upstream contribution | Annotated claim analysis → independent reproduction → independent transfer | Public report including a negative or discrepant result, plus one accepted-or-rejected upstream contribution with the review recorded |

## T0 — harness acceptance specification

Every performance claim in Phases 1–5 gates on T0, so T0 needs an acceptance test rather than a description. The learner is an experienced programmer; the named blind spot in [`inference-dependency-graph.md`](inference-dependency-graph.md) is precisely that **a harness feeling well-engineered is not evidence that its numbers are valid**. This section converts that from a warning into something executable.

### Required capabilities

| # | The harness must | Verified by |
|---|---|---|
| 1 | Count tokens with the model's own tokenizer, and refuse to report a token-derived metric when no tokenizer is bound | Unit test asserting the refusal path |
| 2 | Declare cache state explicitly (`cold` / `warm-prefix`) and provide a reset that actually cools it | A cold run following a warm run reproduces the cold timing distribution |
| 3 | Emit p50/p90/p99, mean, and n for every latency metric — never a bare mean | Schema assertion on the emitted JSON |
| 4 | Record the resolved contract with the result, including engine/model/dtype/driver/versions and a hash over the contract fields | Result JSON fails validation if any contract field is absent |
| 5 | Refuse to run at all on an incomplete contract, rather than filling defaults | Test that an under-specified contract raises before any GPU work |
| 6 | Synchronise correctly around every timed region, and record the synchronisation policy | Test 7.4 below |
| 7 | Capture pre- and post-run clock/thermal state alongside the result | Present in the emitted artifact, as W0 already does |
| 8 | Reproduce from one command against tracked code | The command in the contract's Reproduction field, run from a clean checkout |

### The confound self-test

**The strongest single piece of Phase 0 evidence.** Write `bench/test_harness_catches.py`, which injects each of the Item A2 confounds into an otherwise-correct measurement and asserts the harness **fails, flags, or corrects** it. Not that a careful operator would notice — that the harness does.

| # | Injected confound | Required harness behaviour |
|---|---|---|
| 7.1 | Warmup set to zero | Flag: first-iteration outlier detected, or refuse a run with warmup below the contract's declared minimum |
| 7.2 | Prefix cache left warm while the contract declares `cold` | Refuse: declared cache state and observed cache state disagree |
| 7.3 | Token count taken by whitespace splitting | Refuse: capability 1's refusal path, since no tokenizer is bound |
| 7.4 | `torch.cuda.synchronize()` removed from the timing path | Flag: measured time below a physically achievable floor derived from the constants table |
| 7.5 | A bimodal latency sample reported as a mean | Flag: p99/p50 ratio above a declared threshold, reported rather than averaged away |
| 7.6 | Second configuration run immediately after a sustained sweep, with clocks depressed | Flag: pre/post clock snapshots differ beyond a declared band |

Each check declares its threshold **in the contract, before results exist**. A check that cannot state its threshold in advance is not a check — it is a judgement call wearing a test's clothes.

Two properties make this worth the session it costs. It is the only Phase 0 artifact that produces *uncontrolled* feedback about the harness itself rather than about the model. And 7.4's floor is derived from the measured constants table, which means the self-test cannot be written until A2/A3 are encoded — it is an integration requirement, working exactly as one should.

### What T0 is not

T0 is not a fast benchmark, and it is not a comparison against anything. Its deliverable is a *defended* baseline: one number, under one contract, that survives the six checks above and a cold-cache re-run. Optimising anything before this exists is the failure mode `T0` was placed here to prevent.


# Per-phase control design

| Phase | Encoding resources and outputs | Retrieval / interleaving | Deliberate-practice target | Feedback and milestone |
|---|---|---|---|---|
| 0 | Roofline and inference-arithmetic sources; metric-definition references; profiler documentation (see [`resources.md`](resources.md) §A, §H, §M). Output: the measured constants table, a byte/FLOP derivation for the chosen model, and a workload contract. | Prompts: KV bytes/token for a changed config; which metric answers this question; what confound would explain this number. No mixing until each is individually correct. | Napkin math under time pressure; identifying a confound in a benchmark description; reading one trace. | Wall clock and profiler; a bandwidth microbenchmark against the spec sheet; published numbers under a comparable contract. Milestone: T0. |
| 1 | Reference engine implementations to trace (§E). Output: shape/dtype/offset contract for the KV cache, a parity test suite, a decode-loop trace annotated with where time goes. | Reconstruct the cache-append and position-offset logic from memory; predict which seeded bug produces which symptom. Mix cache, masking, and position bugs once each is individually diagnosable. | Position/offset bookkeeping; GQA head mapping; parity-tolerance selection; sync-correct timing. | Numerical parity against an independent reference — an uncontrolled ground truth. Milestone: T1. |
| 2 | Scheduler and paging sources: the PagedAttention and Orca papers, the vLLM anatomy post (§S). Output: a scheduler state diagram drawn before implementing, and a memory-accounting model that predicts max concurrency. | Predict the frontier's shape before sweeping; diagnose a p99 blowup from a log. Mix queueing, capacity, and interference explanations — this is the first heavy interleaving block. | Block-table and allocator invariants; admission policy under overload; explaining a latency distribution. | A production engine on the identical contract: the single most informative feedback channel in the curriculum. Milestone: T2. |
| 3 | Profiling and kernel material (§K); quantisation schemes and evaluation (§Q). Output: a bottleneck-attribution write-up per optimisation, with the predicted speedup derived before the change. | Attribute a shortfall to a bottleneck class from a trace alone. Interleave overhead / bandwidth / compute / capacity cases — the central discrimination drill. | Trace reading to fluency; predicting a speedup from the roofline; designing an accuracy evaluation. | Profiler ground truth; accuracy evaluation on a held-out task; the pre-declared negative result. Milestone: T3. |
| 4 | Speculative decoding papers and implementations (§D); production engine internals (§V). Output: the acceptance-rate speedup model derived before implementation; a config-decision rationale. | Predict speedup from measured acceptance; discriminate draft-quality, batch-regime, and scheduler explanations of a disappointing result. | Distribution-preservation testing; SLO-constrained configuration search. | Distribution test; the SLO gate itself; comparison against published speculation results. Milestone: T4. |
| 5 | One selected paper or system claim plus its primary implementation (§X). Output: a claim/assumption map produced before any code. | From-memory reconstruction of the core mechanism; discriminate implementation, workload, and hardware explanations of a discrepancy. | Reproducing under a stated contract; writing for a hostile reader. | Public review: maintainers and readers who did not ask to be impressed. Milestone: T5. |

Resources are tools, not completion metrics. Each has an attached output; see [`resources.md`](resources.md).

# Weekly operating system

## Dose

**2–3 hours per week**, secondary track. Typically two sessions of 60–75 minutes, or one 90-minute session plus one 30-minute fallback.

| Phase | Retrieval + prediction | Encoding | Targeted practice | Whole task | Feedback / logging |
|---|---:|---:|---:|---:|---:|
| 0–1 | 15% | 35% | 25% | 20% | 5% |
| 2–3 | 15% | 20% | 25% | 35% | 5% |
| 4–5 | 10% | 15% | 20% | 45% | 10% |

At the 2-hour floor, retain one 45-minute whole-task block and cut new material first. Do not compensate for a short week by extending the next one.

## Session template: 60–75 minutes

1. **5–10 min:** closed-resource retrieval — one napkin-math prediction or one trace diagnosis from prior work.
2. **2 min:** commit the **metric prediction** in writing — the expected value, its tolerance, and its basis — before anything runs. Also note a time estimate for the session. Do not commit a score prediction or a confidence percentage; see below.
3. **20–25 min:** theory/worked example, or continue the active whole task.
4. **20–25 min:** targeted implementation or diagnosis practice.
5. **5–10 min:** run the verification — parity test, fixed-contract measurement, or profile.
6. **5 min:** log actual result, prediction gap, error codes, bottleneck class, and the next smallest action.

### Why the session commits a metric prediction and not a score

These are different instruments and were previously bundled into one step, which taught that they are the same kind of object. They are not.

| Instrument | What it measures | Where it belongs |
|---|---|---|
| **Metric prediction** | The learner's model of the system, against ground truth arriving minutes later from a channel the learner does not control | Every session, before every measurement. Non-negotiable. |
| **Score prediction** | Metacognitive accuracy — specifically, the illusion-of-fluency failure where material feels known and is therefore under-tested | The macrocycle cumulative checkpoint only |
| **Time estimate** | Whether the dose is real and the session card is scoped correctly | Every session, as operations data, logged against weekly hours actual — not as learning evidence |
| **Confidence percentage** | Nothing interpretable at one number per session | Removed. Collect confidence only as per-item binary bets on a multi-item assessment, where a hit rate per bucket can be computed. |

The knob-setting argument applies here as it does everywhere else in this curriculum. Score and confidence calibration earn their cost in domains where the learner grades their own work, because there they are the only check on self-assessment. **This domain does not have that weakness.** The wall clock, numerical parity, the profiler, and a production engine on an identical contract are four channels the learner does not control. Spending session minutes on introspective self-prediction, when the bench will answer honestly and shortly, is a poor trade at a 2–3 h/week dose.

## Four-week macrocycle

| Week | Function |
|---|---|
| 1 | Encode one bounded node; worked examples; derive its cost model |
| 2 | Independent implementation; delayed retrieval of Week 1 |
| 3 | Mixed diagnosis practice; a measurement under the locked contract; external comparison |
| 4 | Closed-resource cumulative check, transfer task, prediction-ledger review, plan adjustment |

Prior nodes reappear after roughly 2 days, 1 week, 3–4 weeks, then inside later whole tasks and 12-week checks. Distributed retrieval is **[A]**; the exact intervals are adaptive planning heuristics, not findings.

## Rotation and parallel load

This track is secondary by design. The primary is `../q2`.

| | `../q2` robot learning (primary) | This track (secondary) | Photography / Q1 (maintenance) |
|---|---|---|---|
| Sessions/week | 3–4 | 2 | 1 or fewer |
| Content | New material, encoding, targeted practice | Whole tasks, prediction practice, bounded new material | Retrieval and one whole task |
| Assessment | Full stack | Cumulative checkpoints; delayed checks before phase gates | Delayed retention only |

Rotate the primary at a macrocycle boundary, never mid-phase. When this track becomes primary, the dose rises and the phase pace changes; the gates do not.

**A gap of one or two weeks here is spacing, not neglect.** After a gap longer than about three weeks, re-diagnose rather than resume: run one napkin-math prediction and one parity test before touching the frontier.

## Implementation intention

> If it is my planned study window on Tuesday or Sunday, then I open this repository, commit one prediction in writing before running anything, and work until the next committed artifact, parity result, or measured number.

Adapt the days, not the cue–action structure **[B]**.

## Fallback session: 20 minutes

1. One napkin-math prediction from a changed config, closed-resource.
2. Run one existing parity test or one measurement under the locked contract.
3. Log one error code, one prediction gap, and the next smallest action.

No new material in fallback mode.

## Collapse protocol

1. Drop to the fallback session rather than to zero.
2. Never drop retrieval and prediction before dropping new material.
3. If two consecutive weeks fall under the floor, formally park this track at maintenance and record it — an unrecorded silent lapse is what turns into abandonment.
4. After a gap over three weeks, re-diagnose before resuming.

# Error-routing rules

Classify every substantive miss before altering the plan. **Dominant** means at least three instances or one-third of substantive errors across two sessions.

| Dominant pattern | Remedy | Do not do |
|---|---|---|
| `K` | Pause the dependent task; guided derivation and worked example; retry a near variant | Schedule retrieval for unencoded material |
| `R` | Raise retrieval to 25%; closed-resource derivation next session and next week | Add more explanation |
| `M` | Contrast cases from the misconception bank; predict-then-measure the case that breaks the belief | Repeat the correct statement — it entrenches the belief |
| `D` | Interleave the relevant confusable family, especially bottleneck classes, once isolated competence exists | Return to isolated study |
| `P` | Faded skeleton plus focused unit and parity tests on the specific invariant | Re-teach broad theory |
| `F` | Timed napkin-math or trace-reading drill, after accuracy is established | Introduce new techniques |
| `T` | Changed-surface whole task: different model shape, different workload mix, or the GB10 | Add flashcards |
| `C` | Checklist, commit-before-run discipline, pacing break | Treat it as a content deficit |

New material labelled `K` is the active frontier, not a personal failure.

## Bottleneck-class routing

Separate from attempt errors: when a *system* underperforms, classify the bottleneck before changing anything.

| Class | Evidence that establishes it | Remedy family | Common wrong move |
|---|---|---|---|
| Memory-bandwidth-bound | Achieved GB/s near measured peak; time scales with bytes moved | Quantisation, KV reduction, larger batch, fusion | Adding compute-side optimisation |
| Compute-bound | Achieved TFLOP/s near measured peak; time scales with FLOPs | Better kernels, lower-precision compute, less speculation | Quantising weights only |
| Launch/CPU-overhead-bound | Trace is gaps, not kernels; time is roughly constant in batch size | CUDA graphs, `torch.compile`, batching, less Python per step | Writing a faster kernel |
| Synchronisation-bound | Time collapses when a sync is removed; timings are implausibly fast or slow | Remove sync points; fix the measurement | Trusting the timing |
| Queueing-bound | p99 ≫ p50; latency grows with offered load at constant per-step cost | Admission control, chunked prefill, capacity | Optimising the model path |
| KV-capacity-bound | Preemption/recompute events; concurrency capped below the arithmetic limit | Paging, KV quantisation, shorter context, more memory | Raising batch size |
| Host-side | Time in tokenise/detokenise/serialise; GPU idle | Fix the host path | Any GPU-side change |

# Assessment stack

| Measure | Cadence | Evidence |
|---|---|---|
| Prediction commitment | **Every measurement** | A written metric prediction with tolerance and basis, before the run |
| Closed-resource retrieval | Most sessions | Napkin math, an implementation fragment from memory, or a trace diagnosis |
| Parity check | Every engine change | Numerical agreement with an independent reference at a declared tolerance |
| Implementation check | Weekly | Passing tests plus a predicted failure mode stated before execution |
| Cumulative checkpoint | Every macrocycle | Mixed prior/current problems plus a seeded diagnosis task |
| Transfer measure | Every macrocycle | Changed model shape, changed workload mix, or the GB10 |
| Phase-gate delayed check | 7–14 days after the qualifying independent/transfer attempt | Alternate-form implementation or diagnosis before advancing |
| Maintenance delayed measure | 4–12 weeks after a node exits active study | Alternate-form implementation or diagnosis; regression reopens remediation |
| Long retention | 6 and 12 months | Rebuild or modify a representative subsystem and run a novel diagnostic experiment |
| Metric calibration | Continuous, reviewed each macrocycle | The prediction ledger: predicted vs actual, recorded as log₁₀(actual/predicted), with the gap explained. Review statistic: median absolute log ratio, plus hit-rate within declared tolerance. |
| Score calibration | Macrocycle checkpoint only | Predicted vs actual score on the cumulative check. Not collected per session. |
| Explanation defence | Each phase exit | Recorded technical defence of the contract, metric, and configuration choices |

## Phase exit rule

Advance only with all of:

1. accurate independent performance;
2. justified choice among confusable alternatives;
3. a changed-surface transfer result;
4. one delayed recheck;
5. a reproducible experiment record naming its workload contract.

Scores are task-local and require an explicit rubric. A score never implies a node state by itself. Every assessed point records assistance as `scaffolded`, `independent`, or `transfer`. Critical failures named in a gate override aggregate scores.

## Phase exit scorecards

Every row is a binary gate. Thresholds that depend on a task, model, or device are declared in `bench/workload-contract.md` **before** results are run; they may not be chosen after observing results.

### Phase 0 — measurement and inference arithmetic

| Gate | Required evidence |
|---|---|
| Independent mechanism | Closed-resource, from an unseen model config: parameter bytes, KV bytes/token, decode bytes/step, arithmetic intensity, the compute-bound crossover batch, and an upper-bound tokens/s. Correct within the declared tolerance without consulting prior work. |
| Debugging and discrimination | Identify the confound and its direction in seeded benchmark descriptions covering: absent warmup, warm prefix cache, whitespace token counting, missing CUDA synchronisation, mean-over-bimodal reporting. No unresolved critical case. |
| Harness self-test | `bench/test_harness_catches.py` passes: all six injected confounds in the T0 specification are caught by the harness itself, each against a threshold declared in the contract before the test was written. A confound that only a careful operator would notice is not caught. |
| Whole task / transfer | T0 runs on a second model shape or a materially changed workload contract, producing a defended metric, percentile-reported latency, a fixed-seed result, and a documented measurement bug found in the learner's own harness. **A result reported without its contract invalidates the gate.** |
| Delayed | After 7–14 days, reproduce the napkin-math chain and the confound identification on alternate forms, closed-resource. |
| Reproducibility | One command recreates the environment and the fixed-contract result from tracked code; the record includes contract, versions, constants table, and explanation defence. |

### Phase 1 — the engine core

| Gate | Required evidence |
|---|---|
| Independent mechanism | From the written contract alone, implement the KV cache, cached attention with correct GQA mapping and position offsets, and the decode loop; parity against an independent reference within declared tolerance. |
| Debugging and discrimination | Diagnose, before execution, seeded faults covering: stale position/RoPE offset, cache written at the wrong offset under padding, wrong GQA head mapping, missing synchronisation in the timing path, and a sampling-nondeterminism false alarm. |
| Whole task / transfer | T1's speedup over the T0 baseline is *predicted from arithmetic first*, then measured, and the residual gap is attributed to a named bottleneck class. Repeat on a second model shape. |
| Delayed | After 7–14 days, reconstruct or repair an alternate-form cache/attention variant without the prior implementation. |
| Reproducibility | Parity test suite, fixed-contract measurement, versions, and a one-command reproduction. |

### Phase 2 — scheduling and memory

| Gate | Required evidence |
|---|---|
| Independent mechanism | Implement iteration-level scheduling and a paged KV allocator; state and test the block-table and allocator invariants; predict maximum concurrency from memory accounting and match the observed limit within tolerance. |
| Correctness under concurrency | Per-request output parity between engine v1 under concurrency N and engine v0 at batch 1: token-identical under greedy decoding on the same seeds and the same prompts, at the highest concurrency the frontier reports, and again at a concurrency that forces preemption or recompute. **No frontier number from a run that has not passed this is admissible.** |
| Debugging and discrimination | From logs and traces, distinguish queueing, KV-capacity, prefill/decode interference, admission-policy, and per-step-cost explanations of a latency distribution; confirm at least one diagnosis with a controlled intervention. |
| Whole task / transfer | T2 produces a throughput–latency frontier on at least two locked workloads (prefill-heavy and decode-heavy), compared against a production engine on the identical contract, with every material gap attributed rather than excused. |
| Delayed | After 7–14 days, repair or extend an alternate-form scheduler or allocator without the completed implementation. |
| Reproducibility | Contract version, seeds, frontier data, comparison table, and one-command reproduction. |

### Phase 3 — making it fast

| Gate | Required evidence |
|---|---|
| Independent mechanism | For each optimisation: derive the expected speedup from the bottleneck class *before* implementing, implement it, and measure. One verified Triton kernel with a parity test, or a documented decision not to write one backed by profile evidence. |
| Debugging and discrimination | Attribute five shortfall cases to the correct bottleneck class from traces and arithmetic, and name the remedy that would *not* work for each. |
| Whole task / transfer | T3 reports two optimisations with predicted-then-measured speedups under single-variable contracts, a quantisation accuracy evaluation on a held-out task, and **one pre-declared negative result that was published rather than discarded**. |
| Delayed | After 7–14 days, attribute a fresh trace and propose the matching remedy, closed-resource. |
| Reproducibility | Per-optimisation contracts, profiles, accuracy results, and reproduction commands. |

### Phase 4 — speculation and the production stack

| Gate | Required evidence |
|---|---|
| Independent mechanism | Derive the acceptance-rate speedup model; implement speculation in the engine; **pass a distribution-preservation test**, not merely an output-looks-fine check. |
| Debugging and discrimination | Explain a disappointing speculation result by discriminating draft quality, batch regime, scheduler interaction, and verification overhead, with evidence for the chosen cause. |
| Whole task / transfer | T4 meets a pre-registered SLO on a locked workload with a production engine, with the configuration defended knob-by-knob against the bottleneck classes it addresses, and re-tuned for a changed workload mix. |
| Delayed | After 7–14 days, reconstruct the speedup model and re-derive a configuration for a changed SLO, closed-resource. |
| Reproducibility | Acceptance-rate measurements, distribution test, SLO evidence, config rationale, and reproduction commands. |

### Phase 5 — transfer and public artifact

| Gate | Required evidence |
|---|---|
| Independent mechanism | A claim/assumption map for the selected paper or system claim, produced before any code, naming which choices are load-bearing and which are benchmark artifacts. |
| Transfer | The Phase 2–4 competence is re-exercised on the GB10 or another materially changed surface: predictions re-derived for the new memory hierarchy, and the tuning outcome explained by the hardware difference rather than by trial and error. |
| Whole task | A public write-up under a stated contract including a negative or discrepant result, plus one upstream contribution (reproducible issue, benchmark, documentation fix, or PR). |
| Delayed | After 7–14 days, reconstruct the core mechanism and diagnose an alternate discrepancy without the completed work. |
| External review | Review actually requested and its response recorded — including no response, which is recorded as no response and never read as approval. |

# Feedback channels not controlled by the learner

This domain has unusually strong uncontrolled feedback. Use it instead of self-assessment.

| Channel | Used in | Role |
|---|---|---|
| Wall clock and profiler (torch profiler, Nsight) | 0–5 | Refuses to be persuaded; the primary channel |
| Numerical parity with a reference implementation | 1–5 | Correctness ground truth; gates every performance claim |
| A production engine on an identical workload contract | 2–5 | The most informative single comparison available |
| Published reference numbers and papers | 0, 3–5 | External target for reproduction and discrepancy analysis |
| Accuracy evaluation on a held-out task | 3–5 | Prevents quantisation from silently trading quality for speed |
| Requested public review: a write-up posted for critique, or an upstream issue/PR | 3–5 | Critique outside self-authored tests |
| The GB10's different memory hierarchy | 5 | Punishes a model that only fits one machine |

If human review does not arrive within a week, do not interpret silence as approval. The machine channels remain valid regardless.

# Revision triggers

Set in advance so plan changes are evidence-driven.

| Trigger | Mandatory response |
|---|---|
| Two failed exit attempts | Audit graph dependencies; do not simply repeat the phase |
| `K`/`M` dominates | Reduce scope to prerequisite nodes; restore worked examples; work the misconception bank |
| `R` dominates | Increase delayed closed-resource retrieval; reduce new material |
| `D` dominates | Increase interleaved bottleneck-attribution drilling; reduce isolated study |
| **Metric prediction off by more than 2× on three consecutive predictions** | Stop optimising. Return to A2/A3 arithmetic until predictions land inside tolerance |
| **Three consecutive predictions recorded with basis `guess`** | The arithmetic is not encoded; treat as `K` regardless of whether outcomes were met |
| **A reported speedup fails to reproduce on a cold-cache re-run** | Measurement-hygiene remediation before any further optimisation; audit every result since the last clean re-run |
| **A performance number is recorded without its contract** | Invalidate the number; re-run under contract |
| Median absolute log ratio above 0.3 (≈2×) across a macrocycle's predictions | Return to the arithmetic for the nodes involved before further optimisation work |
| Tolerance hit-rate at or near 100% across a macrocycle | The tolerances are too wide to be falsifiable; tighten them before the next measurement |
| Score calibration misses by more than 20 points on three macrocycle checkpoints | The sense of what is known is uninformative; raise delayed closed-resource retrieval rather than adding explanation |
| Whole-task work absent for two weeks | Reserve the next session for integration before more theory |
| No delayed measure in a macrocycle | Block advancement until completed |
| Two weeks under the floor | Park at maintenance and record it; resume with fallback sessions; never catch up |
| Gap longer than three weeks | Re-diagnose before resuming the frontier |

# Design log

| Decision | Rationale |
|---|---|
| Serving-system outcome primary; from-scratch engine as the mechanism route | The learner asked for both. Making the engine the *outcome* would produce a toy that loses to vLLM and teaches flag-blindness; making it the *encoding mechanism* means every production behaviour has a built implementation behind it. The engine is judged on what it explains, not on what it beats. |
| Predict-before-measure as the curriculum spine | This domain supplies free, immediate, honest ground truth. That makes calibration nearly costless and turns retrieval into the exact format future performance takes. It is also the only reliable defence against measurement theatre. |
| Measurement hygiene (T0) before any optimisation | The deliberately unglamorous early project the framework calls for. It is the direct analogue of the tabular-baseline/leakage task in `../q2`: it cheaply exposes the gaps an impressive demo would hide. |
| PyTorch + Triton rather than JAX | The reference implementations, profilers, benchmarks, and reviewable community are there, so the uncontrolled feedback channels are much stronger. It also satisfies Part VII's rule to keep parallel tracks in dissimilar idioms — the primary track is JAX. |
| Bottleneck-class discrimination promoted to a first-class assessment target | Type 3 pattern recognition is the working skill of this domain and is normally left implicit. It gets its own routing table, interleaving set, and gate rows. |
| Scope bounded to single-GPU, one model family, one quantisation scheme | A 2–3 h/week secondary track. Multi-GPU, disaggregation, and long-context are recognition leaves. Cutting scope preserves depth; stretching it would produce coverage without evidence. |
| Minimum viable completion declared at end of Phase 2 | So that a track ending early ends as a success with a real artifact, rather than as an abandonment. |
| Fluency constants measured, not quoted | A measured bandwidth is both more accurate for the actual bench and better encoded than a spec-sheet number. It also makes the first session produce evidence. |
| Negative result required in T3 and T5 | The domain's literature is dominated by favourable contracts. Producing and publishing one honest negative result is the clearest available evidence of measurement integrity. |
| GB10 reserved as the transfer surface rather than used early | Its value is precisely that it is unfamiliar. Spending it early as a second dev box would destroy the strongest transfer measure the learner owns. |
| Separate node state from attempt errors | Inherited from `../q2` and `CONTEXT.md`: evidence maturity persists across attempts; `K/R/M/D/P/F/T/C` diagnoses only the miss that routes a remedy. |
| **Item A1 decomposed into seven independently scored sub-items with definitional unlocks (2026-09-01)** | The first attempt halted before sub-item 1, yielding one bit of information — `K` somewhere in A2/A3 — when seven were available. A chained cold-start derivation is a gate, not a diagnostic: the decode-byte accounting is a convention plus arithmetic, not a discoverable result, so a learner never taught the convention cannot reason to it. Unlocks supply definitions only, are recorded as `scaffolded` per sub-item, and separate "does not know the convention" from "cannot do the arithmetic" — two misses with different remedies. Error carried forward is not penalised twice, so a wrong link ends a sub-item rather than the block. |
| **T0 given an acceptance specification with a confound self-test (2026-09-01)** | Every performance claim in Phases 1–5 gates on T0, which was described but not specified. The learner's named blind spot is that a well-engineered-feeling harness is not evidence of valid numbers; the countermeasure has to be executable. `bench/test_harness_catches.py` injects the six Item A2 confounds and asserts the harness catches them, which is the only Phase 0 artifact producing uncontrolled feedback about the *harness* rather than the model. Its synchronisation check derives its floor from the measured constants table, so the test cannot be written before A2/A3 are encoded — an integration requirement behaving as one should. |
| **Phase 2 gate given a correctness row (2026-09-01)** | Phases 1 and 3 required parity and Phase 2 did not, despite continuous batching plus paged KV being where ragged-batch position and mask bugs live. Those bugs produce plausible text rather than crashes, so a frontier can be measured, published, and wrong. Per-request greedy parity against engine v0 at batch 1 is cheap to write and gates every frontier number, including at a concurrency that forces preemption. |
| **Score and confidence prediction demoted; metric prediction kept (2026-09-01)** | The session template bundled three instruments into one commitment, teaching that they are the same object. Metric prediction tests the domain model against ground truth from a channel the learner does not control, and stays mandatory. Score prediction detects the illusion-of-fluency failure but is very noisy at n=1 per session; it moves to macrocycle checkpoints and multi-item assessments. A single confidence percentage per session supports no calibration computation at all and is replaced by per-item binary bets where a hit rate is computable. The general argument for self-assessment calibration assumes the learner grades their own work; this domain has four uncontrolled channels, so the knob is turned down here — consistent with the rule that knobs are re-derived per domain rather than inherited. |
| **Calibration recorded in log space (2026-09-01)** | The revision triggers mixed ratio-valued and point-valued misses. Averaging a 948× miss with a 1.2× miss linearly is meaningless; median absolute log₁₀ ratio plus tolerance hit-rate is interpretable, and a hit-rate near 100% is now read as tolerances too wide to falsify rather than as good calibration. |

# Self-critique

Scored against the rubric in `evidence-adaptive-curriculum-architecture.md` §V.2.

| Criterion | Score / 2 | Note |
|---|---:|---|
| Outcome specificity | 2 | Observable performance under a declared SLO and contract |
| Domain-type fit | 2 | Type 5 dominant with Type 1 base and an explicit Type 3 diagnosis component; knobs differ deliberately from `../q2` |
| Prerequisite mapping | 2 | Explicit DAG; fluency nodes marked; sequence constraints kept separate from prerequisites |
| Diagnostic placement | 1 | Designed but **not yet run**; every node state is provisional until it is |
| Encoding quality | 2 | Worked examples with an explicit fading path per phase |
| Retrieval layer | 2 | Prediction, from-memory implementation, and trace diagnosis — the formats future performance takes |
| Spacing | 2 | Crosses week boundaries; delayed and maintenance measures specified |
| Discrimination | 2 | Seven confusable families and a twelve-item misconception bank, with a dedicated routing table |
| Whole-task integration | 2 | A whole task in every phase beginning at Phase 0 |
| Feedback integrity | 2 | Profiler, parity, production-engine comparison, published numbers, and public review — most of it uncontrolled |
| Measurement | 2 | Delayed and transfer measures in every phase gate; prediction ledger is continuous |
| Sustainability | 1 | 2–3 h/week against six phases is a long horizon; mitigated by the Phase 2 minimum-viable-completion declaration, the collapse protocol, and bounded scope, but the risk is real |
| **Total** | **22 / 24** | |

**Weakest two:** (1) *diagnostic placement* — the frontier is assumed, not measured, until `phase-0-entry-diagnostic.md` is run; treat every node state in the graph as provisional. (2) *sustainability* — at this dose Phases 3–5 are far out, and the honest mitigation is the declared minimum viable completion rather than a claim that the full plan will be finished.

Checked for the two failure signatures: this is not **over-atomised** (whole tasks start in Phase 0 and carry the highest time share from Phase 2 onward), and not **premature authenticity** (the arithmetic and measurement foundations gate the engine, and the engine gates the optimisation work).
