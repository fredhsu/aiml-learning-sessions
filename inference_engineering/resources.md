# Resource Map

**Design stage:** confirmed; revisable as the field moves
**Indexed by:** node IDs in [`inference-dependency-graph.md`](inference-dependency-graph.md)

## How to use this file

Resources are **tools, not completion metrics**. Nothing here is "done" when it has been read or watched. Every entry has an **attached output** — the artifact that makes engagement with it evidence of anything. Consuming a resource without producing its output is exposure, and exposure is not competence.

Three rules for this domain specifically:

1. **Read for the cost model, not the conclusion.** The durable content of an inference post is *why* a technique helps and *in which regime*. The numbers are contract-dependent and go stale; the reasoning does not.
2. **Every number you read has a contract you were not shown.** Reconstructing that missing contract is a Phase 5 skill (X1) and a useful habit from day one. Ask: what model, what dtype, what hardware, what input/output lengths, what concurrency, what cache state?
3. **This field has an unusually bad signal-to-noise ratio.** It is full of SEO content that reproduces the same diagram set with unreproducible numbers and confident claims about which framework is fastest. The list below is restricted to primary sources, engineering posts by people who built the thing, and course material. If a source is not in this file and is not a primary source, treat it as a hypothesis.

**Evidence tiering [A]/[B]/[C]** is used where a resource makes a *learning-design* claim. Technical sources are labelled by what they are: `primary` (paper or the actual implementation), `engineering` (a post by an implementer), `course`, `reference`, or `vendor` (useful, but written by someone selling the conclusion).

---

## §A — Inference arithmetic (nodes A1–A4)

The load-bearing section. If only three things on this page are read closely, they are the first three here.

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| kipply, **"Transformer Inference Arithmetic"** — https://kipp.ly/transformer-inference-arithmetic/ | engineering | A1–A3 | The canonical first-principles treatment: KV cache, capacity, latency equations, FLOP counting, and a comparison of the predicted numbers against real benchmarks. It reasons rather than benchmarks, which is exactly the skill being built. | Re-derive its latency equations for **your** chosen model and the 4090, before reading its worked numbers. Record the derivation in the constants table. |
| Horace He, **"Making Deep Learning Go Brrrr From First Principles"** — https://horace.io/brrr_intro.html | engineering | A2, K1 | The clearest statement of the compute / bandwidth / overhead trichotomy that the entire bottleneck-class taxonomy rests on. Written by a PyTorch performance engineer. | Write the three-way decision procedure from memory, then classify your Phase 0 baseline with it before profiling. |
| DeepMind, **"How To Scale Your Model"**, ch. 7 *All About Transformer Inference* — https://jax-ml.github.io/scaling-book/inference/ (and ch. 4, *All the Transformer Math You Need to Know* — https://jax-ml.github.io/scaling-book/transformers/) | course | A1–A4 | The most rigorous free treatment of inference arithmetic and the latency/throughput tradeoff. **Caveat: it is TPU-centric.** Use the reasoning and the derivations; do not import its hardware constants. | Redo one of its worked examples with 4090 numbers instead of TPU numbers, and state which conclusions change and which survive. |
| Lilian Weng, **"Large Transformer Model Inference Optimization"** — https://lilianweng.github.io/posts/2023-01-10-inference-optimization/ | engineering | A2, Q1, K4 | A well-organised map of the technique landscape. Best used as a **taxonomy** to hang later phases on, not as a how-to. | One page: for each technique named, which bottleneck class it addresses. This becomes a Phase 3 discrimination drill. |
| Pierre Lienhart, **"LLM Inference Series"** — https://medium.com/@plienhar/llm-inference-series-1-introduction-9c78e56ef49d | engineering | A1–A4 | A patient multi-part walk through the same material at a slower pace. Use it as the fallback when a derivation elsewhere does not land. | None on its own; it is remediation material for a specific `K`. |
| Yuan et al., **"LLM Inference Unveiled: Survey and Roofline Model Insights"** — https://arxiv.org/abs/2402.16363 | primary | A2 | Applies the roofline model to LLM inference systematically. Useful for seeing the method applied across many techniques at once. | Reproduce its roofline placement for your model's decode step, from your own measured constants. |

**Section exit output:** the measured constants table plus a from-memory derivation, on an unseen model config, of parameter bytes → KV bytes/token → decode bytes/step → arithmetic intensity → crossover batch → upper-bound tokens/s.

---

## §H — Hardware model (nodes H1–H3)

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| Modal, **GPU Glossary** — https://modal.com/gpu-glossary | reference | H1, H2 | A hyperlinked glossary spanning device hardware (SM, warp scheduler, tensor core) through the CUDA programming model. Solves the specific problem that GPU documentation is fragmented across abstraction levels. | Not read front-to-back. Use it as a lookup, and after Phase 1 write the memory hierarchy and execution model from memory, checking against it. |
| GPU MODE, **resource stream** — https://github.com/gpu-mode/resource-stream | reference | H1–H3, K1–K4 | The community's curated index of GPU programming material, including the lecture series index. The best single entry point for going deeper on demand. | None; it is an index. Use it to select, not to consume. |
| Your own bench | — | H1 | Spec sheets state theoretical peaks that no kernel achieves. The number you need is the achievable one. | **A bandwidth and matmul microbenchmark, run on the 4090 and later the GB10, whose results populate the constants table.** This is the first session's artifact. |

---

## §M — Measurement (nodes M1–M3)

Deliberately placed before anything interesting. This is the section that decides whether the rest of the curriculum produces knowledge or noise.

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| vLLM, **Benchmark CLI documentation** — https://docs.vllm.ai/en/latest/benchmarking/cli/ | reference | M1, M2 | Shows what a serious serving benchmark actually parameterises: dataset, request rate, concurrency, percentile metrics. Read it as a specification of what a workload contract must contain. | Derive your own `bench/workload-contract.md` fields from it, then justify each field you omitted. |
| **GuideLLM** — https://pypi.org/project/guidellm/ | reference | M1, M2 | A benchmarking harness built around SLO-driven evaluation with full TTFT/ITL distributions. Useful as a design reference and later as an independent cross-check on your own harness. | After T0, run one workload through both your harness and GuideLLM and explain any disagreement. A disagreement you cannot explain is a bug in one of them. |
| PyTorch profiler and Nsight Systems documentation | reference | M3 | The tools themselves. CUDA's asynchronous execution model makes naive timing wrong in a way that is invisible without them. | An annotated trace of your Phase 0 baseline decode loop, with the dominant kernel and the gap structure identified. |
| Anyscale, **"How continuous batching enables 23x throughput in LLM inference"** — https://www.anyscale.com/blog/continuous-batching-llm-inference | engineering / vendor | M1, M2, S1 | Read here **twice**: once in Phase 0 as a claim-analysis exercise (what is the contract behind "23×"? what baseline? which model? what lengths?), and again in Phase 2 for the mechanism. | Phase 0: reconstruct the contract behind the headline number and state what workload would make the number much smaller. |

**A note on published numbers.** Almost every impressive inference number in public is true under a contract chosen to make it true. This is not usually dishonesty; it is the absence of a norm requiring the contract. Treat every such number as a claim about a specific regime, and make reconstructing that regime a reflex.

---

## §E — Engine core (nodes E1–E4)

Read these as **implementations to trace**, per step 2 of the theory-to-code loop — not as code to copy.

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| Karpathy, **nanochat** `engine.py` — https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py | primary | E1–E4 | A small, readable, dependency-light engine with KV cache and a clean prefill/decode split. The right size to trace line by line — small enough to hold entirely in your head. | A shape/dtype/offset contract for the KV cache, written from the trace, then a from-memory reimplementation on a different model config. |
| **gpt-fast** — https://github.com/meta-pytorch/gpt-fast, with its writeup https://pytorch.org/blog/accelerating-generative-ai-2/ | primary + engineering | E1–E4, K2, Q1, D1 | Under 1000 lines of native PyTorch reaching genuinely fast numbers via `torch.compile`, int4/int8 quantisation, and speculative decoding. The best demonstration that most of the win is available without exotic machinery. | Predict, from the blog's technique list, the *order* in which the techniques help your bottleneck — before reading its results. Check afterwards. |
| Andrew Chan, **"Fast LLM Inference From Scratch"** — https://andrewkchan.dev/posts/yalm.html | engineering | E1–E4, K1 | A complete account of building a single-GPU inference engine in C++/CUDA without libraries, with the optimisation reasoning kept visible and benchmarked against llama.cpp on an RTX 4090 — the same bench as this curriculum. | Extract its optimisation sequence and map each step to a bottleneck class. Predict which steps will and will not transfer to PyTorch. |
| HuggingFace `transformers` generation code | reference | E1–E3 | Not a model of good inference engineering — it is the **parity reference**. Its outputs are the correctness ground truth for engine v0. | A parity test suite: your engine vs `transformers` on identical inputs, at a declared tolerance, under declared determinism settings. |

---

## §S — Scheduling and memory management (nodes S1–S4)

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| Kwon et al., **"Efficient Memory Management for LLM Serving with PagedAttention"** (SOSP '23) — https://arxiv.org/abs/2309.06180 | primary | S2 | The paged KV cache, argued from the OS virtual-memory analogy. The core paper of the phase. | A block-table and allocator design written **before** reading their implementation section, then a diff against theirs with each difference justified or conceded. |
| Yu et al., **"Orca: A Distributed Serving System for Transformer-Based Generative Models"** (OSDI '22, USENIX) | primary | S1 | The origin of iteration-level scheduling. Reading it after Phase 1 makes continuous batching feel inevitable rather than clever. | A scheduler state diagram drawn from the paper, then implemented. |
| vLLM, **"Inside vLLM: Anatomy of a High-Throughput LLM Inference System"** — https://vllm.ai/blog/2025-09-05-anatomy-of-vllm (also at https://www.aleksagordic.com/blog/vllm) | engineering | S1–S4, V1 | A deep, current walkthrough of how a real engine is actually structured — scheduler, executor, attention backend, and the paths between them. The single best bridge from "I built a toy" to "I can read the real one". | Locate, in the vLLM source, the code implementing one behaviour you observed in your own engine's frontier curve. |
| **vLLM** source — https://github.com/vllm-project/vllm · **SGLang** source — https://github.com/sgl-project/sglang | primary | S1–S4, V1 | The production systems. SGLang's RadixAttention is the sharpest available contrast to vLLM's approach to prefix reuse — a genuine design disagreement, which makes it good discrimination material. | One page comparing how each handles prefix sharing, and under which workload each choice wins. |

---

## §K — Kernels and compute efficiency (nodes K1–K4)

Entered **after** T2, and only to the depth a profile justifies. K4 is recognition-level unless a project requires more.

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| **GPU MODE** lecture series (YouTube channel + https://github.com/gpu-mode/resource-stream) | course / video | K1–K4 | The best free video series on practical GPU kernel work, taught by practitioners. Lecture 1 (profiling CUDA kernels in PyTorch) and the Triton lecture are the two directly on the critical path. | Per lecture: one runnable artifact. A profile you took, or a kernel you wrote and verified. Never notes alone. |
| Christian Mills, **GPU MODE lecture notes** — https://christianjmills.com/series/notes/cuda-mode-notes.html (e.g. Triton: /posts/cuda-mode-notes/lecture-014/) | reference | K1–K4 | Written notes for the lecture series. Use them to *decide which lecture to watch* and to review afterwards — not as a substitute for doing the work. **[C]** Reading notes in place of writing code is the classic false-fluency trap in this domain. | None. This is a navigation aid. |
| **Triton tutorials** — https://triton-lang.org/main/getting-started/tutorials/index.html (start with fused softmax: .../02-fused-softmax.html) | reference | K3 | The official progression from vector add through fused softmax to matmul and fused attention. Fused softmax is the ideal first kernel: the memory-traffic argument for fusing it is exactly the argument the whole curriculum runs on. | One fused kernel you wrote, with a parity test against the PyTorch reference and a measured speedup you predicted first. |
| Dao et al., **FlashAttention** — https://arxiv.org/abs/2205.14135 (and FlashAttention-2 / -3 for the hardware-specific refinements) | primary | K4 | The canonical IO-aware kernel argument: the win comes from not materialising the attention matrix in HBM. Read it as roofline reasoning applied to a real kernel. | Explain, from the roofline alone and without the paper open, why fusing attention wins and in which regime the win shrinks. |
| Stanford **CS336**, *Language Modeling from Scratch* — https://cs336.stanford.edu/ · playlist https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_ | course / video | K3, A1–A4 | Lecture 6 (**Kernels, Triton**) and Lecture 10 (**Inference**) are directly on-curriculum and taught at research depth. The rest of the course is a training curriculum and is out of scope here. | Watch only those two lectures against a specific open question; produce the answer as an artifact. |

---

## §Q — Quantisation (nodes Q1–Q3)

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| Lin et al., **AWQ** — https://arxiv.org/abs/2306.00978 | primary | Q1, Q2 | The salient-weight argument: protecting roughly 1% of weights, identified from the *activation* distribution, recovers most of the loss. A clean, memorable mechanism. | State the salient-weight argument from memory, then predict which of your model's layers it protects. |
| Frantar et al., **GPTQ** — https://arxiv.org/abs/2210.17323 | primary | Q2 | The second-order, layer-by-layer error-redistribution approach. The contrast with AWQ is the discrimination target — two different theories of what quantisation error *is*. | A one-page comparison: what each method assumes, what each needs (calibration data, compute), and when you would choose each. |
| **llm-compressor** documentation — https://docs.vllm.ai/projects/llm-compressor/en/latest/ | reference | Q1, Q3 | The currently maintained path for producing quantised checkpoints for vLLM (AutoAWQ is archived). Its FP8 W8A8 example is the practical starting point. | One quantised checkpoint you produced, with the recipe recorded in the workload contract. |
| GPU MODE lecture on quantisation — https://christianjmills.com/posts/cuda-mode-notes/lecture-007/ | course / video | Q1 | Connects quantisation to the kernels that make it actually fast, which is where the naive "fewer bits must be faster" story breaks. | Predict, then measure, whether your quantised path is faster — and if it is not, attribute it to dequantisation overhead or kernel selection. |
| A held-out task evaluation of your choice | — | Q3 | The uncontrolled channel that stops quantisation from silently trading quality for speed. | **An accuracy result reported alongside every quantisation speedup.** A speed number without an accuracy number is not a result in this curriculum. |

---

## §D — Speculative decoding (nodes D1–D2)

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| Leviathan, Kalman & Matias, **"Fast Inference from Transformers via Speculative Decoding"** — https://arxiv.org/abs/2211.17192 | primary | D1 | The original method and, more importantly, the proof that the output distribution is preserved exactly. That proof is the difference between implementing speculation and implementing a subtle sampling bug. | Reproduce the acceptance/rejection sampling argument on paper, then write the distribution-preservation test **before** the implementation. |
| NVIDIA, **"An Introduction to Speculative Decoding for Reducing Latency in AI Inference"** — https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/ | vendor / engineering | D1, D2 | A current overview of the draft-strategy landscape (draft models, n-gram, trained heads such as Medusa and EAGLE). Vendor-published, so treat its numbers as regime-specific claims. | Predict the speedup for your own measured acceptance rate before reading theirs; explain any discrepancy by regime. |
| gpt-fast's speculative decoding implementation (see §E) | primary | D1 | A short, readable implementation to trace after the math is derived. | A faded-skeleton implementation in your own engine, passing the distribution test. |

---

## §V — Production stack (nodes V1–V3)

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| **vLLM documentation** — https://docs.vllm.ai/ | reference | V2 | The configuration surface you will be tuning against an SLO, and the semantics of each knob. | A configuration defended knob-by-knob against the bottleneck classes it addresses. |
| Red Hat / DeepLearning.AI, **"Fast & Efficient LLM Inference with vLLM"** (free course) — https://vllm.ai/blog/2026-06-03-deeplearning-ai-vllm-course · https://developers.redhat.com/blog/2026/06/03/learn-optimize-deploy-and-benchmark-llms-vllm-new-free-course | course / video | V2 | A structured optimise → deploy → benchmark walkthrough from the people maintaining the engine. Best taken **in Phase 4**, after building your own engine, when it becomes a design comparison rather than a tutorial. | For each recommendation the course makes, state the bottleneck class it targets and whether it applies to your workload. |
| "Inside vLLM" (see §S) | engineering | V1 | Re-read in Phase 4 for the operational layer rather than the scheduler. | Locate the code path for one knob you tuned. |

---

## §X — Claim analysis and public artifact (nodes X1–X2)

| Resource | Type | Node | Why this one | Attached output |
|---|---|---|---|---|
| Any vendor benchmark post claiming a framework is fastest | vendor | X1 | Ideal claim-analysis material precisely because the contract is chosen. **[C]** as evidence, high-value as an exercise. | Reconstruct the missing contract; name the workload under which the claim would reverse. |
| vLLM / SGLang issue trackers | primary | X2 | Where a reproducible benchmark report is actually useful to someone, and where review is free and honest. | One upstream contribution: a reproducible issue, a benchmark, a documentation fix, or a PR — with the response recorded, including no response. |
| Your own negative result | — | X1, X2 | The rarest and most credible artifact in this field. | A published write-up of an optimisation that did not work, with the arithmetic explaining why. |

---

## Deliberately excluded

| Excluded | Why |
|---|---|
| Framework comparison round-ups ("vLLM vs SGLang vs TensorRT-LLM in 2026") | Contracts are rarely stated, numbers are rarely reproducible, and conclusions change per release. Useful only as X1 exercise material. |
| Broad "LLM optimisation" video courses | The technique inventory is available free in §A at higher density. Video's advantage is watching someone *work*, which is what GPU MODE and CS336 provide. |
| CUDA C++ textbooks | Out of scope at this dose; Triton covers the fluency target. Revisit only if a Triton limitation blocks a real need. |
| Distributed-serving and multi-GPU material | Recognition-level (V3) on a single-GPU bench. Reading it now would consume the dose without producing evidence. |
| Anything requiring a subscription | Not a quality judgement — the free primary sources here are sufficient, and paid material would add a cost without adding a feedback channel. |
