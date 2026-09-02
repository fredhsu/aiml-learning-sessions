# Workload contracts

**Status:** template only. No contract is active until the entry diagnostic records one.

A **workload contract** is the complete specification that makes a measurement meaningful. In this curriculum, a number without its contract is not weak evidence — it is not evidence. Contracts are versioned; every result names the contract version it ran under.

## Rules

1. Fill the contract **before** running. A field discovered afterwards is a confound.
2. **Change one dimension at a time.** A ratio between contracts differing in more than one declared dimension is an attribution error, not a speedup.
3. Version on every change: `W1`, `W1.1`, `W2`. Never edit a contract that has results attached; supersede it.
4. Declare thresholds (SLO targets, tolerances) **in the contract, before results exist**. A threshold chosen after seeing results is not a threshold.

## Template

```markdown
### Contract <ID> — <short name>
Status: active | superseded by <ID> | retired
Declared on: YYYY-MM-DD, before any result under this contract

**System under test**
- Engine + version/commit:
- Model + revision/hash:
- dtype / quantisation scheme + recipe:
- Attention backend / kernel:
- Compilation: eager | torch.compile (mode) | CUDA graphs — and capture policy

**Hardware and environment**
- Device, driver, CUDA/PyTorch versions:
- Power/clock state, cooling, and whether clocks were locked:
- Other GPU tenants: none | named

**Workload**
- Input length distribution (source, not just mean):
- Output length policy (fixed, sampled, or EOS-terminated):
- Request arrival: closed-loop concurrency N | open-loop at R req/s
- Sampling parameters (temperature, top-p, seed):
- Total requests / duration:

**Measurement protocol**
- Warmup: iterations discarded, and why that count
- Cache state: cold | warm prefix cache (declare which, and how it was reset)
- Token counting: model tokenizer (name it) — never whitespace
- Repetitions and seeds:
- Synchronisation policy for timing:
- Reported statistics: p50/p90/p99 + mean + n (never mean alone for latency)

**Declared thresholds (before results)**
- SLO targets:
- Parity tolerance and determinism settings:
- What result would falsify the hypothesis under test:

**Reproduction**
- One command:
```

### Contract W0 — 1 GiB streaming source-read bandwidth
Status: active
Declared on: 2026-08-31, before any W0 result

**System under test**
- Engine + version/commit: PyTorch eager `torch.sum`; PyTorch 2.13.0+cu130, resolved by `uv sync` on 2026-08-31
- Model + revision/hash: not applicable — hardware microbenchmark
- dtype / quantisation scheme + recipe: contiguous CUDA float32 source tensor; no quantisation
- Attention backend / kernel: not applicable
- Compilation: eager; no `torch.compile` or CUDA graph capture

**Hardware and environment**
- Device, driver, CUDA/PyTorch versions: NVIDIA GeForce RTX 4090 (24564 MiB); driver 610.57.04; CUDA 13.0 / PyTorch 2.13.0+cu130; Python 3.12.7
- Power/clock state, cooling, and whether clocks were locked: air-cooled; clocks unlocked. Record `nvidia-smi` state immediately before and after the run.
- Other GPU tenants: none known; no other process will be intentionally launched during the run

**Workload**
- Input length distribution (source, not just mean): one contiguous 1 GiB (1,073,741,824-byte) CUDA float32 tensor per repetition; this exceeds L2 capacity and is reduced once with `torch.sum(dtype=torch.float32)`
- Output length policy (fixed, sampled, or EOS-terminated): not applicable
- Request arrival: closed-loop; one reduction at a time
- Sampling parameters (temperature, top-p, seed): `torch.manual_seed(0)`; no model sampling
- Total requests / duration: 10 warmup reductions discarded; 30 timed reductions

**Measurement protocol**
- Warmup: 10 reductions, to initialise CUDA context, allocator, and kernel path; excluded from results
- Cache state: no model or prefix cache. The 1 GiB source cannot reside in L2 in full; allocation and an initial synchronisation are outside timing.
- Token counting: not applicable
- Repetitions and seeds: 30 timed samples; seed 0
- Synchronisation policy for timing: `torch.cuda.synchronize()` after every warmup and timed reduction; `time.perf_counter()` surrounds only the reduction plus that synchronization
- Reported statistics: source-read GB/s and elapsed milliseconds as p50/p90/p99 + mean + n=30

**Declared thresholds (before results)**
- SLO targets: not applicable
- Parity tolerance and determinism settings: checksum is recorded to establish an executed reduction; no independent numerical reference is needed because W0 does not change an inference engine
- What result would falsify the hypothesis under test: the committed 1 GB/s ±10% prediction is falsified if W0 p50 effective source-read bandwidth is outside [0.9, 1.1] GB/s

**Reproduction**
- One command: `uv run python bench/bandwidth_microbench.py --tensor-bytes 1073741824 --warmup 10 --repetitions 30`

## Planned locked workloads

Two contrasting shapes, so that the throughput–latency frontier is measured where the two regimes actually differ. Both are declared before Phase 2 and then frozen.

| ID | Name | Shape | Purpose |
|---|---|---|---|
| `W1` | decode-heavy | short prompts, long generations | The memory-bandwidth-bound regime; where KV pressure, quantisation, and speculation matter most |
| `W2` | prefill-heavy | long prompts, short generations | The compute-bound regime; where chunked prefill, interference, and TTFT behaviour matter most |
| `W3` | mixed under load | blend of `W1` and `W2` at rising offered load | Where queueing, admission control, and p99 behaviour appear at all |

`W3` is the SLO contract for the Phase 4 gate. A configuration that only meets its SLO on `W1` has not met it.
