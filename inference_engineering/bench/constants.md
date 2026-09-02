# Measured hardware constants

Measured constants are contract-scoped observations, not universal GPU properties.

| Device | Constant | Value | Contract / artifact | Conditions and limitations |
|---|---|---:|---|---|
| RTX 4090 | Effective streaming source-read bandwidth, p50 | 948.648 GB/s | W0; `bench/results/W0-2026-08-31.json` | 1 GiB contiguous CUDA float32 tensor; eager `torch.sum`; 10 warmups, 30 synchronized samples; unlocked clocks; measures this reduction's source-read path, not generic model-decode bandwidth. |
| RTX 4090 | Effective streaming source-read bandwidth, mean | 919.987 GB/s | W0; `bench/results/W0-2026-08-31.json` | Same contract. The mean is pulled below p50 by a slow-tail sample; use the latency distribution, not the mean alone, for diagnosis. |
| RTX 4090 | Reduction elapsed time, p50 / p99 | 1.132 ms / 5.743 ms | W0; `bench/results/W0-2026-08-31.json` | Same contract. Cause of the slow tail is unconfirmed; do not attribute it without a controlled follow-up. |

W0 was run with driver 610.57.04, CUDA 13.0, PyTorch 2.13.0+cu130, Python 3.12.7. Pre/post clock-state snapshots are stored beside the result.
