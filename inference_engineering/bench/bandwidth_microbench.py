"""W0: streaming source-read bandwidth microbenchmark.

Reports effective bandwidth as input bytes read once per torch.sum kernel divided by
CUDA-synchronised wall time. It is deliberately a contract-specific source-read
bandwidth measurement, not a claim about universal HBM bandwidth.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-bytes", type=int, default=1 << 30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("W0 requires CUDA")
    if args.tensor_bytes % torch.tensor([], dtype=torch.float32).element_size():
        raise ValueError("tensor bytes must be divisible by float32 element size")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    elements = args.tensor_bytes // torch.tensor([], dtype=torch.float32).element_size()
    source = torch.rand(elements, device=device, dtype=torch.float32)

    # Allocate and touch outside the timed region. The source is 1 GiB, far above
    # the RTX 4090 L2 cache; each timed reduction streams the complete input.
    torch.cuda.synchronize(device)
    for _ in range(args.warmup):
        checksum = source.sum(dtype=torch.float32)
        torch.cuda.synchronize(device)

    elapsed_s: list[float] = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        checksum = source.sum(dtype=torch.float32)
        torch.cuda.synchronize(device)
        elapsed_s.append(time.perf_counter() - start)

    # Keep the result observable after all timed repetitions without timing .item().
    checksum_value = float(checksum.cpu())
    bandwidth_gbps = [args.tensor_bytes / sample / 1e9 for sample in elapsed_s]
    result = {
        "contract": "W0",
        "operation": "torch.sum(float32) over one contiguous CUDA float32 tensor",
        "source_bytes_per_rep": args.tensor_bytes,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "synchronization": "torch.cuda.synchronize after every warmup and timed repetition",
        "checksum": checksum_value,
        "elapsed_ms": {
            "mean": statistics.mean(elapsed_s) * 1e3,
            "p50": percentile(elapsed_s, 0.50) * 1e3,
            "p90": percentile(elapsed_s, 0.90) * 1e3,
            "p99": percentile(elapsed_s, 0.99) * 1e3,
        },
        "effective_source_read_gbps": {
            "mean": statistics.mean(bandwidth_gbps),
            "p50": percentile(bandwidth_gbps, 0.50),
            "p90": percentile(bandwidth_gbps, 0.90),
            "p99": percentile(bandwidth_gbps, 0.99),
        },
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
