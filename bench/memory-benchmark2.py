#!/usr/bin/env python3
import argparse
import statistics
import time

import torch


def benchmark_gpu(size_gib, warmup, iters):
    if not torch.cuda.is_available():
        print("GPU: CUDA/ROCm not available, skipping.")
        return

    device = torch.device("cuda")

    # FP16 = 2 bytes/element
    nbytes = int(size_gib * 1024**3)
    nelems = nbytes // 2
    actual_bytes = nelems * 2

    print("=== GPU ===")
    print(f"Device : {torch.cuda.get_device_name(device)}")
    print(f"Buffer : {actual_bytes / 1024**3:.2f} GiB FP16")

    x = torch.empty(nelems, dtype=torch.float16, device=device)
    x.fill_(1.0)

    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        y = x.sum()

    torch.cuda.synchronize()

    times_ms = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        start.record()
        y = x.sum()
        end.record()

        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    result = y.item()

    median_ms = statistics.median(times_ms)
    best_ms = min(times_ms)

    def bw(ms):
        return actual_bytes / (ms / 1000) / 1e9

    print(f"Result : {result:.0f}")
    print(f"Best   : {best_ms:.3f} ms  ({bw(best_ms):.2f} GB/s)")
    print(f"Median : {median_ms:.3f} ms  ({bw(median_ms):.2f} GB/s)")
    print()


def benchmark_cpu(size_gib, warmup, iters, threads):
    # FP32 = 4 bytes/element
    nbytes = int(size_gib * 1024**3)
    nelems = nbytes // 4
    actual_bytes = nelems * 4

    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)

    print("=== CPU DRAM ===")
    print(f"Threads : {torch.get_num_threads()}")
    print(f"Buffer  : {actual_bytes / 1024**3:.2f} GiB FP32")

    # Allocate in CPU memory.
    x = torch.empty(nelems, dtype=torch.float32)

    # Touch every page / initialize the allocation.
    # This also avoids measuring first-touch/page-fault overhead.
    x.fill_(1.0)

    # Warmup
    for _ in range(warmup):
        y = x.sum()

    times_ms = []

    for _ in range(iters):
        t0 = time.perf_counter()

        y = x.sum()

        # Force materialization of the scalar.
        result = y.item()

        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    median_ms = statistics.median(times_ms)
    best_ms = min(times_ms)

    def bw(ms):
        return actual_bytes / (ms / 1000) / 1e9

    print(f"Result  : {result:.0f}")
    print(f"Best    : {best_ms:.3f} ms  ({bw(best_ms):.2f} GB/s)")
    print(f"Median  : {median_ms:.3f} ms  ({bw(median_ms):.2f} GB/s)")
    print()

    torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch CPU DRAM + CUDA/ROCm GPU memory benchmark"
    )

    parser.add_argument(
        "--gb",
        type=float,
        default=4.0,
        help="Buffer size in GiB (default: 4)"
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Timed iterations (default: 20)"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (default: 5)"
    )

    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=torch.get_num_threads(),
        help="CPU threads (default: PyTorch default)"
    )

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Only run CPU benchmark"
    )

    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Only run GPU benchmark"
    )

    args = parser.parse_args()

    print(f"PyTorch : {torch.__version__}")
    print()

    if not args.gpu_only:
        benchmark_cpu(
            args.gb,
            args.warmup,
            args.iters,
            args.cpu_threads,
        )

    if not args.cpu_only:
        benchmark_gpu(
            args.gb,
            args.warmup,
            args.iters,
        )


if __name__ == "__main__":
    main()

