
#!/usr/bin/env python3
"""
benchmark_tensor_vs_cuda.py
--------------------------------
A minimal benchmark that compares the raw TFLOPs of a GPU's CUDA cores
(FP32) vs Tensor Cores (FP16 / BF16) using PyTorch.

Usage:
    python benchmark_tensor_vs_cuda.py [--size N] [--iters M] [--dtype fp16|bf16]

Options:
    --size   : matrix dimension N (default: 8192)
    --iters  : number of timed iterations (default: 20)
    --dtype  : mixed‑precision type to force Tensor‑Core usage (default: fp16)
               (bf16 works on newer GPUs; fp16 works on all Tensor‑Core GPUs)

The script prints a short summary table with average time and TFLOPs for
both FP32 (CUDA cores) and the chosen mixed‑precision (Tensor Cores).
"""

import argparse
import time
import statistics
import os
import sys

import torch
from torch import nn
from tqdm import tqdm


print(torch.cuda.get_device_name())
print("\n\n")


# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CUDA vs Tensor‑Core TFLOPs benchmark")
    parser.add_argument("--size", type=int, default=8192,
                        help="Matrix dimension N (default: 8192)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of timed iterations (default: 20)")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                        help="Mixed‑precision data type for Tensor‑Core path (default: fp16)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warm‑up iterations (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


# ----------------------------------------------------------------------
def get_device():
    if not torch.cuda.is_available():
        print("❌ No CUDA device detected. Exiting.")
        sys.exit(1)

    device_idx = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_idx}")

    # Basic device info
    prop = torch.cuda.get_device_properties(device)
    name = prop.name
    cc = f"{prop.major}.{prop.minor}"
    total_mem_gb = prop.total_memory / 1e9

    # Try to read Tensor‑Core count from the driver (available on recent drivers)
    # This is optional – not all drivers expose it.
    tensor_cores = getattr(prop, "multi_processor_count", None)  # placeholder
    # The exact number of Tensor cores per SM varies by architecture; we just
    # print the SM count.
    sm_count = prop.multi_processor_count

    print(f"🔧 Device: {name} (CUDA {cc})")
    print(f"   SMs (Streaming Multiprocessors): {sm_count}")
    print(f"   Total memory: {total_mem_gb:.2f} GB")
    print(f"   Compute Capability: {cc}")
    print("-" * 60)

    return device


# ----------------------------------------------------------------------
def create_matrices(N: int, device: torch.device, dtype: torch.dtype):
    """
    Allocate two random N×N matrices on the GPU.
    dtype must be torch.float32, torch.float16 or torch.bfloat16.
    """
    torch.manual_seed(0)  # deterministic values (does not affect timing)
    A = torch.randn(N, N, device=device, dtype=dtype, requires_grad=False)
    B = torch.randn(N, N, device=device, dtype=dtype, requires_grad=False)
    return A, B


# ----------------------------------------------------------------------
def timed_matmul_fp32(A: torch.Tensor, B: torch.Tensor, iters: int, warmup: int):
    """
    Runs C = A @ B using FP32 (CUDA cores). Returns a list of elapsed times (seconds).
    """
    # Use CUDA events for high‑resolution timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warm‑up
    for _ in range(warmup):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()

    times = []
    for _ in tqdm(range(iters), desc="FP32 (CUDA cores)"):
        start.record()
        C = torch.matmul(A, B)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)  # ms
        times.append(elapsed_ms / 1e3)       # convert to seconds
    return times


# ----------------------------------------------------------------------
def timed_matmul_tensor_core(A: torch.Tensor, B: torch.Tensor,
                             iters: int, warmup: int, dtype: torch.dtype):
    """
    Runs C = A @ B under autocast so that Tensor Cores are used (FP16/BF16).
    Returns a list of elapsed times (seconds).
    """
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # we only need autocast, not scaling

    # Warm‑up
    for _ in range(warmup):
        with torch.cuda.amp.autocast(dtype=dtype):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in tqdm(range(iters), desc=f"{dtype} (Tensor Cores)"):
        start.record()
        with torch.cuda.amp.autocast(dtype=dtype):
            C = torch.matmul(A, B)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms / 1e3)
    return times


# ----------------------------------------------------------------------
def compute_tflops(N: int, seconds: float) -> float:
    """
    For a dense matrix multiply C = A·B (N×N), the FLOP count is 2·N³.
    TFLOPs = (2·N³) / (seconds·1e12)
    """
    flops = 2.0 * (N ** 3)
    return flops / (seconds * 1e12)


# ----------------------------------------------------------------------
def summarize(times: list[float], N: int, label: str):
    avg = statistics.mean(times)
    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    tflops = compute_tflops(N, avg)

    print(f"\n=== {label} ===")
    print(f"  Runs          : {len(times)}")
    print(f"  Avg time (s)  : {avg:.6f}")
    print(f"  Median (s)    : {med:.6f}")
    print(f"  Std dev (s)   : {std:.6f}")
    print(f"  TFLOPs (avg)  : {tflops:.3f} TFLOPs")
    return tflops


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device()
    N = args.size
    iters = args.iters
    warmup = args.warmup

    # ------------------------------------------------------------------
    # 1️⃣ FP32 path (CUDA cores)
    # ------------------------------------------------------------------
    A_fp32, B_fp32 = create_matrices(N, device, torch.float32)
    fp32_times = timed_matmul_fp32(A_fp32, B_fp32, iters, warmup)
    fp32_tflops = summarize(fp32_times, N, "FP32 – CUDA cores")

    # ------------------------------------------------------------------
    # 2️⃣ Tensor‑Core path (mixed precision)
    # ------------------------------------------------------------------
    # Choose the dtype for autocast
    mixed_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # Note: Tensor Cores require the inputs to be in the same low‑precision type.
    # We'll cast the FP32 matrices once and reuse them.
    A_mixed = A_fp32.to(mixed_dtype)
    B_mixed = B_fp32.to(mixed_dtype)

    tensor_times = timed_matmul_tensor_core(A_mixed, B_mixed,
                                            iters, warmup, mixed_dtype)
    tensor_tflops = summarize(tensor_times, N, f"{args.dtype.upper()} – Tensor Cores")

    # ------------------------------------------------------------------
    # 3️⃣ Relative speed‑up
    # ------------------------------------------------------------------
    speedup = fp32_tflops / tensor_tflops
    print("\n🚀 Relative performance")
    print(f"   FP32 TFLOPs : {fp32_tflops:.3f}")
    print(f"   {args.dtype.upper()} TFLOPs : {tensor_tflops:.3f}")
    print(f"   Speed‑up (FP32 / {args.dtype.upper()}) : {speedup:.2f}×")

    # ------------------------------------------------------------------
    # 4️⃣ Optional: Write a CSV log for later analysis
    # ------------------------------------------------------------------
    csv_line = f"{device.index},{N},{iters},{fp32_tflops:.6f},{tensor_tflops:.6f},{speedup:.3f}\n"
    csv_path = os.getenv("TFLOPS_BENCH_CSV", "tflops_benchmark.csv")
    header = "device_id,matrix_N,iters,fp32_tflops,mixed_tflops,speedup\n"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write(header)
        f.write(csv_line)
    print(f"\n📊 Results appended to {csv_path}")

if __name__ == "__main__":
    main()

