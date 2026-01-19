import torch
import time

def benchmark_gemm_flops(
    M=8192,
    N=8192,
    K=8192,
    dtype=torch.float16,
    iters=50
):
    assert torch.cuda.is_available(), "CUDA not available"

    device = "cuda"

    A = torch.randn((M, K), device=device, dtype=dtype)
    B = torch.randn((K, N), device=device, dtype=dtype)

    # Optional: enable TF32 for FP32
    if dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Warm-up
    for _ in range(10):
        C = A @ B
    torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    for _ in range(iters):
        C = A @ B
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start

    flops_per_iter = 2 * M * N * K
    total_flops = flops_per_iter * iters
    tflops = total_flops / elapsed / 1e12

    print("GEMM FLOPS Benchmark")
    print(f"Matrix sizes: M={M}, N={N}, K={K}")
    print(f"dtype: {dtype}")
    print(f"Iterations: {iters}")
    print(f"Elapsed time: {elapsed:.4f} s")
    print(f"Achieved performance: {tflops:.2f} TFLOPS")
    print("\n")


def benchmark_memory_bandwidth(
    size_mb=1024,
    dtype=torch.float32,
    iters=100,
    device="cuda"
):
    assert torch.cuda.is_available(), "CUDA is not available"

    # Convert MB → number of elements
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    num_elems = (size_mb * 1024 * 1024) // bytes_per_elem

    # Allocate tensors
    src = torch.empty(num_elems, device=device, dtype=dtype)
    dst = torch.empty_like(src)

    # Warm-up
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    total_bytes = size_mb * 1024 * 1024 * iters
    bandwidth_gbps = total_bytes / elapsed / 1e9

    print("GPU → GPU Bandwidth Benchmark")
    print(f"  Elapsed time: {elapsed:.4f} s, {iters} x {size_mb} MB")
    print(f"  Effective bandwidth: {bandwidth_gbps:.2f} GB/s")




def benchmark_cpu_to_gpu_bandwidth(
    size_mb=1024,
    dtype=torch.float32,
    iters=100
):
    assert torch.cuda.is_available(), "CUDA not available"

    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    num_elems = (size_mb * 1024 * 1024) // bytes_per_elem

    # Pinned (page-locked) host memory
    host_tensor = torch.empty(
        num_elems,
        dtype=dtype,
        pin_memory=True
    )

    device_tensor = torch.empty(
        num_elems,
        dtype=dtype,
        device="cuda"
    )

    # Warm-up
    for _ in range(10):
        device_tensor.copy_(host_tensor, non_blocking=True)
    torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    for _ in range(iters):
        device_tensor.copy_(host_tensor, non_blocking=True)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    total_bytes = size_mb * 1024 * 1024 * iters
    bandwidth_gbps = total_bytes / elapsed / 1e9

    print("CPU → GPU Bandwidth Benchmark")
    print(f"  Elapsed time: {elapsed:.4f} s, {iters} x {size_mb} MB")
    print(f"  Effective bandwidth: {bandwidth_gbps:.2f} GB/s")



if __name__ == "__main__":


    n = 1024
    benchmark_gemm_flops(
        M=n,
        N=n,
        K=n,
        dtype=torch.float16,  # try float32, float16
        iters=200
    )        

    benchmark_gemm_flops(
        M=8192,
        N=8192,
        K=8192,
        dtype=torch.float16,  # try float32, float16
        iters=50
    )

    n = 16384
    benchmark_gemm_flops(
        M=n,
        N=n,
        K=n,
        dtype=torch.float16,  # try float32, float16
        iters=20
    )        


    benchmark_memory_bandwidth(1024, iters = 200)
    print('\n')
    benchmark_cpu_to_gpu_bandwidth(
        size_mb=256,  # try 256, 512, 2048
        iters=200
    )
