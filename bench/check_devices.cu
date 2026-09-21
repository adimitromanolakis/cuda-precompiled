#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }
    printf("CUDA Devices (%d):\n", deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f MB\n", totalMem / 1024.0 / 1024.0);
        printf("  Free Memory:  %.2f MB\n", freeMem / 1024.0 / 1024.0);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("\n");
    }
    return 0;
}
