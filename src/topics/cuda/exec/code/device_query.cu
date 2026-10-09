#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        printf("== Device %d: %s (compute capability %d.%d)\n", d, p.name, p.major, p.minor);
        printf("  SMs                      : %d\n", p.multiProcessorCount);
        printf("  warp size                : %d\n", p.warpSize);
        printf("  max threads / block      : %d\n", p.maxThreadsPerBlock);
        printf("  max threads / SM         : %d  (= %d warps)\n", p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / p.warpSize);
        printf("  max blocks / SM          : %d\n", p.maxBlocksPerMultiProcessor);
        printf("  registers / SM, / block  : %d, %d\n", p.regsPerMultiprocessor, p.regsPerBlock);
        printf("  shared mem / SM, / block : %zu KB, %zu KB\n", p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerBlock / 1024);
        printf("  L2 cache                 : %d MB\n", p.l2CacheSize / (1024 * 1024));
        printf("  global memory            : %.1f GB\n", p.totalGlobalMem / 1e9);
        int clk = 0, memclk = 0;
        cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, d);          // kHz
        cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, d); // kHz
        double bw = 2.0 * memclk * 1e3 * (p.memoryBusWidth / 8) / 1e9;  // DDR: x2
        printf("  memory bus / bandwidth   : %d-bit, ~%.0f GB/s\n", p.memoryBusWidth, bw);
        printf("  boost clock              : %.2f GHz\n", clk / 1e6);
    }
    return 0;
}
