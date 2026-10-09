// nvcc -O3 -arch=sm_80 latency_hide.cu -o latency_hide
//
// Same grid, same number of FMAs. `chain` is one dependent accumulator:
// each FMA waits on the previous one, so extra warps are the only
// latency cover. `wide` keeps 8 independent accumulators, so one warp
// has other work while a result is in flight. Wide should need far
// fewer resident warps to reach the same FMA throughput.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void chain(float* a, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    for (int k = 0; k < iters; ++k)
        x = x * 1.0001f + 0.001f;          // I = 1, latency ≈ 4 cycles
    a[i] = x;
}

__global__ void wide(float* a, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x0 = a[i], x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
    float x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
    for (int k = 0; k < iters; ++k) {
        x0 = x0 * 1.0001f + 0.001f; x1 = x1 * 1.0001f + 0.001f;
        x2 = x2 * 1.0001f + 0.001f; x3 = x3 * 1.0001f + 0.001f;
        x4 = x4 * 1.0001f + 0.001f; x5 = x5 * 1.0001f + 0.001f;
        x6 = x6 * 1.0001f + 0.001f; x7 = x7 * 1.0001f + 0.001f;
    }
    a[i] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

static float time_ms(void (*launch)(float*, int, int), float* a, int n, int iters, int grid, int block) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    launch(a, n, iters);                       // warm the launch path
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int r = 0; r < 8; ++r) launch(a, n, iters);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 8.f;
}

// Kernels are launched through a thin wrapper so the timer stays generic.
static void launch_chain(float* a, int n, int iters) { chain<<<(n + 255) / 256, 256>>>(a, n, iters); }
static void launch_wide(float* a, int n, int iters)  { wide<<<(n + 255) / 256, 256>>>(a, n, iters); }

int main() {
    const int n = 1 << 20;
    const int iters = 4096;
    float* a;
    cudaMalloc(&a, n * sizeof(float));
    cudaMemset(a, 0, n * sizeof(float));
    float ms_c = time_ms(launch_chain, a, n, iters, 0, 0);
    float ms_w = time_ms(launch_wide, a, n, iters, 0, 0);
    // chain does `iters` FMAs/thread; wide does 8*iters. Compare per FMA.
    double fma_c = (double)n * iters;
    double fma_w = (double)n * iters * 8;
    printf("chain  %7.3f ms   %.0f GFLOP/s  (one dependent accumulator)\n",
           ms_c, fma_c / ms_c / 1e6);
    printf("wide   %7.3f ms   %.0f GFLOP/s  (eight independent accumulators)\n",
           ms_w, fma_w / ms_w / 1e6);
    cudaFree(a);
    return 0;
}
