// nvcc -O3 -arch=sm_80 --ptxas-options=-v launch_bounds.cu -o launch_bounds
//
// The unconstrained kernel keeps many live values. launch_bounds asks
// the compiler for 8 resident blocks of 256 threads, which forces the
// register budget down. Watch ptxas: "used N registers" falls, and
// "spill stores" may appear. Spills are local memory, not a free win.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void unconstrained(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    // A long dependent chain of named values. The compiler will keep
    // as many as the register file allows; it has no hint to do otherwise.
    float a0 = x, a1 = x + 1, a2 = x + 2, a3 = x + 3;
    float a4 = x + 4, a5 = x + 5, a6 = x + 6, a7 = x + 7;
    #pragma unroll 1
    for (int k = 0; k < 64; ++k) {
        a0 = a0 * 1.001f + a1; a1 = a1 * 1.001f + a2;
        a2 = a2 * 1.001f + a3; a3 = a3 * 1.001f + a4;
        a4 = a4 * 1.001f + a5; a5 = a5 * 1.001f + a6;
        a6 = a6 * 1.001f + a7; a7 = a7 * 1.001f + a0;
    }
    a[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

// minBlocksPerMultiprocessor = 8, max threads = 256 → 2048 resident threads,
// the whole SM on A100/H100. The compiler must fit 8 blocks in 65536 registers.
__global__ void __launch_bounds__(256, 8)
bounded(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float a0 = x, a1 = x + 1, a2 = x + 2, a3 = x + 3;
    float a4 = x + 4, a5 = x + 5, a6 = x + 6, a7 = x + 7;
    #pragma unroll 1
    for (int k = 0; k < 64; ++k) {
        a0 = a0 * 1.001f + a1; a1 = a1 * 1.001f + a2;
        a2 = a2 * 1.001f + a3; a3 = a3 * 1.001f + a4;
        a4 = a4 * 1.001f + a5; a5 = a5 * 1.001f + a6;
        a6 = a6 * 1.001f + a7; a7 = a7 * 1.001f + a0;
    }
    a[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

int main() {
    int u = 0, b = 0;
    cudaError_t eu = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&u, unconstrained, 256, 0);
    cudaError_t eb = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, bounded, 256, 0);
    if (eu != cudaSuccess || eb != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(eu != cudaSuccess ? eu : eb));
        return 1;
    }
    printf("unconstrained: %d blocks/SM (%d warps)\n", u, u * 8);
    printf("launch_bounds(256, 8): %d blocks/SM (%d warps)\n", b, b * 8);
    printf("if the bounded kernel spilled, the higher count can still be slower — time both\n");
    return 0;
}
