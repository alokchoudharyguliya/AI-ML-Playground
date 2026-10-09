// nvcc -O3 -arch=sm_80 divergence.cu -o divergence
//
// Two kernels do the same arithmetic. uniform() is one path for the whole
// warp. split() takes both arms, so the warp issues the heavy loop twice.
// Time them with Nsight Compute (smsp__thread_inst_executed vs eligible
// warps). The source difference is only the predicate.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void uniform(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                 // the tail warp may be partial; that is fine
    float x = a[i];                     // this branch is uniform inside a full warp
    if ((blockIdx.x & 1) == 0) {        // predicate is the same for every lane
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) x = x * 1.0001f + 0.001f;
    } else {
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) x = x * 0.9999f - 0.001f;
    }
    a[i] = x;
}

__global__ void split(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    if ((threadIdx.x & 1) == 0) {       // even and odd lanes both exist in every warp
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) x = x * 1.0001f + 0.001f;
    } else {
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) x = x * 0.9999f - 0.001f;
    }
    a[i] = x;
}

// Do NOT write this. The else-lanes never reach the barrier:
//
//   if (threadIdx.x < 16) { ... __syncthreads(); }
//   else                  { ... __syncthreads(); }
//
// A block barrier must be textually reached by every thread of the block,
// on every iteration, with no divergent guard around it.

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    const int n = 1 << 20;
    float *a;
    CHECK(cudaMalloc(&a, n * sizeof(float)));
    CHECK(cudaMemset(a, 0, n * sizeof(float)));
    int block = 256;
    int grid = (n + block - 1) / block;
    uniform<<<grid, block>>>(a, n);
    split<<<grid, block>>>(a, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(a));
    printf("launched uniform and split over %d threads\n", n);
    return 0;
}
