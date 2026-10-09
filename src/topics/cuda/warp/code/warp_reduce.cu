// nvcc -O3 -arch=sm_80 warp_reduce.cu -o warp_reduce
//
// Block sum. Each warp reduces in registers (5 shuffles, no barrier).
// Lane 0 of each warp writes one partial, the block synchronises once,
// and warp 0 reduces the partials. Every lane of the warp must execute
// the shuffles — including lanes that have no element left.

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ float warp_sum_down(float v) {
    // Result is valid in lane 0 only.
    const unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(mask, v, offset);
    return v;
}

__device__ float warp_sum_all(float v) {
    // Result is valid in every lane. Partner is lane ^ offset, always in range.
    const unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(mask, v, offset);
    return v;
}

__global__ void block_sum(const float* in, float* partial, int n) {
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        v += in[i];

    // Full mask: padding lanes hold 0 and still have to reach the shuffle.
    v = warp_sum_down(v);

    __shared__ float smem[32];                 // at most 32 warps in a block
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    int nwarps = blockDim.x >> 5;
    float w = (lane < nwarps) ? smem[lane] : 0.f;
    if (warp == 0) w = warp_sum_down(w);
    if (threadIdx.x == 0) partial[blockIdx.x] = w;
}

// warp_sum_all is the same five steps when the caller needs the sum
// on every lane (a broadcast without a sixth shuffle).

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    const int n = 1 << 16;
    const int block = 256;
    const int grid = 64;
    float *d_in, *d_part, *h_in, *h_part;
    CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK(cudaMalloc(&d_part, grid * sizeof(float)));
    h_in = new float[n];
    h_part = new float[grid];
    double cpu = 0.0;
    for (int i = 0; i < n; ++i) { h_in[i] = 1.f; cpu += 1.0; }
    CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    block_sum<<<grid, block>>>(d_in, d_part, n);
    CHECK(cudaMemcpy(h_part, d_part, grid * sizeof(float), cudaMemcpyDeviceToHost));
    double gpu = 0.0;
    for (int i = 0; i < grid; ++i) gpu += h_part[i];
    printf("cpu %.0f  gpu %.0f  %s\n", cpu, gpu, std::fabs(cpu - gpu) < 1e-2 ? "ok" : "MISMATCH");

    delete[] h_in; delete[] h_part;
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_part));
    return 0;
}
