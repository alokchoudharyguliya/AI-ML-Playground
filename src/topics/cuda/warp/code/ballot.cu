// nvcc -O3 -arch=sm_80 ballot.cu -o ballot
//
// Warp-local stream compaction. __ballot_sync packs the 32 predicates
// into one word; popc of the bits below this lane is the dense index.
// No shared memory. The warp writes popc(bits) outputs, tightly packed.

#include <cstdio>
#include <cuda_runtime.h>

__device__ void warp_compact(int pred, int value, int* out, int* counter) {
    const unsigned mask = 0xffffffffu;
    // Every lane calls ballot, whether or not pred is true.
    unsigned bits = __ballot_sync(mask, pred);
    int lane = threadIdx.x & 31;
    int idx = __popc(bits & ((1u << lane) - 1u));   // how many hits are below me
    int total = __popc(bits);

    __shared__ int base;
    if (lane == 0) base = atomicAdd(counter, total);
    __syncwarp(mask);                               // publish `base` to the rest of this warp
    if (pred) out[base + idx] = value;
}

// One warp per block so the shared `base` does not need a block barrier.
// A multi-warp block would stage per-warp totals, __syncthreads(), then prefix them.
__global__ void compact_positive(const int* in, int* out, int* counter, int n) {
    int i = blockIdx.x * 32 + (threadIdx.x & 31);
    int pred = 0, value = 0;
    if (i < n) { value = in[i]; pred = value > 0; }
    warp_compact(pred, value, out, counter);
}

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    const int n = 1024;
    int *h = new int[n];
    int expect = 0;
    for (int i = 0; i < n; ++i) { h[i] = (i % 3 == 0) ? i : -i; if (h[i] > 0) ++expect; }
    // i == 0 is not > 0, so expect counts 3,6,...,1023 → 341.

    int *d_in, *d_out, *d_count;
    CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    CHECK(cudaMalloc(&d_count, sizeof(int)));
    CHECK(cudaMemcpy(d_in, h, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_count, 0, sizeof(int)));

    compact_positive<<<(n + 31) / 32, 32>>>(d_in, d_out, d_count, n);
    int got = 0;
    CHECK(cudaMemcpy(&got, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("positives: gpu %d  cpu %d  %s\n", got, expect, got == expect ? "ok" : "MISMATCH");

    delete[] h;
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_count));
    return 0;
}
