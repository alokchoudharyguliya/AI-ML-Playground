// What __syncthreads() is for: a cooperative load, then a neighbour read.
// Variant 0 skips the barrier — some threads compute with a neighbour that has not yet written.
// Variant 1 inserts the barrier and the result is deterministic.
//
// build: nvcc -O3 -arch=sm_80 syncthreads_race.cu -o race && ./race
#include <cstdio>
#include <cuda_runtime.h>

__global__ void with_race(int* out) {
    __shared__ int s[32];
    s[threadIdx.x] = threadIdx.x;               // each thread publishes its id
    // MISSING __syncthreads();
    out[threadIdx.x] = s[(threadIdx.x + 1) & 31];   // read the neighbour — may be stale
}

__global__ void with_sync(int* out) {
    __shared__ int s[32];
    s[threadIdx.x] = threadIdx.x;
    __syncthreads();                            // all 32 writes are visible
    out[threadIdx.x] = s[(threadIdx.x + 1) & 31];
}

int main() {
    int *d0, *d1, h0[32], h1[32];
    cudaMalloc(&d0, 32 * sizeof(int));
    cudaMalloc(&d1, 32 * sizeof(int));

    with_race<<<1, 32>>>(d0);
    with_sync<<<1, 32>>>(d1);
    cudaDeviceSynchronize();
    cudaMemcpy(h0, d0, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h1, d1, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("with    sync: ");
    for (int i = 0; i < 32; ++i) printf("%2d ", h1[i]);   // always 1,2,...,31,0
    printf("\nwithout sync: ");
    for (int i = 0; i < 32; ++i) printf("%2d ", h0[i]);   // often looks right (one warp!), but is a data race
    printf("\n");
    printf("On a *single warp* the race is often invisible because the 32 lanes already run in lockstep.\n");
    printf("Launch the same pattern with 256 threads (8 warps) and the missing barrier bites: some warps\n");
    printf("read s[] before other warps have written.  compute-sanitizer --tool racecheck ./race flags it.\n");
    cudaFree(d0); cudaFree(d1);
    return 0;
}
