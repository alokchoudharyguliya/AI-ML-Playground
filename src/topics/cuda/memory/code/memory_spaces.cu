// Every CUDA memory space in one kernel.
// build:   nvcc -O3 -arch=sm_80 -Xptxas -v memory_spaces.cu -o spaces
// -Xptxas -v prints per-kernel register count, shared bytes, constant bytes and stack frame / spills.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__constant__ float c_coef[4];          // CONSTANT: 64 KB total, read-only in kernels, broadcast through the constant cache
__device__   unsigned int g_counter;   // GLOBAL: statically allocated, visible to every thread and kernel

#define BLOCK 256

__global__ void spaces(const float* __restrict__ in, float* __restrict__ out, const int* __restrict__ pick, int n) {
    __shared__ float tile[BLOCK];                       // SHARED: one copy per block, on-chip

    int i = blockIdx.x * blockDim.x + threadIdx.x;      // REGISTER (copied from special registers)
    float x = (i < n) ? in[i] : 0.f;                    // REGISTER, filled from GLOBAL memory (HBM -> L2 -> L1 -> reg)

    tile[threadIdx.x] = x;                              // register -> shared
    __syncthreads();
    float nb = tile[(threadIdx.x + 1) % BLOCK];         // shared -> register (a neighbour's value: reuse!)

    float y = c_coef[0] * x + c_coef[1] * nb;           // every lane reads the SAME constant address: 1 broadcast

    // Static indices after unrolling -> registers. No memory traffic at all.
    float acc[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = y * (j + 1);

    // A RUNTIME index into a per-thread array cannot live in registers (registers are not addressable).
    // The compiler puts `scratch` in LOCAL memory: private to the thread, physically in device memory.
    float scratch[16];
    for (int j = 0; j < 16; ++j) scratch[j] = y + j;
    float dyn = (i < n) ? scratch[pick[i] & 15] : 0.f;  // dynamic index -> local-memory load (look for "stack frame" in -Xptxas -v)

    if (i < n) out[i] = acc[0] + acc[3] + dyn;
    if (threadIdx.x == 0) atomicAdd(&g_counter, 1u);    // atomics are resolved in L2
}

// Same computation, but dynamic indexing replaced by a shared-memory slice per thread:
// the spill disappears at the price of shared memory (and potential bank conflicts, see next chapters).
__global__ void spaces_shared_scratch(const float* __restrict__ in, float* __restrict__ out, const int* __restrict__ pick, int n) {
    __shared__ float scratch[16][BLOCK + 1];            // +1 column of padding to dodge bank conflicts
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (i < n) ? in[i] : 0.f;
    float y = c_coef[0] * x + c_coef[1] * x;
    for (int j = 0; j < 16; ++j) scratch[j][threadIdx.x] = y + j;
    float dyn = (i < n) ? scratch[pick[i] & 15][threadIdx.x] : 0.f;
    if (i < n) out[i] = dyn;
}

int main() {
    const int n = 1 << 22;
    float h_c[4] = {0.5f, 0.25f, 0.f, 0.f};
    CUDA_CHECK(cudaMemcpyToSymbol(c_coef, h_c, sizeof(h_c)));

    float *d_in, *d_out; int* d_pick;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pick, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_in, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pick, 0, n * sizeof(int)));

    // Opt in to more shared memory per block / change the L1-vs-shared split (hint, in percent of the maximum shared size).
    CUDA_CHECK(cudaFuncSetAttribute(spaces, cudaFuncAttributePreferredSharedMemoryCarveout, 25));

    int blocks = (n + BLOCK - 1) / BLOCK;
    cudaEvent_t t0, t1; float ms;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    for (int variant = 0; variant < 2; ++variant) {
        for (int w = 0; w < 2; ++w) {                   // second pass is the timed one (first warms up)
            CUDA_CHECK(cudaEventRecord(t0));
            if (variant == 0) spaces<<<blocks, BLOCK>>>(d_in, d_out, d_pick, n);
            else              spaces_shared_scratch<<<blocks, BLOCK>>>(d_in, d_out, d_pick, n);
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        printf("%-22s %.3f ms\n", variant == 0 ? "local-memory scratch" : "shared-memory scratch", ms);
    }

    unsigned int h_counter = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_counter, g_counter, sizeof(h_counter)));
    printf("blocks that incremented the global counter: %u (expected %d x 2 launches)\n", h_counter, blocks);

    // Try:  nvcc -Xptxas -v ...                 -> compare "stack frame" of the two kernels
    //       nvcc -maxrregcount=32 -Xptxas -v ... -> force spills and watch "spill stores / loads" appear
    //       ncu --section MemoryWorkloadAnalysis ./spaces   -> local-memory traffic per level
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_pick);
    return 0;
}
