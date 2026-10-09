// Shared-memory bank conflicts, isolated from global traffic.
// One warp has consecutive threadIdx.x and a constant threadIdx.y.
//   row  s[y][tx]     → 32 consecutive words → 32 different banks → no conflict
//   col  s[tx][0]     → stride 32            → all 32 threads hit bank 0 → 32-way
//   pad  s[tx][0]     → leading dim 33       → bank = tx → no conflict
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 smem_banks.cu -o banks && ./banks
#include <cstdio>
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

constexpr int ITERS = 4096;

__global__ void row_read(float* sink) {
    __shared__ float s[32][32];
    s[threadIdx.y][threadIdx.x] = threadIdx.x;
    __syncthreads();
    float a = 0.f;
    for (int it = 0; it < ITERS; ++it)
        for (int k = 0; k < 32; ++k)
            a += s[threadIdx.y][(threadIdx.x + k) & 31];         // rotated row: 32 distinct banks, no conflict
    if (threadIdx.x == 0) sink[blockIdx.x * blockDim.y + threadIdx.y] = a;
}

__global__ void col_read(float* sink) {
    __shared__ float s[32][32];
    s[threadIdx.y][threadIdx.x] = threadIdx.x;
    __syncthreads();
    float a = 0.f;
    for (int it = 0; it < ITERS; ++it)
        for (int k = 0; k < 32; ++k) a += s[threadIdx.x][k];     // warp: 32 different rows, one column → 32-way
    if (threadIdx.x == 0) sink[blockIdx.x * blockDim.y + threadIdx.y] = a;
}

__global__ void col_padded(float* sink) {
    __shared__ float s[32][33];                                  // +1 column: leading dimension coprime with 32
    s[threadIdx.y][threadIdx.x] = threadIdx.x;
    __syncthreads();
    float a = 0.f;
    for (int it = 0; it < ITERS; ++it)
        for (int k = 0; k < 32; ++k) a += s[threadIdx.x][k];     // bank = (tx*33 + k) % 32 = (tx + k) % 32
    if (threadIdx.x == 0) sink[blockIdx.x * blockDim.y + threadIdx.y] = a;
}

template <typename Fn>
float time_ms(Fn launch) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < 6; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    return best;
}

int main() {
    const dim3 block(32, 8);                 // 8 warps; each warp is one row (consecutive threadIdx.x)
    const int blocks = 128;
    float* sink;
    CUDA_CHECK(cudaMalloc(&sink, blocks * block.y * sizeof(float)));

    float row = time_ms([&] { row_read<<<blocks, block>>>(sink); });
    float col = time_ms([&] { col_read<<<blocks, block>>>(sink); });
    float pad = time_ms([&] { col_padded<<<blocks, block>>>(sink); });
    printf("row (broadcast)     %7.2f ms\n", row);
    printf("column, ld=32       %7.2f ms   (32-way bank conflict)\n", col);
    printf("column, ld=33       %7.2f ms   (padded, expect ~row speed)\n", pad);
    printf("conflict / padded   %.1f×\n", col / pad);
    cudaFree(sink);
    return 0;
}
