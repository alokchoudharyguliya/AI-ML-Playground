// Matrix transpose is the kernel that gets both lessons wrong at once.
//   naive:          coalesced read, stride-N write          (global)
//   tiled, no pad:  coalesced global read AND write, but the
//                   shared-memory column read is a 32-way bank conflict
//   tiled, padded:  tile[TILE][TILE+1] removes the conflict
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 transpose.cu -o transpose && ./transpose 2048
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
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

constexpr int TILE = 32;

__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;          // consecutive across the warp
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) out[x * N + y] = in[y * N + x];     // read coalesced, write stride N
}

__global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out, int N) {
    __shared__ float tile[TILE][TILE];                      // no padding
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    int x2 = blockIdx.y * TILE + threadIdx.x;               // consecutive → coalesced store
    int y2 = blockIdx.x * TILE + threadIdx.y;
    if (x2 < N && y2 < N) out[y2 * N + x2] = tile[threadIdx.x][threadIdx.y];   // column read: 32-way
}

__global__ void transpose_padded(const float* __restrict__ in, float* __restrict__ out, int N) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    int x2 = blockIdx.y * TILE + threadIdx.x;
    int y2 = blockIdx.x * TILE + threadIdx.y;
    if (x2 < N && y2 < N) out[y2 * N + x2] = tile[threadIdx.x][threadIdx.y];   // bank = (tx*(TILE+1) + ty) % 32
}

template <typename Fn>
float best_ms(Fn launch) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < 8; ++r) {
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

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 2048;
    const size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> h(N * N), back(N * N);
    for (int i = 0; i < N * N; ++i) h[i] = (float)i;

    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemcpy(in, h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);                                 // 32-wide: each warp is one row (consecutive threadIdx.x)
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    auto report = [&](const char* name, float ms) {
        double gb = 2.0 * bytes / 1e9;                      // read N² + write N²
        printf("%-18s %7.3f ms   %6.0f GB/s\n", name, ms, gb / (ms * 1e-3));
    };
    report("naive", best_ms([&] { transpose_naive<<<grid, block>>>(in, out, N); }));
    report("tiled, no pad", best_ms([&] { transpose_tiled<<<grid, block>>>(in, out, N); }));
    float ms = best_ms([&] { transpose_padded<<<grid, block>>>(in, out, N); });
    report("tiled, pad +1", ms);

    CUDA_CHECK(cudaMemcpy(back.data(), out, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N && bad < 3; i += N / 7)
        for (int j = 0; j < N && bad < 3; j += N / 5)
            if (back[i * N + j] != h[j * N + i]) ++bad;
    printf("spot-check %s\n", bad ? "MISMATCH" : "ok");
    cudaFree(in); cudaFree(out);
    return 0;
}
