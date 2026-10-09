// Tiled GEMM: C = A B. Each block owns a TILE×TILE output tile.
// Threads cooperatively stage TILE×TILE slabs of A and B in shared memory, reuse them T times, then
// slide along K. Global traffic drops by ~T; both loads are coalesced (x = threadIdx.x is contiguous).
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 tiled_gemm.cu -o tiled && ./tiled 1024
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
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

#ifndef TILE
#define TILE 16                                 // 16×16 = 256 threads; 2×16×16×4 B = 2 KB shared
#endif

__global__ void gemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    // Number of TILE×TILE slabs along K. Works for any N: out-of-range loads become 0.
    const int ntiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < ntiles; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.f;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();                        // every thread must see the whole tiles before MAC

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];      // As: broadcast along the warp; Bs: consecutive banks

        __syncthreads();                        // don't let the next load overwrite tiles still in use
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 1024;
    const size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> hA(N * N), hB(N * N), hC(N * N);
    for (int i = 0; i < N * N; ++i) { hA[i] = ((i * 17) % 11) * 0.1f; hB[i] = ((i * 13) % 7) * 0.1f; }

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t t0, t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    gemm_tiled<<<grid, block>>>(A, B, C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    gemm_tiled<<<grid, block>>>(A, B, C, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));

    // Spot-check a few C[i,j] against a CPU inner product (not the whole N³ — too slow at 1024).
    int bad = 0;
    for (int sample = 0; sample < 8; ++sample) {
        int i = (sample * 97) % N, j = (sample * 53) % N;
        double gold = 0;
        for (int k = 0; k < N; ++k) gold += (double)hA[i * N + k] * hB[k * N + j];
        if (fabs(hC[i * N + j] - gold) > 1e-2 * (1 + fabs(gold))) ++bad;
    }
    const double gflops = 2.0 * N * (double)N * N / (ms * 1e6);
    const double loads_gb = 2.0 * N * (double)N * N / TILE * 4 / 1e9;     // model: 2 N³/T floats
    printf("tiled GEMM  TILE=%d  N=%d: %.3f ms, %.1f GFLOP/s\n", TILE, N, ms, gflops);
    printf("  model traffic %.2f GB  →  %.0f GB/s effective  (spot-check %s)\n",
           loads_gb, loads_gb / (ms * 1e-3), bad ? "MISMATCH" : "ok");

    // Try TILE=32: nvcc -DTILE=32 ...  (1024 threads/block, 8 KB shared). Bigger tiles ⇒ more reuse
    // but fewer resident blocks (Occupancy chapter). Bank conflicts: see the next chapter.
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
