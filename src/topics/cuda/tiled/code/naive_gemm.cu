// Naive GEMM: C = A B, square N×N, row-major.
// Each thread owns one C[row, col] and streams a whole row of A and column of B from global memory.
// Global traffic: 2 N³ loads + N² stores  →  AI = 1/s  (0.25 FLOP/B in FP32)  — always memory-bound.
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 naive_gemm.cu -o naive && ./naive 1024
#include <cstdio>
#include <cstdlib>
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

__global__ void gemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];     // A row coalesced; B column is strided (N floats)
    C[row * N + col] = acc;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 1024;
    const size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> hA(N * N), hB(N * N), hC(N * N);
    for (int i = 0; i < N * N; ++i) { hA[i] = 1.f; hB[i] = 1.f; }   // C should be N everywhere

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t t0, t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    gemm_naive<<<grid, block>>>(A, B, C, N);                        // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    gemm_naive<<<grid, block>>>(A, B, C, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
    printf("naive GEMM N=%d: %.3f ms, %.1f GFLOP/s  (C[0]=%.0f, expected %d)\n",
           N, ms, 2.0 * N * (double)N * N / (ms * 1e6), hC[0], N);

    // Traffic model: 2 N³ loads + N² stores of 4 B. At N=1024 that is 8.6 GB of loads.
    // Peak A100 HBM ~2 TB/s ⇒ lower bound ~4 ms even with perfect coalescing — and B's
    // column reads are *not* coalesced. Tiling (next file) cuts the loads by T and coalesces both.
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
