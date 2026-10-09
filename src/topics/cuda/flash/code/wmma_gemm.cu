// nvcc -O3 -arch=sm_80 wmma_gemm.cu -o wmma_gemm
//
// One warp computes one 16×16 output tile, looping over K in steps of 16.
// Inputs are FP16. The accumulator is FP32: tensor cores round the
// products, then the sum stays in the wider type.
// Every lane of the warp must execute the wmma:: calls. Launch <<<grid, 32>>>.

#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>

using namespace nvcuda;

__global__ void wmma_gemm(const half* A, const half* B, float* C, int N) {
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.f);
    for (int k = 0; k < N; k += 16) {
        const half* a_tile = A + (tile_row * 16) * N + k;
        const half* b_tile = B + k * N + tile_col * 16;
        wmma::load_matrix_sync(a, a_tile, N);
        wmma::load_matrix_sync(b, b_tile, N);
        wmma::mma_sync(c, a, b, c);
    }
    float* c_tile = C + (tile_row * 16) * N + tile_col * 16;
    wmma::store_matrix_sync(c_tile, c, N, wmma::mem_row_major);
}

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    const int N = 64;                                  // multiple of 16
    std::vector<half> A(N * N, __float2half(1.f));
    std::vector<half> B(N * N, __float2half(1.f));
    std::vector<float> C(N * N, 0.f);
    half *dA, *dB; float *dC;
    CHECK(cudaMalloc(&dA, A.size() * sizeof(half)));
    CHECK(cudaMalloc(&dB, B.size() * sizeof(half)));
    CHECK(cudaMalloc(&dC, C.size() * sizeof(float)));
    CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(N / 16, N / 16);
    wmma_gemm<<<grid, block>>>(dA, dB, dC, N);
    CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // All-ones inputs: every output is the dot of two length-N vectors of ones, so N.
    float max_err = 0.f;
    for (float v : C) max_err = std::fmax(max_err, std::fabs(v - N));
    printf("N=%d  max |C - %d| = %g  %s\n", N, N, max_err, max_err < 1e-2f ? "ok" : "MISMATCH");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
