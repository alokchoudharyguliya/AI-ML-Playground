// 1-D 3-point stencil with a *halo* in shared memory.
// Each block owns a contiguous chunk of the output. Interior points need their neighbours, so the
// block also loads one extra element on each side (the halo / ghost cells) into shared memory.
// Without the halo every thread would re-read its neighbours from global memory — 3× the traffic.
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 stencil1d.cu -o stencil && ./stencil
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

constexpr int BLK = 256;

__global__ void stencil_smem(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float s[];                // BLK + 2 floats, sized at launch
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = threadIdx.x + 1;              // interior slots are s[1 .. BLK]

    s[t] = (g < n) ? in[g] : 0.f;
    if (threadIdx.x == 0)     s[0]       = (g > 0)     ? in[g - 1] : 0.f;          // left halo
    if (threadIdx.x == BLK-1) s[BLK + 1] = (g + 1 < n) ? in[g + 1] : 0.f;          // right halo
    __syncthreads();

    if (g < n) out[g] = 0.25f * s[t - 1] + 0.5f * s[t] + 0.25f * s[t + 1];
}

__global__ void stencil_naive(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    const float l = (g > 0)     ? in[g - 1] : 0.f;
    const float c = in[g];
    const float r = (g + 1 < n) ? in[g + 1] : 0.f;
    out[g] = 0.25f * l + 0.5f * c + 0.25f * r;
}

int main() {
    const int n = 1 << 24;                      // 16 M elements
    const size_t bytes = n * sizeof(float);
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = (float)i;

    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemcpy(in, h.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = (n + BLK - 1) / BLK;
    cudaEvent_t t0, t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    auto time = [&](auto launch) {
        launch(); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0)); launch(); CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        CUDA_CHECK(cudaGetLastError());
        return ms;
    };

    float ms_n = time([&] { stencil_naive<<<blocks, BLK>>>(in, out, n); });
    float ms_s = time([&] { stencil_smem <<<blocks, BLK, (BLK + 2) * sizeof(float)>>>(in, out, n); });

    // Naive: ~3 n reads + n writes.  Smem: ~n reads (halo is 2/BLK extra) + n writes.
    const double naive_gb = 4.0 * n * 4 / 1e9, smem_gb = 2.0 * n * 4 / 1e9;
    printf("naive  %.3f ms  model %.2f GB  →  %.0f GB/s\n", ms_n, naive_gb, naive_gb / (ms_n * 1e-3));
    printf("smem   %.3f ms  model %.2f GB  →  %.0f GB/s   (halo = 2 extra loads / block)\n",
           ms_s, smem_gb, smem_gb / (ms_s * 1e-3));
    cudaFree(in); cudaFree(out);
    return 0;
}
