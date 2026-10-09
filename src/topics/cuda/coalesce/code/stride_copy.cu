// How stride destroys coalescing.
// out[i] = in[i * stride]. Stride 1 is one 128-byte line per warp (4 × 32-byte sectors).
// Stride 32 is one sector per thread: 8× the bytes moved for the same useful data.
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 stride_copy.cu -o stride && ./stride
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

__global__ void copy_stride(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i * stride];
}

int main() {
    const int n = 1 << 22;                              // outputs
    const int maxStride = 32;
    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, (size_t)n * maxStride * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(in, 0, (size_t)n * maxStride * sizeof(float)));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    printf("stride   ms    useful GB/s   bytes moved / useful   (model)\n");
    for (int stride = 1; stride <= maxStride; stride *= 2) {
        auto launch = [&] { copy_stride<<<blocks, threads>>>(in, out, n, stride); };
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
        // Model: each warp of 32 floats touches ceil-unique 32-byte sectors.
        // For an aligned stride-s float walk the sector count per warp is min(32, s==0?1: ...).
        // Stride s (elements): 32 threads span 32*s floats = 128*s bytes, touching 4*s sectors
        // when s is a power of two and the base is aligned. Useful bytes stay 128 per warp.
        const double useful = (double)n * 4;
        const double moved = useful * stride;           // power-of-two stride, aligned: s× overfetch
        printf("%6d  %6.3f   %8.1f      %5.1f×\n",
               stride, best, useful / (best * 1e6), moved / useful);
    }
    cudaFree(in); cudaFree(out);
    return 0;
}
