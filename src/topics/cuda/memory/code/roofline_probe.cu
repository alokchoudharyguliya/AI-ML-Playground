// Measure YOUR GPU's roofline.
// Each thread loads one float, performs K fused multiply-adds on it (two independent chains for ILP),
// and stores it back. K sets the arithmetic intensity:
//     FLOP/element = 2*K*2 (two chains, FMA = 2 FLOP)   bytes/element = 8 (4 read + 4 write)
//     AI = (4*K) / 8 = K/2 FLOP/B
// Sweep K from 1 to 1024 and the kernel walks from the memory roof to the compute roof.
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 roofline_probe.cu -o probe && ./probe > roofline.csv
// plot:  python plot_roofline.py roofline.csv
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

template <int K>
__global__ void fma_kernel(float* __restrict__ x, size_t n, float a, float b) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v0 = x[i], v1 = v0 + 1.0f;
    #pragma unroll 16
    for (int j = 0; j < K; ++j) {              // K is a compile-time constant -> unrolled, no loop overhead
        v0 = fmaf(v0, a, b);                   // chain 0
        v1 = fmaf(v1, a, b);                   // chain 1 (independent: instruction-level parallelism)
    }
    x[i] = v0 + v1;                            // use both results so nothing is optimised away
}

template <int K>
void run(float* d, size_t n, double* gflops, double* gbs) {
    const int threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    const float a = 0.9999f, b = 1e-4f;        // |a|<1: values stay bounded for any K
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    fma_kernel<K><<<blocks, threads>>>(d, n, a, b);              // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < 10; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        fma_kernel<K><<<blocks, threads>>>(d, n, a, b);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    const double flops = (double)n * (4.0 * K + 1.0);          // 2 chains x K FMAs x 2 FLOP + the final add
    const double bytes = (double)n * 8.0;
    *gflops = flops / (best * 1e-3) / 1e9;
    *gbs = bytes / (best * 1e-3) / 1e9;
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
}

int main() {
    const size_t n = (size_t)1 << 26;           // 256 MB: larger than L2, so HBM traffic is real
    float* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));

    printf("fma_per_chain,flop_per_byte,gflops,gbs\n");
    double gf, gb;
#define PROBE(K) run<K>(d, n, &gf, &gb); printf("%d,%.4f,%.1f,%.1f\n", K, (4.0 * K + 1.0) / 8.0, gf, gb);
    PROBE(1) PROBE(2) PROBE(4) PROBE(8) PROBE(16) PROBE(32)
    PROBE(64) PROBE(128) PROBE(256) PROBE(512) PROBE(1024)
#undef PROBE
    cudaFree(d);
    return 0;
}
