// STREAM-style effective-bandwidth benchmark.
//   copy : c = a            (2 arrays touched)
//   scale: b = q * c        (2)
//   add  : c = a + b        (3)
//   triad: a = b + q * c    (3)
// plus a float4 (16-byte) copy to show what wider loads do.
// build: nvcc -O3 -std=c++17 -arch=sm_80 bandwidth_stream.cu -o stream && ./stream
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

__global__ void k_copy(const float* __restrict__ a, float* __restrict__ c, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        c[i] = a[i];
}
__global__ void k_scale(const float* __restrict__ c, float* __restrict__ b, float q, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        b[i] = q * c[i];
}
__global__ void k_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        c[i] = a[i] + b[i];
}
__global__ void k_triad(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, float q, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        a[i] = b[i] + q * c[i];
}
// 16 bytes per thread per access: 4x fewer load instructions for the same bytes
__global__ void k_copy4(const float4* __restrict__ a, float4* __restrict__ c, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x)
        c[i] = a[i];
}

// time `reps` launches of fn() and return the best single-launch time in ms
template <typename F>
float best_ms(F fn, int reps = 20) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    fn();                                           // warm-up (also loads the kernel)
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        fn();
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
    int dev = 0, sms = 0, memclk = 0, bus = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, dev));       // kHz
    CUDA_CHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));     // bits
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    const double peak = 2.0 * memclk * 1e3 * (bus / 8) / 1e9;                           // GB/s (DDR x2)
    printf("%s: %d SMs, theoretical bandwidth %.0f GB/s (%d-bit bus)\n", p.name, sms, peak, bus);

    const size_t n = (size_t)1 << 27;               // 128M floats = 512 MB per array: far larger than L2
    const size_t bytes = n * sizeof(float);
    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMemset(a, 0, bytes));
    CUDA_CHECK(cudaMemset(b, 0, bytes));
    CUDA_CHECK(cudaMemset(c, 0, bytes));

    const int threads = 256, blocks = sms * 16;     // hardware-sized grid + grid-stride loop
    const float q = 3.0f;

    auto report = [&](const char* name, float ms, double moved) {
        double gbs = moved / (ms * 1e-3) / 1e9;
        printf("%-10s %8.3f ms  %8.1f GB/s  (%5.1f%% of peak)\n", name, ms, gbs, 100.0 * gbs / peak);
    };

    report("copy",  best_ms([&] { k_copy <<<blocks, threads>>>(a, c, n); }),       2.0 * bytes);
    report("scale", best_ms([&] { k_scale<<<blocks, threads>>>(c, b, q, n); }),    2.0 * bytes);
    report("add",   best_ms([&] { k_add  <<<blocks, threads>>>(a, b, c, n); }),    3.0 * bytes);
    report("triad", best_ms([&] { k_triad<<<blocks, threads>>>(a, b, c, q, n); }), 3.0 * bytes);
    report("copy f4", best_ms([&] { k_copy4<<<blocks, threads>>>((const float4*)a, (float4*)c, n / 4); }), 2.0 * bytes);

    // What to look for:
    //  * Each kernel does 0-2 FLOP per 8-12 bytes moved (arithmetic intensity < 0.2): memory-bound by construction.
    //  * Healthy streaming kernels reach ~85-95% of the theoretical bandwidth.
    //  * float4 often adds a few percent: fewer instructions, more bytes in flight per thread (Little's law).
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
