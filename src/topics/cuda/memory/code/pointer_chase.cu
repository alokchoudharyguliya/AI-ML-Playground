// Pointer-chasing latency microbenchmark.
// ONE thread follows a random cyclic chain of dependent loads: p = next[p].
// Every load depends on the previous one, so nothing can overlap and time/iteration IS the latency
// of whichever memory level the working set fits in.
//
// build: nvcc -O3 -arch=sm_80 pointer_chase.cu -o chase && ./chase > chase.csv
// Expect steps in latency at roughly: L1 size  ->  L2 size  ->  (TLB reach) beyond.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
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

__global__ void chase(const unsigned* __restrict__ next, unsigned start, long iters,
                      unsigned* sink, long long* cycles) {
    unsigned p = start;
    long long t0 = clock64();
    for (long i = 0; i < iters; ++i) p = next[p];      // dependent load chain
    long long t1 = clock64();
    *sink = p;                                         // keep the chain alive
    *cycles = t1 - t0;
}

int main() {
    const int LINE = 32;                               // one element per 128-byte cache line (32 x 4 B)
    int dev = 0, clk_khz = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
    const double ghz = clk_khz / 1e6;

    unsigned* d_sink; long long* d_cyc;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_cyc, sizeof(long long)));

    printf("bytes,cycles_per_load,ns_per_load\n");
    for (size_t bytes = 4 << 10; bytes <= ((size_t)1 << 30); bytes = bytes * 3 / 2 + 1024) {
        size_t lines = bytes / (LINE * sizeof(unsigned));
        if (lines < 2) continue;

        // random cyclic permutation over cache lines (Sattolo) -> defeats hardware prefetching
        std::vector<unsigned> perm(lines);
        std::iota(perm.begin(), perm.end(), 0u);
        std::mt19937 rng(42);
        for (size_t i = lines - 1; i > 0; --i) {
            std::uniform_int_distribution<size_t> d(0, i - 1);
            std::swap(perm[i], perm[d(rng)]);
        }
        std::vector<unsigned> next(lines * LINE, 0);
        for (size_t i = 0; i < lines; ++i)
            next[(size_t)perm[i] * LINE] = perm[(i + 1) % lines] * LINE;   // element index of the next line

        unsigned* d_next;
        CUDA_CHECK(cudaMalloc(&d_next, next.size() * sizeof(unsigned)));
        CUDA_CHECK(cudaMemcpy(d_next, next.data(), next.size() * sizeof(unsigned), cudaMemcpyHostToDevice));

        long iters = (long)std::min<size_t>(lines * 4, (size_t)1 << 20);
        chase<<<1, 1>>>(d_next, perm[0] * LINE, iters, d_sink, d_cyc);      // warm-up: fills the caches
        CUDA_CHECK(cudaDeviceSynchronize());
        chase<<<1, 1>>>(d_next, perm[0] * LINE, iters, d_sink, d_cyc);      // measured pass
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        long long cyc = 0;
        CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, sizeof(cyc), cudaMemcpyDeviceToHost));
        double per = (double)cyc / iters;
        printf("%zu,%.1f,%.1f\n", bytes, per, per / ghz);
        CUDA_CHECK(cudaFree(d_next));
    }
    // Reading the CSV: the first plateau is L1 latency, the second is L2, the third is HBM.
    // Where the curve bends tells you the effective capacity of each level for this access pattern.
    return 0;
}
