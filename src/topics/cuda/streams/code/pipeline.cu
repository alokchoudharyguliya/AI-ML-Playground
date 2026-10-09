// nvcc -O3 -arch=sm_80 pipeline.cu -o pipeline
//
// Chunked saxpy. Each chunk is H2D, kernel, D2H on one stream.
// Chunks are dealt round-robin, so three streams keep the copy-in,
// the kernel and the copy-out in flight together.
// Reusing one device buffer per stream is safe: the next H2D on that
// stream is queued behind the previous D2H.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void saxpy(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

static float run(int nstreams, float* h_x, float* h_y, int n, int chunk, float a) {
    cudaError_t e;
    #define FAIL(cmd) do { e = (cmd); if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return -1.f; } } while (0)
    cudaStream_t streams[4];
    float* d_x[4];
    float* d_y[4];
    for (int s = 0; s < nstreams; ++s) {
        FAIL(cudaStreamCreate(&streams[s]));
        FAIL(cudaMalloc(&d_x[s], chunk * sizeof(float)));
        FAIL(cudaMalloc(&d_y[s], chunk * sizeof(float)));
    }
    int block = 256;
    int grid = (chunk + block - 1) / block;
    cudaEvent_t start, stop;
    FAIL(cudaEventCreate(&start));
    FAIL(cudaEventCreate(&stop));
    FAIL(cudaEventRecord(start));
    int chunks = n / chunk;
    for (int c = 0; c < chunks; ++c) {
        int s = c % nstreams;
        size_t off = (size_t)c * chunk;
        FAIL(cudaMemcpyAsync(d_x[s], h_x + off, chunk * sizeof(float),
                              cudaMemcpyHostToDevice, streams[s]));
        FAIL(cudaMemcpyAsync(d_y[s], h_y + off, chunk * sizeof(float),
                              cudaMemcpyHostToDevice, streams[s]));
        saxpy<<<grid, block, 0, streams[s]>>>(d_y[s], d_x[s], a, chunk);
        FAIL(cudaMemcpyAsync(h_y + off, d_y[s], chunk * sizeof(float),
                              cudaMemcpyDeviceToHost, streams[s]));
    }
    FAIL(cudaEventRecord(stop));
    FAIL(cudaEventSynchronize(stop));   // one wait, after the whole pipeline is queued
    float ms = 0;
    FAIL(cudaEventElapsedTime(&ms, start, stop));
    for (int s = 0; s < nstreams; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_x[s]);
        cudaFree(d_y[s]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int chunk = 1 << 20;
    const int chunks = 12;
    const int n = chunk * chunks;
    float *h_x, *h_y;
    CHECK(cudaMallocHost(&h_x, n * sizeof(float)));
    CHECK(cudaMallocHost(&h_y, n * sizeof(float)));
    for (int i = 0; i < n; ++i) { h_x[i] = 1.f; h_y[i] = 2.f; }

    float one = run(1, h_x, h_y, n, chunk, 3.f);
    for (int i = 0; i < n; ++i) h_y[i] = 2.f;
    float three = run(3, h_x, h_y, n, chunk, 3.f);
    printf("1 stream  %.3f ms\n", one);
    printf("3 streams %.3f ms\n", three);
    printf("speedup   %.2fx   (copies are small next to a tiny saxpy — raise the chunk or the\n"
           "                    arithmetic if the kernel finishes before the next copy is queued)\n",
           one / three);

    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    return 0;
}
