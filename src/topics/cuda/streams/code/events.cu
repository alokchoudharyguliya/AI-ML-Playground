// nvcc -O3 -arch=sm_80 events.cu -o events
//
// Stream B's copy waits for stream A's kernel. The host does not.
// cudaDeviceSynchronize() would also order them, and it would drain
// every other stream at the same time.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void fill(float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.f;
}

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    const int n = 1 << 20;
    float *d, *h;
    cudaStream_t a, b;
    cudaEvent_t ready;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    CHECK(cudaMallocHost(&h, n * sizeof(float)));
    CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));

    fill<<<(n + 255) / 256, 256, 0, a>>>(d, n);
    CHECK(cudaEventRecord(ready, a));          // becomes signaled when `fill` finishes
    CHECK(cudaStreamWaitEvent(b, ready, 0));   // b's later work waits; the host returns now
    CHECK(cudaMemcpyAsync(h, d, n * sizeof(float), cudaMemcpyDeviceToHost, b));

    // The host could enqueue more chunks here. Nothing above has stopped it.
    CHECK(cudaStreamSynchronize(b));
    printf("h[0] = %.0f (1 means B's copy saw A's kernel)\n", h[0]);

    cudaEventDestroy(ready);
    cudaStreamDestroy(a);
    cudaStreamDestroy(b);
    cudaFreeHost(h);
    cudaFree(d);
    return 0;
}
