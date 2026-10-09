// nvcc -O3 -arch=sm_80 pinned.cu -o pinned
//
// Pageable cudaMemcpyAsync is allowed to block the host: the driver
// copies into an internal pinned bounce buffer first. A buffer from
// cudaMallocHost is already DMA-able, so the call returns and the copy
// engine overlaps whatever kernel is running in another stream.
//
// Time the two against the pipeline in pipeline.cu. This file only
// shows the allocation, and the flag that keeps a stream out of the
// legacy default-stream barrier.

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    const size_t n = 1 << 20;
    float* pageable = new float[n];
    float* pinned = nullptr;
    float* d = nullptr;
    CHECK(cudaMallocHost(&pinned, n * sizeof(float)));   // pair with cudaFreeHost
    CHECK(cudaMalloc(&d, n * sizeof(float)));

    cudaStream_t stream;
    // Non-blocking: this stream does not wait for, and is not waited on by,
    // work launched into the legacy default stream (the NULL stream).
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CHECK(cudaMemcpyAsync(d, pinned, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    // cudaMemcpyAsync(d, pageable, ...) can return only after a staging copy.
    // Don't use it on the overlap path.

    CHECK(cudaStreamSynchronize(stream));
    printf("pinned %zu MB, stream is non-blocking\n", (n * sizeof(float)) >> 20);
    printf("pageable pointer %p was not used for the DMA\n", (void*)pageable);

    cudaStreamDestroy(stream);
    cudaFree(d);
    cudaFreeHost(pinned);
    delete[] pageable;
    return 0;
}
