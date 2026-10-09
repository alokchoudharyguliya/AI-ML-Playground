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

// y = a*x + y   — one element per thread
__global__ void saxpy(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global 1-D index
    if (i < n) y[i] = a * x[i] + y[i];               // guard: grid is rounded up
}

// grid-stride version: any n, any grid size
__global__ void saxpy_stride(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 24;                            // 16M elements
    const size_t bytes = n * sizeof(float);

    float *h_x = (float*)malloc(bytes), *h_y = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;         // ceil(n / threads)

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    saxpy<<<blocks, threads>>>(n, 2.0f, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());                   // launch-configuration errors
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));             // execution errors surface here

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    // bytes moved: read x, read y, write y = 3 * n * 4
    printf("saxpy: %.3f ms, %.1f GB/s\n", ms, 3.0 * bytes / (ms * 1e6));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    printf("y[0] = %f (expected 4.0)\n", h_y[0]);

    // grid-stride variant sized to the hardware
    int dev = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    saxpy_stride<<<sms * 8, threads>>>(n, 2.0f, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_x); cudaFree(d_y); free(h_x); free(h_y);
    return 0;
}
// build: nvcc -O3 -arch=sm_80 vecadd.cu -o vecadd && ./vecadd
