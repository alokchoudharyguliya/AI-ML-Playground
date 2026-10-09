// nvcc -O3 -arch=sm_80 occupancy_query.cu -o occupancy_query
//
// Ask the runtime how many blocks of this kernel fit on one SM.
// The count includes the compiler's register usage and the static
// __shared__ array. Compare it with the lab: 256 threads, the array
// is 256*4 = 1 KB of user shared memory, plus the 1 KB reserve.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void saxpy_smem(float* a, const float* x, float alpha, int n) {
    __shared__ float tile[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tile[threadIdx.x] = x[i];
    __syncthreads();
    if (i < n) a[i] = a[i] + alpha * tile[threadIdx.x];
}

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    int block = 256;
    int blocksPerSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, saxpy_smem, block, /* dynamic smem */ 0));

    int warps = blocksPerSM * (block / 32);
    int maxWarps = p.maxThreadsPerMultiProcessor / 32;
    printf("%s  sm_%d  SMs %d\n", p.name, p.major * 10 + p.minor, p.multiProcessorCount);
    printf("registers/SM %d   shared/SM %zu B   max warps/SM %d   max blocks/SM %d\n",
           p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor,
           maxWarps, p.maxBlocksPerMultiProcessor);
    printf("this kernel: %d blocks/SM, %d warps/SM, occupancy %.0f%%\n",
           blocksPerSM, warps, 100.0 * warps / maxWarps);
    printf("compile with --ptxas-options=-v to see the register count behind that percentage\n");
    return 0;
}
