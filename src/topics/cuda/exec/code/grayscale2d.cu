#include <cstdio>
#include <cuda_runtime.h>

// RGB (interleaved, 3 bytes/pixel) -> grayscale. One thread per pixel, 2-D grid.
__global__ void rgb2gray(const unsigned char* __restrict__ rgb, unsigned char* __restrict__ gray,
                         int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // column (fastest-varying -> coalesced)
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (x >= width || y >= height) return;           // bottom/right edge guard

    int idx = y * width + x;
    const unsigned char* p = rgb + 3 * idx;
    gray[idx] = (unsigned char)(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
}

int main() {
    const int W = 1920, H = 1080;
    unsigned char *d_rgb, *d_gray;
    cudaMalloc(&d_rgb, 3 * W * H);
    cudaMalloc(&d_gray, W * H);
    cudaMemset(d_rgb, 128, 3 * W * H);

    dim3 block(32, 8);                               // 256 threads; 32 wide = one warp per row
    dim3 grid((W + block.x - 1) / block.x,           // 60
              (H + block.y - 1) / block.y);          // 135
    printf("grid (%d,%d) x block (%d,%d) = %d threads for %d pixels\n",
           grid.x, grid.y, block.x, block.y, grid.x * grid.y * block.x * block.y, W * H);

    rgb2gray<<<grid, block>>>(d_rgb, d_gray, W, H);
    cudaError_t e = cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString(e));

    cudaFree(d_rgb); cudaFree(d_gray);
    return 0;
}
