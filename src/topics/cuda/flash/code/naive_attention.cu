// nvcc -O3 -arch=sm_80 naive_attention.cu -o naive_attention
//
// Textbook attention. S and P are N×N matrices in HBM.
// Bytes (one head, s = 4 here): about 4 N² s for the score traffic,
// plus 4 N d s for Q, K, V, O. The lab's FP16 model is the same
// count with s = 2. Nothing here is wrong numerically — it is just
// the kernel FlashAttention exists to avoid.

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

__global__ void scores(const float* Q, const float* K, float* S, int n, int d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    float acc = 0.f;
    for (int t = 0; t < d; ++t) acc += Q[i * d + t] * K[j * d + t];
    S[i * n + j] = acc * rsqrtf((float)d);          // N² write
}

__global__ void softmax_rows(float* P, const float* S, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = S + i * n;                   // N² read
    float m = row[0];
    for (int j = 1; j < n; ++j) m = fmaxf(m, row[j]);
    float sum = 0.f;
    for (int j = 0; j < n; ++j) {
        float e = expf(row[j] - m);
        P[i * n + j] = e;                           // N² write
        sum += e;
    }
    for (int j = 0; j < n; ++j) P[i * n + j] /= sum;
}

__global__ void attend(const float* P, const float* V, float* O, int n, int d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || t >= d) return;
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc += P[i * n + j] * V[j * d + t];  // N² read of P
    O[i * d + t] = acc;
}

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    const int n = 32, d = 16;
    std::vector<float> Q(n * d, 0.f), K(n * d, 0.f), V(n * d, 1.f), O(n * d);
    for (int i = 0; i < n * d; ++i) Q[i] = K[i] = ((i * 17) % 5) * 0.1f;
    float *dQ, *dK, *dV, *dS, *dP, *dO;
    CHECK(cudaMalloc(&dQ, Q.size() * sizeof(float)));
    CHECK(cudaMalloc(&dK, K.size() * sizeof(float)));
    CHECK(cudaMalloc(&dV, V.size() * sizeof(float)));
    CHECK(cudaMalloc(&dS, n * (size_t)n * sizeof(float)));   // the matrix FlashAttention does not allocate
    CHECK(cudaMalloc(&dP, n * (size_t)n * sizeof(float)));
    CHECK(cudaMalloc(&dO, O.size() * sizeof(float)));
    CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 b2(16, 16);
    dim3 g2((n + 15) / 16, (n + 15) / 16);
    scores<<<g2, b2>>>(dQ, dK, dS, n, d);
    softmax_rows<<<(n + 127) / 128, 128>>>(dP, dS, n);
    dim3 bo(16, 16);
    dim3 go((d + 15) / 16, (n + 15) / 16);
    attend<<<go, bo>>>(dP, dV, dO, n, d);
    CHECK(cudaMemcpy(O.data(), dO, O.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double bytes = (4.0 * n * n + 4.0 * n * d) * sizeof(float);
    printf("N=%d d=%d  score matrices %d B  model traffic %.0f B\n",
           n, d, 2 * n * n * (int)sizeof(float), bytes);
    printf("O[0] = %g (finite: %s)\n", O[0], std::isfinite(O[0]) ? "yes" : "no");
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dS); cudaFree(dP); cudaFree(dO);
    return 0;
}
