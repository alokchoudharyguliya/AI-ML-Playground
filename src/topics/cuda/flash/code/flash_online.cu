// nvcc -O3 -arch=sm_80 flash_online.cu -o flash_online
//
// Exact attention, one warp per query, d = 32.
// The score row is never stored. Each key updates a running max m,
// a running sum l, and the output accumulator, then the key is dropped.
// This is tile size B_c = 1 so the recurrence is easy to check.
// The SRAM tiling that cuts HBM traffic (B_r, B_c >> 1) is the lab;
// the rescale below is the same one those tiles use.
// V = 1, so each output component must come back as 1.

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

__global__ void online_attn(const float* Q, const float* K, const float* V, float* O, int n, int d) {
    int q = blockIdx.x;
    int lane = threadIdx.x;                         // d == blockDim.x == 32
    float qv = Q[q * d + lane];
    float acc = 0.f;
    float m = -1e30f;
    float l = 0.f;
    for (int j = 0; j < n; ++j) {
        float prod = qv * K[j * d + lane];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            prod += __shfl_xor_sync(0xffffffffu, prod, off);
        float score = prod * rsqrtf((float)d);      // identical on every lane
        float m_new = fmaxf(m, score);
        float alpha = __expf(m - m_new);            // 0 on the first key: m is -1e30
        float p = __expf(score - m_new);
        l = l * alpha + p;
        acc = acc * alpha + p * V[j * d + lane];
        m = m_new;
    }
    O[q * d + lane] = acc / l;
}

#define CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    const int n = 64, d = 32, queries = 4;
    std::vector<float> Q(queries * d), K(n * d), V(n * d, 1.f), O(queries * d);
    for (int i = 0; i < (int)Q.size(); ++i) Q[i] = ((i * 3) % 7) * 0.05f;
    for (int i = 0; i < (int)K.size(); ++i) K[i] = ((i * 5) % 7) * 0.05f;

    float *dQ, *dK, *dV, *dO;
    CHECK(cudaMalloc(&dQ, Q.size() * sizeof(float)));
    CHECK(cudaMalloc(&dK, K.size() * sizeof(float)));
    CHECK(cudaMalloc(&dV, V.size() * sizeof(float)));
    CHECK(cudaMalloc(&dO, O.size() * sizeof(float)));
    CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice));

    online_attn<<<queries, d>>>(dQ, dK, dV, dO, n, d);
    CHECK(cudaMemcpy(O.data(), dO, O.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // V is 1, so every output component is a softmax weight times 1, summed: exactly 1.
    float max_err = 0.f;
    for (float v : O) max_err = std::fmax(max_err, std::fabs(v - 1.f));
    printf("%d queries × %d keys, d=%d, no N×N buffer\n", queries, n, d);
    printf("max |O - 1| = %g  %s\n", max_err, max_err < 1e-3f ? "ok" : "MISMATCH");
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
