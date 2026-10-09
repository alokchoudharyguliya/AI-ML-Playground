import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{g as r,n as i,s as a,t as o,y as s}from"./viz-CPys2405.js";import{t as c}from"./Stage3D-CWOT2b2l.js";import{a as l,c as u,i as d,l as f,o as p,r as m,s as h,t as g}from"./hooks-Dw7oo1m3.js";import{a as _,n as v,t as y}from"./specs-CrKDU_SL.js";var b=e(t(),1);function x(e,t){let n=2*e*e*e,r=3*e*e*t;return{flops:n,moved:r,intensity:n/r}}function S(e,t,n){let r=t*e/1e3;return Math.min(r,n)}function C({N:e,d:t,Br:n,Bc:r,bytes:i}){let a=4*e*e*t,o=4*e*e+4*e*t,s=Math.ceil(e/n),c=Math.ceil(e/r),l=2*e*t*(1+s),u=(2*n*t+2*r*t+n*r)*i,d=o*i,f=l*i;return{flops:a,standard:d,flash:f,Tr:s,Tc:c,sram:u,iStd:a/d,iFlash:a/f}}var w=[1,2,0,1,5,1,0,2,3,1,4,0,2,2,1,3];function T(e=w,t=4){let n=[{m:null,l:0,seen:0}],r=-1/0,i=0;for(let a=0;a<e.length;a+=t){let o=e.slice(a,a+t),s=Math.max(...o),c=Math.max(r,s),l=Number.isFinite(r)?Math.exp(r-c):0,u=o.reduce((e,t)=>e+Math.exp(t-c),0);i=i*l+u,r=c,n.push({m:r,l:i,alpha:l,mBlock:s,seen:a+o.length})}return n}var E=`## A tensor core multiplies a tile, not a scalar

A CUDA core computes one FMA per instruction. A **tensor core** computes a small matrix multiply-accumulate per instruction: a warp (or, on Hopper, a warpgroup) feeds it tiles of A, B and C and gets back C + AB. The CUDA WMMA API exposes this as a 16×16×16 tile for FP16 and BF16, accumulated in FP32:

\`\`\`cuda
wmma::load_matrix_sync(a, A, lda);
wmma::load_matrix_sync(b, B, ldb);
wmma::mma_sync(c, a, b, c);          // c is FP32 even when a and b are FP16
wmma::store_matrix_sync(C, c, ldc, wmma::mem_row_major);
\`\`\`

Every lane of the warp must execute those calls together. The tile is the unit of work; a scalar loop never reaches the unit.

The peaks in the memory chapter are this hardware, dense, with no 2:4 sparsity:

| GPU | FP32 CUDA cores | TF32 tensor | FP16 tensor |
|---|---|---|---|
| A100 | 19.5 TFLOP/s | 156 | 312 |
| H100 SXM | 67 | 494 | 989 |
| RTX 4090 | 82.6 | 82.6 | 165 |

On the 4090 the TF32 column equals FP32: that chip's tensor cores show up in FP16/BF16, not as a separate TF32 peak. Hopper adds FP8 (about twice FP16) and a larger **warpgroup** MMA (\`wgmma\`) fed by the tensor memory accelerator. The idea is the same tile, with a wider tile.

A higher peak moves the roofline's ridge to the right. A100 FP32 turns compute-bound near 10 FLOP/byte. A100 FP16 tensor cores turn compute-bound near **153 FLOP/byte**. A GEMM that was sitting on the FP32 roof is often still on the memory slope for FP16, and the tensor cores idle. The lab's N = 256 case is exactly that.

## Attention is a GEMM that writes an N×N matrix

One head:

$$
S = QK^\\top / \\sqrt{d}, \\qquad P = \\mathrm{softmax}(S), \\qquad O = PV.
$$

Two GEMMs, so the FLOPs are the familiar $4N^2 d$. The trouble is $S$ and $P$. Each is $N \\times N$, and the textbook implementation writes $S$, reads it back for the softmax, writes $P$, and reads $P$ for the second GEMM. That is $4N^2$ elements of HBM on top of $Q, K, V, O$.

Arithmetic intensity then tends to $d/s$ (element size $s$ in bytes), **not** to something that grows with $N$. For FP16 and $d = 64$ that is 32 FLOP/byte. A100's FP16 ridge is 153. Making the sequence longer does not cross the ridge. The kernel is memory-bound at every length, and the bytes it moves are the score matrix.

## Online softmax, then never store the tile

Softmax needs the row max and the row sum, which you do not know until the row is finished. The online algorithm keeps a running max $m$ and a running sum $\\ell$, and rescales when a later tile raises the max. For a block of scores with max $m_{\\text{block}}$:

$$
m' = \\max(m, m_{\\text{block}}),
\\qquad
\\ell' = e^{m - m'}\\,\\ell + \\sum_j e^{s_j - m'},
\\qquad
o' = e^{m - m'}\\,o + \\sum_j e^{s_j - m'}\\,v_j.
$$

The factor $e^{m-m'}$ is 1 when the max did not move, and a fraction when it did. After the last block, $o/\\ell$ is the exact softmax-weighted output. No approximation, and no $N$-long row resident anywhere.

FlashAttention is that recurrence applied to tiles that fit in SRAM:

- A $B_r \\times d$ tile of $Q$ stays on chip while every $B_c \\times d$ tile of $K$ and $V$ streams through.
- The $B_r \\times B_c$ score tile is computed, softmaxed online, multiplied into $V$, and dropped.
- $O$'s tile is updated in SRAM and written once.

$K$ and $V$ are re-read once per $Q$ tile. $S$ and $P$ are not written at all. The backward pass recomputes the tiles instead of storing them; saving them would put the $N \\times N$ matrix back in HBM.

<div class="callout">

**The ridge still has to be reachable.** Asymptotically the flash kernel moves $2B_r/s$ FLOP/byte. On A100 FP16 that needs $B_r$ above about 153, so a tile of 128 is close and a tile of 256 clears it — if the $Q$, $K$, $V$, $O$ and score tiles fit in the SM's SRAM together. A tile that misses the SRAM spills to HBM and the intensity collapses back toward the standard kernel.

</div>

Causal masking is free inside the tile: scores past the diagonal are set to $-\\infty$ before the row max. Nothing extra is stored.
`,D=`## GEMM intensity, three precisions

A square product $C = AB$ that reads each input once and writes $C$ once:

$$
\\text{FLOPs} = 2N^3,
\\qquad
\\text{bytes} = 3N^2 s,
\\qquad
I = \\frac{2N}{3s}.
$$

$s$ is 4 for FP32 and for TF32 (the values stored are FP32; TF32 is the compute format) and 2 for FP16.

The roofline from the memory chapter, with $\\pi$ in TFLOP/s and $\\beta$ in GB/s:

$$
\\text{achieved} = \\min\\big(\\pi,\\; \\beta I / 1000\\big),
\\qquad
I^\\star = \\pi \\cdot 1000 / \\beta.
$$

| GPU | $I^\\star$ FP32 | $I^\\star$ TF32 | $I^\\star$ FP16 |
|---|---|---|---|
| A100 | 9.6 | 76.5 | 153 |
| H100 SXM | 20 | 147 | 295 |
| RTX 4090 | 82 | 82 | 164 |

### N = 256 on A100

| Format | $s$ | $I = 2N/(3s)$ | Roof it hits | Achieved |
|---|---|---|---|---|
| FP32 | 4 | 42.7 | compute, 19.5 | 19.5 TFLOP/s |
| TF32 | 4 | 42.7 | memory | $2039 \\times 42.7 / 1000 \\approx 87$ |
| FP16 | 2 | 85.3 | memory | $2039 \\times 85.3 / 1000 \\approx 174$ |

Same matrix. FP32 is already on its roof, so a faster CUDA core would not help. TF32's roof is 156 and this GEMM only feeds it 87. FP16's roof is 312 and the GEMM feeds it 174. The tensor core pays off in full only once $I$ passes $I^\\star$, which for A100 FP16 means

$$
\\frac{2N}{3 \\cdot 2} \\ge 153 \\implies N \\ge 459.
$$

At $N = 1024$, $I = 341$ and the FP16 dot sits on the 312 TFLOP/s roof.

## Standard attention does not get there by growing N

Forward FLOPs, one head: $4N^2 d$ (the two GEMMs; softmax is lower order).

Bytes, materializing $S$ and $P$ at $s$ bytes/element:

$$
4N^2 s + 4Nds.
$$

Divide:

$$
I_{\\text{std}} = \\frac{4N^2 d}{s(4N^2 + 4Nd)} = \\frac{d}{s}\\cdot\\frac{N}{N+d} \\;\\xrightarrow{N \\gg d}\\; \\frac{d}{s}.
$$

FP16, $d = 64$: $I \\to 32$ FLOP/byte. A100's FP16 ridge is 153. The gap does not close as $N$ grows. At $N = 2048$ the finite formula gives $31.0$, already at the limit.

## FlashAttention bytes

$T_r = \\lceil N / B_r \\rceil$ tiles of $Q$. Each stays in SRAM while $K$ and $V$ are read in full, so $K$ and $V$ are read $T_r$ times. $Q$ and $O$ move once. $S$ and $P$ do not move.

$$
\\text{elements} = 2Nd\\,(1 + T_r),
\\qquad
I_{\\text{flash}} = \\frac{4N^2 d}{2Nd\\,(1+T_r)\\,s} = \\frac{2N}{s(1+T_r)}.
$$

For $N \\gg B_r$, $T_r \\approx N/B_r$ and

$$
I_{\\text{flash}} \\to \\frac{2 B_r}{s}.
$$

FP16 ($s = 2$): the limit is $B_r$ itself, in FLOP/byte. $B_r = 128$ tends to 128, just under A100's ridge of 153. $B_r = 256$ tends to 256 and clears it.

SRAM for the live tiles, in bytes:

$$
\\big(2 B_r d + 2 B_c d + B_r B_c\\big)\\, s.
$$

$Q$ and $O$ contribute the two $B_r d$ terms, $K$ and $V$ the two $B_c d$ terms, and the score tile is $B_r B_c$.

Worked point, the lab's default. $N = 2048$, $d = 64$, $B_r = B_c = 128$, FP16:

| | Elements | Bytes | Intensity |
|---|---|---|---|
| Standard | $4N^2 + 4Nd$ | 33.0 MB | 31.0 |
| Flash | $2Nd(1 + 16)$ | 8.50 MB | 120 |

SRAM $= (2\\cdot128\\cdot64 + 2\\cdot128\\cdot64 + 128^2)\\cdot 2 = 96$ KB, inside an A100's 164 KB and inside an H100's 228 KB. A 4090's 100 KB still holds it; $B_r = 256$ at this $d$ generally does not.

The speedup on bytes is $33.0 / 8.50 = 3.9\\times$ here. It grows as $N/B_r$ grows, because the standard kernel's $N^2$ term keeps pulling ahead of the flash kernel's $N^2 d / B_r$ term.

## The rescale

Scores $[1, 2, 0, 1]$ then $[5, 1, 0, 2]$, which are the first two tiles in the lab.

After tile 0, $m = 2$. Tile 1 contains 5, so $m' = 5$ and the mass already accumulated is multiplied by

$$
e^{m - m'} = e^{2 - 5} = e^{-3} \\approx 0.0498.
$$

Tiles that do not raise $m$ multiply by $e^0 = 1$. With $V = 1$ the normalized output is exactly 1: the weights are a softmax.
`,O=`## Exercises

**Q1.** A100, square GEMM, $N = 256$, FP32. Intensity, and which roof does it hit?

<details>
<summary>Show answer</summary>

$I = 2 \\cdot 256 / (3 \\cdot 4) = 42.7$ FLOP/byte. The FP32 ridge is 9.6, so the dot is **compute-bound at 19.5 TFLOP/s**. A faster memory would not move it.

</details>

**Q2.** Same GEMM, TF32 tensor cores. Why is the answer not 156 TFLOP/s?

<details>
<summary>Show answer</summary>

Storage is still 4 bytes, so $I$ is still 42.7. The TF32 ridge is 76.5, and 42.7 is on the memory slope: $2039 \\times 42.7 / 1000 \\approx 87$ TFLOP/s. The 156 peak is real and this matrix does not feed it.

</details>

**Q3.** Same GEMM, FP16. Intensity and achieved rate on A100?

<details>
<summary>Show answer</summary>

$I = 2 \\cdot 256 / (3 \\cdot 2) = 85.3$ FLOP/byte. Ridge is 153, so still memory-bound: about **174 TFLOP/s**, not 312. FP16 needs $N \\ge 459$ before this GEMM shape reaches the tensor-core roof.

</details>

**Q4.** Standard attention, FP16, $d = 64$. What does the intensity tend to as $N$ grows, and does A100's FP16 ridge (153) ever get crossed?

<details>
<summary>Show answer</summary>

$I \\to d/s = 64/2 = 32$ FLOP/byte. **No.** The $N^2$ score matrix grows as fast as the FLOPs. At $N = 2048$ the finite value is already 31.

</details>

**Q5.** FlashAttention, FP16, $B_r = 128$. Asymptotic intensity? Does that clear the A100 FP16 ridge?

<details>
<summary>Show answer</summary>

$I \\to 2 B_r / s = 128$ FLOP/byte. **Not quite** — 128 is under 153. $B_r = 256$ tends to 256 and does clear it, provided the tiles still fit in SRAM.

</details>

**Q6.** $N = 2048$, $d = 64$, $B_r = B_c = 128$, FP16. Standard bytes, flash bytes, and SRAM?

<details>
<summary>Show answer</summary>

Standard **33.0 MB**, flash **8.50 MB** (3.9× less). SRAM is **96 KB**: two $Q$/$O$ tiles, two $K$/$V$ tiles, and the $128 \\times 128$ score tile, at 2 bytes.

</details>

**Q7.** The lab's scores start $[1, 2, 0, 1,\\; 5, 1, 0, 2]$. After the second tile, what is the running max, and by what factor is the first tile's mass rescaled?

<details>
<summary>Show answer</summary>

$m = 5$. The first tile's max was 2, so the rescale is $e^{2-5} = e^{-3} \\approx 0.0498$. The output is still exact; the earlier exponentials were computed against a max that later moved.

</details>

**Q8.** The backward pass needs $P$ as well. Why does FlashAttention recompute the score tiles instead of storing them from the forward pass?

<details>
<summary>Show answer</summary>

Storing $P$ writes the $N \\times N$ matrix the forward pass just avoided. Recomputing a tile from $Q$ and $K$, which are $N \\times d$, is cheaper in HBM than reading an $N \\times N$ matrix back. The extra FLOPs land on the tensor cores; the bytes were the bottleneck.

</details>
`,k=`// nvcc -O3 -arch=sm_80 wmma_gemm.cu -o wmma_gemm
//
// One warp computes one 16×16 output tile, looping over K in steps of 16.
// Inputs are FP16. The accumulator is FP32: tensor cores round the
// products, then the sum stays in the wider type.
// Every lane of the warp must execute the wmma:: calls. Launch <<<grid, 32>>>.

#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>

using namespace nvcuda;

__global__ void wmma_gemm(const half* A, const half* B, float* C, int N) {
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.f);
    for (int k = 0; k < N; k += 16) {
        const half* a_tile = A + (tile_row * 16) * N + k;
        const half* b_tile = B + k * N + tile_col * 16;
        wmma::load_matrix_sync(a, a_tile, N);
        wmma::load_matrix_sync(b, b_tile, N);
        wmma::mma_sync(c, a, b, c);
    }
    float* c_tile = C + (tile_row * 16) * N + tile_col * 16;
    wmma::store_matrix_sync(c_tile, c, N, wmma::mem_row_major);
}

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
} while (0)

int main() {
    const int N = 64;                                  // multiple of 16
    std::vector<half> A(N * N, __float2half(1.f));
    std::vector<half> B(N * N, __float2half(1.f));
    std::vector<float> C(N * N, 0.f);
    half *dA, *dB; float *dC;
    CHECK(cudaMalloc(&dA, A.size() * sizeof(half)));
    CHECK(cudaMalloc(&dB, B.size() * sizeof(half)));
    CHECK(cudaMalloc(&dC, C.size() * sizeof(float)));
    CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(N / 16, N / 16);
    wmma_gemm<<<grid, block>>>(dA, dB, dC, N);
    CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // All-ones inputs: every output is the dot of two length-N vectors of ones, so N.
    float max_err = 0.f;
    for (float v : C) max_err = std::fmax(max_err, std::fabs(v - N));
    printf("N=%d  max |C - %d| = %g  %s\\n", N, N, max_err, max_err < 1e-2f ? "ok" : "MISMATCH");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
`,A=`// nvcc -O3 -arch=sm_80 naive_attention.cu -o naive_attention
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

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
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
    printf("N=%d d=%d  score matrices %d B  model traffic %.0f B\\n",
           n, d, 2 * n * n * (int)sizeof(float), bytes);
    printf("O[0] = %g (finite: %s)\\n", O[0], std::isfinite(O[0]) ? "yes" : "no");
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dS); cudaFree(dP); cudaFree(dO);
    return 0;
}
`,j=`// nvcc -O3 -arch=sm_80 flash_online.cu -o flash_online
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

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
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
    printf("%d queries × %d keys, d=%d, no N×N buffer\\n", queries, n, d);
    printf("max |O - 1| = %g  %s\\n", max_err, max_err < 1e-3f ? "ok" : "MISMATCH");
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
`,M=n(),N=[[`fp32`,`FP32 CUDA cores`],[`tf32`,`TF32 tensor cores`],[`fp16`,`FP16 tensor cores`]],P={fp32:4,tf32:4,fp16:2},F={fp32:o.a,tf32:o.d,fp16:o.e},I=e=>Math.round(e*10)/10;function L(){let[e,t]=(0,b.useState)(`a100`),[n,r]=(0,b.useState)(`fp16`),[i,a]=(0,b.useState)(256),c=y[e],f=P[n],C=c.peak[n],w=x(i,f).intensity,T=S(w,c.bw,C),E=T>=C-.05,D=_(C,c.bw),[O]=g(380,(e,t,r)=>{let a=t-16,l=r-32,u=2048,d=e=>48+(Math.log2(e)-Math.log2(1))/(Math.log2(u)-Math.log2(1))*(a-48),f=e=>l-(Math.log2(Math.max(e,1))-Math.log2(1))/(Math.log2(2048)-Math.log2(1))*(l-28);e.strokeStyle=`#1c2136`,e.lineWidth=1;for(let t of[1,4,16,64,256,1024])e.beginPath(),e.moveTo(d(t),28),e.lineTo(d(t),l),e.stroke(),s(e,String(t),d(t),l+14,{size:10,color:o.dim,align:`center`,mono:!0});s(e,`FLOP / byte`,(48+a)/2,r-6,{size:11,color:o.mute,align:`center`}),s(e,`TFLOP/s`,8,14,{size:11,color:o.mute});let p=(t,n)=>{let r=c.peak[t];P[t],e.strokeStyle=F[t],e.lineWidth=n,e.beginPath();let i=!1;for(let t=0;t<=64;t++){let n=1*(u/1)**(t/64),a=Math.min(r,c.bw*n/1e3),o=d(n),s=f(a);i?e.lineTo(o,s):(e.moveTo(o,s),i=!0)}e.stroke()};N.forEach(([e])=>p(e,e===n?2.5:1)),N.forEach(([t])=>{let r=x(i,P[t]).intensity,a=S(r,c.bw,c.peak[t]),o=d(r),s=f(a);e.fillStyle=F[t],e.beginPath(),e.arc(o,s,t===n?6:3.5,0,Math.PI*2),e.fill()}),s(e,`N = `+i,48,14,{size:12,color:o.ink,weight:700})});return(0,M.jsxs)(M.Fragment,{children:[(0,M.jsx)(`canvas`,{...O}),(0,M.jsxs)(d,{children:[(0,M.jsx)(h,{label:`GPU`,value:e,onChange:t,options:v}),(0,M.jsx)(h,{label:`Format`,value:n,onChange:r,options:N}),(0,M.jsx)(u,{label:`N`,min:128,max:4096,step:128,value:i,onChange:a})]}),(0,M.jsxs)(`div`,{className:`controls`,children:[(0,M.jsx)(m,{onClick:()=>a(256),children:`N = 256`}),(0,M.jsx)(m,{onClick:()=>a(1024),children:`N = 1024`})]}),(0,M.jsx)(l,{items:N.map(([e,t])=>[F[e],t])}),(0,M.jsxs)(p,{children:[c.name,`, `,N.find(e=>e[0]===n)[1],`. Intensity `,(0,M.jsx)(`b`,{children:I(w)}),` FLOP/byte, ridge `,(0,M.jsx)(`b`,{children:I(D)}),`.`,E?(0,M.jsxs)(M.Fragment,{children:[` The GEMM is on the compute roof at `,(0,M.jsx)(`b`,{className:`g`,children:I(T)}),` TFLOP/s.`]}):(0,M.jsxs)(M.Fragment,{children:[` The GEMM is on the memory slope at `,(0,M.jsx)(`b`,{className:`w`,children:I(T)}),` TFLOP/s, short of the `,(0,M.jsx)(`b`,{children:I(C)}),` peak.`]}),n===`tf32`&&c.peak.tf32===c.peak.fp32&&(0,M.jsx)(M.Fragment,{children:` On this GPU the TF32 column equals FP32. The tensor-core gap shows up at FP16.`})]})]})}var R=e=>e===0?`start`:`tile `+e;function z(){let e=(0,b.useMemo)(()=>T(),[]),[t,n]=(0,b.useState)(0),a=e[t],c=w.length/4,[f]=g(280,(e,n)=>{let l=Math.min(36,(n-16-16-60)/16);s(e,t===0?`no keys yet`:`after tile `+t+` of `+c,16,18,{size:13,color:o.b,weight:700}),s(e,a.m==null?`m = −∞`:`m = `+a.m,n-16,18,{size:13,color:o.e,align:`right`,weight:700,mono:!0}),w.forEach((n,a)=>{let c=Math.floor(a/4),u=t>0&&c<t,d=t>0&&c===t-1,f=16+a*(l+4);e.fillStyle=d?o.d:u?i(o.e,.85):`#1a1e32`,r(e,f,78,l,48,4),e.fill(),s(e,String(n),f+l/2,102,{size:14,align:`center`,weight:800,mono:!0,color:u||d?`#0a0c14`:o.dim}),a%4==0&&s(e,`t`+c,f,66,{size:10,color:o.mute,mono:!0})}),s(e,`ℓ = `+(t===0?`0`:a.l.toFixed(3)),16,156,{size:13,color:o.ink,mono:!0}),t>0&&a.alpha<.999&&s(e,`earlier mass × `+a.alpha.toFixed(4),16,178,{size:12,color:o.c,mono:!0})}),m=t>0&&a.alpha<.999;return(0,M.jsxs)(M.Fragment,{children:[(0,M.jsx)(`canvas`,{...f}),(0,M.jsx)(d,{children:(0,M.jsx)(u,{label:`Tile`,min:0,max:c,value:t,onChange:n,fmt:R})}),(0,M.jsx)(l,{items:[[o.d,`tile just absorbed`],[o.e,`already in the running softmax`]]}),(0,M.jsxs)(p,{children:[t===0&&(0,M.jsx)(M.Fragment,{children:` Sixteen integer scores, tiles of 4. V = 1, so the normalized output has to come back as 1.`}),t>0&&!m&&(0,M.jsxs)(M.Fragment,{children:[` Running max `,(0,M.jsx)(`b`,{children:a.m}),`. This tile did not raise it, so the rescale factor is `,(0,M.jsx)(`b`,{children:`1`}),`.`]}),m&&(0,M.jsxs)(M.Fragment,{children:[` Running max moved to `,(0,M.jsx)(`b`,{children:a.m}),`. Mass from earlier tiles is multiplied by `,(0,M.jsx)(`b`,{className:`p`,children:a.alpha.toFixed(4)}),t===2&&(0,M.jsxs)(M.Fragment,{children:[`, which is e`,(0,M.jsx)(`sup`,{children:`−3`})]}),`.`]}),t===c&&(0,M.jsxs)(M.Fragment,{children:[` ℓ covers every key. Dividing the accumulator by ℓ is the softmax. With V = 1 that quotient is `,(0,M.jsx)(`b`,{className:`g`,children:`1`}),`.`]})]})]})}function B({rows:e,cols:t,step:n}){let r=n%t,i=Math.floor(n/t)%e;return(0,M.jsx)(`group`,{children:Array.from({length:e*t},(n,a)=>{let o=a%t,s=Math.floor(a/t),c=s===i&&o===r,l=s<i||s===i&&o<r,u=(o-(t-1)/2)*.52,d=(s-(e-1)/2)*.52,f=c?.55:l?.16:0,p=c?`#fbbf24`:l?`#4ade80`:`#2a3148`;return(0,M.jsxs)(`mesh`,{position:[u,f,d],children:[(0,M.jsx)(`boxGeometry`,{args:[.4,c?.7:l?.22:.08,.4]}),(0,M.jsx)(`meshStandardMaterial`,{color:p,emissive:p,emissiveIntensity:c?.45:l?.2:0,roughness:.45})]},a)})})}function V(){let[e,t]=(0,b.useState)(`a100`),[n,r]=(0,b.useState)(2048),[i,o]=(0,b.useState)(64),[s,f]=(0,b.useState)(128),[m,g]=(0,b.useState)(128),[x,S]=(0,b.useState)(0),w=y[e],T=(0,b.useMemo)(()=>C({N:n,d:i,Br:s,Bc:m,bytes:2}),[n,i,s,m]),E=Math.min(8,T.Tr),D=Math.min(8,T.Tc),O=E*D,k=(x%O+O)%O,A=T.sram/1024,j=A<=w.smemKB,N=_(w.peak.fp16,w.bw),P=T.iFlash>=N,F=T.iStd>=N,L=(0,M.jsxs)(M.Fragment,{children:[`Q tile `,Math.floor(k/D),` · K tile `,k%D,(0,M.jsx)(`br`,{}),`amber tile is the only scores alive · green already folded into O`]});return(0,M.jsxs)(M.Fragment,{children:[(0,M.jsx)(c,{height:420,camera:[0,5.6,7.2],target:[0,.15,0],fov:42,overlay:L,hint:`drag to orbit · the flat tiles are the N×N matrix that HBM never receives`,children:(0,M.jsx)(B,{rows:E,cols:D,step:k})}),(0,M.jsxs)(d,{children:[(0,M.jsx)(h,{label:`GPU`,value:e,onChange:t,options:v}),(0,M.jsx)(h,{label:`N`,value:String(n),onChange:e=>r(+e),options:[[`512`,`512`],[`1024`,`1024`],[`2048`,`2048`],[`4096`,`4096`]]}),(0,M.jsx)(h,{label:`d`,value:String(i),onChange:e=>o(+e),options:[[`32`,`32`],[`64`,`64`],[`128`,`128`]]}),(0,M.jsx)(h,{label:`Bᵣ`,value:String(s),onChange:e=>f(+e),options:[[`64`,`64`],[`128`,`128`],[`256`,`256`]]}),(0,M.jsx)(h,{label:`Bᶜ`,value:String(m),onChange:e=>g(+e),options:[[`64`,`64`],[`128`,`128`]]}),(0,M.jsx)(u,{label:`Tile step`,min:0,max:O-1,value:k,onChange:S})]}),(0,M.jsx)(l,{items:[[`#fbbf24`,`score tile in SRAM`],[`#4ade80`,`already accumulated into O`],[`#2a3148`,`never stored`]]}),(0,M.jsxs)(p,{children:[`FP16, one head. Standard attention moves `,(0,M.jsx)(`b`,{children:a(T.standard)}),`. Flash moves `,(0,M.jsx)(`b`,{className:`g`,children:a(T.flash)}),` (`,(T.standard/T.flash).toFixed(1),`× less). Intensity `,(0,M.jsx)(`b`,{children:I(T.iStd)}),` vs `,(0,M.jsx)(`b`,{children:I(T.iFlash)}),` FLOP/byte. `,w.name,` FP16 ridge is `,(0,M.jsx)(`b`,{children:I(N)}),`, so standard is `,F?`on the roof`:`memory-bound`,` and flash is `,P?`on the roof`:`memory-bound`,`. SRAM for the live tiles is `,(0,M.jsxs)(`b`,{className:j?`g`:`r`,children:[A.toFixed(0),` KB`]}),` against `,w.smemKB,` KB on the SM`,j?``:` — this tile does not fit, so the scores spill and the intensity collapses`,`.`,E<T.Tr||D<T.Tc?(0,M.jsxs)(M.Fragment,{children:[` The picture shows the first `,E,`×`,D,` of `,T.Tr,`×`,T.Tc,` tiles.`]}):null]})]})}function H(){return(0,M.jsx)(f,{views:[{id:`t`,label:`Tensor-core roofline`,render:()=>(0,M.jsx)(L,{})},{id:`s`,label:`Online softmax`,render:()=>(0,M.jsx)(z,{})},{id:`f`,label:`3D: the score tile`,render:()=>(0,M.jsx)(V,{})}]})}var U={Lab:H,vizTitle:`See the tensor-core roof move, then the score matrix that never lands in HBM`,tryIt:[`Leave **A100**, **FP16**, **N = 256**. Intensity is 85, the ridge is 153, and the dot sits on the memory slope at about 174 TFLOP/s — not on the 312 peak.`,`Switch the format to **FP32**. The same matrix is already on the 19.5 TFLOP/s roof. A faster CUDA core would not help; a tensor core is not fed yet.`,`Click **N = 1024**. FP16 intensity passes the ridge and the dot lands on 312.`,`Open **Online softmax** and step to **tile 2**. The max moves from 2 to 5, and the first tile’s mass is multiplied by e⁻³.`,`Step to the last tile. V = 1, so the normalized output is exactly 1. The N-long score row was never stored.`,`Open **3D**. Standard attention at N = 2048, d = 64, FP16 moves 33 MB. The tiled kernel moves 8.5 MB, and the amber block is the only score tile that exists.`],theory:E,math:D,practice:O,code:[{title:`16×16×16 FP16 tile, FP32 accumulator`,lang:`cuda`,note:`One warp per output tile. All-ones inputs make every entry N, which is the check. Tensor-core rounding stays in the accumulator.`,src:k},{title:`Attention that allocates the N×N matrices`,lang:`cuda`,note:`S and P are real allocations. The byte count printed is the 4N² + 4Nd model. Correct, and the reason the next file exists.`,src:A},{title:`Online softmax, one warp per query, no score row`,lang:`cuda`,note:`Running max, running sum, rescale. V = 1, so every output is exactly 1. Tile size is one key; the SRAM blocking is the lab.`,src:j}],quiz:[{q:`A tensor-core WMMA tile of FP16 values accumulates into:`,options:[`FP16`,`FP32`,`INT8`,`The register file as a single FMA`],answer:1,why:`The products are low precision. The sum is kept in FP32 so a long K reduction does not wash out.`},{q:`A100, N = 256, TF32. The GEMM runs at about:`,options:[`156 TFLOP/s, the TF32 peak`,`19.5 TFLOP/s, the FP32 peak`,`87 TFLOP/s, on the memory slope`,`312 TFLOP/s`],answer:2,why:`Intensity is 42.7 FLOP/byte and the TF32 ridge is 76.5. $\\beta I$ is about 87 TFLOP/s. The peak is not the rate of this matrix.`},{q:`Standard attention’s intensity, as N grows, tends to:`,options:[`Something proportional to N`,`d / (element size)`,`The tensor-core peak`,`Zero`],answer:1,why:`FLOPs and the N×N score traffic grow together. FP16 with d = 64 tends to 32 FLOP/byte, under every tensor-core ridge in the lab.`},{q:`FlashAttention’s asymptotic intensity depends on:`,options:[`N only`,`The Q-tile rows Bᵣ and the element size`,`The batch size`,`How many streams you launch`],answer:1,why:`$I \\to 2 B_r / s$. A bigger tile re-reads K and V fewer times. N cancels out.`},{q:`The online-softmax factor e^(m − m′) is less than 1 when:`,options:[`The new tile raises the running max`,`The new tile is empty`,`You use FP16`,`The kernel is memory-bound`],answer:0,why:`Exponentials computed against the old max are too large once a bigger max appears. Multiplying by e^(m−m′) corrects them. If the max does not move, the factor is 1.`},{q:`N = 2048, d = 64, tiles 128, FP16. Flash moves:`,options:[`The same 33 MB as standard attention`,`8.5 MB, about 3.9× less`,`Nothing; Q and K stay in registers`,`N² bytes`],answer:1,why:`K and V are re-read once per Q tile (16 times). S and P are not written. 33.0 / 8.50 = 3.9.`},{q:`Why does the backward pass recompute score tiles?`,options:[`Softmax is not differentiable`,`Storing P would write the N×N matrix the forward pass avoided`,`Tensor cores cannot multiply V`,`Recomputing uses less SRAM than one tile`],answer:1,why:`The bytes were the bottleneck. Recomputing from Q and K, which are N×d, is cheaper in HBM than rereading N×N.`}]};export{U as default};