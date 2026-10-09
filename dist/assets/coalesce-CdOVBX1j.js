import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{g as r,n as i,s as a,t as o,y as s}from"./viz-CPys2405.js";import{t as c}from"./Stage3D-CWOT2b2l.js";import{a as l,c as u,i as d,l as f,o as p,r as m,s as h,t as g}from"./hooks-Dw7oo1m3.js";var _=e(t(),1),v=`## A warp does not load 32 times

The hardware does not issue one memory request per thread. It looks at the 32 addresses a warp produces and **covers them with as few transactions as possible**. Two different memories, two different rules:

| Memory | Unit that gets merged or split | What you want |
|---|---|---|
| **Global** (HBM, via L1/L2) | **32-byte sectors** | many threads inside the same few sectors |
| **Shared** | **32 banks** | every thread in a different bank, or all of them on the *same address* |

The previous chapter cut *how many* bytes you load. This one is about *how the hardware actually moves them*. A tiled GEMM that reloads every tile through stride-32 accesses gives the bandwidth back.

## Sectors, not threads

Global memory is transferred in **32-byte sectors**. Four sectors make the **128-byte cache line** mentioned in the memory chapter. A fully coalesced warp load of 32 \`float\`s is exactly one cache line: **4 sectors, 128 bytes, nothing wasted**.

\`\`\`cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;   // consecutive threadIdx.x
out[i] = in[i];                                  // stride 1: 4 sectors / warp
\`\`\`

\`threadIdx.x\` is the consecutive lane (the linear thread id runs x, then y, then z). So the *fast* dimension of every array you touch from a warp must be \`threadIdx.x\`.

Stride breaks it. \`in[i * stride]\` with \`stride == 32\` puts each lane in its own sector: **32 sectors, 1024 bytes moved, 128 bytes used** — 12.5 % efficiency. The kernel is still "touching N floats". The memory system is moving 8× more.

<div class="callout">

**Efficiency, not sector count.** A \`float4\` load also touches 16 sectors per warp (32 × 16 B = 512 B). That is fine: every byte is useful, and each instruction puts more bytes in flight (Little's law). A stride-2 \`float\` load touches 8 sectors for only 128 useful bytes. Same "more sectors", opposite meaning. The lab reports **useful bytes / bytes moved**.

</div>

### What else splits a warp

- **Misalignment.** 32 consecutive floats starting at element 1 cover bytes 4…131, which is **5** sectors instead of 4. \`cudaMalloc\` is aligned; a pointer plus an odd column offset is not. \`float4\` loads must be 16-byte aligned or they split (and on some toolchains they fail to vectorise).
- **AoS.** \`struct Point { float x, y, z, w; }\` and then reading only \`.x\` is stride 4. Lay the same data out as four arrays (SoA) and the load is stride 1. This is the single most common coalescing bug in particle and graph code.
- **The transpose write.** \`out[x * N + y] = in[y * N + x]\` reads coalesced and **writes** with stride N. Stores coalesce by the same sector rule as loads. \`transpose.cu\` fixes it by staging a tile and writing the transpose back out along \`threadIdx.x\`.

## Banks

Shared memory is 32 banks wide. Successive **32-bit words** go to successive banks:

$$
\\text{bank} = \\text{wordIndex} \\bmod 32.
$$

A warp's shared-memory instruction is one access if every lane hits a **different** bank, or if several lanes hit the **same address** (a **broadcast** — the hardware serves it once). It splits into several serialised passes when two lanes hit **different addresses in the same bank**. The number of passes is the **conflict degree**: the busiest bank's count of distinct addresses. An n-way conflict makes that instruction about n times slower. It does not change the answer.

\`\`\`cuda
__shared__ float s[32][32];
float row = s[threadIdx.y][threadIdx.x];   // lanes read s[y][0..31] → banks 0..31, no conflict
float col = s[threadIdx.x][0];             // lanes read s[0..31][0] → wordIndex = tx*32, bank 0 for every lane
\`\`\`

The column is a **32-way conflict**. Every lane wants bank 0, and each wants a different row.

### The +1 padding

\`\`\`cuda
__shared__ float s[32][33];
float col = s[threadIdx.x][0];             // wordIndex = tx*33, bank = tx
\`\`\`

A leading dimension that is **odd** (coprime with 32) rotates the column across the banks. \`33\` is the usual choice: one extra column, and \`tile[32][33]\` is the line the tiled-matmul chapter pointed at. \`34\` is not a fix — \`tx*34\` only lands on even banks, a 2-way conflict.

Broadcast is not a conflict. In the plain tiled GEMM, \`As[ty][k]\` is the *same* address for every lane of a warp (one row, one k). That is a broadcast, which is why that kernel did not need padding. The conflict appears when you **transpose on the way through shared memory**, because the read back is a column. \`transpose.cu\` times both.

64-bit words (\`double\`, \`float2\`) occupy two consecutive banks. Two threads whose 64-bit values start on the same bank still conflict. The lab and the rule of thumb below are the 32-bit model; widen the word and the same padding still works.

## Putting both on one kernel

A tiled transpose does four memory operations:

1. **Global load** \`in[y, x]\` with \`x = threadIdx.x\` — coalesced.
2. **Shared store** \`tile[ty][tx]\` — a row, no bank conflict. \`__syncthreads()\`.
3. **Shared load** \`tile[tx][ty]\` — a column. Conflict degree 32 unless the row stride is 33.
4. **Global store** \`out[y2, x2]\` with \`x2 = threadIdx.x\` — coalesced again.

Step 3 is why "I coalesced the global accesses" can still leave the kernel slow. Nsight Compute's Memory Workload Analysis shows shared-memory bank conflicts separately from global sector counts; the Speed-of-Light section will not tell you which of the two you hit.

<div class="callout tip">

**Rules that fall out of the hardware.**
1. Make \`threadIdx.x\` the contiguous index of every global array you read or write.
2. Count sectors when you stride, gather, or start at an odd offset. Efficiency is useful bytes over sectors × 32.
3. Shared rows are free. Shared columns of a multiple-of-32 width are not. Pad the leading dimension to an odd number.
4. Same address many times is a broadcast, not a conflict. Different addresses, same bank, is a conflict.
5. Fix the access pattern before you touch instruction scheduling. A 32-way conflict or an 8× overfetch dominates.

</div>
`,y=`## Sectors touched by one warp

Lane $\\ell$ issues a load of $w$ bytes at byte address

$$
a_\\ell = (b + \\ell\\, s)\\, w_0,
$$

where $s$ is the stride in elements, $b$ the starting element, and $w_0$ the element size. A \`float\` has $w = w_0 = 4$; a \`float4\` has $w = 16$. The load covers the half-open range $[a_\\ell,\\ a_\\ell + w)$.

$$
\\text{sectors}
  = \\left|\\left\\{\\ \\left\\lfloor \\frac{t}{32} \\right\\rfloor
    : t \\in \\bigcup_{\\ell=0}^{31} [a_\\ell,\\ a_\\ell+w)\\ \\right\\}\\right|.
$$

Bytes moved $= 32 \\times \\text{sectors}$ (the hardware always transfers a whole sector). Useful bytes $= 32 w$ only when the ranges do not overlap; they don't for $s \\ge 1$.

$$
\\eta = \\frac{32 w}{32 \\cdot \\text{sectors}} = \\frac{w}{\\text{sectors}}.
$$

### Aligned power-of-two strides, $w = 4$

The 32 lanes span $128 s$ bytes starting on a sector boundary, so $\\text{sectors} = 4s$ and $\\eta = 1/s$.

| Stride $s$ | Sectors | Bytes moved | $\\eta$ |
|---|---|---|---|
| 1 | 4 | 128 B | 100 % |
| 2 | 8 | 256 B | 50 % |
| 4 | 16 | 512 B | 25 % |
| 8 | 32 | 1 KiB | 12.5 % |
| 32 | 32 | 1 KiB | 12.5 % |

Stride 8 and stride 32 land on the same sector count: there are only 32 lanes, so you cannot touch more than 32 sectors with a 4-byte load that sits inside one sector. Past $s = 8$ the *efficiency* stays at $4/32$; you are already at one sector per lane.

### The misaligned float

$s = 1$, $w = 4$, $b = 1$. Bytes $[4,\\ 132)$ cover sectors $\\lfloor 4/32 \\rfloor = 0$ through $\\lfloor 131/32 \\rfloor = 4$: **5 sectors**, $\\eta = 4/5 = 80\\%$. One element of offset costs a whole extra sector on every warp.

### float4

$w = 16$, $s = 1$, $b$ a multiple of 4 (16-byte aligned): $32 \\times 16 = 512$ useful bytes $= 16$ sectors, $\\eta = 1$. The same \`float4\` with $b = 1$ (4-byte aligned only) straddles a sector on every lane and $\\eta$ drops. Alignment is part of the type, not just the pointer.

## Banks

Word index $i$ (32-bit words from the base of the array) maps to

$$
\\text{bank}(i) = i \\bmod 32.
$$

For a 2-D array with leading dimension $L$ (in words), element $(r, c)$ has $i = r L + c$.

A warp that reads addresses $i_0, \\ldots, i_{31}$:

$$
d = \\max_{0 \\le b < 32} \\left|\\{\\, i_\\ell : \\text{bank}(i_\\ell) = b \\,\\}\\right|.
$$

$d$ is the conflict degree. The instruction is issued in $d$ serial passes, so shared-memory throughput on that instruction scales as $1/d$. If all 32 lanes share one address, each bank's set has size 1: a **broadcast**, $d = 1$.

### Column of \`tile[32][L]\`

Lane $\\ell$ reads row $\\ell$, column $0$: $i = \\ell L$.

$$
\\text{bank} = (\\ell L) \\bmod 32.
$$

- $L = 32$: bank $= 0$ for every $\\ell$. All 32 addresses differ. $d = 32$.
- $L = 16$: bank $= 0$ for even $\\ell$ and $16$ for odd $\\ell$. Sixteen distinct rows on each of those banks. $d = 16$.
- $L = 33$: $(\\ell \\cdot 33) \\bmod 32 = \\ell$. All different. $d = 1$.
- $L = 34$: $(\\ell \\cdot 34) \\bmod 32 = (2\\ell) \\bmod 32$. Only even banks, two rows each. $d = 2$.

$d = 1$ for every column (any $c$) exactly when $L$ is **odd**: $L$ and $32$ are coprime, so $\\ell \\mapsto \\ell L$ is a bijection on the banks. That is the whole content of "pad to \`TILE+1\`".

### Row

Lane $\\ell$ reads $i = r L + \\ell$. Those are 32 consecutive words, hence 32 consecutive banks, regardless of $L$ (a row never wraps inside 32 words). $d = 1$. Padding does nothing for pure row access, and it costs $T$ extra words of shared memory per tile.

### Cost of the pad

\`tile[32][33]\` instead of \`tile[32][32]\` is $32$ extra floats $= 128$ bytes per tile. A block that keeps one such tile spends 128 B more shared memory. Against a 32× slowdown on every column read, it is the right trade until shared memory is the occupancy limiter (the Occupancy chapter).

## What the transpose moves

$N \\times N$ floats, read once and written once: $Q = 2 N^2 \\cdot 4$ bytes. That number is identical for the naive and the tiled kernels — tiling a transpose does not change arithmetic intensity, because there is no reuse. The naive kernel just *fails to reach* $\\beta$, because the stores touch $N/32$ times more sectors than a coalesced write (stride $N$, and $N$ is a multiple of 32 in every interesting case). The unpadded tiled kernel reaches the global roof on paper and then loses a factor of up to 32 in the shared-memory column read. The padded kernel is the one whose measured GB/s should sit near the STREAM numbers from the memory chapter.
`,b=`## Exercises

**Q1.** A warp loads 32 \`float\`s at stride 1, starting at element 0. How many 32-byte sectors, and what is the efficiency?

<details>
<summary>Show answer</summary>

128 consecutive bytes = **4 sectors**. Useful = 128 B, so $\\eta = 100\\%$. This is the access \`in[blockIdx.x * blockDim.x + threadIdx.x]\` generates.

</details>

**Q2.** Same warp, but the pointer is offset by one float (\`b = 1\`). Sectors and efficiency?

<details>
<summary>Show answer</summary>

Bytes $[4, 132)$ cover sectors 0, 1, 2, 3 and 4: **5 sectors**, 160 B moved, $\\eta = 128/160 = 80\\%$. Every subsequent warp of a misaligned column pays the same extra sector.

</details>

**Q3.** Stride 4, \`float\`, aligned. A kernel copies $2^{22}$ outputs. How many bytes does the memory system move, against how many useful bytes?

<details>
<summary>Show answer</summary>

$\\eta = 1/4$. Useful $= 2^{22}\\cdot 4 = 16$ MiB. Moved $= 64$ MiB. From the table, stride 4 is 16 sectors per warp against 4 for stride 1.

</details>

**Q4.** \`float s[32][32]\`. A warp reads \`s[threadIdx.x][5]\` (each lane a different row, column 5). Conflict degree?

<details>
<summary>Show answer</summary>

Word index $= \\ell \\cdot 32 + 5$, bank $= 5$ for every lane, 32 different addresses. **32-way** conflict. The column index does not matter when the leading dimension is a multiple of 32.

</details>

**Q5.** You change the declaration to \`s[32][33]\` and read \`s[threadIdx.x][5]\`. Degree now? What about \`s[32][34]\`?

<details>
<summary>Show answer</summary>

$L = 33$: bank $= (\\ell \\cdot 33 + 5) \\bmod 32 = (\\ell + 5) \\bmod 32$. All 32 banks, **degree 1**. $L = 34$: bank $= (2\\ell + 5) \\bmod 32$, only 16 distinct banks, **degree 2**. Padding must be odd, not merely "a bit bigger".

</details>

**Q6.** \`float tile[32][16]\`. A warp reads the column \`tile[threadIdx.x][0]\`. Conflict degree?

<details>
<summary>Show answer</summary>

Bank $= (\\ell \\cdot 16) \\bmod 32$, which is 0 for even $\\ell$ and 16 for odd $\\ell$. Sixteen distinct rows on each bank: **degree 16**. Half the pain of $L = 32$, still a 16× serialisation. \`tile[32][17]\` (or \`[16][17]\`) brings it back to 1.

</details>

**Q7.** Why is \`As[ty][k] * Bs[k][tx]\` in the *unpadded* tiled GEMM from the previous chapter not a 32-way conflict, while \`tile[tx][ty]\` in the transpose is?

<details>
<summary>Show answer</summary>

In the GEMM a warp has one \`ty\` and consecutive \`tx\`. \`As[ty][k]\` is **one address** read by every lane: a broadcast, degree 1. \`Bs[k][tx]\` is a consecutive row: degree 1. The transpose reads \`tile[tx][ty]\` — consecutive lanes, *different rows*, same column — which is the column pattern, degree 32 when the width is 32.

</details>

**Q8 (code).** Build \`transpose.cu\` for $N = 2048$ and $N = 4096$. Order the three kernels by GB/s and say which limit each one hits.

<details>
<summary>Show answer</summary>

Naive is limited by uncoalesced stores (stride $N$): expect a small fraction of peak. Tiled without padding fixes global traffic but the shared column read is 32-way; it lands in between. Padded tiled should approach the STREAM copy bandwidth from the memory chapter (both kernels move $2 N^2$ bytes, coalesced). The gap between "tiled, no pad" and "padded" is the bank-conflict term with no change in global bytes.

</details>

## In practice

- **Profile the pattern, then the kernel.** A one-warp microbenchmark (\`stride_copy.cu\`, \`smem_banks.cu\`) tells you the factor before you rewrite a 200-line kernel.
- **\`float4\` only helps if it is aligned and stride-1.** A vectorised load of a stride-2 array just wastes a wider transaction.
- **SoA by default** for any per-thread record you stream. Convert AoS at the boundary.
- **Pad every shared tile whose second index is not \`threadIdx.x\`.** If you only ever read rows, padding is pure overhead.
- **Nsight Compute → Memory Workload Analysis** splits "sectors" (global) from "bank conflicts" (shared). The two fixes are different; the roofline will not tell them apart.

## Common pitfalls

- Assuming "consecutive \`threadIdx.y\`" coalesces. The warp's consecutive lanes are \`threadIdx.x\`.
- Counting a broadcast as a conflict, and "fixing" it with padding that changes nothing.
- Padding by 1 on an already-odd width, or padding by 2 (makes a 2-way conflict).
- Declaring the pad but still indexing \`[TILE][TILE]\` in the column read.
- Calling a kernel memory-bound because it is slow, when it is actually moving 8× more bytes than you think, or serialising 32 ways inside the SM.
- Testing bank conflicts with a single timing of 32 threads and no inner loop: launch and latency noise swamp a conflict that is obvious over a few thousand iterations.
`,x=`// How stride destroys coalescing.
// out[i] = in[i * stride]. Stride 1 is one 128-byte line per warp (4 × 32-byte sectors).
// Stride 32 is one sector per thread: 8× the bytes moved for the same useful data.
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 stride_copy.cu -o stride && ./stride
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \\
    do {                                                                        \\
        cudaError_t err = (call);                                               \\
        if (err != cudaSuccess) {                                               \\
            fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__,       \\
                    cudaGetErrorString(err));                                   \\
            exit(EXIT_FAILURE);                                                 \\
        }                                                                       \\
    } while (0)

__global__ void copy_stride(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i * stride];
}

int main() {
    const int n = 1 << 22;                              // outputs
    const int maxStride = 32;
    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, (size_t)n * maxStride * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(in, 0, (size_t)n * maxStride * sizeof(float)));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    printf("stride   ms    useful GB/s   bytes moved / useful   (model)\\n");
    for (int stride = 1; stride <= maxStride; stride *= 2) {
        auto launch = [&] { copy_stride<<<blocks, threads>>>(in, out, n, stride); };
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 8; ++r) {
            CUDA_CHECK(cudaEventRecord(t0));
            launch();
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
            if (ms < best) best = ms;
        }
        CUDA_CHECK(cudaGetLastError());
        // Model: each warp of 32 floats touches ceil-unique 32-byte sectors.
        // For an aligned stride-s float walk the sector count per warp is min(32, s==0?1: ...).
        // Stride s (elements): 32 threads span 32*s floats = 128*s bytes, touching 4*s sectors
        // when s is a power of two and the base is aligned. Useful bytes stay 128 per warp.
        const double useful = (double)n * 4;
        const double moved = useful * stride;           // power-of-two stride, aligned: s× overfetch
        printf("%6d  %6.3f   %8.1f      %5.1f×\\n",
               stride, best, useful / (best * 1e6), moved / useful);
    }
    cudaFree(in); cudaFree(out);
    return 0;
}
`,S=`// Shared-memory bank conflicts, isolated from global traffic.
// One warp has consecutive threadIdx.x and a constant threadIdx.y.
//   row  s[y][tx]     → 32 consecutive words → 32 different banks → no conflict
//   col  s[tx][0]     → stride 32            → all 32 threads hit bank 0 → 32-way
//   pad  s[tx][0]     → leading dim 33       → bank = tx → no conflict
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 smem_banks.cu -o banks && ./banks
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \\
    do {                                                                        \\
        cudaError_t err = (call);                                               \\
        if (err != cudaSuccess) {                                               \\
            fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__,       \\
                    cudaGetErrorString(err));                                   \\
            exit(EXIT_FAILURE);                                                 \\
        }                                                                       \\
    } while (0)

constexpr int ITERS = 4096;

__global__ void row_read(float* sink) {
    __shared__ float s[32][32];
    s[threadIdx.y][threadIdx.x] = threadIdx.x;
    __syncthreads();
    float a = 0.f;
    for (int it = 0; it < ITERS; ++it)
        for (int k = 0; k < 32; ++k)
            a += s[threadIdx.y][(threadIdx.x + k) & 31];         // rotated row: 32 distinct banks, no conflict
    if (threadIdx.x == 0) sink[blockIdx.x * blockDim.y + threadIdx.y] = a;
}

__global__ void col_read(float* sink) {
    __shared__ float s[32][32];
    s[threadIdx.y][threadIdx.x] = threadIdx.x;
    __syncthreads();
    float a = 0.f;
    for (int it = 0; it < ITERS; ++it)
        for (int k = 0; k < 32; ++k) a += s[threadIdx.x][k];     // warp: 32 different rows, one column → 32-way
    if (threadIdx.x == 0) sink[blockIdx.x * blockDim.y + threadIdx.y] = a;
}

__global__ void col_padded(float* sink) {
    __shared__ float s[32][33];                                  // +1 column: leading dimension coprime with 32
    s[threadIdx.y][threadIdx.x] = threadIdx.x;
    __syncthreads();
    float a = 0.f;
    for (int it = 0; it < ITERS; ++it)
        for (int k = 0; k < 32; ++k) a += s[threadIdx.x][k];     // bank = (tx*33 + k) % 32 = (tx + k) % 32
    if (threadIdx.x == 0) sink[blockIdx.x * blockDim.y + threadIdx.y] = a;
}

template <typename Fn>
float time_ms(Fn launch) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < 6; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    return best;
}

int main() {
    const dim3 block(32, 8);                 // 8 warps; each warp is one row (consecutive threadIdx.x)
    const int blocks = 128;
    float* sink;
    CUDA_CHECK(cudaMalloc(&sink, blocks * block.y * sizeof(float)));

    float row = time_ms([&] { row_read<<<blocks, block>>>(sink); });
    float col = time_ms([&] { col_read<<<blocks, block>>>(sink); });
    float pad = time_ms([&] { col_padded<<<blocks, block>>>(sink); });
    printf("row (broadcast)     %7.2f ms\\n", row);
    printf("column, ld=32       %7.2f ms   (32-way bank conflict)\\n", col);
    printf("column, ld=33       %7.2f ms   (padded, expect ~row speed)\\n", pad);
    printf("conflict / padded   %.1f×\\n", col / pad);
    cudaFree(sink);
    return 0;
}
`,C=`// Matrix transpose is the kernel that gets both lessons wrong at once.
//   naive:          coalesced read, stride-N write          (global)
//   tiled, no pad:  coalesced global read AND write, but the
//                   shared-memory column read is a 32-way bank conflict
//   tiled, padded:  tile[TILE][TILE+1] removes the conflict
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 transpose.cu -o transpose && ./transpose 2048
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \\
    do {                                                                        \\
        cudaError_t err = (call);                                               \\
        if (err != cudaSuccess) {                                               \\
            fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__,       \\
                    cudaGetErrorString(err));                                   \\
            exit(EXIT_FAILURE);                                                 \\
        }                                                                       \\
    } while (0)

constexpr int TILE = 32;

__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;          // consecutive across the warp
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) out[x * N + y] = in[y * N + x];     // read coalesced, write stride N
}

__global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out, int N) {
    __shared__ float tile[TILE][TILE];                      // no padding
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    int x2 = blockIdx.y * TILE + threadIdx.x;               // consecutive → coalesced store
    int y2 = blockIdx.x * TILE + threadIdx.y;
    if (x2 < N && y2 < N) out[y2 * N + x2] = tile[threadIdx.x][threadIdx.y];   // column read: 32-way
}

__global__ void transpose_padded(const float* __restrict__ in, float* __restrict__ out, int N) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    int x2 = blockIdx.y * TILE + threadIdx.x;
    int y2 = blockIdx.x * TILE + threadIdx.y;
    if (x2 < N && y2 < N) out[y2 * N + x2] = tile[threadIdx.x][threadIdx.y];   // bank = (tx*(TILE+1) + ty) % 32
}

template <typename Fn>
float best_ms(Fn launch) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < 8; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    return best;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 2048;
    const size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> h(N * N), back(N * N);
    for (int i = 0; i < N * N; ++i) h[i] = (float)i;

    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemcpy(in, h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);                                 // 32-wide: each warp is one row (consecutive threadIdx.x)
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    auto report = [&](const char* name, float ms) {
        double gb = 2.0 * bytes / 1e9;                      // read N² + write N²
        printf("%-18s %7.3f ms   %6.0f GB/s\\n", name, ms, gb / (ms * 1e-3));
    };
    report("naive", best_ms([&] { transpose_naive<<<grid, block>>>(in, out, N); }));
    report("tiled, no pad", best_ms([&] { transpose_tiled<<<grid, block>>>(in, out, N); }));
    float ms = best_ms([&] { transpose_padded<<<grid, block>>>(in, out, N); });
    report("tiled, pad +1", ms);

    CUDA_CHECK(cudaMemcpy(back.data(), out, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N && bad < 3; i += N / 7)
        for (int j = 0; j < N && bad < 3; j += N / 5)
            if (back[i * N + j] != h[j * N + i]) ++bad;
    printf("spot-check %s\\n", bad ? "MISMATCH" : "ok");
    cudaFree(in); cudaFree(out);
    return 0;
}
`,w=n(),T=32,E=32,D=e=>{let t=Math.round(1e3*e)/10;return t>=99.95?`100`:String(t)};function O(e,t,n){let r=[],i=new Map;for(let a=0;a<T;a++){let o=(t+a*e)*n,s=[];for(let e=o;e<o+n;e++){let t=Math.floor(e/E);s[s.length-1]!==t&&s.push(t)}r.push({lane:a,start:o,secs:s}),s.forEach(e=>{i.has(e)||i.set(e,[]),i.get(e).push(a)})}let a=[...i.keys()].sort((e,t)=>e-t),o=T*n,s=a.length*E;return{hits:r,sectors:a,bySector:i,useful:o,moved:s,eta:o/s}}function k(e){let t=Array.from({length:32},()=>new Set);return e.forEach(e=>t[(e%32+32)%32].add(e)),{byBank:t,degree:Math.max(...t.map(e=>e.size)),broadcast:e.every(t=>t===e[0])}}var A=[[`s1`,`stride 1`,1,0,4],[`mis`,`offset +1`,1,1,4],[`s2`,`stride 2`,2,0,4],[`aos`,`AoS .x (stride 4)`,4,0,4],[`f4`,`float4`,1,0,16],[`col`,`column, stride 32`,32,0,4]];function j(){let[e,t]=(0,_.useState)(1),[n,c]=(0,_.useState)(0),[f,v]=(0,_.useState)(4),y=O(e,n,f),[b]=g(400,(e,t,n)=>{s(e,`one warp · 32 lanes`,16,16,{size:12,color:o.mute,weight:600}),s(e,`${D(y.eta)}% of bytes useful`,t-16,16,{size:13,color:y.eta>.95?o.e:y.eta>.45?o.d:o.r,align:`right`,weight:700});let c=t-16-16;e.fillStyle=`#1a1e32`,r(e,16,32,c,16,5),e.fill(),e.fillStyle=y.eta>.95?o.e:y.eta>.45?o.d:o.r,r(e,16,32,Math.max(4,c*y.eta),16,5),e.fill(),s(e,`${a(y.useful)} useful  /  ${a(y.moved)} moved  ·  ${y.sectors.length} sectors`,16,62,{size:12,color:o.ink,mono:!0});let l=Math.min(18,(c-93)/32),u=(c-32*l)/31,d=e=>16+e*(l+u);s(e,`lanes`,16,80,{size:11,color:o.mute}),y.hits.forEach(t=>{e.fillStyle=t.secs.length>1?o.c:o.a,r(e,d(t.lane),92,l,28,3),e.fill()});let p=y.sectors,m=Math.min(16,Math.max(4,p.length)),h=Math.ceil(p.length/m),g=Math.min(72,(c-(m-1)*6)/m);s(e,`32-byte sectors this warp touches  ·  number = lanes in that sector`,16,156,{size:11,color:o.mute}),p.forEach((t,n)=>{let a=n%m,c=Math.floor(n/m),l=16+a*(g+6),u=168+c*44,d=y.bySector.get(t).length;e.fillStyle=d*f>=E?i(o.e,.9):d===1?i(o.r,.85):i(o.d,.9),r(e,l,u,g,36,6),e.fill(),s(e,String(d),l+g/2,u+14,{align:`center`,size:14,color:`#0a0c14`,weight:800,mono:!0}),s(e,`s`+t,l+g/2,u+28,{align:`center`,size:9,color:i(`#0a0c14`,.7),mono:!0})});let _=168+h*44+6;if(_<n-4){let t=y.hits.some(e=>e.secs.length>1);s(e,t?`pink lanes: that load straddles two sectors (misaligned or wider than the gap)`:`every load sits inside one sector`,16,Math.min(n-12,_),{size:11,color:t?o.c:o.mute})}}),x=(e,n,r)=>{t(e),c(n),v(r)};return(0,w.jsxs)(w.Fragment,{children:[(0,w.jsx)(`canvas`,{...b}),(0,w.jsx)(d,{children:A.map(([t,r,i,a,o])=>(0,w.jsx)(m,{primary:e===i&&n===a&&f===o,onClick:()=>x(i,a,o),children:r},t))}),(0,w.jsxs)(d,{children:[(0,w.jsx)(u,{label:`Stride (elements)`,min:1,max:32,value:e,onChange:t}),(0,w.jsx)(u,{label:`Start element`,min:0,max:32,value:n,onChange:c}),(0,w.jsx)(h,{label:`Load width`,value:String(f),onChange:e=>v(+e),options:[[`4`,`float · 4 B`],[`8`,`float2 · 8 B`],[`16`,`float4 · 16 B`]]})]}),(0,w.jsx)(l,{items:[[o.e,`sector fully used`],[o.d,`sector partly used`],[o.r,`one lane, one sector`],[o.c,`lane straddles two sectors`]]}),(0,w.jsxs)(p,{children:[y.sectors.length,` sector`,y.sectors.length===1?``:`s`,` × 32 B = `,(0,w.jsx)(`b`,{children:a(y.moved)}),` moved for `,(0,w.jsx)(`b`,{children:a(y.useful)}),` wanted → efficiency `,(0,w.jsxs)(`b`,{className:y.eta>.95?`g`:`w`,children:[D(y.eta),`%`]}),`.`,e===1&&n===0&&f===4&&(0,w.jsxs)(w.Fragment,{children:[` This is the access a warp gets from `,(0,w.jsx)(`b`,{children:`threadIdx.x`}),` on a contiguous `,(0,w.jsx)(`b`,{children:`float`}),` array: one cache line.`]}),e===1&&n===1&&f===4&&(0,w.jsxs)(w.Fragment,{children:[` One float of misalignment turns 4 sectors into `,(0,w.jsx)(`b`,{children:`5`}),`. `,(0,w.jsx)(`b`,{children:`cudaMalloc`}),` is aligned; a column that starts at `,(0,w.jsx)(`b`,{children:`x = 1`}),` is not.`]}),e>1&&f===4&&n===0&&(0,w.jsxs)(w.Fragment,{children:[` Aligned stride `,e,`: about `,(0,w.jsxs)(`b`,{children:[e<=8?e:8,`×`]}),` the traffic of stride 1. Past stride 8 a 4-byte load is already one sector per lane, so it cannot get worse.`]}),f===16&&e===1&&(0,w.jsxs)(w.Fragment,{children:[` `,(0,w.jsx)(`b`,{children:`float4`}),` touches more sectors and still scores 100% when 16-byte aligned — every extra byte is useful, and one instruction fills more of the memory pipe.`]})]})]})}function M({stride:e,offset:t,word:n}){let r=(0,_.useMemo)(()=>O(e,t,n),[e,t,n]),i=(0,_.useMemo)(()=>{let e=0,t=r.sectors.map(t=>{let n=r.bySector.get(t).length,i=Math.max(.34,n*.2),a=e+i/2;return e+=i+.1,{s:t,n,width:i,x:a}}),n=e/2;return t.forEach(e=>{e.x-=n}),t},[r]),a=Object.fromEntries(i.map(e=>[e.s,e]));return(0,w.jsxs)(`group`,{children:[i.map(e=>{let t=e.n*n>=E?`#4ade80`:e.n===1?`#fb7185`:`#fbbf24`;return(0,w.jsxs)(`mesh`,{position:[e.x,0,0],children:[(0,w.jsx)(`boxGeometry`,{args:[e.width*.92,.4,.85]}),(0,w.jsx)(`meshStandardMaterial`,{color:t,roughness:.45,emissive:t,emissiveIntensity:.25})]},e.s)}),r.hits.map(e=>{let t=a[e.secs[0]],n=r.bySector.get(e.secs[0]),i=n.indexOf(e.lane),o=t.x+(i-(n.length-1)/2)*(t.width*.86/n.length);return(0,w.jsxs)(`mesh`,{position:[o,1.05,0],children:[(0,w.jsx)(`sphereGeometry`,{args:[.075,12,12]}),(0,w.jsx)(`meshStandardMaterial`,{color:e.secs.length>1?`#f472b6`:`#8b7bff`,emissive:`#8b7bff`,emissiveIntensity:.4})]},e.lane)})]})}function N(){let[e,t]=(0,_.useState)(1),[n,r]=(0,_.useState)(0),[i,a]=(0,_.useState)(4),o=O(e,n,i),s=(0,w.jsxs)(w.Fragment,{children:[`stride `,(0,w.jsx)(`b`,{children:e}),` · start element `,(0,w.jsx)(`b`,{children:n}),` · `,i,` B loads`,(0,w.jsx)(`br`,{}),(0,w.jsx)(`b`,{children:o.sectors.length}),` sectors · efficiency `,(0,w.jsxs)(`b`,{children:[D(o.eta),`%`]}),(0,w.jsx)(`br`,{}),(0,w.jsx)(`span`,{style:{color:`#4ade80`},children:`green`}),` full · `,(0,w.jsx)(`span`,{style:{color:`#fbbf24`},children:`amber`}),` partial · `,(0,w.jsx)(`span`,{style:{color:`#fb7185`},children:`red`}),` one lane`]});return(0,w.jsxs)(w.Fragment,{children:[(0,w.jsx)(c,{height:420,camera:[0,2.8,9.5],target:[0,.35,0],fov:42,overlay:s,hint:`drag to orbit · spheres are the 32 lanes, boxes are the sectors they hit`,children:(0,w.jsx)(M,{stride:e,offset:n,word:i})}),(0,w.jsxs)(d,{children:[(0,w.jsx)(u,{label:`Stride (elements)`,min:1,max:32,value:e,onChange:t}),(0,w.jsx)(u,{label:`Start element`,min:0,max:16,value:n,onChange:r}),(0,w.jsx)(h,{label:`Load width`,value:String(i),onChange:e=>a(+e),options:[[`4`,`float · 4 B`],[`8`,`float2 · 8 B`],[`16`,`float4 · 16 B`]]})]}),(0,w.jsx)(p,{children:`Each sphere is a lane; each box is a 32-byte sector that warp actually requests. Stride 1 packs all 32 lanes into 4 green boxes. Stride 32 leaves a red box per lane — the same 128 useful bytes, eight times the traffic.`})]})}function P(){let[e,t]=(0,_.useState)(`col`),[n,r]=(0,_.useState)(32),[a,c]=(0,_.useState)(0),[f,m]=(0,_.useState)(0),v=n+a,y=(0,_.useMemo)(()=>e===`stride`?Array.from({length:32},(e,t)=>t*v):e===`row`?Array.from({length:32},(e,t)=>f*v+t):Array.from({length:32},(e,t)=>t*v+f),[e,n,a,f,v]),b=(0,_.useMemo)(()=>k(y),[y]),[x]=g(340,(e,t)=>{let n=(t-36-12)/32;s(e,`shared-memory banks 0–31`,36,16,{size:12,color:o.mute,weight:600}),s(e,b.broadcast?`broadcast`:b.degree===1?`no conflict`:`${b.degree}-way conflict`,t-12,16,{align:`right`,size:13,weight:700,color:b.degree===1?o.e:o.r});for(let t=0;t<32;t++){let r=36+t*n,a=[...b.byBank[t]];e.fillStyle=a.length>1?i(o.r,.18):`#14182a`,e.fillRect(r+1,36,n-2,250),a.forEach((t,i)=>{let s=44+i*7;e.fillStyle=a.length>1?o.r:o.e,e.fillRect(r+2,s,n-4,5)}),t%4==0&&s(e,t,r+n/2,300,{align:`center`,size:10,color:o.mute,mono:!0})}s(e,`each dash is a distinct address in that bank · a full red column is one serialised pass per dash`,36,328,{size:11,color:o.mute})});return(0,w.jsxs)(w.Fragment,{children:[(0,w.jsx)(`canvas`,{...x}),(0,w.jsxs)(d,{children:[(0,w.jsx)(h,{label:`Access`,value:e,onChange:t,options:[[`row`,`row  tile[r][lane]`],[`col`,`column  tile[lane][c]`],[`stride`,`stride  s[lane * L]`]]}),(0,w.jsx)(h,{label:e===`stride`?`Stride`:`Width L`,value:String(n),onChange:e=>r(+e),options:[[`8`,`8`],[`16`,`16`],[`32`,`32`],[`33`,`33`]]}),(0,w.jsx)(h,{label:`Padding`,value:String(a),onChange:e=>c(+e),options:[[`0`,`+0`],[`1`,`+1 (odd)`],[`2`,`+2 (even)`]]}),e!==`stride`&&(0,w.jsx)(u,{label:e===`row`?`Row`:`Column`,min:0,max:31,value:f,onChange:m})]}),(0,w.jsx)(l,{items:[[o.e,`one address in this bank`],[o.r,`several addresses → serialised`]]}),(0,w.jsxs)(p,{children:[`Leading dimension `,(0,w.jsx)(`b`,{children:v}),`. Conflict degree `,(0,w.jsx)(`b`,{className:b.degree===1?`g`:`r`,children:b.broadcast?`1 (broadcast)`:b.degree}),b.degree>1&&(0,w.jsxs)(w.Fragment,{children:[` — this instruction takes `,(0,w.jsxs)(`b`,{children:[b.degree,`×`]}),` as long as a conflict-free one.`]}),e===`col`&&n===32&&a===0&&(0,w.jsxs)(w.Fragment,{children:[` Every lane hits `,(0,w.jsxs)(`b`,{children:[`bank `,f%32]}),` at a different row. Add padding `,(0,w.jsx)(`b`,{children:`+1`}),`.`]}),e===`col`&&a===1&&(0,w.jsxs)(w.Fragment,{children:[` `,(0,w.jsxs)(`b`,{children:[`L = `,v]}),` is odd, so a column is a permutation of the 32 banks. Degree 1.`]}),e===`col`&&a===2&&n%2==0&&(0,w.jsxs)(w.Fragment,{children:[` `,(0,w.jsx)(`b`,{children:`+2`}),` keeps L even. Even padding does not fix a power-of-two width — it leaves a `,b.degree,`-way conflict.`]}),e===`row`&&(0,w.jsx)(w.Fragment,{children:` A row is 32 consecutive words. Padding does not change it, and the column index is only a shift of which bank lane 0 starts on.`}),e===`stride`&&(0,w.jsxs)(w.Fragment,{children:[` Lane ℓ reads word `,(0,w.jsxs)(`b`,{children:[`ℓ·`,v]}),`. `,b.degree===1?`Those land on distinct banks.`:`Several lanes share a bank at different addresses.`]})]})]})}function F(){return(0,w.jsx)(f,{views:[{id:`c`,label:`Count the sectors`,render:()=>(0,w.jsx)(j,{})},{id:`d`,label:`3D: a warp over memory`,render:()=>(0,w.jsx)(N,{})},{id:`b`,label:`Shared-memory banks`,render:()=>(0,w.jsx)(P,{})}]})}var I={Lab:F,vizTitle:`Count the sectors a warp really moves, then the banks a column access serialises`,tryIt:[`Leave **stride 1**. Four green sectors, 100%. Click **offset +1**: a fifth sector appears and efficiency falls to 80%.`,`Click **column, stride 32**. Thirty-two red sectors — 128 useful bytes, 1 KiB moved.`,`Click **float4**. More sectors than the float case, efficiency still 100%. Width is not waste when every byte is used.`,`Open **Banks**, access = column, width 32, padding +0. One bank stacks 32 deep. Set padding to **+1** and it flattens.`,`Set padding to **+2** instead. The conflict drops from 32-way to 2-way and then stops. Even padding is not the fix.`,`Switch access to **row**. Degree stays 1 at every width — rows were never the problem.`],theory:v,math:y,practice:b,code:[{title:`Stride copy: same useful bytes, s× the traffic`,lang:`cuda`,note:`Prints useful GB/s for power-of-two strides. Stride 32 should move about 8× more bytes than stride 1.`,src:x},{title:`Row vs conflicted column vs padded column`,lang:`cuda`,note:`The three kernels do the same number of shared-memory reads. Only the bank mapping changes.`,src:S},{title:`Transpose: naive, tiled, and tiled with tile[32][33]`,lang:`cuda`,note:`Global bytes are identical for all three. The gap is coalescing on the store, then bank conflicts on the way out of shared memory.`,src:C}],quiz:[{q:`A coalesced warp load of 32 aligned floats moves:`,options:[`32 separate 4-byte transactions`,`4 sectors (128 bytes), all of them useful`,`One 32-byte sector`,`1024 bytes`],answer:1,why:`32 × 4 B = 128 B = four 32-byte sectors, which is one cache line.`},{q:`Stride-2 aligned float loads have an efficiency of about:`,options:[`100%`,`50%`,`12.5%`,`2%`],answer:1,why:`The warp spans 256 B and uses 128 B, so $\\eta = 1/s = 1/2$. Sectors go from 4 to 8.`},{q:"Why does `struct { float x, y, z, w; } p[n]` loaded as `p[i].x` coalesce badly?",options:[`Structs cannot live in global memory`,`Consecutive lanes are 16 bytes apart: stride 4`,`The compiler refuses struct loads`,`x is a register`],answer:1,why:"Each lane skips y, z and w. SoA (four separate arrays) makes `.x` stride 1."},{q:`A shared-memory broadcast is:`,options:[`A 32-way bank conflict`,`Many lanes reading one address, served in a single access`,`A global-memory sector`,`Illegal`],answer:1,why:`Conflict requires *different* addresses in the same bank. The same address is a broadcast, degree 1.`},{q:"`float s[32][32];` and a warp reads `s[threadIdx.x][0]`. The conflict degree is:",options:[`1`,`2`,`16`,`32`],answer:3,why:`Word index $= \\ell \\cdot 32$, so every lane maps to bank 0 at a different row.`},{q:`Which padding removes that conflict?`,options:["`s[32][34]`","`s[32][33]`","`s[32][32]` with `__syncthreads`","`s[31][32]`"],answer:1,why:`The leading dimension must be odd. 33 is coprime with 32; 34 still collides on even banks (degree 2).`},{q:"The unpadded tiled GEMM reads `As[ty][k]` from a warp with one `ty`. That shared load is:",options:[`A 32-way conflict`,`A broadcast`,`Uncoalesced global traffic`,`A stride-32 sector walk`],answer:1,why:`Every lane reads the same address. The conflict shows up when the read is a *column*, as in the transpose.`}]};export{I as default};