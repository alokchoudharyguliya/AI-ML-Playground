import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,f as n,l as r,n as i,v as a}from"./r3f-x1z21uF6.js";import{c as o,f as s,g as c,i as l,n as u,s as d,t as f,y as p}from"./viz-CPys2405.js";import{t as m}from"./Stage3D-CWOT2b2l.js";import{a as h,c as g,i as _,l as v,n as y,o as b,r as x,s as S,t as C,u as w}from"./hooks-Dw7oo1m3.js";var T=e(t(),1),E=`## The reuse problem

The last chapter left GEMM sitting on the roofline: a *naive* kernel that streams a whole row of $A$ and column of $B$ for every output has arithmetic intensity $I=1/s$ (0.25 FLOP/B in FP32) — pinned to the memory roof, a few percent of peak. The *ideal* kernel that reads each input once has $I=2N/(3s)$, which crosses the ridge as soon as $N$ is a few hundred. The gap is **reuse**: each $A_{ik}$ is needed by $N$ outputs in its row, each $B_{kj}$ by $N$ outputs in its column.

Registers cannot hold an $N\\times N$ panel. **Shared memory** can hold a $T\\times T$ *tile*, visible to every thread of the block. Load the tile once, reuse it $T$ times, slide along $K$. Global traffic drops by $\\sim T$. That is this chapter.

## Shared memory in one paragraph

\`\`\`cuda
__shared__ float tile[16][16];          // static: size known at compile time, one copy per block
extern __shared__ float dyn[];          // dynamic: size is the 3rd launch parameter
kernel<<<grid, block, bytes>>>(...);
\`\`\`

- **Scope:** the thread block. Not visible to other blocks, not to the host.
- **Lifetime:** the block. Contents are gone when the block finishes.
- **Location:** on-chip SRAM, same physical array as L1 (you pick the split). ~20–30 cycles, ~10× HBM bandwidth.
- **Layout:** 32 banks, 4 bytes wide. Two threads of a warp hitting different addresses in the *same* bank serialise — **bank conflicts**, the next chapter. Tiled GEMM with consecutive \`threadIdx.x\` is naturally conflict-free; we will not chase padding here.

A thread writes \`tile[ty][tx] = ...\`. That write is **not** guaranteed visible to other threads until \`__syncthreads()\`.

## \`__syncthreads()\` is a contract

\`__syncthreads()\` is a **block-wide barrier and memory fence**. After it returns:

1. Every thread of the block has reached the call.
2. Every shared *and* global write issued before it is visible to every thread of the block.

<div class="callout warn">

**The deadlock rule.** *Every* thread of the block must execute the *same* \`__syncthreads()\`. Putting it inside \`if (threadIdx.x < 16)\` (or behind \`if (row < N)\`) deadlocks the block: the taken threads wait forever for the skipped ones. Bounds checks go *around the memory operation*, not around the barrier.

</div>

A second barrier is needed at the *bottom* of a tile loop: otherwise the next cooperative load overwrites \`As\`/\`Bs\` while another warp is still multiplying the old tile.

On a *single warp* (32 threads) a missing barrier often *looks* fine, because the 32 lanes already issue together. The race is real as soon as two warps share the array — \`compute-sanitizer --tool racecheck\` is the authority, not a lucky run. The third lab tab and \`syncthreads_race.cu\` make this concrete.

## Cooperative load

The pattern for almost every shared-memory kernel:

1. **Each thread loads one (or a few) elements** of a tile from global memory into shared memory. Together they cover the whole tile. This is a *cooperative load*.
2. \`__syncthreads()\`.
3. **Every thread reads many elements** of the tile, now hitting SRAM instead of HBM.
4. \`__syncthreads()\` if the tile will be overwritten.
5. Repeat, or write the result back.

The load should be **coalesced**: \`threadIdx.x\` walks consecutive addresses so a warp's 32 loads become one 128-byte transaction (next chapter). In the GEMM kernel below, \`As[ty][tx] = A[row, t+tx]\` does exactly that.

## Tiled matrix multiply

$C = AB$, all $N\\times N$, row-major. Launch a 2-D grid of **TILE×TILE** thread blocks. Block \`(bx, by)\` owns the output tile

$$
C\\big[by\\cdot T :\\, by\\cdot T+T,\\ \\ bx\\cdot T :\\, bx\\cdot T+T\\big].
$$

Thread \`(tx, ty)\` inside the block owns one output, $C[by\\cdot T+ty,\\ bx\\cdot T+tx]$, and keeps a running product in a **register** \`acc\`.

Along $K$ the block slides a window of width $T$:

\`\`\`
k-tile t = 0, 1, …, N/T − 1
   load  As = A[by·T : by·T+T,  t·T : t·T+T]     // T×T, cooperatively
   load  Bs = B[t·T : t·T+T,  bx·T : bx·T+T]
   __syncthreads()
   acc += row ty of As  ·  column tx of Bs       // T FMAs, all in shared memory
   __syncthreads()
write C
\`\`\`

Each input element that lands in \`As\` is then read by **T threads** (a whole output-tile row). Same for \`Bs\` and a column. That is the factor-of-$T$ traffic cut.

When $N$ is not a multiple of $T$, **pad with zeros** on the load (the \`?: 0\` in the code tab) rather than skipping the barrier. The extra FMAs on zeros are free next to an HBM round-trip.

Open the **3D** lab: the two floating tiles *are* \`As\` and \`Bs\`; the floor is the C-tile. Step through a k-slab and watch one thread's inner product.

## How big a tile?

| TILE | Threads / block | Shared bytes (2 tiles, FP32) | Reuse $T$ | Notes |
|---|---|---|---|---|
| 8 | 64 | 0.5 KB | 8 | Occupancy-friendly, little reuse |
| 16 | 256 | 2 KB | 16 | The sweet spot to start with |
| 32 | 1024 | 8 KB | 32 | Max threads/block; fewer resident blocks |

Bigger $T$ raises arithmetic intensity ($I=T/s$) and so moves you right on the roofline, but it burns shared memory and thread slots, which can *drop occupancy* enough that you can no longer hide HBM latency (Occupancy chapter). Measure; don't assume 32 wins.

A further trick, **register blocking**, has each thread own a small $r\\times r$ patch of $C$ (e.g. 4×4) so the inner kernel is $T$ FMAs into 16 accumulators. CUTLASS / cuBLAS do this; it is how you actually reach Tensor Core peak. We stay at one output per thread so the algorithm stays visible.

## A smaller cousin: stencils and halos

Not every kernel is GEMM. A 3-point stencil $y_i = (x_{i-1}+2x_i+x_{i+1})/4$ looks like 3 global loads per output. Have the block load its interior **plus one halo cell on each side** into shared memory, barrier, then every output is 3 shared reads. Halo cost is $2/\\text{blockDim}$ extra loads — ~1 % at 256 threads. The same idea in 2-D is the 5-point / 9-point stencil with a 1-cell apron, and in attention it is the $B_r\\times B_c$ tiles of FlashAttention.

## What this chapter is not

- **Bank conflicts and padding \`As[TILE][TILE+1]\`** — next chapter. The GEMM kernel here is already conflict-free for the inner product.
- **Tensor Cores / WMMA / \`mma.sync\`** — FlashAttention chapter. Tiling is the *software* analogue of what those units do in hardware.
- **cuBLAS.** For production GEMM, call the library. The point of writing this kernel is to *see* why tiling works, and to reuse the pattern on fusions the library cannot write for you.

<div class="callout tip">

**The checklist.** (1) Identify data reused by many threads of a block. (2) Stage it in \`__shared__\` with a coalesced cooperative load. (3) Barrier. (4) Compute from shared memory / registers. (5) Barrier again if you will overwrite the tile. (6) Count bytes: if $I$ is still left of the ridge, fuse, widen the tile, or drop to a smaller dtype — don't micro-optimise the FMAs.

</div>
`,D=`## Naive GEMM traffic

$C=AB$ with $N\\times N$ matrices, $s$ bytes per element, $W=2N^3$ FLOP.

Each of the $N^2$ outputs reads $N$ elements of $A$ and $N$ of $B$:

$$
Q_{\\text{naive}}=2N^3 s + N^2 s,\\qquad
I_{\\text{naive}}=\\frac{2N^3}{2N^3 s+N^2 s}\\approx\\frac{1}{s}.
$$

FP32: $I=1/4$ FLOP/B. On an A100 ($\\pi=19.5$ TFLOP/s, $\\beta=2.04$ TB/s) the ridge is $9.6$ FLOP/B, so this kernel is **~38× below the ridge** no matter how large $N$ is.

## Tiled traffic

A TILE $T$ grid: $(N/T)^2$ output tiles, $N/T$ slabs along $K$. Each slab loads $T^2$ elements of $A$ and $T^2$ of $B$:

$$
Q_{\\text{tiled}}=\\underbrace{\\Big(\\frac{N}{T}\\Big)^2\\cdot\\frac{N}{T}\\cdot 2T^2 s}_{A,B\\text{ loads}} + N^2 s
=\\frac{2N^3 s}{T}+N^2 s,
$$

$$
I_{\\text{tiled}}=\\frac{2N^3}{2N^3 s/T+N^2 s}=\\frac{T}{s}\\cdot\\frac{1}{1+T/(2N)}\\approx\\frac{T}{s}.
$$

Reuse factor versus naive: $Q_{\\text{naive}}/Q_{\\text{tiled}}\\approx T$. FP32, $T=16$: $I\\approx 4$ FLOP/B (still memory-bound on A100, but 16× fewer bytes). $T=32$: $I\\approx 8$, almost at the FP32 ridge.

### Worked numbers, $N=1024$, FP32

| Kernel | Loads | Stores | $Q$ | $I$ | HBM-bound time on 2 TB/s |
|---|---|---|---|---|---|
| Naive | $2\\cdot1024^3$ | $1024^2$ | $8.59$ GB | $0.25$ | $4.2$ ms |
| $T=16$ | $2\\cdot1024^3/16$ | $1024^2$ | $0.54$ GB | $4.0$ | $0.26$ ms |
| $T=32$ | $2\\cdot1024^3/32$ | $1024^2$ | $0.27$ GB | $8.0$ | $0.13$ ms |
| Ideal (read $A,B$ once) | $2\\cdot1024^2$ | $1024^2$ | $12.6$ MB | $170$ | **compute-bound** ($\\approx 0.11$ ms at 19.5 TF) |

$W=2\\cdot1024^3=2.15$ GFLOP. At A100 peak that is $0.11$ ms, so $T=32$ is within ~20 % of the compute roof *if* you actually hit HBM peak. In practice the naive kernel is much slower than 4.2 ms because the $B$ column is uncoalesced; tiling fixes that too.

### When does tiled GEMM become compute-bound?

Ignore the lower-order store: $T/s > \\pi/\\beta$. FP32 A100: $T/4>9.6\\Rightarrow T>38$ — larger than a block can hold at one output per thread. That is why production kernels **register-block**: $T_{\\text{effective}}=T_{\\text{smem}}\\cdot r$ with $r$ outputs per thread, or they switch to Tensor Cores whose ridge is ~150 FLOP/B and demand even more reuse (FlashAttention chapter).

## Shared-memory footprint and occupancy

Two FP32 tiles:

$$
S=2\\,T^2 s=8T^2\\text{ bytes}.
$$

$T=16\\Rightarrow 2$ KB, $T=32\\Rightarrow 8$ KB. An SM with 164 KB shared (A100) could in principle hold $164/2=82$ such blocks, but it is also limited to 16–32 blocks and 2048 threads: at 256 threads/block the thread cap is 8 blocks, so **shared memory is not the limiter** at $T=16$. At $T=32$ (1024 threads) the thread cap is 2 blocks and shared memory is still fine. The Occupancy chapter puts numbers on the three-way fight (registers, shared, threads).

Dynamic shared memory is just $S$ passed at launch; occupancy tools see the same $S$.

## Inner-product work per thread

Per k-tile each thread does $T$ FMAs ($2T$ FLOP) and 2 shared loads per FMA. Over $N/T$ tiles:

$$
\\text{FLOP/thread}=2N,\\qquad \\text{shared loads/thread}=2N.
$$

The global loads per thread are only $2\\cdot(N/T)$ — one $A$ element and one $B$ element per k-tile. That is the whole point.

## Stencil halo

Block of $B$ threads, radius-$r$ stencil (3-point $\\Rightarrow r=1$). Cooperative load of $B+2r$ elements, then $B$ outputs:

$$
\\frac{Q_{\\text{smem}}}{Q_{\\text{naive}}}=\\frac{B+2r}{(2r+1)B}\\approx\\frac{1}{2r+1}\\quad(B\\gg r).
$$

3-point: traffic $\\times 1/3$. 2-D 9-point on a $B\\times B$ tile with a 1-cell apron: $(B+2)^2 / (9B^2)\\approx 1/9$.

## Barrier cost

\`__syncthreads()\` is a few cycles when all warps of the block arrive together, and a *warp stall* for early warps when they don't. In the tile loop it is issued twice per $T$ FMAs. For $T=16$ that is one barrier per 8 FMAs — cheap next to HBM, visible if you have already hit the compute roof. Removing the *second* barrier is a data race, not an optimisation.
`,O=`## Exercises

**Q1.** Naive vs $T=16$ tiled GEMM at $N=2048$, FP32. How many bytes are loaded from global memory in each case, and what is the reuse factor?

<details>
<summary>Show answer</summary>

Naive loads $2N^3=2\\cdot2048^3=17.2\\times10^9$ floats $=68.7$ GB. Tiled loads $2N^3/T=17.2\\times10^9/16=1.07\\times10^9$ floats $=4.29$ GB. Reuse $T=16$. (Stores $N^2\\cdot4=16$ MB are noise in both.)

</details>

**Q2.** A thread at \`threadIdx=(3,5)\` in block \`(1,2)\` with \`TILE=16\`, $N=64$. Which element of $C$ does it write, and which $A$ element does it load on k-tile $t=1$?

<details>
<summary>Show answer</summary>

\`row=2·16+5=37\`, \`col=1·16+3=19\` → writes $C_{37,19}$. On $t=1$, \`As[5][3]=A[37,\\ 16+3]=A_{37,19}$. (It still *reads* a whole row of \`As\` during the MAC: $A_{37,16\\ldots31}$.)

</details>

**Q3.** Why are there **two** \`__syncthreads()\` in the k-loop? What goes wrong if you drop the second?

<details>
<summary>Show answer</summary>

The first makes the cooperative load visible before any thread starts the inner product. The second prevents the *next* iteration's load from overwriting \`As\`/\`Bs\` while another warp is still on the inner product. Dropping it is a write-after-read race on shared memory; \`racecheck\` will flag it. It often passes at TILE=8 (few warps) and fails at TILE=16.

</details>

**Q4.** Can you write \`if (row < N && col < N) { load; __syncthreads(); … }\` to skip padded threads?

<details>
<summary>Show answer</summary>

No — that is the deadlock rule. Threads with \`row ≥ N\` would skip the barrier while the others wait forever. Always barrier unconditionally; mask only the load (\`?: 0\`) and the store.

</details>

**Q5.** Shared memory for two FP32 $T\\times T$ tiles at $T=32$? Could an A100 SM (164 KB shared, max 2048 threads, max 32 blocks) be limited by shared memory, threads, or blocks?

<details>
<summary>Show answer</summary>

$S=2\\cdot32^2\\cdot4=8$ KB. Threads/block $=1024$ ⇒ at most $2048/1024=2$ blocks/SM from the thread cap. Shared would allow $164/8=20$ blocks; the block cap is 32. **Limiter = threads.** (A $T=16$ kernel is also thread-limited: 256 thr ⇒ 8 blocks; shared would allow 82.)

</details>

**Q6.** A 5-point 1-D stencil (radius 2) on 256-thread blocks. What fraction of naive global traffic does a halo load achieve?

<details>
<summary>Show answer</summary>

Naive: 5 loads/point. Halo: $256+4=260$ loads for 256 outputs $\\Rightarrow 260/(5\\cdot256)=20.3\\%$ of naive, essentially $1/5$, plus a $4/256=1.6\\%$ overhead.

</details>

**Q7 (code).** Compile \`tiled_gemm.cu\` at \`TILE=16\` and \`TILE=32\` for the same $N$ (start with 1024, then 4096). Which is faster, and does the ratio match the $2\\times$ traffic cut?

<details>
<summary>Show answer</summary>

At 1024 the $T=32$ kernel has only $(1024/32)^2=1024$ blocks — on an A100 (108 SMs) that is a handful of waves, so occupancy/tail effects compete with the traffic win. At 4096 there are 16× more blocks and $T=32$ should approach $2\\times$ faster *if* HBM-bound, less if you are already near the compute roof (Math tab). Always measure both; bigger tiles are not automatically better.

</details>

**Q8 (code).** Run \`syncthreads_race.cu\` as written (32 threads) and then change both launches to \`<<<1,256>>>\` with \`s[256]\` and \`(threadIdx.x+1)%256\`. When does the unsynchronised variant go wrong? Confirm with \`compute-sanitizer --tool racecheck\`.

<details>
<summary>Show answer</summary>

At 32 threads, one warp: the race is often silent. At 256 threads, 8 warps: the first warps to pass the load read neighbours that later warps have not written — garbage / zeros / stale. \`racecheck\` reports a *Hazard* on \`s\` even in the 32-thread case.

</details>

## In practice

- **Start at 16×16 threads, one output per thread.** It is the picture in the lab. Then try 32×32, then (if you need more) register blocking or cuBLAS.
- **Pad the load, never the barrier.**
- **Coalesce on \`threadIdx.x\`:** \`A[row, t+tx]\` and \`B[t+ty, col]\` are the canonical pair. Swapping x/y in the launch (\`dim3(TILE,TILE)\` is \`(x,y)\`) is a classic slowdown.
- **Fuse after you tile.** A tiled GEMM + bias + ReLU in one kernel saves a whole extra read/write of $C$.
- **Prefer the library for plain GEMM**; steal this *pattern* for the kernels they don't ship (attention tiles, custom epilogues, stencils).

## Common pitfalls

- One \`__syncthreads()\` instead of two in the k-loop (race on the next load).
- Barrier inside a divergent \`if\` (deadlock, especially on the last blocks of a non-multiple $N$).
- \`tile[tx][ty]\` instead of \`tile[ty][tx]\` — transposes the cooperative load, usually uncoalesced.
- Assuming shared memory is "like a cache" that fills itself. Nothing lands in \`__shared__\` unless a thread writes it.
- Counting on a 32-thread test to prove the barrier is unnecessary.
- TILE that is not a multiple of 32: wasted warp lanes (Execution Model) *and* messier coalescing.
- Ignoring that naive $B$ is uncoalesced: a "tiled" kernel that still walks $B$ in the inner loop with a strided global index has not actually tiled $B$.
`,k=`// Naive GEMM: C = A B, square N×N, row-major.
// Each thread owns one C[row, col] and streams a whole row of A and column of B from global memory.
// Global traffic: 2 N³ loads + N² stores  →  AI = 1/s  (0.25 FLOP/B in FP32)  — always memory-bound.
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 naive_gemm.cu -o naive && ./naive 1024
#include <cstdio>
#include <cstdlib>
#include <vector>
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

__global__ void gemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];     // A row coalesced; B column is strided (N floats)
    C[row * N + col] = acc;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 1024;
    const size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> hA(N * N), hB(N * N), hC(N * N);
    for (int i = 0; i < N * N; ++i) { hA[i] = 1.f; hB[i] = 1.f; }   // C should be N everywhere

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t t0, t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    gemm_naive<<<grid, block>>>(A, B, C, N);                        // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    gemm_naive<<<grid, block>>>(A, B, C, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
    printf("naive GEMM N=%d: %.3f ms, %.1f GFLOP/s  (C[0]=%.0f, expected %d)\\n",
           N, ms, 2.0 * N * (double)N * N / (ms * 1e6), hC[0], N);

    // Traffic model: 2 N³ loads + N² stores of 4 B. At N=1024 that is 8.6 GB of loads.
    // Peak A100 HBM ~2 TB/s ⇒ lower bound ~4 ms even with perfect coalescing — and B's
    // column reads are *not* coalesced. Tiling (next file) cuts the loads by T and coalesces both.
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
`,A=`// Tiled GEMM: C = A B. Each block owns a TILE×TILE output tile.
// Threads cooperatively stage TILE×TILE slabs of A and B in shared memory, reuse them T times, then
// slide along K. Global traffic drops by ~T; both loads are coalesced (x = threadIdx.x is contiguous).
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 tiled_gemm.cu -o tiled && ./tiled 1024
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
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

#ifndef TILE
#define TILE 16                                 // 16×16 = 256 threads; 2×16×16×4 B = 2 KB shared
#endif

__global__ void gemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    // Number of TILE×TILE slabs along K. Works for any N: out-of-range loads become 0.
    const int ntiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < ntiles; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.f;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();                        // every thread must see the whole tiles before MAC

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];      // As: broadcast along the warp; Bs: consecutive banks

        __syncthreads();                        // don't let the next load overwrite tiles still in use
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 1024;
    const size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> hA(N * N), hB(N * N), hC(N * N);
    for (int i = 0; i < N * N; ++i) { hA[i] = ((i * 17) % 11) * 0.1f; hB[i] = ((i * 13) % 7) * 0.1f; }

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t t0, t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    gemm_tiled<<<grid, block>>>(A, B, C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    gemm_tiled<<<grid, block>>>(A, B, C, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));

    // Spot-check a few C[i,j] against a CPU inner product (not the whole N³ — too slow at 1024).
    int bad = 0;
    for (int sample = 0; sample < 8; ++sample) {
        int i = (sample * 97) % N, j = (sample * 53) % N;
        double gold = 0;
        for (int k = 0; k < N; ++k) gold += (double)hA[i * N + k] * hB[k * N + j];
        if (fabs(hC[i * N + j] - gold) > 1e-2 * (1 + fabs(gold))) ++bad;
    }
    const double gflops = 2.0 * N * (double)N * N / (ms * 1e6);
    const double loads_gb = 2.0 * N * (double)N * N / TILE * 4 / 1e9;     // model: 2 N³/T floats
    printf("tiled GEMM  TILE=%d  N=%d: %.3f ms, %.1f GFLOP/s\\n", TILE, N, ms, gflops);
    printf("  model traffic %.2f GB  →  %.0f GB/s effective  (spot-check %s)\\n",
           loads_gb, loads_gb / (ms * 1e-3), bad ? "MISMATCH" : "ok");

    // Try TILE=32: nvcc -DTILE=32 ...  (1024 threads/block, 8 KB shared). Bigger tiles ⇒ more reuse
    // but fewer resident blocks (Occupancy chapter). Bank conflicts: see the next chapter.
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
`,j=`// 1-D 3-point stencil with a *halo* in shared memory.
// Each block owns a contiguous chunk of the output. Interior points need their neighbours, so the
// block also loads one extra element on each side (the halo / ghost cells) into shared memory.
// Without the halo every thread would re-read its neighbours from global memory — 3× the traffic.
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 stencil1d.cu -o stencil && ./stencil
#include <cstdio>
#include <cstdlib>
#include <vector>
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

constexpr int BLK = 256;

__global__ void stencil_smem(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float s[];                // BLK + 2 floats, sized at launch
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = threadIdx.x + 1;              // interior slots are s[1 .. BLK]

    s[t] = (g < n) ? in[g] : 0.f;
    if (threadIdx.x == 0)     s[0]       = (g > 0)     ? in[g - 1] : 0.f;          // left halo
    if (threadIdx.x == BLK-1) s[BLK + 1] = (g + 1 < n) ? in[g + 1] : 0.f;          // right halo
    __syncthreads();

    if (g < n) out[g] = 0.25f * s[t - 1] + 0.5f * s[t] + 0.25f * s[t + 1];
}

__global__ void stencil_naive(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    const float l = (g > 0)     ? in[g - 1] : 0.f;
    const float c = in[g];
    const float r = (g + 1 < n) ? in[g + 1] : 0.f;
    out[g] = 0.25f * l + 0.5f * c + 0.25f * r;
}

int main() {
    const int n = 1 << 24;                      // 16 M elements
    const size_t bytes = n * sizeof(float);
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = (float)i;

    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemcpy(in, h.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = (n + BLK - 1) / BLK;
    cudaEvent_t t0, t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    auto time = [&](auto launch) {
        launch(); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0)); launch(); CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        CUDA_CHECK(cudaGetLastError());
        return ms;
    };

    float ms_n = time([&] { stencil_naive<<<blocks, BLK>>>(in, out, n); });
    float ms_s = time([&] { stencil_smem <<<blocks, BLK, (BLK + 2) * sizeof(float)>>>(in, out, n); });

    // Naive: ~3 n reads + n writes.  Smem: ~n reads (halo is 2/BLK extra) + n writes.
    const double naive_gb = 4.0 * n * 4 / 1e9, smem_gb = 2.0 * n * 4 / 1e9;
    printf("naive  %.3f ms  model %.2f GB  →  %.0f GB/s\\n", ms_n, naive_gb, naive_gb / (ms_n * 1e-3));
    printf("smem   %.3f ms  model %.2f GB  →  %.0f GB/s   (halo = 2 extra loads / block)\\n",
           ms_s, smem_gb, smem_gb / (ms_s * 1e-3));
    cudaFree(in); cudaFree(out);
    return 0;
}
`,M=`// What __syncthreads() is for: a cooperative load, then a neighbour read.
// Variant 0 skips the barrier — some threads compute with a neighbour that has not yet written.
// Variant 1 inserts the barrier and the result is deterministic.
//
// build: nvcc -O3 -arch=sm_80 syncthreads_race.cu -o race && ./race
#include <cstdio>
#include <cuda_runtime.h>

__global__ void with_race(int* out) {
    __shared__ int s[32];
    s[threadIdx.x] = threadIdx.x;               // each thread publishes its id
    // MISSING __syncthreads();
    out[threadIdx.x] = s[(threadIdx.x + 1) & 31];   // read the neighbour — may be stale
}

__global__ void with_sync(int* out) {
    __shared__ int s[32];
    s[threadIdx.x] = threadIdx.x;
    __syncthreads();                            // all 32 writes are visible
    out[threadIdx.x] = s[(threadIdx.x + 1) & 31];
}

int main() {
    int *d0, *d1, h0[32], h1[32];
    cudaMalloc(&d0, 32 * sizeof(int));
    cudaMalloc(&d1, 32 * sizeof(int));

    with_race<<<1, 32>>>(d0);
    with_sync<<<1, 32>>>(d1);
    cudaDeviceSynchronize();
    cudaMemcpy(h0, d0, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h1, d1, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("with    sync: ");
    for (int i = 0; i < 32; ++i) printf("%2d ", h1[i]);   // always 1,2,...,31,0
    printf("\\nwithout sync: ");
    for (int i = 0; i < 32; ++i) printf("%2d ", h0[i]);   // often looks right (one warp!), but is a data race
    printf("\\n");
    printf("On a *single warp* the race is often invisible because the 32 lanes already run in lockstep.\\n");
    printf("Launch the same pattern with 256 threads (8 warps) and the missing barrier bites: some warps\\n");
    printf("read s[] before other warps have written.  compute-sanitizer --tool racecheck ./race flags it.\\n");
    cudaFree(d0); cudaFree(d1);
    return 0;
}
`,N=a(),P=new n,F={a100:{name:`A100`,pi:19.5,bw:2.039},h100:{name:`H100`,pi:67,bw:3.35},rtx4090:{name:`RTX 4090`,pi:82.6,bw:1.008}};function I(e,t,n){let r=t/n,i=r,a=1+n,o=i*a+1,s=r*r*o,c=(e%s+s)%s,l=Math.floor(c/o),u=c%o,d=l%r,f=Math.floor(l/r);if(u===o-1)return{bx:d,by:f,t:i-1,phase:`write`,k:n-1,max:s,s:c};let p=Math.floor(u/a),m=u%a;return m===0?{bx:d,by:f,t:p,phase:`load`,k:-1,max:s,s:c}:{bx:d,by:f,t:p,phase:`mac`,k:m-1,max:s,s:c}}function L({N:e,s:t,origin:n,colorFor:r,hover:i,onHover:a,z:o=0}){let s=(0,T.useRef)();return(0,T.useLayoutEffect)(()=>{let a=s.current;if(a){a.count=e*e;for(let s=0;s<e*e;s++){let c=s%e,l=Math.floor(s/e);P.position.set(n[0]+(c-(e-1)/2)*t,n[1]-(l-(e-1)/2)*t,n[2]+o);let u=i&&i[0]===l&&i[1]===c;P.scale.setScalar(u?1.22:1),P.updateMatrix(),a.setMatrixAt(s,P.matrix),a.setColorAt(s,r(l,c,u))}a.instanceMatrix.needsUpdate=!0,a.instanceColor&&(a.instanceColor.needsUpdate=!0)}}),(0,N.jsxs)(`instancedMesh`,{ref:s,args:[null,null,e*e],onPointerMove:t=>{t.stopPropagation();let n=t.instanceId;a&&a([Math.floor(n/e),n%e])},onPointerOut:()=>a&&a(null),children:[(0,N.jsx)(`boxGeometry`,{args:[t*.86,t*.86,t*.34]}),(0,N.jsx)(`meshStandardMaterial`,{roughness:.45,metalness:.08})]})}function R({T:e,s:t,origin:n,fill:i,glow:a}){let o=(0,T.useRef)();return(0,T.useLayoutEffect)(()=>{let s=o.current;if(!s)return;s.count=e*e;let c=new r(a?`#fbbf24`:`#3d4568`);for(let o=0;o<e*e;o++){let l=o%e,u=Math.floor(o/e);P.position.set(n[0]+(l-(e-1)/2)*t,n[1]-(u-(e-1)/2)*t,n[2]),P.scale.setScalar(i[u][l]?1:.55),P.updateMatrix(),s.setMatrixAt(o,P.matrix);let d=i[u][l]?a?c:new r(`#fbbf24`):new r(`#1a1e32`);s.setColorAt(o,d)}s.instanceMatrix.needsUpdate=!0,s.instanceColor&&(s.instanceColor.needsUpdate=!0)}),(0,N.jsxs)(`instancedMesh`,{ref:o,args:[null,null,64],children:[(0,N.jsx)(`boxGeometry`,{args:[t*.86,t*.86,t*.4]}),(0,N.jsx)(`meshStandardMaterial`,{roughness:.4,emissive:a?`#fbbf24`:`#000`,emissiveIntensity:a?.35:0})]})}function z({w:e,h:t,d:n,color:r}){return(0,N.jsxs)(`mesh`,{children:[(0,N.jsx)(`boxGeometry`,{args:[e,t,n]}),(0,N.jsx)(`meshStandardMaterial`,{color:r,transparent:!0,opacity:.07,depthWrite:!1}),(0,N.jsx)(i,{color:r})]})}function B(){let[e,t]=(0,T.useState)(4),[n,i]=(0,T.useState)(0),[a,s]=(0,T.useState)(!0),[c,l]=(0,T.useState)(null),{bx:u,by:d,t:f,phase:p,k:g}=I(n,8,e),v=d*e,C=u*e,E=f*e;y(a,420,()=>i(e=>e+1));let D=(t,n)=>{let r=Math.floor(t/e),i=Math.floor(n/e);return r<d||r===d&&i<u?8:r!==d||i!==u?0:p===`write`?8:f*e+(p===`mac`?g+1:0)},O=(t,n,r,i)=>t>=r&&t<r+e&&n>=i&&n<i+e,k=(e,t)=>{let n=O(e,t,v,E),i=p===`mac`&&c&&e===c[0]&&t===E+g;return new r(i?`#ffffff`:n?p===`load`?`#c4b5fd`:`#8b7bff`:`#2a3152`)},A=(e,t)=>{let n=O(e,t,E,C),i=p===`mac`&&c&&t===c[1]&&e===E+g;return new r(i?`#ffffff`:n?p===`load`?`#67e8f9`:`#22d3ee`:`#2a3152`)},j=(e,t,n)=>{let i=D(e,t);return n?new r(`#ffffff`):O(e,t,v,C)&&p!==`write`?new r().setHSL(.33,.7,.25+.4*i/8):i>=8?new r(`#3f6b28`):new r(`#1c2238`)},M=Array.from({length:e},()=>Array.from({length:e},()=>!0)),P=p===`load`||p===`mac`,F=.32,B=.3,[V,H]=c&&O(c[0],c[1],v,C)?c:[v,C],U=H-C,W=V-v,G=(0,N.jsxs)(N.Fragment,{children:[`Block `,(0,N.jsxs)(`b`,{children:[`(`,u,`,`,d,`)`]}),` owns C[`,v,`:`,v+e,`, `,C,`:`,C+e,`] · k-tile `,(0,N.jsx)(`b`,{children:f}),`/`,8/e,(0,N.jsx)(`br`,{}),p===`load`&&(0,N.jsxs)(N.Fragment,{children:[`Cooperative load → `,(0,N.jsx)(`span`,{style:{color:`#c4b5fd`},children:`As`}),` = A[`,v,`:`,v+e,`, `,E,`:`,E+e,`] · `,(0,N.jsx)(`span`,{style:{color:`#67e8f9`},children:`Bs`}),` = B[`,E,`:`,E+e,`, `,C,`:`,C+e,`]`]}),p===`mac`&&(0,N.jsxs)(N.Fragment,{children:[`MAC k=`,(0,N.jsx)(`b`,{children:g}),`: C[`,V,`,`,H,`] += As[`,W,`][`,g,`] × Bs[`,g,`][`,U,`]  (acc = `,(0,N.jsx)(`b`,{children:D(V,H)}),`/`,8,`)`]}),p===`write`&&(0,N.jsxs)(N.Fragment,{children:[`Write the C-tile (all-ones A,B → each entry is N = `,(0,N.jsx)(`b`,{children:8}),`)`]})]});return(0,N.jsxs)(N.Fragment,{children:[(0,N.jsxs)(m,{height:520,camera:[0,3.4,15.5],target:[0,.2,0],fov:42,overlay:G,hint:`drag to orbit · hover a C cell · play / step the k-tiles`,children:[(0,N.jsxs)(`group`,{position:[-5.4,1.15,0],children:[(0,N.jsx)(z,{w:2.74,h:2.74,d:.16,color:`#8b7bff`}),(0,N.jsx)(L,{N:8,s:F,origin:[0,0,0],colorFor:k})]}),(0,N.jsxs)(`group`,{position:[5.4,1.15,0],children:[(0,N.jsx)(z,{w:2.74,h:2.74,d:.16,color:`#22d3ee`}),(0,N.jsx)(L,{N:8,s:F,origin:[0,0,0],colorFor:A})]}),(0,N.jsxs)(`group`,{position:[0,-2.15,0],children:[(0,N.jsx)(z,{w:2.74,h:2.74,d:.18,color:`#76d12a`}),(0,N.jsx)(L,{N:8,s:F,origin:[0,0,0],colorFor:j,hover:c,onHover:l,z:.05})]}),P&&(0,N.jsxs)(N.Fragment,{children:[(0,N.jsx)(`group`,{position:[-1.15,3.15,.35],children:(0,N.jsx)(R,{T:e,s:B,origin:[0,0,0],fill:M,glow:p===`load`})}),(0,N.jsx)(`group`,{position:[1.15,3.15,.35],children:(0,N.jsx)(R,{T:e,s:B,origin:[0,0,0],fill:M,glow:p===`load`})})]})]}),(0,N.jsx)(h,{items:[[`#8b7bff`,`A (left)`],[`#22d3ee`,`B (right)`],[`#76d12a`,`C (centre)`],[`#fbbf24`,`shared As / Bs`]]}),(0,N.jsxs)(_,{children:[(0,N.jsx)(S,{label:`TILE`,value:String(e),onChange:e=>{t(+e),i(0)},options:[[`2`,`2 × 2`],[`4`,`4 × 4`]]}),(0,N.jsx)(w,{label:`Auto-step`,value:a,onChange:s}),(0,N.jsx)(x,{onClick:()=>i(e=>e-1),children:`←`}),(0,N.jsx)(x,{primary:!0,onClick:()=>i(e=>e+1),children:`Step →`}),(0,N.jsx)(x,{onClick:()=>i(0),children:`Reset`})]}),(0,N.jsxs)(b,{children:[o(8),`×`,o(8),` all-ones GEMM, TILE=`,(0,N.jsx)(`b`,{children:e}),` → `,8/e,`×`,8/e,` blocks, `,8/e,` k-tiles each. Thread (`,(0,N.jsx)(`b`,{children:U}),`,`,(0,N.jsx)(`b`,{children:W}),`) writes C[`,(0,N.jsx)(`b`,{children:V}),`,`,(0,N.jsx)(`b`,{children:H}),`]; over the whole K loop it does `,(0,N.jsx)(`b`,{children:8}),` FMAs and only `,(0,N.jsx)(`b`,{children:8/e*2}),` global loads (one A, one B per k-tile). Naive would issue `,(0,N.jsx)(`b`,{children:16}),` global loads for the same output — `,(0,N.jsxs)(`b`,{className:`g`,children:[e,`×`]}),` more.`,p===`load`&&(0,N.jsxs)(N.Fragment,{children:[` Right now every thread of the block writes `,(0,N.jsx)(`b`,{children:`one`}),` element of As and of Bs, then the (unseen) `,(0,N.jsx)(`b`,{children:`__syncthreads`}),`.`]}),p===`mac`&&(0,N.jsxs)(N.Fragment,{children:[` Inner product step `,g+1,`/`,e,` uses only shared memory. After `,e,` steps, a second barrier, then the next k-tile.`]})]})]})}function V(){let[e,t]=(0,T.useState)(10),[n,r]=(0,T.useState)(4),[i,a]=(0,T.useState)(4),[m,v]=(0,T.useState)(`a100`),y=2**e,x=2**n,w=F[m],E=2*y*y*y,D=(2*y*y*y+y*y)*i,O=(2*y*y*y/x+y*y)*i,k=3*y*y*i,A=E/O,j=E/D,M=E/k,P=w.pi*0xe8d4a51000/(w.bw*0xe8d4a51000),I=D/(w.bw*0xe8d4a51000),L=Math.max(E/(w.pi*0xe8d4a51000),O/(w.bw*0xe8d4a51000)),R=Math.max(E/(w.pi*0xe8d4a51000),k/(w.bw*0xe8d4a51000)),z=E/(w.pi*0xe8d4a51000)>O/(w.bw*0xe8d4a51000)?`compute`:`memory`,[B]=C(280,(e,t,n)=>{let r=t-58-24,a=n-28-36,s=Math.log10(k*.45),l=Math.log10(D),m=e=>58+(Math.log10(e)-s)/(l-s)*r,h=[[`naive`,D,f.r],[`tiled T=${x}`,O,f.d],[`ideal (A,B once)`,k,f.e]];p(e,`global traffic for N=${o(y)}  FP${8*i}`,58,12,{size:12,color:f.mute,weight:600}),h.forEach(([n,i,o],s)=>{let l=34+a/3*s,h=a/3-16,g=Math.max(4,Math.min(r-4,m(i)-58));e.fillStyle=u(o,.9),c(e,58,l,g,h,6),e.fill(),p(e,n,58,l-7,{size:11,color:o,weight:600});let _=d(i),v=g>90&&58+g+8+_.length*7.2>t-12;p(e,_,v?58+g-10:58+g+8,l+h/2,{size:12,color:v?`#0a0c14`:f.ink,mono:!0,weight:700,align:v?`right`:`left`})}),p(e,`log scale · reuse vs naive: ${x}×`,58,n-12,{size:12,color:f.d,mono:!0})}),[V]=C(220,(e,t,n)=>{let r=t-48-16,i=n-18-32,a=e=>48+(Math.log2(e)- -3)/11*r,o=e=>18+i-(Math.log10(e)- -1)/3.2*i,c=w.bw,u=Math.max(w.pi,1);e.strokeStyle=f.grid,e.lineWidth=1,e.strokeRect(48,18,r,i),e.save(),e.beginPath(),e.rect(48,18,r,i),e.clip(),s(e,[[a(2**-3),o(c*2**-3)],[a(u/c),o(u)]],f.a,2.4),s(e,[[a(w.pi/c),o(w.pi)],[48+r,o(w.pi)]],f.g,2.4),[[j,Math.min(w.pi,c*j),f.r,`naive`],[A,Math.min(w.pi,c*A),f.d,`tiled`],[M,Math.min(w.pi,c*M),f.e,`ideal`]].forEach(([t,n,r])=>{e.fillStyle=r,e.beginPath(),e.arc(a(l(t,2**-3,256)),o(n),7,0,7),e.fill()}),e.restore(),p(e,`ridge ${P.toFixed(1)} FLOP/B`,a(l(P,2**-3,256))+6,18+i-8,{size:10.5,color:f.g,mono:!0}),p(e,`I (FLOP/B) →`,48+r/2,n-8,{align:`center`,size:11,color:f.mute})});return(0,N.jsxs)(N.Fragment,{children:[(0,N.jsx)(`canvas`,{...B}),(0,N.jsx)(`canvas`,{...V}),(0,N.jsx)(h,{items:[[f.r,`naive`],[f.d,`tiled`],[f.e,`ideal`]]}),(0,N.jsxs)(_,{children:[(0,N.jsx)(S,{label:`GPU`,value:m,onChange:v,options:Object.entries(F).map(([e,t])=>[e,t.name])}),(0,N.jsx)(S,{label:`dtype`,value:String(i),onChange:e=>a(+e),options:[[`8`,`FP64`],[`4`,`FP32`],[`2`,`FP16`]]}),(0,N.jsx)(g,{label:`N`,min:6,max:14,value:e,onChange:t,fmt:e=>o(2**e)}),(0,N.jsx)(g,{label:`TILE`,min:2,max:6,value:n,onChange:r,fmt:e=>2**e})]}),(0,N.jsxs)(b,{children:[`W = 2N³ = `,(0,N.jsxs)(`b`,{children:[(E/1e9).toFixed(2),` GFLOP`]}),`. Tiled I ≈ T/s = `,(0,N.jsx)(`b`,{children:A.toFixed(2)}),` FLOP/B (naive `,j.toFixed(2),`, ideal `,M.toFixed(1),`) vs `,w.name,` ridge `,(0,N.jsx)(`b`,{children:P.toFixed(1)}),` → tiled is `,(0,N.jsxs)(`b`,{className:z===`memory`?`w`:`g`,children:[z,`-bound`]}),`. Lower bound: naive `,(I*1e3).toFixed(2),` ms · tiled `,(L*1e3).toFixed(2),` ms · ideal `,(R*1e3).toFixed(2),` ms. Shared footprint 2·T²·`,i,` = `,(0,N.jsx)(`b`,{children:d(2*x*x*i)}),`.`,x*x>1024&&(0,N.jsx)(`span`,{className:`r`,children:` TILE² > 1024: not a legal block.`})]})]})}function H({sync:e,play:t,nonce:n}){let r=(0,T.useRef)(0);(0,T.useLayoutEffect)(()=>{r.current=0},[e,n]),y(t,40,()=>{r.current+=.04});let[i]=C(300,(t,n,i)=>{let a=r.current%6,o=Math.min(52,(n-70-24)/8);p(t,`shared s[0..7]  —  each thread writes s[i]=i, then reads s[(i+1) mod 8]`,70,16,{size:11.5,color:f.mute});for(let n=0;n<8;n++){let r=70+n*o,i=n>=4,s=e?a>1.6:i?a>2.4:a>.8,l=a>(e?2.2:i?2.8:1.2)&&a<4.8;if(t.fillStyle=s?i?f.c:f.a:`#1a1e32`,c(t,r+4,36,o-8,36,6),t.fill(),p(t,s?String(n):`?`,r+o/2,54,{align:`center`,size:14,color:s?`#0a0c14`:f.dim,mono:!0,weight:700}),p(t,`t`+n,r+o/2,86,{align:`center`,size:10.5,color:i?f.c:f.a,mono:!0}),l){let i=(n+1)%8;t.strokeStyle=(e?a>1.6:i>=4?a>2.4:a>.8)?f.e:f.r,t.lineWidth=1.6,t.beginPath(),t.moveTo(70+i*o+o/2,72),t.lineTo(r+o/2,100),t.stroke()}}p(t,`warp 0`,70,112,{size:11,color:f.a,weight:600}),p(t,`warp 1  (arrives later)`,70+4*o,112,{size:11,color:f.c,weight:600}),p(t,`out[i] = s[i+1]`,62,170,{size:11,color:f.mute,align:`right`});for(let n=0;n<8;n++){let r=(n+1)%8,i=e||r<4,s=a>3.2;t.fillStyle=s?u(i?f.e:f.r,.85):`#1a1e32`,c(t,70+n*o+4,154,o-8,32,6),t.fill(),s&&p(t,i?String(r):`??`,70+n*o+o/2,170,{align:`center`,size:13,color:`#0a0c14`,mono:!0,weight:700})}p(t,e?`Barrier: every write is visible. out = 1,2,3,4,5,6,7,0.`:`No barrier: warp 0 reads s[4] before warp 1 writes it. out[3] is garbage — a data race, not a “sometimes works”.`,70,i-18,{size:12,color:e?f.e:f.r,weight:600})},{animate:!0});return(0,N.jsx)(`canvas`,{...i})}function U({play:e,nonce:t}){let n=(0,T.useRef)(0);(0,T.useLayoutEffect)(()=>{n.current=0},[t]),y(e,40,()=>{n.current+=.04});let[r]=C(300,(e,t,r)=>{let i=Math.min(2,Math.floor(n.current/3%3)),a=n.current%3<1.35?`load`:`compute`,o=(t-36-20)/24;p(e,`3-point stencil · block ${i} of 3 · ${a===`load`?`cooperative load + halo`:`compute from shared memory`}`,36,18,{size:12,color:f.mute,weight:600});for(let t=0;t<24;t++){let n=Math.floor(t/8),r=(t===i*8-1||t===i*8+8)&&t>=0&&t<24,s=n===i;e.fillStyle=s?f.d:r&&a===`load`?f.c:`#1a1e32`,c(e,36+t*o+1,78,o-2,34,4),e.fill(),p(e,String(t),36+t*o+o/2,95,{align:`center`,size:10,color:s||r?`#0a0c14`:f.dim,mono:!0})}p(e,`global in[]`,36,66,{size:11,color:f.mute});let s=36+i*8*o;p(e,`shared s[0..9]  (8 interior + 2 halo)`,s,134,{size:11,color:f.mute});for(let t=0;t<10;t++){let n=i*8-1+t;e.fillStyle=t===0||t===9?f.c:f.d,c(e,s+(t-.5)*o+1,148,o-2,34,4),e.fill(),p(e,n<0||n>=24?`0`:String(n),s+(t-.5)*o+o/2,165,{align:`center`,size:10,color:`#0a0c14`,mono:!0})}a===`compute`?(s+4*o,e.strokeStyle=f.e,e.lineWidth=1.5,e.beginPath(),e.moveTo(s+3.5*o,182),e.lineTo(s+3.5*o,218),e.stroke(),p(e,`y[i] = (s[i-1] + 2 s[i] + s[i+1]) / 4`,s,230,{size:12,color:f.e,mono:!0})):p(e,`thread 0 also loads in[start-1] · thread 7 loads in[end]  — 2 extra global reads for the whole block`,s,204,{size:11,color:f.c}),p(e,`traffic: naive 3 loads/point · tiled ${10/8} loads/point  (${(10/24*100).toFixed(0)}% of naive)`,36,r-16,{size:12,color:f.d,mono:!0})},{animate:!0});return(0,N.jsx)(`canvas`,{...r})}function W(){let[e,t]=(0,T.useState)(`race`),[n,r]=(0,T.useState)(!1),[i,a]=(0,T.useState)(!0),[o,s]=(0,T.useState)(0);return(0,N.jsxs)(N.Fragment,{children:[e===`race`?(0,N.jsx)(H,{sync:n,play:i,nonce:o}):(0,N.jsx)(U,{play:i,nonce:o}),(0,N.jsxs)(_,{children:[(0,N.jsx)(S,{label:`Demo`,value:e,onChange:t,options:[[`race`,`__syncthreads race`],[`halo`,`Stencil halo`]]}),e===`race`&&(0,N.jsx)(w,{label:`Insert __syncthreads()`,value:n,onChange:r}),(0,N.jsx)(w,{label:`Animate`,value:i,onChange:a}),(0,N.jsx)(x,{onClick:()=>s(e=>e+1),children:`Replay`})]}),(0,N.jsx)(b,{children:e===`race`?(0,N.jsxs)(N.Fragment,{children:[`Two warps share `,(0,N.jsx)(`b`,{children:`s[8]`}),`. Warp 1 is scheduled later. Without a barrier, warp 0's read of `,(0,N.jsx)(`b`,{children:`s[4]`}),` is a data race — it may be empty, stale, or (on a lucky run) already written. A single-warp test hides this; `,(0,N.jsx)(`b`,{children:`compute-sanitizer --tool racecheck`}),` does not. The same bug in tiled GEMM is dropping the `,(0,N.jsx)(`b`,{children:`second`}),` barrier in the k-loop.`]}):(0,N.jsxs)(N.Fragment,{children:[`A radius-1 stencil needs neighbours. The block loads its `,(0,N.jsx)(`b`,{children:8}),` interior points plus a 1-cell `,(0,N.jsx)(`b`,{children:`halo`}),` (pink) = 10 shared slots, then every output is 3 shared reads. Halo overhead is 2/`,8,` = 25% here (tiny blocks for the picture); at 256 threads it is 0.8%. Same pattern as the GEMM tile, just 1-D.`]})})]})}function G(){return(0,N.jsx)(v,{views:[{id:`g`,label:`3D: tiled GEMM, step by step`,render:()=>(0,N.jsx)(B,{})},{id:`t`,label:`Traffic, reuse & the roofline`,render:()=>(0,N.jsx)(V,{})},{id:`b`,label:`__syncthreads & stencil halo`,render:()=>(0,N.jsx)(W,{})}]})}var K={Lab:G,vizTitle:`Stage a tile in shared memory, reuse it, slide along K — watch global traffic fall by T`,tryIt:[`In the 3D tab **hover a C cell** in the active tile during a MAC: the matching A-element and B-element light up white. That pair lives in **As / Bs**, not HBM.`,`Set **TILE = 2** and step: four k-tiles instead of two, twice the global loads. TILE is the reuse factor.`,`Let it run through a **write** phase: the C-tile goes solid green (value N) and the next block starts.`,`In Traffic, set **N = 1024, TILE = 16, FP32, A100**: tiled sits at 4 FLOP/B, still memory-bound but 16× less traffic than naive.`,`Raise TILE to 32, then switch the GPU to **H100 FP16** in your head (ridge ~300): you now understand why FlashAttention tiles.`,"In the race demo, leave **__syncthreads off** and watch `out[3]` go red; turn it on and the vector is a clean rotation."],theory:E,math:D,practice:O,code:[{title:`Naive GEMM: one output per thread, 2N global loads each`,lang:`cuda`,note:`B's column is uncoalesced. This is the kernel the roofline in the previous chapter plotted as "naive GEMM".`,src:k},{title:`Tiled GEMM: cooperative load, two __syncthreads, T× less traffic`,lang:`cuda`,note:`Build with -DTILE=32 to try a bigger tile. Spot-checks a few C[i,j] against a CPU inner product.`,src:A},{title:`1-D stencil with a halo in dynamic shared memory`,lang:`cuda`,src:j},{title:`The missing-barrier data race (and why 32 threads hide it)`,lang:`cuda`,src:M}],quiz:[{q:`Tiling a GEMM with TILE $=T$ changes global-memory traffic by about:`,options:[`×T (more traffic)`,`÷T`,`÷T²`,`no change; only latency changes`],answer:1,why:`Each loaded element is reused $T$ times inside the block, so $Q\\approx 2N^3 s/T$.`},{q:"Why does the k-loop of tiled GEMM contain **two** `__syncthreads()`?",options:[`The compiler requires pairs of barriers`,`One after the load so the tile is visible; one after the MAC so the next load cannot overwrite it`,`One for A and one for B`,`To synchronise with other blocks`],answer:1,why:"The second barrier is a write-after-read fence on `As`/`Bs`. Dropping it is a shared-memory race, often silent at small TILE."},{q:`Which of these is legal?`,options:["`if (row < N) { load; __syncthreads(); }`","`As[ty][tx] = row<N && col<N ? A[...] : 0; __syncthreads();`",`Skipping the barrier on the last k-tile`,`Synchronising two blocks through shared memory`],answer:1,why:`Every thread of the block must reach the same barrier. Mask the **memory op**, never the barrier. Shared memory does not exist across blocks.`},{q:"Thread `(tx,ty)` in block `(bx,by)` with TILE $T$ owns which output?",options:[`$C[tx,ty]$`,`$C[by\\cdot T+ty,\\ bx\\cdot T+tx]$`,`$C[bx\\cdot T+tx,\\ by\\cdot T+ty]$`,`$C[ty, tx]$`],answer:1,why:"`row = blockIdx.y * T + threadIdx.y`, `col = blockIdx.x * T + threadIdx.x`. $x$ is the contiguous dimension so the B-load is coalesced."},{q:`A 3-point stencil on 256-thread blocks with a 1-cell halo loads how many global elements per output, versus 3 naive?`,options:[`3`,`≈1.008`,`256`,`2`],answer:1,why:`$(256+2)/256=1.008$ loads/output. The halo is 2 extra loads for the whole block.`},{q:`At TILE $=16$, FP32, the tiled-GEMM arithmetic intensity is about:`,options:[`0.25 FLOP/B`,`4 FLOP/B`,`16 FLOP/B`,`256 FLOP/B`],answer:1,why:`$I\\approx T/s=16/4=4$ FLOP/B. Still below the A100 FP32 ridge (~9.6), which is why bigger tiles or register blocking still pay off.`},{q:"A missing `__syncthreads` after a cooperative load is often invisible when tested with 32 threads because:",options:[`Barriers are optional inside a block`,`A warp already executes in lockstep, so the 32 writes happen together; a second warp exposes the race`,`Shared memory is coherent globally`,`The compiler inserts a barrier`],answer:1,why:"Correctness requires a block-wide fence. `racecheck` reports the hazard even at 32 threads."}]};export{K as default};