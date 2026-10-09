import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{g as r,n as i,s as a,t as o,y as s}from"./viz-CPys2405.js";import{t as c}from"./Stage3D-CWOT2b2l.js";import{a as l,c as u,i as d,l as f,o as p,r as m,s as h,t as g}from"./hooks-Dw7oo1m3.js";import{n as _,t as v}from"./specs-CrKDU_SL.js";var y=e(t(),1),b={a100:{maxWarps:64,maxBlocks:32,regs:65536,smem:167936,schedulers:4},h100:{maxWarps:64,maxBlocks:32,regs:65536,smem:233472,schedulers:4},rtx4090:{maxWarps:48,maxBlocks:24,regs:65536,smem:102400,schedulers:4}},x=256,S=4,C=128,w=1024,T=(e,t)=>Math.ceil(e/t)*t;function E({threads:e,regs:t,smem:n,sm:r}){let i=Math.ceil(e/32),a=T(t,8),o=T(t*32,x),s=Math.floor(r.regs/o/S)*S,c=r.smem-w,l=n>0?T(n,C):0,u=n>0?l+w:0,d=e>1024||e<1||t>255||n>c,f=[{id:`regs`,name:`registers`,blocks:Math.floor(s/i)},{id:`smem`,name:`shared memory`,blocks:u===0?1/0:Math.floor(r.smem/u)},{id:`warps`,name:`warp slots`,blocks:Math.floor(r.maxWarps/i)},{id:`blocks`,name:`block slots`,blocks:r.maxBlocks}];if(d||f.some(e=>e.blocks<1))return{fit:!1,reason:e>1024?`A block cannot hold more than 1024 threads.`:t>255?`The compiler cannot give a thread more than 255 registers.`:n>c?`Shared memory exceeds the per-block maximum (the SM pool minus the 1 KB reserve).`:`One block already needs more registers or warps than the SM has.`,warpsPerBlock:i,regsAlloc:a,regsPerWarp:o,smemCharged:u,perBlockMax:c,limits:f,activeBlocks:0,activeWarps:0,occ:0,limiters:[]};let p=Math.min(...f.map(e=>e.blocks)),m=f.filter(e=>e.blocks===p),h=p*i;return{fit:!0,reason:``,warpsPerBlock:i,regsAlloc:a,regsPerWarp:o,smemCharged:u,perBlockMax:c,limits:f,activeBlocks:p,activeWarps:h,occ:h/r.maxWarps,limiters:m}}function D(e,t,n){let r=Math.min(1,t/e);return Math.ceil(n/r)}function O(e,t,n,r){return e*1e9/r*(t/(n*1e9))}var k=`## Resident warps are the latency budget

An SM does not context-switch the way a CPU does. Every resident warp keeps its registers, so switching to a ready warp is free — and a warp that is not resident cannot be switched to at all. **Occupancy** is how full that set is:

$$
\\text{occupancy} = \\frac{\\text{resident warps per SM}}{\\text{maximum warps per SM}}.
$$

The maximum is **64** on A100 and H100 (2048 threads) and **48** on the RTX 4090 (1536 threads). Four resources can stop you earlier. The lab takes the minimum:

| Resource | What one block consumes | A100 / H100 ceiling | RTX 4090 |
|---|---|---|---|
| Registers | threads × registers, rounded up | 65 536 × 32-bit | same |
| Shared memory | your allocation + 1 KB reserve | 164 KB / 228 KB | 100 KB |
| Warp slots | \`ceil(threads / 32)\` warps | 64 | 48 |
| Block slots | one | 32 | 24 |

A block of 1024 threads is 32 warps, so an A100 holds **two** of them even when registers and shared memory would allow more. A block of 32 threads is one warp, so the 32-block cap is what stops you, not the 64 warp slots: half the warp slots stay empty.

<div class="callout">

**Full occupancy is not the goal.** It is the budget of ready warps you can spend on stalls. A dependent FMA is ~4 cycles; sixteen resident warps already keep an A100's four schedulers busy. A dependent trip to HBM is a few hundred cycles, and 64 warps do not cover it. Past the stall you actually have, extra warps only help if they don't push the compiler into local memory.

</div>

## How the register file is actually handed out

The compiler's register count is rounded **up to a multiple of 8** per thread (256 registers per warp). The number of warps that fit in the 64K register file is then rounded **down to a multiple of 4**.

So 32 registers/thread is exact: 1024 registers/warp, 64 warps, 100 % on an A100 if nothing else binds. **37 becomes 40.** That is 1280 registers/warp, only 48 warps, **75 %**. The source did not ask for 25 % fewer warps; the allocation unit did.

\`__launch_bounds__(maxThreads, minBlocks)\` tells the compiler to stay inside a register budget that allows \`minBlocks\` resident blocks, spilling to local memory if it has to. Spills bring the occupancy back and add a round trip through the same path as global memory. \`ptxas\` prints both numbers: registers per thread, and spill stores/loads. Trust the spill count over the occupancy percentage.

## Shared memory and the 1 KB you didn't allocate

CUDA reserves **1 KB of shared memory per block**. That is why the per-block maximum is 163 KB on A100 (164 − 1), 227 KB on H100 and 99 KB on the 4090. Static \`__shared__\` is also capped at 48 KB until the kernel opts in with \`cudaFuncSetAttribute\`.

The carveout is the split of one SRAM array between shared memory and L1. Asking for 100 KB of shared memory on an A100 shrinks L1. A kernel that was L1-resident can get slower as its occupancy goes up.

## Waves

The block scheduler places \`SMs × blocksPerSM\` blocks at a time. That many blocks is one **wave**. A grid that is not a multiple of the wave leaves the last wave partly empty — the tail. It matters when the grid is small or each block runs for a long time. A persistent kernel (a fixed grid of about one wave, with a grid-stride loop) simply does not have a tail.

## Hiding the stall you actually have

Each SM has four warp schedulers. A warp that issues a load and then needs the result is not eligible again for the latency of that load, unless it has other independent instructions to issue first. With latency $L$ cycles and $I$ independent instructions before the use:

$$
\\text{warps to keep every scheduler busy} \\approx 4 \\cdot \\frac{L}{I}.
$$

| Stall | $L$ on A100 (≈) | $I = 1$ | What actually saves you |
|---|---|---|---|
| Dependent FMA | 4 | 16 warps | almost any legal block |
| Shared memory | ~20 | ~80 warps | a few independent uses, or just not depending on the load immediately |
| HBM | ~500 | ~2000 warps | impossible by occupancy alone |

Two thousand warps is not a real target. A streaming kernel does not need the schedulers busy: it needs the **memory pipe** full. Little's law, from the memory chapter, says an A100 SM must keep about **6.5 KB** in flight to hit HBM peak, which is about **52** coalesced 128-byte warp loads. Occupancy covers that. It does not cover a pointer chase, where each warp has one 4-byte miss in flight and then waits. That kernel needs more independent misses per thread, not a higher percentage.

\`cudaOccupancyMaxActiveBlocksPerMultiprocessor\` applies this arithmetic to a real kernel, including its compiled register count. The calculator in the lab is the same model with the inputs in your hands.
`,A=`## The four ceilings

A block of $T$ threads uses

$$
W_b = \\left\\lceil \\frac{T}{32} \\right\\rceil
$$

warps. The compiler's register count $R$ is charged as

$$
R' = 8\\left\\lceil \\frac{R}{8} \\right\\rceil,
\\qquad
\\text{registers per warp} = 32 R' = 256\\left\\lceil \\frac{R}{8} \\right\\rceil.
$$

Warps the register file can hold, on a 65 536-register SM:

$$
W_{\\text{regs}} = 4\\left\\lfloor \\frac{65536 / (32 R')}{4} \\right\\rfloor.
$$

Blocks that fit:

$$
B_{\\text{regs}} = \\left\\lfloor W_{\\text{regs}} / W_b \\right\\rfloor.
$$

Shared memory charged per block is $0$ when the kernel allocates none. Otherwise, with the 128-byte allocation unit and the 1 KB reserve,

$$
S' = 128\\left\\lceil \\frac{S}{128} \\right\\rceil + 1024,
\\qquad
B_{\\text{smem}} = \\left\\lfloor \\frac{S_{\\text{SM}}}{S'} \\right\\rfloor.
$$

$S$ above $S_{\\text{SM}} - 1024$ does not launch. The other two ceilings need no rounding:

$$
B_{\\text{warps}} = \\left\\lfloor W_{\\max} / W_b \\right\\rfloor,
\\qquad
B_{\\text{blocks}} = B_{\\max}.
$$

$$
B = \\min(B_{\\text{regs}}, B_{\\text{smem}}, B_{\\text{warps}}, B_{\\text{blocks}}),
\\qquad
\\text{occupancy} = \\frac{B \\cdot W_b}{W_{\\max}}.
$$

The limiter is whichever ceiling equals $B$. Two of them often tie when the register file divides evenly.

### Worked A100 numbers ($W_{\\max} = 64$, $S_{\\text{SM}} = 164$ KB)

| Block | $R$ | $S$ | $R'$ | $B$ | Warps | Occupancy | Limiter |
|---|---|---|---|---|---|---|---|
| 128 | 32 | 0 | 32 | 16 | 64 | 100 % | registers and warp slots |
| 128 | 37 | 0 | 40 | 12 | 48 | 75 % | registers |
| 1024 | 37 | 0 | 40 | 1 | 32 | 50 % | registers (16 slots left over) |
| 256 | 64 | 0 | 64 | 4 | 32 | 50 % | registers |
| 128 | 32 | 32 KB | 32 | 4 | 16 | 25 % | shared memory |

The 37-register row is the allocation unit, not the source. $32 \\times 40 = 1280$ registers/warp, and $4\\lfloor 65536/1280/4 \\rfloor = 48$ warps. Twelve blocks of 4 warps use all 48. A 1024-thread block is 32 warps, so only one fits, and $48 - 32 = 16$ warp slots stay empty.

On an RTX 4090 the same 128-thread, 37-register kernel gets $48/48 = 100\\%$, because that SM's ceiling is 48 warps. The kernel did not get faster; the denominator shrank.

Shared memory: $32 \\times 1024 + 1024 = 33792$ bytes/block. $\\lfloor 164 \\times 1024 / 33792 \\rfloor = 4$ blocks.

## Waves

With $N_{\\text{SM}}$ SMs and $B$ resident blocks per SM, one wave is $N_{\\text{SM}} B$ blocks. A grid of $G$ blocks runs

$$
\\left\\lceil \\frac{G}{N_{\\text{SM}} B} \\right\\rceil
$$

waves, and the last wave occupies $G \\bmod (N_{\\text{SM}} B)$ block slots (or a full wave, when that remainder is 0).

## Warps required to hide a stall

Four schedulers, each wanting one eligible warp per cycle. A warp that has $I$ independent instructions and then waits out a latency of $L$ cycles is eligible a fraction $\\min(1, I/L)$ of the time. Warps that keep every scheduler busy:

$$
W_{\\text{hide}} = \\left\\lceil \\frac{4}{\\min(1, I/L)} \\right\\rceil = \\left\\lceil 4 \\cdot \\frac{L}{I} \\right\\rceil
\\quad\\text{when } I < L.
$$

Dependent FMA, $L = 4$, $I = 1$: $W_{\\text{hide}} = 16$. That is 25 % occupancy on an A100. Raising occupancy from 25 % to 100 % does not speed a pure FMA loop.

Dependent HBM load, $L \\approx 500$, $I = 1$: $W_{\\text{hide}} = 2000$. The SM has 64 warp slots, so the schedulers stay idle $\\tfrac{64}{2000}$ of the time no matter how you launch. The fix is a larger $I$ (more independent misses per thread), not a higher percentage.

## When the pipe, not the scheduler, is the limit

Bytes one SM must keep in flight to hit a link of bandwidth $\\beta$ (bytes/s) and latency $t$ (seconds):

$$
\\text{bytes} = \\frac{\\beta}{N_{\\text{SM}}} \\cdot t,
\\qquad
t = \\frac{L}{f}.
$$

A100 HBM: $\\beta = 2039$ GB/s, $N_{\\text{SM}} = 108$, $L = 500$, $f = 1.41$ GHz.

$$
t = 355\\text{ ns},
\\qquad
\\text{bytes} \\approx 6.5\\text{ KB per SM}.
$$

A coalesced warp load is 128 bytes, so about $6.5 \\times 1024 / 128 \\approx 52$ such loads in flight fill the pipe. Fifty-two resident warps are enough for a streaming kernel. They are not enough for a chase of 4-byte dependent loads, which keeps $52 \\times 4 = 208$ bytes in flight — about 3 % of the pipe.
`,j=`## Exercises

**Q1.** A100, 128 threads/block, 32 registers/thread, no shared memory. How many blocks per SM, how many warps, and what is the occupancy?

<details>
<summary>Show answer</summary>

4 warps/block. 32 × 32 = 1024 registers/warp, and $65536/1024 = 64$ warps exactly. $64/4 = 16$ blocks, which is also the warp-slot cap. **16 blocks, 64 warps, 100 %.** Registers and warp slots tie.

</details>

**Q2.** The compiler reports 37 registers instead of 32. What does the SM actually allocate, and what is the occupancy for the same 128-thread block?

<details>
<summary>Show answer</summary>

37 rounds up to **40** registers/thread (the unit is 8). That is 1280 registers/warp. $4\\lfloor 65536/1280/4 \\rfloor = 48$ warps, so **12 blocks, 75 %.** The limiter is the register file. The source change was 5 registers; the occupancy change was a quarter of the SM.

</details>

**Q3.** Same 37 registers, but the block is 1024 threads. Blocks, warps, occupancy?

<details>
<summary>Show answer</summary>

A 1024-thread block is 32 warps. Only **one** fits in the 48 warps the register file allows, leaving 16 slots empty. Occupancy is $32/64 = 50\\%$. A smaller block would have used those 16 slots.

</details>

**Q4.** 128 threads, 32 registers, 32 KB of shared memory, A100. Which resource limits, and what is the occupancy?

<details>
<summary>Show answer</summary>

Charged shared memory is $32768 + 1024 = 33792$ bytes. $\\lfloor 164 \\times 1024 / 33792 \\rfloor = 4$ blocks. Warp slots would allow 16, so **shared memory** limits. 4 × 4 = 16 warps, occupancy **25 %.**

</details>

**Q5.** Take the kernel from Q2 (128 threads, 37 registers) and move it to an RTX 4090. Occupancy?

<details>
<summary>Show answer</summary>

The register file still allows 48 warps, and a 4090 SM holds 48. Occupancy is $48/48 = 100\\%$. Same kernel, higher percentage, because the maximum shrank from 64 to 48. Check achieved bandwidth or time, not the percentage, when you compare those two chips.

</details>

**Q6.** A dependent FMA has a 4-cycle latency and the kernel has no other independent instruction ($I = 1$). How many resident warps keep four schedulers busy? Is 100 % occupancy useful here?

<details>
<summary>Show answer</summary>

$W = \\lceil 4 \\times 4 / 1 \\rceil = 16$ warps. That is 25 % of an A100. More resident warps have nothing to hide. This is the case for cutting occupancy on purpose (fewer registers, more of the L1 left for data).

</details>

**Q7.** An A100 SM must keep about 6.5 KB in flight to fill HBM. How many coalesced 128-byte warp loads is that, and why doesn't the same count cover a 4-byte pointer chase?

<details>
<summary>Show answer</summary>

$6700 / 128 \\approx 52$ warp loads. One resident warp per load covers a streaming kernel. A chase that puts a single 4-byte load in flight per warp keeps $52 \\times 4 \\approx 208$ bytes moving, a few percent of the pipe. That kernel needs more independent misses per thread. Occupancy cannot invent them.

</details>

**Q8.** \`__launch_bounds__(256, 8)\` is set on a kernel whose unconstrained compile uses 80 registers and spills nothing. What is the compiler being asked to do?

<details>
<summary>Show answer</summary>

Keep **8 blocks of 256 threads** resident. That is 64 warps, the whole A100 SM, so the register budget is about $65536 / 2048 = 32$ registers/thread before rounding. The compiler will cut from 80 toward that budget and **spill** the rest to local memory. Occupancy goes up; local-memory traffic may eat the gain. Read the \`ptxas\` spill lines.

</details>
`,M=`// nvcc -O3 -arch=sm_80 occupancy_query.cu -o occupancy_query
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

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
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
    printf("%s  sm_%d  SMs %d\\n", p.name, p.major * 10 + p.minor, p.multiProcessorCount);
    printf("registers/SM %d   shared/SM %zu B   max warps/SM %d   max blocks/SM %d\\n",
           p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor,
           maxWarps, p.maxBlocksPerMultiProcessor);
    printf("this kernel: %d blocks/SM, %d warps/SM, occupancy %.0f%%\\n",
           blocksPerSM, warps, 100.0 * warps / maxWarps);
    printf("compile with --ptxas-options=-v to see the register count behind that percentage\\n");
    return 0;
}
`,N=`// nvcc -O3 -arch=sm_80 --ptxas-options=-v launch_bounds.cu -o launch_bounds
//
// The unconstrained kernel keeps many live values. launch_bounds asks
// the compiler for 8 resident blocks of 256 threads, which forces the
// register budget down. Watch ptxas: "used N registers" falls, and
// "spill stores" may appear. Spills are local memory, not a free win.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void unconstrained(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    // A long dependent chain of named values. The compiler will keep
    // as many as the register file allows; it has no hint to do otherwise.
    float a0 = x, a1 = x + 1, a2 = x + 2, a3 = x + 3;
    float a4 = x + 4, a5 = x + 5, a6 = x + 6, a7 = x + 7;
    #pragma unroll 1
    for (int k = 0; k < 64; ++k) {
        a0 = a0 * 1.001f + a1; a1 = a1 * 1.001f + a2;
        a2 = a2 * 1.001f + a3; a3 = a3 * 1.001f + a4;
        a4 = a4 * 1.001f + a5; a5 = a5 * 1.001f + a6;
        a6 = a6 * 1.001f + a7; a7 = a7 * 1.001f + a0;
    }
    a[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

// minBlocksPerMultiprocessor = 8, max threads = 256 → 2048 resident threads,
// the whole SM on A100/H100. The compiler must fit 8 blocks in 65536 registers.
__global__ void __launch_bounds__(256, 8)
bounded(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float a0 = x, a1 = x + 1, a2 = x + 2, a3 = x + 3;
    float a4 = x + 4, a5 = x + 5, a6 = x + 6, a7 = x + 7;
    #pragma unroll 1
    for (int k = 0; k < 64; ++k) {
        a0 = a0 * 1.001f + a1; a1 = a1 * 1.001f + a2;
        a2 = a2 * 1.001f + a3; a3 = a3 * 1.001f + a4;
        a4 = a4 * 1.001f + a5; a5 = a5 * 1.001f + a6;
        a6 = a6 * 1.001f + a7; a7 = a7 * 1.001f + a0;
    }
    a[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

int main() {
    int u = 0, b = 0;
    cudaError_t eu = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&u, unconstrained, 256, 0);
    cudaError_t eb = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, bounded, 256, 0);
    if (eu != cudaSuccess || eb != cudaSuccess) {
        fprintf(stderr, "%s\\n", cudaGetErrorString(eu != cudaSuccess ? eu : eb));
        return 1;
    }
    printf("unconstrained: %d blocks/SM (%d warps)\\n", u, u * 8);
    printf("launch_bounds(256, 8): %d blocks/SM (%d warps)\\n", b, b * 8);
    printf("if the bounded kernel spilled, the higher count can still be slower — time both\\n");
    return 0;
}
`,P=`// nvcc -O3 -arch=sm_80 latency_hide.cu -o latency_hide
//
// Same grid, same number of FMAs. \`chain\` is one dependent accumulator:
// each FMA waits on the previous one, so extra warps are the only
// latency cover. \`wide\` keeps 8 independent accumulators, so one warp
// has other work while a result is in flight. Wide should need far
// fewer resident warps to reach the same FMA throughput.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void chain(float* a, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    for (int k = 0; k < iters; ++k)
        x = x * 1.0001f + 0.001f;          // I = 1, latency ≈ 4 cycles
    a[i] = x;
}

__global__ void wide(float* a, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x0 = a[i], x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
    float x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
    for (int k = 0; k < iters; ++k) {
        x0 = x0 * 1.0001f + 0.001f; x1 = x1 * 1.0001f + 0.001f;
        x2 = x2 * 1.0001f + 0.001f; x3 = x3 * 1.0001f + 0.001f;
        x4 = x4 * 1.0001f + 0.001f; x5 = x5 * 1.0001f + 0.001f;
        x6 = x6 * 1.0001f + 0.001f; x7 = x7 * 1.0001f + 0.001f;
    }
    a[i] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

static float time_ms(void (*launch)(float*, int, int), float* a, int n, int iters, int grid, int block) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    launch(a, n, iters);                       // warm the launch path
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int r = 0; r < 8; ++r) launch(a, n, iters);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 8.f;
}

// Kernels are launched through a thin wrapper so the timer stays generic.
static void launch_chain(float* a, int n, int iters) { chain<<<(n + 255) / 256, 256>>>(a, n, iters); }
static void launch_wide(float* a, int n, int iters)  { wide<<<(n + 255) / 256, 256>>>(a, n, iters); }

int main() {
    const int n = 1 << 20;
    const int iters = 4096;
    float* a;
    cudaMalloc(&a, n * sizeof(float));
    cudaMemset(a, 0, n * sizeof(float));
    float ms_c = time_ms(launch_chain, a, n, iters, 0, 0);
    float ms_w = time_ms(launch_wide, a, n, iters, 0, 0);
    // chain does \`iters\` FMAs/thread; wide does 8*iters. Compare per FMA.
    double fma_c = (double)n * iters;
    double fma_w = (double)n * iters * 8;
    printf("chain  %7.3f ms   %.0f GFLOP/s  (one dependent accumulator)\\n",
           ms_c, fma_c / ms_c / 1e6);
    printf("wide   %7.3f ms   %.0f GFLOP/s  (eight independent accumulators)\\n",
           ms_w, fma_w / ms_w / 1e6);
    cudaFree(a);
    return 0;
}
`,F=n(),I={regs:o.c,smem:o.d,warps:o.b,blocks:o.a},L=[o.e,o.b,o.d,o.c,o.a,`#86efac`,`#67e8f9`,`#f9a8d4`],R=e=>{let t=Math.round(1e3*e)/10;return t>=99.95?`100`:String(t)},z=e=>e.map(e=>e.name).join(` and `),B=[[128,32,0,`128 thr · 32 regs`],[128,37,0,`37 regs (rounds to 40)`],[1024,37,0,`1024 thr · 37 regs`],[128,32,32,`32 KB shared`]];function V(e){let[t,n]=(0,y.useState)(`a100`),[r,i]=(0,y.useState)(e.threads),[a,o]=(0,y.useState)(e.regs),[s,c]=(0,y.useState)(e.smemKB),l=Math.floor((b[t].smem-1024)/1024),u=Math.min(s,l)*1024,d=(0,y.useMemo)(()=>E({threads:r,regs:a,smem:u,sm:b[t]}),[r,a,u,t]);return{gpu:t,applyGpu:e=>{n(e);let t=Math.floor((b[e].smem-1024)/1024);c(e=>Math.min(e,t))},threads:r,setThreads:i,regs:a,setRegs:o,smemKB:Math.min(s,l),setSmemKB:c,capKB:l,o:d}}function H({s:e,extra:t}){return(0,F.jsxs)(d,{children:[(0,F.jsx)(h,{label:`GPU`,value:e.gpu,onChange:e.applyGpu,options:_}),(0,F.jsx)(u,{label:`Threads / block`,min:32,max:1024,step:32,value:e.threads,onChange:e.setThreads}),(0,F.jsx)(u,{label:`Registers / thread`,min:8,max:128,value:e.regs,onChange:e.setRegs}),(0,F.jsx)(u,{label:`Shared / block`,min:0,max:e.capKB,value:e.smemKB,onChange:e.setSmemKB,fmt:e=>e+` KB`}),t]})}function U(){let e=V({threads:128,regs:32,smemKB:0}),[t,n]=(0,y.useState)(108),{o:c,gpu:d}=e,f=v[d],h=b[d],_=c.fit?f.sms*c.activeBlocks:0,x=_?Math.ceil(t/_):0,S=_?t%_||_:0,[C]=g(400,(e,n)=>{let a=n-118-16;s(e,c.fit?R(c.occ)+`% occupancy`:`does not fit`,118,18,{size:13,color:c.fit?c.occ>.9?o.e:c.occ>.4?o.d:o.r:o.r,weight:700}),s(e,c.fit?`${c.activeWarps} / ${h.maxWarps} warps`:c.reason,n-16,18,{size:12,color:o.mute,align:`right`});let l=c.limits.map(e=>({...e,shown:e.blocks===1/0?h.maxBlocks:e.blocks,unused:e.blocks===1/0})),u=Math.max(h.maxBlocks,...l.map(e=>e.shown),1);l.forEach((t,n)=>{let l=40+n*52;s(e,t.name,12,l+14,{size:12,color:o.ink}),e.fillStyle=`#1a1e32`,r(e,118,l,a,28,6),e.fill();let d=c.fit&&c.limiters.some(e=>e.id===t.id);e.fillStyle=d?I[t.id]:i(o.ink,.35);let f=Math.max(t.shown>0?4:0,a*t.shown/u);f>0&&(r(e,118,l,Math.min(a,f),28,6),e.fill());let p=t.unused?`not used`:t.shown+(t.shown===1?` block`:` blocks`);s(e,p,128,l+14,{size:12,color:d?`#0a0c14`:o.ink,weight:700,mono:!0})});let d=40+l.length*52+8;c.fit&&(s(e,`one wave = ${f.sms} SMs × ${c.activeBlocks} blocks = ${_} blocks`,12,d,{size:12,color:o.mute}),s(e,`grid ${t} → ${x} wave${x===1?``:`s`}, last wave ${Math.round(100*S/_)}% full`,12,d+18,{size:12,color:o.ink,mono:!0}))});return(0,F.jsxs)(F.Fragment,{children:[(0,F.jsx)(`canvas`,{...C}),(0,F.jsx)(H,{s:e,extra:(0,F.jsx)(u,{label:`Blocks in the grid`,min:1,max:4096,value:t,onChange:n})}),(0,F.jsx)(`div`,{className:`controls`,children:B.map(([t,n,r,i])=>(0,F.jsx)(m,{onClick:()=>{e.setThreads(t),e.setRegs(n),e.setSmemKB(r)},children:i},i))}),(0,F.jsx)(l,{items:[[o.c,`registers bind`],[o.d,`shared memory binds`],[o.b,`warp slots bind`],[o.a,`block slots bind`]]}),(0,F.jsx)(p,{children:c.fit?(0,F.jsxs)(F.Fragment,{children:[(0,F.jsx)(`b`,{className:c.occ>.9?`g`:`w`,children:c.activeBlocks}),` blocks/SM · `,(0,F.jsx)(`b`,{children:c.activeWarps}),` warps · occupancy `,(0,F.jsxs)(`b`,{children:[R(c.occ),`%`]}),`. Limited by `,(0,F.jsx)(`b`,{children:z(c.limiters)}),`.`,c.regsAlloc!==e.regs&&(0,F.jsxs)(F.Fragment,{children:[` The compiler's `,e.regs,` registers allocate as `,(0,F.jsx)(`b`,{children:c.regsAlloc}),` (multiples of 8).`]}),c.smemCharged>0&&(0,F.jsxs)(F.Fragment,{children:[` Shared memory is charged as `,(0,F.jsx)(`b`,{children:a(c.smemCharged)}),`, including the 1 KB reserve.`]})]}):(0,F.jsx)(`b`,{className:`r`,children:c.reason})})]})}function W({o:e,sm:t}){let n=Math.ceil(t.maxWarps/8);return(0,F.jsx)(`group`,{children:Array.from({length:t.maxWarps},(t,r)=>{let i=e.fit&&r<e.activeWarps,a=i?Math.floor(r/e.warpsPerBlock):-1,o=r%8,s=Math.floor(r/8),c=(o-7/2)*.46,l=(s-(n-1)/2)*.46,u=i?L[a%L.length]:`#3d4660`;return(0,F.jsxs)(`mesh`,{position:[c,i?.22:0,l],children:[(0,F.jsx)(`boxGeometry`,{args:[.36,i?.55:.16,.36]}),(0,F.jsx)(`meshStandardMaterial`,{color:u,emissive:i?u:`#000`,emissiveIntensity:i?.4:0,roughness:.45})]},r)})})}function G(){let e=V({threads:128,regs:37,smemKB:0}),{o:t,gpu:n}=e,r=b[n],i=t.fit?(0,F.jsxs)(F.Fragment,{children:[(0,F.jsxs)(`b`,{style:{color:o.e},children:[R(t.occ),`%`]}),` · `,t.activeWarps,`/`,r.maxWarps,` warp slots filled`,(0,F.jsx)(`br`,{}),t.activeBlocks,` blocks · limited by `,z(t.limiters)]}):(0,F.jsxs)(F.Fragment,{children:[`does not fit`,(0,F.jsx)(`br`,{}),t.reason]});return(0,F.jsxs)(F.Fragment,{children:[(0,F.jsx)(c,{height:420,camera:[0,5.4,6.8],target:[0,0,0],fov:42,overlay:i,hint:`drag to orbit · one box is a warp slot, colour groups a block`,children:(0,F.jsx)(W,{o:t,sm:r})}),(0,F.jsx)(H,{s:e}),(0,F.jsx)(p,{children:t.fit?(0,F.jsxs)(F.Fragment,{children:[`Tall boxes are resident warps. A short grey box is a slot this block shape cannot fill. Each colour is one block (`,t.warpsPerBlock,` warps).`]}):(0,F.jsxs)(F.Fragment,{children:[`Nothing is resident. `,t.reason]})})]})}var K=[[`reg`,`dependent FMA`],[`smem`,`shared memory`],[`l2`,`L2`],[`hbm`,`HBM`]];function q(){let[e,t]=(0,y.useState)(`a100`),[n,i]=(0,y.useState)(`hbm`),[c,f]=(0,y.useState)(1),[m,x]=(0,y.useState)(32),S=v[e],C=b[e],w=S.lat[n],T=Math.min(m,C.maxWarps),E=D(w,c,C.schedulers),k=Math.min(1,T/E),A=O(S.bw,S.lat.hbm,S.clk,S.sms),j=A/128,[M]=g(340,(e,t)=>{let n=t-16-16;s(e,`schedulers kept busy`,16,18,{size:12,color:o.mute,weight:600}),s(e,R(k)+`%`,t-16,18,{size:14,align:`right`,weight:700,color:k>.95?o.e:k>.4?o.d:o.r}),e.fillStyle=`#1a1e32`,r(e,16,32,n,22,6),e.fill(),e.fillStyle=k>.95?o.e:k>.4?o.d:o.r,r(e,16,32,Math.max(4,n*k),22,6),e.fill(),s(e,`${T} resident  /  ${E} needed  ·  L = ${w} cycles, I = ${c}`,16,70,{size:12,color:o.ink,mono:!0}),s(e,`the `+C.maxWarps+` warp slots on one SM`,16,100,{size:12,color:o.mute});let i=C.maxWarps>48?16:12,a=Math.min(18,(n-(i-1)*3)/i);for(let t=0;t<C.maxWarps;t++){let n=t%i,s=Math.floor(t/i),c=16+n*(a+3),l=114+s*(a+3);e.fillStyle=t<T?t<E?o.e:o.b:`#1a1e32`,r(e,c,l,a,a,3),e.fill()}let l=114+Math.ceil(C.maxWarps/i)*(a+3)+14;s(e,`green: resident warps this stall can actually use    cyan: extra, with nothing left to hide`,16,Math.min(326,l),{size:11,color:o.mute})});return(0,F.jsxs)(F.Fragment,{children:[(0,F.jsx)(`canvas`,{...M}),(0,F.jsxs)(d,{children:[(0,F.jsx)(h,{label:`GPU`,value:e,onChange:t,options:_}),(0,F.jsx)(h,{label:`Stall`,value:n,onChange:i,options:K}),(0,F.jsx)(u,{label:`Independent instructions`,min:1,max:32,value:c,onChange:f}),(0,F.jsx)(u,{label:`Resident warps`,min:1,max:C.maxWarps,value:T,onChange:x})]}),(0,F.jsx)(l,{items:[[o.e,`covers the stall`],[o.b,`resident but idle for this stall`]]}),(0,F.jsxs)(p,{children:[S.name,`: this stall is ≈ `,(0,F.jsx)(`b`,{children:w}),` cycles. With `,(0,F.jsx)(`b`,{children:c}),` independent instruction`,c>1?`s`:``,` before the result is used, four schedulers want `,(0,F.jsx)(`b`,{className:E>C.maxWarps?`r`:`g`,children:E}),` resident warps.`,E>C.maxWarps?(0,F.jsxs)(F.Fragment,{children:[` The SM only has `,C.maxWarps,`. Occupancy cannot cover it — raise the independent work per thread.`]}):(0,F.jsxs)(F.Fragment,{children:[` `,T>=E?`The resident warps cover it.`:(0,F.jsxs)(F.Fragment,{children:[E-T,` more warps would still find a stall to hide.`]})]}),n===`hbm`&&(0,F.jsxs)(F.Fragment,{children:[` Filling the HBM pipe is a different number: `,(0,F.jsx)(`b`,{children:a(A)}),` in flight per SM, about `,(0,F.jsx)(`b`,{children:Math.round(j)}),` coalesced 128-byte loads. A streaming kernel saturates there; a pointer chase does not.`]})]})]})}function J(){return(0,F.jsx)(f,{views:[{id:`o`,label:`What limits the SM`,render:()=>(0,F.jsx)(U,{})},{id:`d`,label:`3D: warp slots`,render:()=>(0,F.jsx)(G,{})},{id:`h`,label:`Hiding the stall`,render:()=>(0,F.jsx)(q,{})}]})}var Y={Lab:J,vizTitle:`See which resource caps the SM, then which stalls those warps can actually hide`,tryIt:[`Leave **128 threads, 32 registers**. Both register and warp bars stop at 16 blocks: 100% on an A100.`,`Click **37 regs**. The count rounds to 40 and occupancy falls to 75%. Nothing else in the kernel changed.`,`Click **1024 thr · 37 regs**. One block fills 32 of the 48 warps the register file allows. Sixteen slots stay empty.`,`Click **32 KB shared**. Shared memory becomes the short bar: 4 blocks, 25%.`,`Open **3D** (it starts at 37 registers). Tall boxes are resident warps; each colour is one block. Switch the GPU to the **4090** — the same kernel fills every slot, because the ceiling is 48.`,`Open **Hiding the stall**. A dependent FMA needs 16 warps. Switch the stall to **HBM** with one independent instruction: the SM would need 2000 warps, and it has 64.`],theory:k,math:A,practice:j,code:[{title:`Ask the runtime how many blocks fit`,lang:`cuda`,note:`cudaOccupancyMaxActiveBlocksPerMultiprocessor counts compiled registers and static shared memory. --ptxas-options=-v prints the register count the percentage came from.`,src:M},{title:`Ask the compiler for 8 resident blocks`,lang:`cuda`,note:`__launch_bounds__(256, 8) caps registers so eight blocks fit. If ptxas then reports spill stores, the extra occupancy is local-memory traffic.`,src:N},{title:`One dependent chain versus eight independent accumulators`,lang:`cuda`,note:`Same FMAs per accumulator. The chain has I = 1, so only extra warps hide the 4-cycle wait. The wide kernel has other work in the same warp.`,src:P}],quiz:[{q:`Occupancy is:`,options:[`Resident warps per SM, divided by the SM's maximum`,`Achieved bandwidth divided by peak`,`Threads per block divided by 32`,`The fraction of the grid that has finished`],answer:0,why:`It is a resident-warp ratio. Bandwidth and elapsed time are what you measure afterwards; occupancy is only the budget of warps that can hide stalls.`},{q:`A thread that "uses 37 registers" is charged for how many, on these GPUs?`,options:[`37`,`40`,`64`,`256`],answer:1,why:`Register allocation rounds up to a multiple of 8 per thread. 37 becomes 40, which is 1280 registers per warp.`},{q:`128 threads, 37 registers, no shared memory, A100. Occupancy is:`,options:[`100%`,`75%`,`50%`,`25%`],answer:1,why:`40 registers/thread allow 48 warps. A 128-thread block is 4 warps, so 12 blocks. $48/64 = 75\\%$.`},{q:`Why can a 1024-thread block leave warp slots empty even though the register file has some left?`,options:[`Blocks cannot be split across the leftover slots`,`1024 exceeds the thread limit`,`Shared memory is always reserved`,`The warp size changes`],answer:0,why:`Slots are taken in whole blocks. A 32-warp block does not fit into 16 leftover warps, so those slots stay empty.`},{q:`CUDA reserves how much shared memory per block, on top of what you allocate?`,options:[`0`,`128 B`,`1 KB`,`48 KB`],answer:2,why:`1 KB per block. That is why the per-block maximum is 163 KB on A100 when the SM pool is 164 KB.`},{q:`A dependent FMA takes about 4 cycles and the kernel has no other independent instruction. Warps that keep four schedulers busy:`,options:[`4`,`16`,`64`,`2000`],answer:1,why:`$4 \\times 4 / 1 = 16$. Full occupancy does not speed a pure FMA loop past that.`},{q:`A dependent HBM load needs far more than 64 warps to keep the schedulers busy. The useful lever is:`,options:[`Raising occupancy from 90% to 100%`,`More independent misses per thread`,`A smaller grid`,`Switching the block to 32 threads`],answer:1,why:`The SM cannot hold the ~2000 warps the stall would take. Independent loads per thread (and, for streaming, enough bytes in flight) are what fill the wait.`}]};export{Y as default};