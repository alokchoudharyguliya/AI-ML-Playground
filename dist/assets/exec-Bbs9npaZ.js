import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,f as n,l as r,v as i}from"./r3f-x1z21uF6.js";import{c as a,g as o,h as s,n as c,t as l,y as u}from"./viz-CPys2405.js";import{t as d}from"./Stage3D-CWOT2b2l.js";import{c as f,i as p,l as m,o as h,r as g,t as _,u as v}from"./hooks-Dw7oo1m3.js";var y=e(t(),1),b=`## Why GPUs look the way they do

A CPU is built to make **one thread fast**: big caches, branch prediction, out-of-order execution — a latency machine with a handful of cores. A GPU is built for **throughput**: thousands of simple lanes, small caches, and *hide* memory latency by always having other work ready. A modern data-centre GPU (H100) has 132 Streaming Multiprocessors (SMs), each able to keep up to 2048 threads resident — **~270,000 threads in flight**. Latency isn't avoided; it is overlapped.

CUDA exposes this with the **SIMT** (Single Instruction, Multiple Threads) model: you write the program for *one thread* (a **kernel**), then launch it for many.

## The thread hierarchy

\`\`\`
Grid  (one kernel launch)
 └─ Thread blocks (a.k.a. CTAs)     independent, any order, any SM
     └─ Warps                       32 threads, scheduled together
         └─ Threads                 each with its own registers and indices
\`\`\`

| Level | Size limits | What it shares | How it synchronises |
|---|---|---|---|
| **Thread** | — | its own registers / local memory | — |
| **Warp** | 32 threads | instruction stream; \`__shfl_*\` registers | \`__syncwarp()\`, implicit lockstep |
| **Block** | ≤ 1024 threads, 1–3-D | **shared memory**, L1 | \`__syncthreads()\` |
| *Cluster* (Hopper+) | ≤ 8–16 blocks | distributed shared memory | cluster barriers |
| **Grid** | up to $2^{31}-1$ blocks in x | global memory | kernel boundaries, atomics, cooperative groups |

Every thread can read the built-in variables \`threadIdx\`, \`blockIdx\`, \`blockDim\`, \`gridDim\` and derives *which data it owns*:

\`\`\`cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;   // global 1-D index
if (i < n) y[i] = a * x[i] + y[i];               // bounds check: the grid is rounded up!
\`\`\`

In the 3D lab, tile = block, cubes = threads; colour = warp. Hover a cube to see that formula evaluated for it.

## Launching

\`\`\`cuda
saxpy<<<gridDim, blockDim, sharedBytes, stream>>>(args…);
dim3 block(16, 16);                              // 256 threads
dim3 grid((W + 15) / 16, (H + 15) / 16);         // round up so every pixel is covered
\`\`\`

Kernel launches are **asynchronous**: the CPU continues immediately; use streams/events or \`cudaDeviceSynchronize()\` to wait. Launch overhead is ~3–10 µs, so tiny kernels are latency-dominated.

## What happens on the hardware

1. The **block scheduler** assigns each block to an SM that has enough **registers, shared memory, warp slots and block slots** (see *Occupancy*). If none is free, the block waits in the queue.
2. Inside the SM the block's threads are split into **warps** (consecutive \`threadIdx.x\` first, then y, then z).
3. Four **warp schedulers** per SM pick, every cycle, a *ready* warp (operands available) and issue one instruction for all 32 lanes. A warp waiting on memory is simply skipped for another — **zero-cost context switching**, because every warp keeps its registers resident.
4. When all warps of a block finish, its resources are freed and the next queued block is placed.

Second lab tab: blocks are packed onto SMs in "waves". If the number of blocks isn't a multiple of the machine's slots, the **last wave runs partly empty** — the *tail effect*. It matters when blocks are few and long; with thousands of short blocks it vanishes.

<div class="callout">

**Rules that fall out of this design.**
1. Blocks must be **independent** — no ordering, no global barrier inside a kernel (except cooperative launches).
2. Threads in a block may **cooperate** via shared memory + \`__syncthreads()\`.
3. **Always** guard against out-of-range indices when \`n\` isn't a multiple of the block size.
4. Use block sizes that are **multiples of 32** (128–512 is the sweet spot) so no warp is partially empty.

</div>

## SM anatomy (Hopper H100, per SM)

- **4 sub-cores**, each with a warp scheduler, 32 FP32 lanes (128 FP32 per SM), 16 INT32, 1 Tensor Core, register file slice (64K × 32-bit registers per SM = 256 KB).
- **Up to 228 KB shared memory/L1** (software-managed), 64 resident warps (2048 threads), 32 resident blocks.
- Special function units, load/store units, and the **Tensor Memory Accelerator (TMA)** for async bulk copies.

## Writing grid-stride loops

A robust pattern when \`n\` may exceed the launched thread count, or to reuse a fixed, occupancy-sized grid:

\`\`\`cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = a * x[i] + y[i];
\`\`\`

## Compilation pipeline

\`nvcc\` splits host and device code; device code compiles to **PTX** (virtual ISA) and then to **SASS** (the real machine code for a specific architecture, e.g. \`sm_80\`, \`sm_90\`). Compile for your target: \`nvcc -O3 -arch=sm_90 kernel.cu\`. Inspect with \`cuobjdump --dump-sass\`, profile with **Nsight Systems** (timeline) and **Nsight Compute** (per-kernel counters).

## Error handling you should always do

Kernel launches don't return errors directly. Check \`cudaGetLastError()\` right after the launch (configuration errors) and the result of \`cudaDeviceSynchronize()\` / the next API call (execution errors). Wrap calls in a macro like \`CUDA_CHECK(...)\` (see the code tab).
`,x=`## Index arithmetic

**1-D:**

$$
i=\\text{blockIdx.x}\\cdot\\text{blockDim.x}+\\text{threadIdx.x},\\qquad
\\text{gridDim.x}=\\Big\\lceil\\frac{N}{\\text{blockDim.x}}\\Big\\rceil .
$$

**2-D** (row-major image, width $W$):

$$
x=\\text{blockIdx.x}\\cdot\\text{blockDim.x}+\\text{threadIdx.x},\\quad
y=\\text{blockIdx.y}\\cdot\\text{blockDim.y}+\\text{threadIdx.y},\\quad
\\text{idx}=y\\,W+x .
$$

**Linear thread id inside a block** (determines the warp):

$$
\\text{tid}=\\text{threadIdx.x}+\\text{threadIdx.y}\\cdot\\text{blockDim.x}+\\text{threadIdx.z}\\cdot\\text{blockDim.x}\\,\\text{blockDim.y},
$$

$$
\\text{warp}=\\lfloor\\text{tid}/32\\rfloor,\\qquad\\text{lane}=\\text{tid}\\bmod32 .
$$

Warps per block: $\\lceil T_b/32\\rceil$ with $T_b$ threads per block. Wasted lanes: $32\\lceil T_b/32\\rceil-T_b$.

## Counting work

Threads launched $=\\text{gridDim}\\times\\text{blockDim}\\ge N$. Wasted (guarded-off) threads $\\le\\text{blockDim}-1$. For $N=10^6$ and 256-thread blocks: 3907 blocks $\\Rightarrow$ 1,000,192 threads, 192 idle.

## Waves and the tail effect

With $S$ SMs, $R$ resident blocks per SM (from occupancy), $B$ blocks:

$$
\\text{waves}=\\Big\\lceil\\frac{B}{S\\,R}\\Big\\rceil,\\qquad
\\text{tail utilisation}=\\frac{B}{S\\,R\\cdot\\text{waves}} .
$$

Example: $B=1000$, $S=108$, $R=8$ ⇒ capacity 864/wave ⇒ 2 waves, utilisation $1000/1728=58\\%$; but with $B=1728$ it is 100%. Tail waste is $<1/\\text{waves}$, so more, smaller blocks (or persistent kernels) reduce it.

## Peak throughput from the hardware spec

$$
\\text{FP32 peak}=\\underbrace{\\#SM}_{}\\times\\underbrace{\\#\\text{FP32 lanes/SM}}_{}\\times\\underbrace{2}_{\\text{FMA}}\\times f_{clk}.
$$

- A100: $108\\times64\\times2\\times1.41\\,\\mathrm{GHz}=19.5$ TFLOPS.
- H100 SXM: $132\\times128\\times2\\times1.98\\,\\mathrm{GHz}\\approx67$ TFLOPS.

An FMA (fused multiply-add) counts as 2 FLOPs.

## Latency hiding (Little's law)

To sustain throughput $\\lambda$ (operations/cycle) when each takes latency $L$ cycles you need $\\lambda L$ operations in flight:

$$
\\text{in-flight}=\\text{latency}\\times\\text{throughput}.
$$

Memory: A100 HBM $\\approx2\\,\\mathrm{TB/s}$ with $\\approx500$-cycle latency at 1.4 GHz ⇒ bytes in flight $\\approx2\\times10^{12}\\times\\tfrac{500}{1.4\\times10^9}\\approx7\\times10^{5}$ B $\\approx700$ KB ⇒ across 108 SMs ≈ 6.5 KB per SM outstanding — e.g. 1,600 independent 4-byte loads per SM. That is why you want **many resident warps** (or many loads in flight per thread, "ILP").

## Instruction issue

Each SM sub-core issues one warp-instruction per cycle. FP32 FMA throughput per SM per cycle: $4\\times32=128$ lanes (H100). A warp stalled on a dependency simply doesn't issue; the scheduler picks another eligible warp. Fully hiding a 4-cycle ALU latency needs ≥ 4 independent warps per scheduler, i.e. ≥ 16 warps per SM; hiding a 400-cycle memory latency needs far more parallelism or independent loads.

## Launch limits (compute capability 8.x / 9.0)

| Resource | Limit |
|---|---|
| Threads per block | 1024 |
| Block dims (x, y, z) | (1024, 1024, 64) with product ≤ 1024 |
| Grid dims (x, y, z) | ($2^{31}-1$, 65535, 65535) |
| Resident threads / SM | 2048 (1536 on Ada, 2048 on Hopper) |
| Resident blocks / SM | 32 (24 on Ada) |
| Resident warps / SM | 64 |
| Registers / thread | ≤ 255 |
| Registers / SM | 65,536 |
`,S=`## Exercises

**Q1.** You process a $1920\\times1080$ image with $16\\times16$ blocks. What is the grid size, how many threads are launched, and how many do no useful work?

<details>
<summary>Show answer</summary>

Grid $=(\\lceil1920/16\\rceil,\\lceil1080/16\\rceil)=(120,68)$ = 8,160 blocks × 256 = 2,088,960 threads. Useful: $1920\\cdot1080=2{,}073{,}600$. Idle: $15{,}360$ ($0.7\\%$) — the bottom 8 rows of the last block-row (1088−1080 = 8 rows × 1920). They must be guarded by \`if (x < W && y < H)\`.

</details>

**Q2.** A kernel uses blocks of 300 threads. How many warps does each block use and what fraction of issue slots is wasted?

<details>
<summary>Show answer</summary>

$\\lceil300/32\\rceil=10$ warps = 320 lanes; 20 are idle $\\Rightarrow6.25\\%$ waste. Switching to 288 (9 warps) or 320 (10 warps) fixes it — or use 256/512, which are multiples of 32 and typical.

</details>

**Q3.** For a thread at \`threadIdx=(5,3)\` in a block of \`dim3(16,8)\`, what are its linear thread id, warp, and lane?

<details>
<summary>Show answer</summary>

$\\text{tid}=5+3\\cdot16=53$ → warp $\\lfloor53/32\\rfloor=1$, lane $53\\bmod32=21$. Note a warp spans *two rows* of the 16-wide block here (rows 2 and 3): neighbouring \`threadIdx.y\` threads can share a warp.

</details>

**Q4.** 5,000 blocks, A100 (108 SMs), kernel allows 6 resident blocks/SM. How many waves, and what fraction of the final wave is used?

<details>
<summary>Show answer</summary>

Capacity $=108\\cdot6=648$ per wave. Waves $=\\lceil5000/648\\rceil=8$ (7 full = 4536, last has 464 blocks) → last wave utilisation $464/648=72\\%$; overall efficiency $5000/(8\\cdot648)=96\\%$.

</details>

**Q5.** Why can't a kernel use \`__syncthreads()\` to synchronise the whole grid? What are the alternatives?

<details>
<summary>Show answer</summary>

Blocks can be scheduled in waves — block 5000 may not even start until block 0 has finished — so a grid-wide barrier would deadlock. Alternatives: split into **multiple kernels** (the kernel boundary is a global barrier), **atomics** with a counter ("last block done" pattern), or a **cooperative launch** (\`cudaLaunchCooperativeKernel\`, \`grid.sync()\`) which requires the whole grid to be co-resident.

</details>

**Q6 (code).** Write a grid-stride SAXPY kernel and launch it with a grid size of \`numSMs * 8\` blocks. Why is this robust?

<details>
<summary>Show answer</summary>

See \`vecadd.cu\`. A fixed, hardware-sized grid amortises launch overhead, keeps all SMs busy, handles any \`n\`, and lets you tune for occupancy; each thread strides by \`gridDim.x*blockDim.x\`, which also keeps global accesses coalesced.

</details>

## In practice

- **Start with 256 threads/block** (or 128–512), measure, then tune. For 2-D problems use 16×16 or 32×8 (x contiguous for coalescing — see the next chapters).
- **Check errors** after every launch during development; run \`compute-sanitizer\` for out-of-bounds and race detection.
- **Profile before optimising**: Nsight Systems for the timeline (launch gaps, copies), Nsight Compute for per-kernel limiters.
- **Prefer libraries**: cuBLAS, cuDNN, CUTLASS, CUB, Thrust, cuFFT — hand-written kernels rarely beat them for standard operations, but fused custom kernels (Triton/CUDA) win for non-standard fusions.
- **Triton / CuTe / Warp** let you write tile-level GPU code in Python/C++ templates when full CUDA is overkill.

## Common pitfalls

- Forgetting the \`if (i < n)\` guard → out-of-bounds writes (silent corruption).
- Using \`int\` indices for arrays > 2³¹ elements (use \`size_t\`/\`long long\`).
- Assuming blocks run in order or concurrently.
- Reading results on the host without synchronising (or copying with a non-blocking stream).
- Launching with 0 blocks when \`n==0\` (\`<<<0,…>>>\` is an error).
- Block size > 1024 or \`dim3\` order mix-ups (\`dim3(x,y)\` – x is the fastest-varying thread dimension).
- Timing with the CPU clock without \`cudaDeviceSynchronize()\`; use \`cudaEvent\`s.
`,C=`#include <cstdio>
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
    printf("saxpy: %.3f ms, %.1f GB/s\\n", ms, 3.0 * bytes / (ms * 1e6));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    printf("y[0] = %f (expected 4.0)\\n", h_y[0]);

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
`,w=`#include <cstdio>
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
    printf("grid (%d,%d) x block (%d,%d) = %d threads for %d pixels\\n",
           grid.x, grid.y, block.x, block.y, grid.x * grid.y * block.x * block.y, W * H);

    rgb2gray<<<grid, block>>>(d_rgb, d_gray, W, H);
    cudaError_t e = cudaDeviceSynchronize();
    printf("%s\\n", cudaGetErrorString(e));

    cudaFree(d_rgb); cudaFree(d_gray);
    return 0;
}
`,T=`#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        printf("== Device %d: %s (compute capability %d.%d)\\n", d, p.name, p.major, p.minor);
        printf("  SMs                      : %d\\n", p.multiProcessorCount);
        printf("  warp size                : %d\\n", p.warpSize);
        printf("  max threads / block      : %d\\n", p.maxThreadsPerBlock);
        printf("  max threads / SM         : %d  (= %d warps)\\n", p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / p.warpSize);
        printf("  max blocks / SM          : %d\\n", p.maxBlocksPerMultiProcessor);
        printf("  registers / SM, / block  : %d, %d\\n", p.regsPerMultiprocessor, p.regsPerBlock);
        printf("  shared mem / SM, / block : %zu KB, %zu KB\\n", p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerBlock / 1024);
        printf("  L2 cache                 : %d MB\\n", p.l2CacheSize / (1024 * 1024));
        printf("  global memory            : %.1f GB\\n", p.totalGlobalMem / 1e9);
        int clk = 0, memclk = 0;
        cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, d);          // kHz
        cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, d); // kHz
        double bw = 2.0 * memclk * 1e3 * (p.memoryBusWidth / 8) / 1e9;  // DDR: x2
        printf("  memory bus / bandwidth   : %d-bit, ~%.0f GB/s\\n", p.memoryBusWidth, bw);
        printf("  boost clock              : %.2f GHz\\n", clk / 1e6);
    }
    return 0;
}
`,E=i(),D=e=>new r().setHSL(e*.137%1,.72,.58),O=new n;function k({bx:e,by:t,hover:n,setHover:i,mode:a}){let o=(0,y.useRef)(),s=e*t,c=Math.min(.5,5.2/e,3.6/t),l=c*1.18;return(0,y.useLayoutEffect)(()=>{let i=o.current;i.count=s;for(let o=0;o<s;o++){let s=o%e,c=Math.floor(o/e);O.position.set((s-(e-1)/2)*l,-(c-(t-1)/2)*l,0);let u=o===n?1.35:1;O.scale.setScalar(u),O.updateMatrix(),i.setMatrixAt(o,O.matrix);let d=Math.floor(o/32),f=a===`warp`?D(d):new r().setHSL(.6,.5,.35+.3*(o%e/e));o===n&&f.set(`#ffffff`),i.setColorAt(o,f)}i.instanceMatrix.needsUpdate=!0,i.instanceColor&&(i.instanceColor.needsUpdate=!0)},[e,t,n,a,s,l]),(0,E.jsxs)(`instancedMesh`,{ref:o,args:[null,null,1024],onPointerMove:e=>{e.stopPropagation(),i(e.instanceId)},onPointerOut:()=>i(null),children:[(0,E.jsx)(`boxGeometry`,{args:[c,c,c]}),(0,E.jsx)(`meshStandardMaterial`,{roughness:.45,metalness:.1})]})}function A({gx:e,gy:t,sel:n,setSel:r}){let i=Math.min(1.25,6.4/e,4.6/t),a=i*1.14;return(0,E.jsx)(`group`,{position:[-6.6,-.6,0],rotation:[-.35,.28,0],children:Array.from({length:e*t},(o,s)=>{let c=s%e,l=Math.floor(s/e),u=n[0]===c&&n[1]===l;return(0,E.jsxs)(`mesh`,{position:[(c-(e-1)/2)*a*1+.5,-(l-(t-1)/2)*a*.9,0],onClick:e=>{e.stopPropagation(),r([c,l])},onPointerOver:()=>document.body.style.cursor=`pointer`,onPointerOut:()=>document.body.style.cursor=``,children:[(0,E.jsx)(`boxGeometry`,{args:[i,i*.82,u?.5:.25]}),(0,E.jsx)(`meshStandardMaterial`,{color:u?`#76d12a`:`#2d3560`,emissive:u?`#3d6b12`:`#10142a`,roughness:.5})]},s)})})}function j(){let[e,t]=(0,y.useState)(4),[n,r]=(0,y.useState)(3),[i,o]=(0,y.useState)(16),[s,c]=(0,y.useState)(4),[l,u]=(0,y.useState)([1,1]),[m,g]=(0,y.useState)(null),[_,b]=(0,y.useState)(`warp`),x=Math.min(i,Math.floor(1024/s)),S=x*s,C=Math.ceil(S/32),w=C*32-S,T=m??0,D=T%x,O=Math.floor(T/x),j=l[0]*x+D,M=l[1]*s+O,N=O*x+D,P=(0,E.jsxs)(E.Fragment,{children:[(0,E.jsx)(`b`,{children:`Block`}),` (`,l[0],`,`,l[1],`) of grid `,e,`×`,n,` · `,(0,E.jsx)(`b`,{children:S}),` threads = `,(0,E.jsx)(`b`,{children:C}),` warps`,(0,E.jsx)(`br`,{}),m==null?(0,E.jsx)(`span`,{style:{opacity:.7},children:`hover a thread cube · click a block tile on the left`}):(0,E.jsxs)(E.Fragment,{children:[(0,E.jsx)(`b`,{children:`thread`}),` (`,D,`,`,O,`) → linear tid `,(0,E.jsx)(`b`,{children:N}),` · `,(0,E.jsxs)(`span`,{style:{color:`#fbbf24`},children:[`warp `,Math.floor(N/32),`, lane `,N%32]}),(0,E.jsx)(`br`,{}),`global x = `,l[0],`·`,x,`+`,D,` = `,(0,E.jsx)(`b`,{children:j}),` · global y = `,l[1],`·`,s,`+`,O,` = `,(0,E.jsx)(`b`,{children:M})]})]});return(0,E.jsxs)(E.Fragment,{children:[(0,E.jsxs)(d,{height:440,camera:[1.6,2.2,11.5],target:[-.5,-.2,0],overlay:P,hint:`drag to orbit · hover threads · click blocks`,children:[(0,E.jsx)(A,{gx:e,gy:n,sel:l,setSel:u}),(0,E.jsx)(`group`,{position:[2.2,0,0],rotation:[0,-.18,0],children:(0,E.jsx)(k,{bx:x,by:s,hover:m,setHover:g,mode:_})})]}),(0,E.jsxs)(p,{children:[(0,E.jsx)(f,{label:`gridDim.x`,min:1,max:8,value:e,onChange:e=>{t(e),u(t=>[Math.min(t[0],e-1),t[1]])}}),(0,E.jsx)(f,{label:`gridDim.y`,min:1,max:6,value:n,onChange:e=>{r(e),u(t=>[t[0],Math.min(t[1],e-1)])}}),(0,E.jsx)(f,{label:`blockDim.x`,min:1,max:64,value:i,onChange:o}),(0,E.jsx)(f,{label:`blockDim.y`,min:1,max:16,value:s,onChange:c}),(0,E.jsx)(v,{label:`Colour by warp`,value:_===`warp`,onChange:e=>b(e?`warp`:`x`)})]}),(0,E.jsxs)(h,{children:[`<<<dim3(`,e,`,`,n,`), dim3(`,x,`,`,s,`)>>> launches `,(0,E.jsx)(`b`,{children:a(e*n)}),` blocks × `,(0,E.jsx)(`b`,{children:S}),` threads = `,(0,E.jsx)(`b`,{children:a(e*n*S)}),` threads`,i!==x&&(0,E.jsxs)(`span`,{className:`w`,children:[` — blockDim.x clamped to `,x,`: a block holds at most 1024 threads`]}),`.`,w>0?(0,E.jsxs)(E.Fragment,{children:[` The last warp has `,(0,E.jsx)(`b`,{className:`r`,children:w}),` idle lanes (`,(100*w/(C*32)).toFixed(1),`% of issue slots wasted) — pick a multiple of 32.`]}):(0,E.jsxs)(E.Fragment,{children:[` Block size is a multiple of 32 → `,(0,E.jsx)(`b`,{className:`g`,children:`no partially-filled warps`}),`.`]})]})]})}function M(){let[e,t]=(0,y.useState)(40),[n,r]=(0,y.useState)(6),[i,a]=(0,y.useState)(2),[d,m]=(0,y.useState)(.4),[v,b]=(0,y.useState)(1),x=(0,y.useRef)(0),S=(0,y.useMemo)(()=>{let t=s(v*17),r=Array.from({length:e},()=>1+(t()*2-1)*d),a=Array.from({length:n*i},()=>0),o=r.map((e,t)=>{let n=0;for(let e=1;e<a.length;e++)a[e]<a[n]&&(n=e);let r=a[n];return a[n]+=e,{b:t,sm:Math.floor(n/i),slot:n%i,start:r,end:a[n],d:e}}),c=Math.max(...a),l=r.reduce((e,t)=>e+t,0);return{items:o,total:c,busy:l,ideal:l/(n*i)}},[e,n,i,d,v]),[C]=_(340,(e,t,r,a,s)=>{x.current=(x.current+s*S.total*.18)%(S.total*1.25);let d=t-64-20,f=Math.min(46,(r-70)/n),p=(f-6)/i,m=e=>64+e/(S.total*1.02)*d;for(let t=0;t<n;t++){let n=22+t*f;e.fillStyle=`#0f1220`,o(e,64,n,d,f-4,6),e.fill(),u(e,`SM ${t}`,56,n+f/2-2,{size:11,align:`right`,color:l.mute,mono:!0})}let h=n*i;S.items.forEach(t=>{let n=22+t.sm*f+1+t.slot*p,r=Math.floor(t.b/h),i=x.current>=t.end;e.fillStyle=x.current>=t.start&&!i?l.g:i?c([`#8b7bff`,`#22d3ee`,`#f472b6`,`#fbbf24`,`#4ade80`][r%5],.55):c(`#8b7bff`,.12),e.fillRect(m(t.start)+1,n+1,Math.max(1,m(t.end)-m(t.start)-2),p-3),m(t.end)-m(t.start)>22&&u(e,t.b,(m(t.start)+m(t.end))/2,n+p/2,{size:9.5,align:`center`,color:`#fff`,mono:!0})});let g=m(Math.min(x.current,S.total));e.strokeStyle=`#fff`,e.lineWidth=1.5,e.beginPath(),e.moveTo(g,16),e.lineTo(g,22+n*f),e.stroke(),e.strokeStyle=l.d,e.setLineDash([4,3]),e.beginPath(),e.moveTo(m(S.ideal),16),e.lineTo(m(S.ideal),22+n*f),e.stroke(),e.setLineDash([]),u(e,`ideal`,m(S.ideal)+4,r-28,{size:10.5,color:l.d,mono:!0}),u(e,`actual`,m(S.total)+4,r-12,{size:10.5,color:`#fff`,mono:!0}),u(e,`time →`,64+d,22+n*f+14,{size:10.5,align:`right`,color:l.dim})},{animate:!0}),w=S.busy/(S.total*n*i),T=Math.ceil(e/(n*i));return(0,E.jsxs)(E.Fragment,{children:[(0,E.jsx)(`canvas`,{...C}),(0,E.jsxs)(p,{children:[(0,E.jsx)(f,{label:`Thread blocks`,min:4,max:160,value:e,onChange:t}),(0,E.jsx)(f,{label:`SMs`,min:2,max:16,value:n,onChange:r}),(0,E.jsx)(f,{label:`Resident blocks / SM`,min:1,max:4,value:i,onChange:a}),(0,E.jsx)(f,{label:`Duration variance`,min:0,max:.9,step:.05,value:d,onChange:m,fmt:e=>e.toFixed(2)}),(0,E.jsx)(g,{onClick:()=>b(e=>e+1),children:`Re-roll durations`})]}),(0,E.jsxs)(h,{children:[e,` blocks on `,n,` SMs × `,i,` resident = `,n*i,` slots → `,(0,E.jsx)(`b`,{children:T}),` wave(s) · makespan `,(0,E.jsx)(`b`,{children:S.total.toFixed(2)}),` vs ideal `,S.ideal.toFixed(2),` · SM utilisation `,(0,E.jsxs)(`b`,{className:w>.9?`g`:`w`,children:[(100*w).toFixed(0),`%`]}),`. When `,e,` is not a multiple of `,n*i,`, the `,(0,E.jsx)(`b`,{children:`last wave leaves SMs idle (the “tail effect”)`}),`; blocks run in no guaranteed order and never wait for each other.`]})]})}function N(){return(0,E.jsx)(m,{views:[{id:`h`,label:`Grid → block → warp → thread (3D)`,render:()=>(0,E.jsx)(j,{})},{id:`s`,label:`Blocks scheduled onto SMs`,render:()=>(0,E.jsx)(M,{})}]})}var P={Lab:N,vizTitle:`Fly through a CUDA launch: grid, blocks, warps, threads — and the SMs that run them`,tryIt:[`Hover any cube: read its **threadIdx**, **global index** and which **warp / lane** it belongs to (colour = warp).`,`Set **blockDim = (20, 1)**: the single warp has 12 idle lanes. Set (32,1) or (16,2): none.`,"Click different **block tiles** on the left; the global index formula changes with `blockIdx`.",`In the scheduler tab choose 41 blocks on 4 SMs×2: see the tail wave leave most SMs idle.`],theory:b,math:x,practice:S,code:[{title:`Vector add: 1-D indexing, bounds check, grid-stride loop, event timing`,lang:`cuda`,src:C},{title:`2-D launch: RGB → grayscale on an image`,lang:`cuda`,src:w},{title:`Query your GPU: SMs, warp size, limits`,lang:`cuda`,src:T}],quiz:[{q:`What is the maximum number of threads in one thread block on current NVIDIA GPUs?`,options:[`256`,`512`,`1024`,`2048`],answer:2,why:`The hardware limit is 1024 threads per block (e.g. 32×32 or 1024×1).`},{q:`Threads of a warp execute:`,options:[`Independently on different SMs`,`In lockstep (SIMT), 32 at a time`,`Sequentially`,`In random order of blocks`],answer:1,why:`A warp is 32 consecutive threads issued together; divergence is handled by masking.`},{q:`Which statement about thread blocks is true?`,options:["Blocks of a grid can synchronise with `__syncthreads()`",`Blocks execute in a guaranteed order`,`Blocks are independent and may run in any order on any SM`,`A block spans several SMs`],answer:2,why:"`__syncthreads()` only syncs within a block; independence is what lets the same code scale across GPUs of any size."},{q:`For $N=1{,}000{,}000$ elements and 256 threads/block, the grid size should be:`,options:[`3906`,`3907`,`1000000`,`256`],answer:1,why:"$\\lceil10^6/256\\rceil=3907$; the last block needs a bounds check `if (i < N)`."},{q:`Why choose block sizes that are multiples of 32?`,options:[`The compiler requires it`,`To avoid partially filled warps that waste issue slots`,`To use more shared memory`,`To enable Tensor Cores`],answer:1,why:`A block of 100 threads still occupies 4 warps (128 lanes); 28 lanes do nothing.`}]};export{P as default};