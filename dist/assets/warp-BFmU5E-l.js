import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{g as r,n as i,t as a,y as o}from"./viz-CPys2405.js";import{t as s}from"./Stage3D-CWOT2b2l.js";import{a as c,c as l,i as u,l as d,o as f,s as p,t as m}from"./hooks-Dw7oo1m3.js";var h=e(t(),1),g="## The warp is the real instruction\n\nA CUDA thread is the program you write. The hardware runs a **warp**: 32 consecutive threads that share one instruction pointer. Lanes have their own registers and their own predicates, but they are issued together. That is SIMT — single instruction, multiple threads.\n\nThreads of a block are grouped with `threadIdx.x` varying fastest, then `y`, then `z`. Lane `ℓ` of a 1-D block is `threadIdx.x & 31`. The warp id is `threadIdx.x >> 5`.\n\nOn Volta and later the scheduler *can* move lanes independently (**independent thread scheduling**). The compiler still reconverges them, but you may not assume lockstep. Every warp primitive takes an explicit mask and ends in `_sync`. The examples below use the full mask `0xffffffff`, which is correct only when all 32 lanes actually reach the call.\n\n## Divergence\n\n```cuda\nif (threadIdx.x < 16)  a();   // lanes 0–15\nelse                   b();   // lanes 16–31\n```\n\nBoth arms run, one after the other. While `a()` is issued the odd half of the warp is masked off; then the mask flips for `b()`. A uniform warp (every lane takes the same arm) issues only that arm.\n\nThe cost is the **sum of the taken arms**, not the max. Sixteen lanes in a long `else` still make the other sixteen wait. Nested data-dependent branches multiply the number of passes. A four-way `switch (lane % 4)` issues four times and each pass is one quarter full.\n\n<div class=\"callout\">\n\n**Two rules that fall out of the mask.**\n\n1. `__syncthreads()` inside a branch that not every thread of the block takes is undefined. The missing threads never arrive.\n2. A `_sync` primitive waits for every lane in its mask. `if (valid) __shfl_sync(0xffffffff, …)` deadlocks when some lanes skip the call. Invalid lanes must still execute the shuffle; give them a zero and let them participate.\n\n</div>\n\nVery short bodies are often **predicated** instead of branched: both sides become ordinary instructions with a per-lane flag. The issue slot is still spent, but there is no reconvergence point to get wrong.\n\n## Shuffles\n\nA shuffle reads a register from another lane of the same warp. No shared memory, no barrier.\n\n| Intrinsic | Source lane | Typical use |\n|---|---|---|\n| `__shfl_sync(mask, v, src)` | absolute lane `src` | broadcast one lane |\n| `__shfl_up_sync(mask, v, Δ)` | `lane − Δ` | inclusive scan |\n| `__shfl_down_sync(mask, v, Δ)` | `lane + Δ` | reduction into lane 0 |\n| `__shfl_xor_sync(mask, v, maskBit)` | `lane XOR maskBit` | butterfly; every lane gets the result |\n\n`__shfl_down_sync` returns the caller's own value when `lane + Δ` falls outside the warp. The reduction below relies on that.\n\n## Reduction\n\nFive instructions reduce 32 values. With `__shfl_down_sync` and offsets 16, 8, 4, 2, 1, **lane 0** holds the sum. With `__shfl_xor_sync` and the same offsets, **every lane** holds it, because each step is a perfect pairing.\n\nA block reduction is that warp reduction plus one barrier:\n\n1. Each warp reduces its own registers. Lane 0 writes one partial to shared memory (at most 32 floats per block).\n2. `__syncthreads()`.\n3. The first warp loads those partials and reduces again. Thread 0 writes the block result.\n\nThat is one barrier per block, not the five barriers of a shared-memory tree. The grid then finishes with an atomic add or a second small kernel. `warp_reduce.cu` is the pattern; production sums call CUB.\n\n## Ballot\n\n`__ballot_sync(mask, pred)` returns a 32-bit word whose bit `ℓ` is lane `ℓ`'s predicate. `__popc` of the bits below you is your index among the lanes that passed — a warp-local stream compaction, done entirely in registers. `ballot.cu` uses it to pack the hits from one warp into a dense list.\n",_=`## What one divergent region costs

A warp has 32 lanes and a branch region with arms $A_1, \\ldots, A_k$. Arm $j$ has length $T_j$ (issue slots) and is **taken** when at least one lane enters it. The hardware issues each taken arm once, with the other lanes masked off:

$$
T = \\sum_{j : \\text{arm } j \\text{ taken}} T_j.
$$

A uniform warp pays only the arm it actually runs. If every arm has the same length $T_0$ and $P$ arms are taken,

$$
T = P\\, T_0, \\qquad \\text{lane utilization} = \\frac{1}{P}.
$$

Each lane does useful work in exactly one arm, but every arm occupies all 32 lanes of the issue slot.

| Predicate | Taken arms $P$ | Utilization |
|---|---|---|
| same arm for every lane | 1 | 100 % |
| \`threadIdx.x < 16\` | 2 | 50 % |
| \`threadIdx.x & 1\` | 2 | 50 % |
| \`threadIdx.x % 4\` | 4 | 25 % |

Unequal arms do not change the rule, only the weights. Sixteen lanes taking a body of length 1 and sixteen taking a body of length 20 cost $21$ slots. The short arm is almost free; the long one is paid in full.

Predication of a tiny body is the same arithmetic with $T_j$ equal to a few instructions and no join point. It does not make the inactive lanes free.

## Warp sum, five steps

Let $v_\\ell^{(0)}$ be lane $\\ell$'s register. The down-sweep with offset $\\Delta \\in \\{16, 8, 4, 2, 1\\}$ is

$$
v_\\ell \\leftarrow v_\\ell + v_{\\ell+\\Delta}
\\quad \\text{when } \\ell+\\Delta < 32,
$$

and $v_\\ell$ unchanged otherwise (that is what \`__shfl_down_sync\` returns past the end of the warp). After all five offsets, lane 0 holds $\\sum_{\\ell=0}^{31} v_\\ell^{(0)}$. Lane $k$ holds the sum of the original values from $k$ through $31$.

The xor (butterfly) sweep is a permutation at every step:

$$
v_\\ell \\leftarrow v_\\ell + v_{\\ell \\oplus \\Delta}.
$$

$\\ell \\oplus \\Delta$ is always a lane in $0 \\ldots 31$, so nobody is idle. Each step adds a disjoint partner, and after $\\Delta = 1$ **every** lane holds the full sum. Same five instructions, a different final distribution.

Check on the all-ones input: the down-sweep writes $2, 4, 8, 16, 32$ into lane 0, and the butterfly writes that same sequence into every lane.

## From a warp to a block

A block of $B$ threads has $W = B/32$ warps (use a multiple of 32). Each warp reduces independently, then one lane per warp stores a partial:

$$
s_w = \\sum_{\\ell=0}^{31} v_{32w+\\ell}, \\qquad w = 0 \\ldots W-1.
$$

Shared memory holds $W$ floats — 128 bytes at $B = 1024$. After one \`__syncthreads()\`, warp 0 reduces $(s_0, \\ldots, s_{W-1})$, padding with zeros so the shuffle mask stays a full warp. Thread 0 holds the block sum.

A shared-memory tree over the same block needs $\\log_2 B$ barriers (10 at $B = 1024$). The shuffle version needs **one**.

## The mask is part of the call

\`__shfl_*_sync(mask, …)\` and \`__ballot_sync(mask, …)\` wait until every lane whose bit is set in \`mask\` has executed the instruction. Two consequences:

- All 32 lanes must *reach* the call when \`mask = 0xffffffff\`. A divergent \`if\` around the shuffle leaves some lanes out, and the warp never completes the wait.
- A partial last warp (fewer than 32 live elements) still has 32 hardware lanes. The live lanes and the padding lanes both execute the primitive. Padding contributes $0$. The mask stays \`0xffffffff\` as long as the whole warp is converged, which it is when the surrounding kernel has no divergent branch around the reduction.

\`__ballot_sync\` packs 32 predicates into one unsigned int. The compaction index of lane $\\ell$ is the population count of the bits strictly below it:

$$
d_\\ell = \\operatorname{popcount}\\big( \\textit{ballot} \\mathbin{\\&} (2^\\ell - 1) \\big).
$$

Lanes with a false predicate do not write. The number of outputs from the warp is \`popcount(ballot)\`.
`,v=`## Exercises

**Q1.** A warp executes \`if (threadIdx.x < 16) heavy(); else heavy();\` and \`heavy\` is 40 instructions. How many instruction issues does the warp spend inside the region, and what is the lane utilization?

<details>
<summary>Show answer</summary>

Both arms are taken, so the warp issues $40 + 40 = 80$ instructions. Each issue does useful work on only 16 lanes, so utilization is **50 %**. A uniform warp would have issued 40.

</details>

**Q2.** Same region, but the predicate is \`threadIdx.x < 32\` (always true). How many issues?

<details>
<summary>Show answer</summary>

Only the \`if\` arm is taken. The \`else\` is never issued. **40** instructions, utilization **100 %**. Divergence is about taken paths, not about the source text containing an \`else\`.

</details>

**Q3.** \`switch (threadIdx.x % 4)\` has four arms of 10 instructions, and every residue occurs in the warp. Issues and utilization?

<details>
<summary>Show answer</summary>

$P = 4$, so $T = 40$ issues and utilization is $1/4 = 25\\%$. Eight lanes are active on each pass.

</details>

**Q4.** All-ones registers, \`__shfl_down_sync\` with offsets 16, 8, 4, 2, 1. What does lane 0 hold after the offset-8 step, and after the full sweep?

<details>
<summary>Show answer</summary>

After offset 16, lanes 0–15 hold 2. After offset 8, lanes 0–7 hold **4**. After all five offsets lane 0 holds **32**, the sum of the warp. Lane 31 still holds 1: nothing was ever added into it.

</details>

**Q5.** Same input, \`__shfl_xor_sync\` with the same offsets. What does lane 31 hold at the end, and how many lanes were idle on the last step?

<details>
<summary>Show answer</summary>

Every lane holds **32**. The partner of lane $\\ell$ is $\\ell \\oplus 1$, which is always inside the warp, so **no lane is idle** on any step. That is the difference from the down-sweep.

</details>

**Q6.** A block of 256 threads reduces with the shuffle pattern. How many \`__syncthreads()\` calls, and how many shared-memory floats?

<details>
<summary>Show answer</summary>

256 / 32 = 8 warps. Each warp's lane 0 writes one partial: **8 floats**. Then **one** \`__syncthreads()\`, and warp 0 reduces those 8 values (lanes 8–31 of that warp add zeros). A shared-memory tree would have needed 8 barriers.

</details>

**Q7.** Why is this deadlock, even though the arithmetic looks right?

\`\`\`cuda
if (tid < n)
    v += __shfl_down_sync(0xffffffff, v, 16);
\`\`\`

<details>
<summary>Show answer</summary>

\`0xffffffff\` means "wait for all 32 lanes". Lanes with \`tid >= n\` never execute the shuffle, so the wait cannot complete. Every lane of the warp must call the primitive. The lanes past \`n\` should pass \`0\` and participate.

</details>

**Q8.** A ballot returns \`0b0000…0101\` (bits 0 and 2 set). What compaction indices do lanes 0 and 2 get, and how many outputs does the warp write?

<details>
<summary>Show answer</summary>

Lane 0 has no set bits below it, so its index is **0**. Lane 2 has one set bit below it (bit 0), so its index is **1**. \`popcount\` of the word is **2**, which is the number of writes.

</details>
`,y=`// nvcc -O3 -arch=sm_80 divergence.cu -o divergence
//
// Two kernels do the same arithmetic. uniform() is one path for the whole
// warp. split() takes both arms, so the warp issues the heavy loop twice.
// Time them with Nsight Compute (smsp__thread_inst_executed vs eligible
// warps). The source difference is only the predicate.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void uniform(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                 // the tail warp may be partial; that is fine
    float x = a[i];                     // this branch is uniform inside a full warp
    if ((blockIdx.x & 1) == 0) {        // predicate is the same for every lane
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) x = x * 1.0001f + 0.001f;
    } else {
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) x = x * 0.9999f - 0.001f;
    }
    a[i] = x;
}

__global__ void split(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    if ((threadIdx.x & 1) == 0) {       // even and odd lanes both exist in every warp
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) x = x * 1.0001f + 0.001f;
    } else {
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) x = x * 0.9999f - 0.001f;
    }
    a[i] = x;
}

// Do NOT write this. The else-lanes never reach the barrier:
//
//   if (threadIdx.x < 16) { ... __syncthreads(); }
//   else                  { ... __syncthreads(); }
//
// A block barrier must be textually reached by every thread of the block,
// on every iteration, with no divergent guard around it.

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
} while (0)

int main() {
    const int n = 1 << 20;
    float *a;
    CHECK(cudaMalloc(&a, n * sizeof(float)));
    CHECK(cudaMemset(a, 0, n * sizeof(float)));
    int block = 256;
    int grid = (n + block - 1) / block;
    uniform<<<grid, block>>>(a, n);
    split<<<grid, block>>>(a, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(a));
    printf("launched uniform and split over %d threads\\n", n);
    return 0;
}
`,b=`// nvcc -O3 -arch=sm_80 warp_reduce.cu -o warp_reduce
//
// Block sum. Each warp reduces in registers (5 shuffles, no barrier).
// Lane 0 of each warp writes one partial, the block synchronises once,
// and warp 0 reduces the partials. Every lane of the warp must execute
// the shuffles — including lanes that have no element left.

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ float warp_sum_down(float v) {
    // Result is valid in lane 0 only.
    const unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(mask, v, offset);
    return v;
}

__device__ float warp_sum_all(float v) {
    // Result is valid in every lane. Partner is lane ^ offset, always in range.
    const unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(mask, v, offset);
    return v;
}

__global__ void block_sum(const float* in, float* partial, int n) {
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        v += in[i];

    // Full mask: padding lanes hold 0 and still have to reach the shuffle.
    v = warp_sum_down(v);

    __shared__ float smem[32];                 // at most 32 warps in a block
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    int nwarps = blockDim.x >> 5;
    float w = (lane < nwarps) ? smem[lane] : 0.f;
    if (warp == 0) w = warp_sum_down(w);
    if (threadIdx.x == 0) partial[blockIdx.x] = w;
}

// warp_sum_all is the same five steps when the caller needs the sum
// on every lane (a broadcast without a sixth shuffle).

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
} while (0)

int main() {
    const int n = 1 << 16;
    const int block = 256;
    const int grid = 64;
    float *d_in, *d_part, *h_in, *h_part;
    CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK(cudaMalloc(&d_part, grid * sizeof(float)));
    h_in = new float[n];
    h_part = new float[grid];
    double cpu = 0.0;
    for (int i = 0; i < n; ++i) { h_in[i] = 1.f; cpu += 1.0; }
    CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    block_sum<<<grid, block>>>(d_in, d_part, n);
    CHECK(cudaMemcpy(h_part, d_part, grid * sizeof(float), cudaMemcpyDeviceToHost));
    double gpu = 0.0;
    for (int i = 0; i < grid; ++i) gpu += h_part[i];
    printf("cpu %.0f  gpu %.0f  %s\\n", cpu, gpu, std::fabs(cpu - gpu) < 1e-2 ? "ok" : "MISMATCH");

    delete[] h_in; delete[] h_part;
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_part));
    return 0;
}
`,x=`// nvcc -O3 -arch=sm_80 ballot.cu -o ballot
//
// Warp-local stream compaction. __ballot_sync packs the 32 predicates
// into one word; popc of the bits below this lane is the dense index.
// No shared memory. The warp writes popc(bits) outputs, tightly packed.

#include <cstdio>
#include <cuda_runtime.h>

__device__ void warp_compact(int pred, int value, int* out, int* counter) {
    const unsigned mask = 0xffffffffu;
    // Every lane calls ballot, whether or not pred is true.
    unsigned bits = __ballot_sync(mask, pred);
    int lane = threadIdx.x & 31;
    int idx = __popc(bits & ((1u << lane) - 1u));   // how many hits are below me
    int total = __popc(bits);

    __shared__ int base;
    if (lane == 0) base = atomicAdd(counter, total);
    __syncwarp(mask);                               // publish \`base\` to the rest of this warp
    if (pred) out[base + idx] = value;
}

// One warp per block so the shared \`base\` does not need a block barrier.
// A multi-warp block would stage per-warp totals, __syncthreads(), then prefix them.
__global__ void compact_positive(const int* in, int* out, int* counter, int n) {
    int i = blockIdx.x * 32 + (threadIdx.x & 31);
    int pred = 0, value = 0;
    if (i < n) { value = in[i]; pred = value > 0; }
    warp_compact(pred, value, out, counter);
}

#define CHECK(cmd) do { \\
    cudaError_t e = (cmd); \\
    if (e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
        return 1; \\
    } \\
} while (0)

int main() {
    const int n = 1024;
    int *h = new int[n];
    int expect = 0;
    for (int i = 0; i < n; ++i) { h[i] = (i % 3 == 0) ? i : -i; if (h[i] > 0) ++expect; }
    // i == 0 is not > 0, so expect counts 3,6,...,1023 → 341.

    int *d_in, *d_out, *d_count;
    CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    CHECK(cudaMalloc(&d_count, sizeof(int)));
    CHECK(cudaMemcpy(d_in, h, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_count, 0, sizeof(int)));

    compact_positive<<<(n + 31) / 32, 32>>>(d_in, d_out, d_count, n);
    int got = 0;
    CHECK(cudaMemcpy(&got, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("positives: gpu %d  cpu %d  %s\\n", got, expect, got == expect ? "ok" : "MISMATCH");

    delete[] h;
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_count));
    return 0;
}
`,S=n(),C=32,w=[a.e,a.d,a.c,a.b];function T(e,t){let n=e=>Array.from({length:C},(t,n)=>e(n));if(e===`all`)return[{name:`A`,color:a.e,mask:n(()=>!0)}];if(e===`cut`){let e=[],r=n(e=>e<t),i=r.map(e=>!e);return r.some(Boolean)&&e.push({name:`then`,color:a.e,mask:r}),i.some(Boolean)&&e.push({name:`else`,color:a.d,mask:i}),e}return e===`even`?[{name:`even`,color:a.e,mask:n(e=>e%2==0)},{name:`odd`,color:a.d,mask:n(e=>e%2==1)}]:[0,1,2,3].map(e=>({name:`%`+e,color:w[e],mask:n(t=>t%4===e)}))}function E(){let[e,t]=(0,h.useState)(`cut`),[n,i]=(0,h.useState)(16),[s,d]=(0,h.useState)(8),g=(0,h.useMemo)(()=>T(e,n),[e,n]),_=g.length*s,[v]=m(360,(e,t)=>{let n=t-36-12,i=Math.min(18,(n-93)/32),c=(n-32*i)/31,l=e=>36+e*(i+c);o(e,`which arm each lane takes`,36,16,{size:12,color:a.mute,weight:600}),o(e,`${g.length} pass${g.length>1?`es`:``} · ${_} issues`,t-12,16,{size:13,align:`right`,weight:700,color:g.length===1?a.e:g.length===2?a.d:a.r}),g.forEach(t=>t.mask.forEach((n,a)=>{n&&(e.fillStyle=t.color,r(e,l(a),32,i,26,3),e.fill())})),o(e,`0`,l(0),70,{size:10,color:a.mute,mono:!0}),o(e,`31`,l(31)+i,70,{size:10,color:a.mute,mono:!0,align:`right`}),o(e,`issued one arm at a time  ·  dim lanes are masked off`,36,96,{size:12,color:a.mute,weight:600}),g.forEach((t,n)=>{let s=112+n*42;o(e,t.name,4,s+16,{size:10,color:a.mute,align:`left`});for(let n=0;n<C;n++)e.fillStyle=t.mask[n]?t.color:`#1a1e32`,r(e,l(n),s,i,32,3),e.fill();let c=t.mask.filter(Boolean).length;o(e,c+` live`,l(31)+i+4,s+16,{size:10,color:a.dim,align:`left`})});let u=112+g.length*42+8;o(e,`each row is ${s} instruction${s>1?`s`:``} issued for the whole warp`,36,Math.min(344,u),{size:11,color:a.mute})});return(0,S.jsxs)(S.Fragment,{children:[(0,S.jsx)(`canvas`,{...v}),(0,S.jsxs)(u,{children:[(0,S.jsx)(p,{label:`Predicate`,value:e,onChange:t,options:[[`all`,`every lane takes A`],[`cut`,`lane < K`],[`even`,`even / odd`],[`mod4`,`lane % 4`]]}),e===`cut`&&(0,S.jsx)(l,{label:`K`,min:0,max:32,value:n,onChange:i}),(0,S.jsx)(l,{label:`Arm length`,min:1,max:20,value:s,onChange:d,fmt:e=>e+` inst`})]}),(0,S.jsx)(c,{items:g.map(e=>[e.color,e.name])}),(0,S.jsxs)(f,{children:[`The warp issues `,(0,S.jsx)(`b`,{className:g.length===1?`g`:`r`,children:_}),` instructions for this region`,g.length===1?(0,S.jsx)(S.Fragment,{children:`, one arm, every lane live.`}):(0,S.jsxs)(S.Fragment,{children:[`. A uniform warp would issue `,(0,S.jsx)(`b`,{children:s}),`. Lane utilization is `,(0,S.jsxs)(`b`,{children:[Math.round(100/g.length),`%`]}),`.`]}),e===`cut`&&n>0&&n<32&&(0,S.jsxs)(S.Fragment,{children:[` `,n,` lanes take `,(0,S.jsx)(`b`,{children:`then`}),`, `,32-n,` take `,(0,S.jsx)(`b`,{children:`else`}),`. Both arms are paid in full.`]}),e===`cut`&&(n===0||n===32)&&(0,S.jsx)(S.Fragment,{children:` The predicate is uniform, so the empty arm is never issued.`}),e===`mod4`&&(0,S.jsx)(S.Fragment,{children:` Eight lanes are live on each pass. A four-way split costs four issue passes.`})]})]})}function D({paths:e,step:t}){let n=e[Math.min(t,e.length-1)];return(0,S.jsx)(`group`,{children:Array.from({length:C},(e,t)=>{let r=n.mask[t],i=t%16,a=t<16?0:1,o=(i-7.5)*.46;return(0,S.jsxs)(`mesh`,{position:[o,r?.28:0,a===0?.5:-.5],children:[(0,S.jsx)(`boxGeometry`,{args:[.34,r?.7:.22,.34]}),(0,S.jsx)(`meshStandardMaterial`,{color:r?n.color:`#232838`,emissive:r?n.color:`#000`,emissiveIntensity:r?.55:0,roughness:.45})]},t)})})}function O(){let[e,t]=(0,h.useState)(`cut`),[n,r]=(0,h.useState)(16),[i,a]=(0,h.useState)(0),o=(0,h.useMemo)(()=>T(e,n),[e,n]),c=Math.min(i,o.length-1),d=o[c],m=d.mask.filter(Boolean).length,g=(0,S.jsxs)(S.Fragment,{children:[`pass `,(0,S.jsx)(`b`,{style:{color:d.color},children:c+1}),` of `,o.length,` · `,`arm `,(0,S.jsx)(`b`,{style:{color:d.color},children:d.name}),(0,S.jsx)(`br`,{}),m,` lanes live · `,C-m,` masked off`]});return(0,S.jsxs)(S.Fragment,{children:[(0,S.jsx)(s,{height:420,camera:[0,4.6,9.5],target:[0,.2,0],fov:42,overlay:g,hint:`drag to orbit · tall boxes are the lanes issued on this pass`,children:(0,S.jsx)(D,{paths:o,step:c})}),(0,S.jsxs)(u,{children:[(0,S.jsx)(p,{label:`Predicate`,value:e,onChange:e=>{t(e),a(0)},options:[[`all`,`every lane takes A`],[`cut`,`lane < K`],[`even`,`even / odd`],[`mod4`,`lane % 4`]]}),e===`cut`&&(0,S.jsx)(l,{label:`K`,min:0,max:32,value:n,onChange:e=>{r(e),a(0)}}),(0,S.jsx)(l,{label:`Pass`,min:0,max:Math.max(0,o.length-1),value:c,onChange:a,fmt:e=>e+1+` / `+o.length})]}),(0,S.jsx)(f,{children:o.length===1?(0,S.jsx)(S.Fragment,{children:`One pass. All 32 lanes stay tall — there is nothing to mask off.`}):(0,S.jsx)(S.Fragment,{children:`Step the pass slider. Each pass stands up a different subset, and the warp waits until every pass has been issued.`})})]})}function k(e){return e===`ones`?Array.from({length:C},()=>1):e===`lane`?Array.from({length:C},(e,t)=>t):Array.from({length:C},(e,t)=>t*3%7)}function A(e,t){let n=[{vals:t.slice(),delta:null}],r=t.slice();for(let t=16;t>=1;t>>=1){let i=r.slice();for(let n=0;n<C;n++){let a=e===`xor`?n^t:n+t;a<C&&(i[n]=r[n]+r[a])}r=i,n.push({vals:r.slice(),delta:t})}return n}var j=e=>e===0?`start`:`Δ `+(32>>e);function M(){let[e,t]=(0,h.useState)(`down`),[n,s]=(0,h.useState)(`ones`),[d,g]=(0,h.useState)(0),[_,v]=(0,h.useState)(0),y=(0,h.useMemo)(()=>k(n),[n]),b=(0,h.useMemo)(()=>A(e,y),[e,y]),x=b[d],w=y.reduce((e,t)=>e+t,0),T=x.delta==null?null:e===`xor`?_^x.delta:_+x.delta,E=T!=null&&T<C,[D]=m(390,(t,n)=>{let s=Math.min(34,(n-16-8-60)/16),c=e=>{let t=e%16,n=e<16?0:1;return{x:16+t*(s+4),y:86+n*92,cw:s,ch:48}};o(t,e===`xor`?`__shfl_xor_sync`:`__shfl_down_sync`,16,16,{size:12,color:a.mute,weight:600});let l=x.delta==null?`initial registers`:`after Δ = ${x.delta}`;o(t,l,n-12,16,{size:13,color:a.b,align:`right`,weight:700}),o(t,`warp sum ${w}`,16,36,{size:12,color:a.ink,mono:!0});let u=Math.max(...x.vals,1);for(let e=0;e<C;e++){let{x:n,y:l}=c(e),d=e===_||E&&e===T;t.fillStyle=d?i(a.a,.95):i(a.b,.18+.55*(x.vals[e]/u)),r(t,n,l,s,48,4),t.fill(),o(t,String(e),n+s/2,l-8,{size:9,color:a.dim,align:`center`,mono:!0}),o(t,String(x.vals[e]),n+s/2,l+24,{size:s<26?9:12,color:d?`#0a0c14`:a.ink,align:`center`,weight:700,mono:!0})}if(E){let e=c(T),n=c(_),r=e.x+s/2,i=e.y+48+2,o=n.x+s/2,l=n.y-2;t.strokeStyle=a.c,t.lineWidth=1.5,t.beginPath(),t.moveTo(r,i),t.bezierCurveTo(r,i+16,o,l-16,o,l),t.stroke()}let d=x.delta==null?`step forward to add a partner into each lane`:E?`lane ${_} adds lane ${T}`:`lane ${_} is past the warp — it keeps its own value`;o(t,d,16,254,{size:12,color:E?a.c:a.mute})}),O=d===b.length-1,M=x.vals[0],N=x.vals.every(e=>e===w);return(0,S.jsxs)(S.Fragment,{children:[(0,S.jsx)(`canvas`,{...D}),(0,S.jsxs)(u,{children:[(0,S.jsx)(p,{label:`Primitive`,value:e,onChange:e=>{t(e),g(0)},options:[[`down`,`shfl_down → lane 0`],[`xor`,`shfl_xor → every lane`]]}),(0,S.jsx)(p,{label:`Input`,value:n,onChange:e=>{s(e),g(0)},options:[[`ones`,`all ones`],[`lane`,`lane index`],[`mix`,`small mix`]]}),(0,S.jsx)(l,{label:`Step`,min:0,max:5,value:d,onChange:g,fmt:j}),(0,S.jsx)(l,{label:`Watch lane`,min:0,max:31,value:_,onChange:v})]}),(0,S.jsx)(c,{items:[[a.a,`watched lane and its partner`],[a.b,`register value (brighter = larger)`]]}),(0,S.jsxs)(f,{children:[x.delta==null&&(0,S.jsx)(S.Fragment,{children:`Five steps, offsets 16, 8, 4, 2, 1. No shared memory and no barrier.`}),x.delta!=null&&e===`down`&&(0,S.jsxs)(S.Fragment,{children:[`Lane `,_,` `,E?(0,S.jsxs)(S.Fragment,{children:[`reads lane `,(0,S.jsx)(`b`,{children:T}),` and adds it.`]}):(0,S.jsx)(S.Fragment,{children:`has no source lane, so the shuffle returns its own value.`})]}),x.delta!=null&&e===`xor`&&(0,S.jsxs)(S.Fragment,{children:[`Lane `,_,` reads lane `,(0,S.jsx)(`b`,{children:_^x.delta}),` (`,(0,S.jsxs)(`b`,{children:[_,` XOR `,x.delta]}),`). Every partner lands inside the warp.`]}),O&&e===`down`&&(0,S.jsxs)(S.Fragment,{children:[` Lane 0 holds `,(0,S.jsx)(`b`,{className:M===w?`g`:`r`,children:M}),M===w?`, the full sum.`:`.`,` The other lanes hold suffixes, not the total.`]}),O&&e===`xor`&&N&&(0,S.jsxs)(S.Fragment,{children:[` Every lane holds `,(0,S.jsx)(`b`,{className:`g`,children:w}),`. The butterfly broadcasts the sum for free.`]})]})]})}function N(){return(0,S.jsx)(d,{views:[{id:`d`,label:`Divergence`,render:()=>(0,S.jsx)(E,{})},{id:`3`,label:`3D: masked passes`,render:()=>(0,S.jsx)(O,{})},{id:`s`,label:`Shuffle reduction`,render:()=>(0,S.jsx)(M,{})}]})}var P={Lab:N,vizTitle:`Watch a warp issue each arm, then reduce 32 registers in five shuffles`,tryIt:[`Predicate **lane < K**, K = 16. Two rows light up, 16 lanes each. The issue count is twice the arm length.`,`Drag **K** to 0, then to 32. One row disappears — a uniform predicate never issues the empty arm.`,`Switch to **lane % 4**. Four passes, eight lanes live on each, utilization 25%.`,`Open **3D** and step **Pass**. Each pass stands up a different subset of the warp.`,`Open **Shuffle reduction**, input **all ones**. Step Δ 16 through Δ 1. Lane 0 grows 2, 4, 8, 16, 32.`,`Switch the primitive to **shfl_xor**. The same five steps leave 32 in every lane, and the watched lane always has a partner.`],theory:g,math:_,practice:v,code:[{title:`Uniform predicate vs even/odd split`,lang:`cuda`,note:`Same arithmetic, different predicate. The split warp issues the heavy loop twice. A __syncthreads inside one arm is undefined.`,src:y},{title:`Block sum with five shuffles and one barrier`,lang:`cuda`,note:`warp_sum_down leaves the result in lane 0. warp_sum_all leaves it in every lane. Padding lanes must still execute the shuffle.`,src:b},{title:`Compact a warp with ballot and popc`,lang:`cuda`,note:`The dense index of a lane is the number of set bits below it. The warp writes popc(ballot) outputs and no shared-memory scan.`,src:x}],quiz:[{q:`A warp takes both arms of an if/else, and each arm is 10 instructions. How many instructions does the warp issue?`,options:[`10`,`20`,`320`,`32`],answer:1,why:`Taken arms are issued one after another. $T = 10 + 10$. Inactive lanes are masked, not skipped.`},{q:`Which predicate does **not** diverge a full warp?`,options:[`threadIdx.x & 1`,`threadIdx.x < 16`,`blockIdx.x == 0`,`threadIdx.x % 4`],answer:2,why:`blockIdx.x is the same for every lane of the warp. The other three take more than one arm inside a single warp.`},{q:`After a full __shfl_down_sync reduction, where is the sum of the warp?`,options:[`In every lane`,`In lane 0 only`,`In shared memory`,`In lane 31`],answer:1,why:`Each lane adds the lane Δ above it. Only lane 0 has absorbed all 31 neighbours. Lane 31 never gains a source.`},{q:`What does the xor (butterfly) reduction produce that the down-sweep does not?`,options:[`A sum in every lane`,`Fewer than five steps`,`A shared-memory write`,`A block barrier`],answer:0,why:`lane ^ Δ stays inside the warp, so every lane accumulates the full sum. Still five steps, still no shared memory.`},{q:`A 256-thread block reduces with warp shuffles. How many __syncthreads() calls does the reduction need?`,options:[`0`,`1`,`5`,`8`],answer:1,why:`Warps reduce independently. One barrier publishes the per-warp partials, then warp 0 reduces those 8 values.`},{q:`Why can if (tid < n) __shfl_down_sync(0xffffffff, v, 16) hang?`,options:[`Shuffles cannot add`,`The mask waits for lanes that never reach the call`,`n must be a power of two`,`Down-sweeps require shared memory`],answer:1,why:`0xffffffff means all 32 lanes participate. Lanes that skip the call never arrive. They must execute the shuffle too, contributing 0.`},{q:`__ballot_sync returns 0b1010 (bits 1 and 3). The compaction index of lane 3 is:`,options:[`0`,`1`,`3`,`4`],answer:1,why:`Index = popcount of bits below lane 3. Only bit 1 is set below bit 3, so the index is 1.`}]};export{P as default};