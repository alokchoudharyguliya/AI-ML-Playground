import{r as e}from"./rolldown-runtime-hePW80VL.js";import{a as t,b as n,c as r,h as i,n as a,v as o}from"./r3f-x1z21uF6.js";import{c as s,f as c,g as l,i as u,n as d,p as f,s as p,t as m,y as h}from"./viz-CPys2405.js";import{n as g,t as _}from"./Md-CSdqn6Ho.js";import{t as v}from"./Stage3D-CWOT2b2l.js";import{a as y,c as b,i as x,l as S,n as C,o as w,r as T,s as E,t as D,u as O}from"./hooks-Dw7oo1m3.js";var k=e(n(),1),A={a100:{name:`A100 80GB SXM`,arch:`Ampere · sm_80`,sms:108,clk:1.41,lanes:64,regKB:256,smemKB:164,l1KB:192,l2MB:40,hbmGB:80,memType:`HBM2e`,bw:2039,l2bw:5500,peak:{fp64:9.7,fp32:19.5,tf32:156,fp16:312},lat:{reg:4,smem:23,l1:33,l2:200,hbm:500}},h100:{name:`H100 SXM`,arch:`Hopper · sm_90`,sms:132,clk:1.98,lanes:128,regKB:256,smemKB:228,l1KB:256,l2MB:50,hbmGB:80,memType:`HBM3`,bw:3350,l2bw:7500,peak:{fp64:34,fp32:67,tf32:494,fp16:989},lat:{reg:4,smem:29,l1:37,l2:260,hbm:580}},rtx4090:{name:`RTX 4090`,arch:`Ada · sm_89`,sms:128,clk:2.52,lanes:128,regKB:256,smemKB:100,l1KB:128,l2MB:72,hbmGB:24,memType:`GDDR6X`,bw:1008,l2bw:5e3,peak:{fp64:1.29,fp32:82.6,tf32:82.6,fp16:165},lat:{reg:4,smem:23,l1:35,l2:270,hbm:520}}},j=Object.entries(A).map(([e,t])=>[e,t.name]),M=e=>e>=1e3?`${(e/1e3).toFixed(e>=1e4?0:e>=5e3?1:2)} TB/s`:`${Math.round(e)} GB/s`;function N(e){let t=1024;return[{id:`reg`,name:`Registers`,color:`#f472b6`,where:`register file inside each SM sub-core`,capPerSM:e.regKB*t,capTotal:e.regKB*t*e.sms,capText:`${e.regKB} KB per SM · ${p(e.regKB*t*e.sms)} chip-wide`,lat:e.lat.reg,bw:e.sms*e.lanes*12*e.clk,scope:`one thread`,life:`thread`,kw:`automatic local variables`,note:`Bandwidth is operand bandwidth (3 × 4 B per lane per cycle). Latency shown is a dependent FMA.`},{id:`smem`,name:`Shared memory`,color:`#fbbf24`,where:`on-chip SRAM, software-managed`,capPerSM:e.smemKB*t,capTotal:e.smemKB*t*e.sms,capText:`up to ${e.smemKB} KB per SM (${p(e.smemKB*t*e.sms)} chip-wide)`,lat:e.lat.smem,bw:e.sms*128*e.clk,scope:`thread block`,life:`block`,kw:`__shared__ / extern __shared__`,note:`32 banks × 4 B per cycle per SM.`},{id:`l1`,name:`L1 cache`,color:`#76d12a`,where:`same SRAM array as shared memory, hardware-managed`,capPerSM:e.l1KB*t,capTotal:e.l1KB*t*e.sms,capText:`${e.l1KB} KB per SM shared with shared memory`,lat:e.lat.l1,bw:e.sms*128*e.clk,scope:`one SM`,life:`transparent`,kw:`automatic (ld.ca / __ldg path)`,note:`The L1/shared split is configurable per kernel (carveout).`},{id:`l2`,name:`L2 cache`,color:`#22d3ee`,where:`chip-wide, sliced across memory partitions`,capPerSM:null,capTotal:e.l2MB*1048576,capText:`${e.l2MB} MB, shared by all ${e.sms} SMs`,lat:e.lat.l2,bw:e.l2bw,scope:`whole GPU`,life:`transparent`,kw:`automatic (cudaAccessPolicyWindow for persistence)`,note:`All global traffic passes through L2; it is also where atomics execute.`},{id:`hbm`,name:e.memType===`GDDR6X`?`GDDR6X (global)`:`HBM (global)`,color:`#8b7bff`,where:`off-chip ${e.memType} DRAM`,capPerSM:null,capTotal:e.hbmGB*1073741824,capText:`${e.hbmGB} GB ${e.memType}`,lat:e.lat.hbm,bw:e.bw,scope:`whole GPU`,life:`until cudaFree / app exit`,kw:`cudaMalloc / __device__ / local-memory backing`,note:`Highest capacity, lowest bandwidth-per-byte and highest latency on the chip.`}]}var P=`## The memory wall

In the last chapter you saw that a GPU hides latency with thousands of threads. This chapter asks the follow-up: *what are those threads waiting for?* Almost always: **data**.

Compute throughput has grown much faster than memory bandwidth. From A100 to H100 the peak FP32 rate went from 19.5 to 67 TFLOP/s (**3.4×**) and the FP16 Tensor rate from 312 to 989 TFLOP/s (**3.2×**), while HBM bandwidth went from about 2.0 to 3.35 TB/s (**1.6×**). Each generation, a fixed kernel gets *more* memory-bound, not less. Moving a 32-bit word from DRAM also costs on the order of **100× more energy** than the floating-point operation you do with it.

So the central skill in GPU optimisation is not "make the math faster". It is **"move fewer bytes, and move them from closer."**

## The ladder

Every byte a thread uses lives somewhere on this ladder. Faster means smaller, closer to the ALUs, and shared by fewer threads.

| Space | Physical location | Scope | Lifetime | Approx. latency | How you get it |
|---|---|---|---|---|---|
| **Register** | register file in each SM sub-core (256 KB/SM) | one thread | thread | ~4 cycles | automatic locals |
| **Shared memory** | on-chip SRAM in the SM | one block | block | ~20–30 cycles | \`__shared__\` |
| **L1 cache** | *same* SRAM array as shared memory | one SM | transparent | ~30–40 cycles | automatic |
| **Constant** | device memory + small constant cache | whole grid (read-only) | application | cache hit ≈ L1-like, broadcast | \`__constant__\` |
| **Local** | **device memory** (cached in L1/L2) | one thread | thread | like global | dynamic-index arrays, spills |
| **L2 cache** | chip-wide SRAM (40 MB A100, 50 MB H100) | whole GPU | transparent | ~200–270 cycles | automatic |
| **Global (HBM/GDDR)** | off-chip DRAM stacks | whole GPU | until \`cudaFree\` | ~400–600 cycles | \`cudaMalloc\`, \`__device__\` |
| **Host over PCIe / NVLink** | CPU DRAM | CPU + GPU | allocation | microseconds | \`cudaMemcpy\`, pinned / managed |

Latencies are *approximate*, from public microbenchmark studies; they depend on clock, access pattern and driver. The point is the **order of magnitude**: from registers to HBM is roughly a **100× latency** gap, and about **50×** in bandwidth.

<div class="callout">

**Scope ≠ location.** *Local* memory is private to a thread, yet physically lives in slow device memory. *Constant* memory is read-only and global in scope, yet is served by a cache. Always ask two questions: *who can see it?* and *where are the bytes?*

</div>

Bandwidth also falls down the ladder, and the numbers are aggregate across the whole GPU. On an A100, registers can deliver on the order of 100 TB/s of operands, shared memory/L1 about 19 TB/s, L2 about 5 TB/s, HBM about 2 TB/s. Open the **3D** and **latency** tabs of the lab to see both axes at once, and switch to an H100 or RTX 4090.

## Anatomy of one global load

When a warp executes \`y = x[i]\` where \`x\` is in global memory:

1. The **load/store unit** collects the 32 lane addresses and merges them into the minimum number of **128-byte cache lines**, tracked as **32-byte sectors**. How well this merging works is *coalescing* (next chapters).
2. **L1** is checked. A hit returns in tens of cycles. L1 is *write-through*: stores always continue to L2.
3. On a miss the request crosses the on-chip network to the right **L2 slice**. L2 is write-back and is the point of coherence for the whole GPU: atomics execute here.
4. On an L2 miss, the **memory controller** fetches the sectors from an **HBM stack** (or GDDR chip).
5. Data returns up the chain; the warp, which was *parked* the entire time, becomes eligible again.

That 5-step round trip is the 400–600 cycles. Other warps run meanwhile. This is exactly the latency hiding from the *Execution Model* chapter, now with a price tag attached.

## Registers

The register file is the largest and fastest storage on the SM: 64K 32-bit registers per SM (256 KB), more than the L1/shared array. Each thread gets up to **255** registers.

- **Registers are not addressable.** Use \`float a[4]\` with only compile-time-constant indices (e.g. after \`#pragma unroll\`) and the compiler keeps it in registers. Index it with a runtime value and the array moves to **local memory**.
- If a kernel needs more registers than are allowed, values **spill** to local memory (slow, and visible in \`nvcc -Xptxas -v\` as *spill stores/loads*).
- Registers are allocated per thread for the **lifetime of the block**. More registers per thread means fewer resident warps (the *Occupancy* chapter).

## Shared memory and L1

Shared memory and L1 are carved from one physical array per SM (192 KB on A100, 256 KB on H100). You choose the split per kernel:

\`\`\`cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50);          // percent hint
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);     // opt in above 48 KB
\`\`\`

Shared memory is a **software-managed scratchpad**: you decide what is cached and when. It is organised into **32 banks** (4 B wide). Its payoff is *data reuse inside a block*, covered in **Tiled MatMul**; its pitfall is *bank conflicts*, covered in **Coalescing & Bank Conflicts**.

## Constant and read-only memory

\`__constant__\` data (64 KB total) is stored in device memory but read through a small **constant cache** that can **broadcast** one value to all 32 lanes in a single access. It is ideal for coefficients and small tables that every thread reads *uniformly*, and a poor fit when lanes read different addresses (that serialises).

For large read-only arrays, declaring pointers \`const __restrict__\` or loading with \`__ldg()\` routes loads through the read-only data path, which lets the compiler assume no aliasing.

## L2 and HBM

**L2** (40 MB on A100, 50 MB on H100, 72 MB on RTX 4090) is shared by all SMs. If your *working set* fits in L2, repeated passes run at L2 bandwidth rather than HBM bandwidth (toggle **Show L2 roof** in the roofline tab). On A100 and later you can **reserve part of L2 for persisting data** with \`cudaAccessPolicyWindow\`.

**HBM** is a stack of DRAM dies on a silicon interposer next to the GPU. The enormous 5,120-bit bus (A100) is what delivers terabytes per second, but each byte is still ~100× more expensive in time and energy than a register operand.

### Hopper additions

- **Thread-block clusters and distributed shared memory** let blocks on neighbouring SMs read each other's shared memory, a new rung between shared memory and L2.
- The **Tensor Memory Accelerator (TMA)** copies whole tiles global ⇄ shared asynchronously, freeing threads from issuing loads.

## Latency × bandwidth = parallelism

*Little's law* (see the Math tab) says that to keep a pipe full you need \`bandwidth × latency\` bytes **in flight**. For HBM on an A100 that is roughly **6–7 KB per SM**. You can supply it with:

- **many warps** (occupancy), or
- **many independent loads per thread** (instruction-level parallelism, ILP), or
- **wider loads** (\`float4\` moves 16 B per instruction).

The second tab of the lab lets you trade these against each other.

## The roofline model

Raw numbers do not tell you what to optimise. The **roofline model** does. For a kernel, count

- $W$: floating-point operations performed, and
- $Q$: bytes moved to/from the memory level you care about (usually HBM).

Their ratio is the **arithmetic intensity** $I = W/Q$ in FLOP/byte. A GPU has a peak compute rate $\\pi$ and a peak bandwidth $\\beta$. Then

$$
P \\le \\min\\big(\\pi,\\ \\beta\\, I\\big).
$$

On a log-log plot this is a **roof**: a slanted line (bandwidth-limited) meeting a flat line (compute-limited) at the **ridge point** $I^*=\\pi/\\beta$. Plot your kernel at its intensity:

- **Left of the ridge:** *memory-bound*. Making arithmetic faster does nothing. Only moving fewer bytes helps.
- **Right of the ridge:** *compute-bound*. Memory traffic is not the issue; you need better instruction throughput or Tensor Cores.

On the H100, the FP32 ridge is about **20 FLOP/B** but the FP16 Tensor ridge is about **300 FLOP/B**: with Tensor Cores, almost everything becomes memory-bound, which is why data movement dominates modern AI kernels.

### What to do about it

| Where you are | What helps | What does not |
|---|---|---|
| **Memory-bound**, far left | fuse kernels, avoid re-reads, vectorise (\`float4\`), fewer bits (FP16/FP8), better caching/reuse, compression | more FLOP/s, Tensor Cores |
| **Memory-bound**, near ridge | tiling, register blocking, larger tiles (raise $I$) | micro-optimising instructions |
| **Compute-bound** on CUDA cores | Tensor Cores, lower precision, ILP, fewer instructions per FLOP | memory tweaks |
| **Neither** (far below the roof) | latency problem: occupancy/ILP, coalescing, bank conflicts, divergence, launch overhead | changing precision |

The last row is important: the roofline gives an *upper bound*. A dot far below it means something else (latency, divergence, bad access patterns) is the limiter. That is the topic of the next chapters.

## How to measure it

- **Effective bandwidth** = bytes actually read + written ÷ kernel time (use \`cudaEvent\` timing). Compare it with the theoretical peak from the device attributes (see \`bandwidth_stream.cu\`). Well-tuned streaming kernels reach **85–95 %** of peak.
- **Nsight Compute** reports a *Speed of Light* section (percent of peak compute and memory throughput), a *Memory Workload Analysis* section (hit rates, bytes per level) and a **Roofline chart** for the kernel, so you do not have to count bytes by hand.
- **Microbenchmarks**: \`pointer_chase.cu\` reveals each cache level's size and latency; \`roofline_probe.cu\` measures your GPU's real roof.

<div class="callout tip">

**A 30-second triage.** (1) Estimate FLOPs and bytes → arithmetic intensity. (2) Compare with the ridge. (3) Measure achieved GB/s and GFLOP/s. If you are near the roof on the correct side, you are done. If you are far below it, look for latency problems. If you want to go *above* it, change the algorithm (fusion, tiling, lower precision) so that $I$ increases.

</div>
`,F=`## Effective bandwidth

For a kernel that reads $R$ bytes and writes $W$ bytes in time $t$:

$$
\\text{BW}_{\\text{eff}}=\\frac{R+W}{t}.
$$

Theoretical peak from the device attributes (DDR doubles the effective rate):

$$
\\text{BW}_{\\text{peak}}=2\\times f_{\\text{mem}}\\times\\frac{\\text{bus width}}{8}.
$$

A100 80GB: $2\\times1.593\\,\\text{GHz}\\times\\frac{5120}{8}=2.04$ TB/s. STREAM *triad* $a_i=b_i+q\\,c_i$ moves $3\\,n\\,s$ bytes ($s$ = bytes per element) and does $2n$ FLOP.

## Average memory access time (AMAT)

Let a load be served by L1 with probability $p_1$, by L2 with $p_2$ and by HBM with $p_3$, where $p_1=h_1$, $p_2=(1-h_1)h_2$, $p_3=(1-h_1)(1-h_2)$. With *end-to-end* latencies $L_1<L_2<L_3$:

$$
\\text{AMAT}=p_1L_1+p_2L_2+p_3L_3 .
$$

Example (A100: $L_1=33$, $L_2=200$, $L_3=500$ cycles) with $h_1=0.5,\\ h_2=0.6$:

$$
0.5(33)+0.3(200)+0.2(500)=16.5+60+100=176.5\\ \\text{cycles},
$$

about $2.8\\times$ better than sending everything to HBM. Note that the *misses* dominate: the 20 % of loads that reach HBM contribute 57 % of the average.

## Little's law

For a pipeline with latency $\\ell$ cycles and throughput $\\lambda$ bytes/cycle, the bytes that must be in flight to keep it full are

$$
B_{\\text{inflight}}=\\lambda\\,\\ell .
$$

Per SM, with aggregate bandwidth $\\beta$ (B/s), $N_{SM}$ SMs and clock $f$: $\\lambda=\\dfrac{\\beta}{N_{SM}\\,f}$.

**A100 / HBM:**

$$
\\lambda=\\frac{2.039\\times10^{12}}{108\\times1.41\\times10^{9}}\\approx13.4\\ \\text{B/cycle/SM},\\qquad
B_{\\text{inflight}}\\approx13.4\\times500\\approx6.7\\ \\text{KB per SM}.
$$

Your supply: $B_{\\text{supply}}=N_{\\text{warps}}\\cdot32\\cdot b\\cdot k$ for loads of $b$ bytes with $k$ independent loads in flight per thread. Achievable fraction of peak:

$$
\\eta=\\min\\!\\Big(1,\\ \\frac{N_{\\text{warps}}\\cdot32\\cdot b\\cdot k}{\\lambda\\,\\ell}\\Big),\\qquad
N_{\\text{warps}}^{\\text{needed}}=\\frac{\\lambda\\,\\ell}{32\\,b\\,k}.
$$

With 4-byte loads and $k=1$: $6{,}700/128\\approx52$ warps (of 64 possible). With \`float4\` ($b=16$): 13 warps. With $b=4,\\ k=4$: 13 warps. **Wider loads and ILP substitute for occupancy.**

## Arithmetic intensity

$$
I=\\frac{W}{Q}\\ \\ \\text{[FLOP/byte]},\\qquad W=\\text{FLOPs},\\quad Q=\\text{bytes moved at the level of interest.}
$$

With $s$ bytes per element (FP64 8, FP32 4, FP16 2):

| Kernel | $W$ per element | $Q$ per element | $I$ |
|---|---|---|---|
| vector add $c=a+b$ | 1 | $3s$ | $\\dfrac{1}{3s}$ ($=\\tfrac1{12}$ in FP32) |
| SAXPY $y=ax+y$ | 2 | $3s$ | $\\dfrac{2}{3s}$ ($=\\tfrac16$) |
| sum reduction | 1 | $s$ | $\\dfrac1s$ |
| 3-point stencil (cached neighbours) | 5 | $2s$ | $\\dfrac{5}{2s}$ |
| LayerNorm (read + write) | ≈8 | $2s$ | $\\approx\\dfrac{4}{s}$ |

### GEMM, naive vs tiled vs ideal

$C=AB$ with $N\\times N$ matrices does $W=2N^3$ FLOP.

- **Naive** (every output re-reads a row of $A$ and a column of $B$ from global memory): $Q\\approx 2N^3 s$, so $I=\\dfrac{1}{s}$, independent of $N$.
- **Tiled** with $T\\times T$ shared-memory tiles: each tile load is reused $T$ times, so $Q\\approx\\dfrac{2N^3s}{T}$ and
$$
I_{\\text{tiled}}=\\frac{T}{s}.
$$
- **Ideal** (each of $A,B$ read once, $C$ written once, $Q=3N^2s$):
$$
I_{\\text{ideal}}=\\frac{2N^3}{3N^2s}=\\frac{2N}{3s}\\quad\\Big(=\\frac N6\\ \\text{in FP32}\\Big).
$$

## The roofline

$$
P_{\\text{attainable}}(I)=\\min\\big(\\pi,\\ \\beta I\\big),\\qquad I^*=\\frac{\\pi}{\\beta},
$$

with $\\pi$ in FLOP/s and $\\beta$ in B/s. Equivalently the **time model**

$$
t\\ \\ge\\ \\max\\Big(\\frac{W}{\\pi},\\ \\frac{Q}{\\beta}\\Big),
$$

whose first term is compute time and second is memory time; whichever is larger is the bound.

### Ridge points

| GPU | $\\pi$ (TFLOP/s) | $\\beta$ (TB/s) | $I^*=\\pi/\\beta$ |
|---|---|---|---|
| A100, FP32 | 19.5 | 2.04 | **9.6** FLOP/B |
| A100, FP16 Tensor | 312 | 2.04 | **153** |
| H100, FP32 | 67 | 3.35 | **20** |
| H100, FP16 Tensor | 989 | 3.35 | **295** |
| RTX 4090, FP32 | 82.6 | 1.01 | **82** |

Unit trick: $1\\ \\text{TB/s}\\times1\\ \\text{FLOP/B}=1$ TFLOP/s, so on the plot the memory roof is just $P=\\beta_{\\text{TB/s}}\\cdot I$.

### Worked examples (A100)

**Vector add (FP32):** $I=\\tfrac1{12}=0.083$. Attainable $=0.083\\times2039\\ \\text{GB/s}=170$ GFLOP/s, only **0.87 %** of the 19.5 TFLOP/s peak, yet that is the *best possible*. For $2^{26}$ elements: $Q=12\\cdot2^{26}=805$ MB, so $t\\ge805\\ \\text{MB}/2.039\\ \\text{TB/s}=0.40$ ms.

**GEMM becomes compute-bound at** $I>I^*$. FP32 ideal: $N/6>9.6\\Rightarrow N>58$. FP16 Tensor ideal ($s=2$, $I=N/3$): $N/3>153\\Rightarrow N>459$. On H100 the same test gives $N>885$. Small GEMMs are memory-bound even with perfect kernels.

**LLM decode:** each weight (2 B in FP16) is used for 2 FLOP per sequence in the batch, so

$$
I_{\\text{decode}}=\\frac{2B}{s}=B\\quad(\\text{FP16}),
$$

and the batch needed to reach the H100 FP16 Tensor ridge is $B\\approx295$. At $B=1$ a 7B-parameter FP16 model streams 14 GB per token: $14\\ \\text{GB}/3.35\\ \\text{TB/s}\\approx4.2$ ms, or $\\lesssim240$ tokens/s no matter how many TFLOPs the GPU has.

## Multiple ceilings

Real GPUs have a roof per memory level (L1, L2, HBM) and per instruction type (FP64, FP32, TF32/FP16 Tensor). The *cache-aware* roofline draws all of them. The binding one is the lowest roof at your kernel's intensity *for the traffic at that level*. If the working set fits in L2, the relevant slope is $\\beta_{L2}$, not $\\beta_{HBM}$.

## Efficiency against the roof

$$
\\text{efficiency}=\\frac{P_{\\text{achieved}}}{P_{\\text{attainable}}(I)}.
$$

An efficiency near 1 means the kernel is on the roof: only an algorithmic change (raising $I$) can help. A low efficiency means a latency or inefficiency problem the roofline cannot see: low occupancy, uncoalesced access, bank conflicts, divergence, launch overhead.
`,I=`## Exercises

**Q1.** SAXPY on $2^{28}$ FP32 elements runs on an A100 (19.5 TFLOP/s, 2.039 TB/s). What is its arithmetic intensity, is it compute- or memory-bound, and what is the fastest it can possibly run?

<details>
<summary>Show answer</summary>

$W=2\\cdot2^{28}$ FLOP, $Q=3\\cdot4\\cdot2^{28}=3.22$ GB, so $I=2/12=0.167$ FLOP/B, far below the ridge ($9.6$): **memory-bound**. Time bound $t\\ge Q/\\beta=3.22\\ \\text{GB}/2.039\\ \\text{TB/s}=1.58$ ms, i.e. at most $0.167\\times2039=340$ GFLOP/s, **1.7 %** of the FP32 peak. Even a perfect kernel cannot beat this; a *good* one reaches 85–95 % of it.

</details>

**Q2.** A copy kernel moves a 1 GiB array to another 1 GiB array in 1.2 ms on an A100. What effective bandwidth is that and what fraction of peak?

<details>
<summary>Show answer</summary>

Bytes moved $=2\\times1.0737\\times10^9=2.147\\times10^9$ (read + write). $\\text{BW}_{\\text{eff}}=2.147\\times10^9/1.2\\times10^{-3}=1.79$ TB/s $=88\\ \\%$ of $2.039$ TB/s. Counting only the written bytes (a classic mistake) would report half of that.

</details>

**Q3.** Using the A100 latencies L1 = 33, L2 = 200, HBM = 500 cycles, compute the AMAT for $h_1=0.8$, $h_2=0.5$. What happens if a code change lifts $h_2$ to 0.9?

<details>
<summary>Show answer</summary>

$p_1=0.8$, $p_2=0.2\\cdot0.5=0.1$, $p_3=0.1$: $\\text{AMAT}=0.8(33)+0.1(200)+0.1(500)=26.4+20+50=96.4$ cycles. With $h_2=0.9$: $p_2=0.18$, $p_3=0.02$: $26.4+36+10=72.4$ cycles, a **25 %** drop from improving only the L2 hit rate. Misses to HBM dominate.

</details>

**Q4.** An H100 (132 SMs, 1.98 GHz, 3.35 TB/s, HBM latency ≈ 600 cycles). How many bytes must be in flight per SM to saturate HBM? How many warps does that require with 4-byte loads at one load per thread, and with four independent loads per thread?

<details>
<summary>Show answer</summary>

$\\lambda=3350/(132\\times1.98)=12.8$ B/cycle/SM. $B_{\\text{inflight}}=12.8\\times600\\approx7.7$ KB. Per warp-load (4 B × 32) $=128$ B, so $\\approx60$ warps with one load/thread (nearly the SM's 64-warp maximum!). With 4 independent loads/thread each warp holds 512 B in flight: $\\approx15$ warps. **Moral: ILP and wide loads are cheaper than occupancy.**

</details>

**Q5.** At what matrix size $N$ does an *ideal* square GEMM become compute-bound on an H100 (a) in FP32 on CUDA cores, (b) in FP16 on Tensor Cores?

<details>
<summary>Show answer</summary>

FP32: $I=N/6>20\\Rightarrow N>120$. FP16 Tensor: $I=2N/(3\\cdot2)=N/3>295\\Rightarrow N>885$. Below those sizes even a perfect GEMM is memory-bound, so batched/small GEMMs (e.g. attention heads) rarely hit Tensor Core peak.

</details>

**Q6.** A 7B-parameter model in FP16 is served at batch 1 on an RTX 4090 (1.008 TB/s, 165 TFLOP/s FP16 Tensor). What is the maximum tokens/s? What batch makes it compute-bound?

<details>
<summary>Show answer</summary>

Weights $=14$ GB. Each token streams them once: $14/1008\\approx13.9$ ms $\\Rightarrow\\lesssim72$ tokens/s. $I=B$ FLOP/B vs ridge $165/1.008=164$, so $B\\gtrsim164$ (in practice the KV cache grows too and limits the batch on a 24 GB card).

</details>

**Q7.** Where does \`float t[16]; ... t[k % 16] = v;\` live (with $k$ a runtime value) and how do you check it? Give two ways to fix it.

<details>
<summary>Show answer</summary>

In **local memory** (device memory, cached in L1/L2): registers cannot be indexed dynamically. Check with \`nvcc -Xptxas -v\` (stack frame, spill stores/loads) or Nsight Compute's local-memory counters. Fixes: make the index a compile-time constant (\`#pragma unroll\` plus constant indices), restructure into scalar variables / \`switch\`, or move the array to **shared memory** (one slice per thread, padded to avoid bank conflicts).

</details>

**Q8 (code).** Run \`pointer_chase.cu\`. At which working-set sizes does latency jump, and how do those sizes compare with your GPU's L1 and L2 capacity?

<details>
<summary>Show answer</summary>

You should see three plateaus: a low one (tens of cycles) while the array fits in L1 (roughly 128–256 KB), a middle one (a couple of hundred cycles) up to the L2 size (40 MB on A100), then a high one (400+ cycles) beyond it, sometimes with a further step from TLB misses on very large arrays. The jumps are fuzzy because of associativity and because L1 and shared memory share SRAM.

</details>

**Q9 (code).** Run \`roofline_probe.cu\`, plot the CSV with \`plot_roofline.py\`, and compare your measured knee with the theoretical ridge $\\pi/\\beta$.

<details>
<summary>Show answer</summary>

Expect the measured memory roof at ~85–92 % of theoretical bandwidth and the compute roof at 85–95 % of the FP32 peak (clocks may throttle below boost). Therefore the measured knee sits a bit away from the datasheet ridge. Always use the *measured* roof for tuning decisions.

</details>

## In practice

- **Estimate $I$ before writing code.** If a kernel is hopelessly memory-bound on paper, no instruction-level tuning will rescue it. Reduce bytes first.
- **Fuse element-wise operations** (bias + activation + residual + norm) into one kernel: each fused op saves a full read and write of the tensor.
- **Use fewer bits**: FP16/BF16/FP8 storage halves or quarters $Q$ with almost no extra work.
- **Vectorise** with \`float4\`/\`int4\` loads (align to 16 B) to raise bytes-in-flight per instruction.
- **Keep reuse in registers first**, then shared memory, then rely on L2. Each step down the ladder costs 5–10× more.
- **Measure with Nsight Compute** (Speed of Light + Roofline) rather than guessing: it reports achieved percent of peak for both compute and memory.

## Common pitfalls

- Counting only *useful* bytes: the hardware moves whole 32-byte sectors, so strided accesses move far more than you asked for (see *Coalescing*).
- Comparing against the wrong ceiling: FP32-peak ridge for a kernel that uses Tensor Cores, or HBM roof for a working set that lives in L2.
- Trusting boost-clock peaks: sustained clocks under power or thermal limits are lower, so measure.
- Assuming *local* memory is "local" in the speed sense. It is global memory with a thread-private view.
- Declaring a kernel "memory-bound" because it is slow. Memory-bound means near the *memory roof*; if you are far below it you have a latency or access-pattern problem.
- Forgetting that the L1/shared split is a per-kernel choice: a kernel that uses little shared memory can run with a larger L1.
`,ee=`// STREAM-style effective-bandwidth benchmark.
//   copy : c = a            (2 arrays touched)
//   scale: b = q * c        (2)
//   add  : c = a + b        (3)
//   triad: a = b + q * c    (3)
// plus a float4 (16-byte) copy to show what wider loads do.
// build: nvcc -O3 -std=c++17 -arch=sm_80 bandwidth_stream.cu -o stream && ./stream
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

__global__ void k_copy(const float* __restrict__ a, float* __restrict__ c, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        c[i] = a[i];
}
__global__ void k_scale(const float* __restrict__ c, float* __restrict__ b, float q, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        b[i] = q * c[i];
}
__global__ void k_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        c[i] = a[i] + b[i];
}
__global__ void k_triad(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, float q, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        a[i] = b[i] + q * c[i];
}
// 16 bytes per thread per access: 4x fewer load instructions for the same bytes
__global__ void k_copy4(const float4* __restrict__ a, float4* __restrict__ c, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x)
        c[i] = a[i];
}

// time \`reps\` launches of fn() and return the best single-launch time in ms
template <typename F>
float best_ms(F fn, int reps = 20) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    fn();                                           // warm-up (also loads the kernel)
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        fn();
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
    int dev = 0, sms = 0, memclk = 0, bus = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, dev));       // kHz
    CUDA_CHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));     // bits
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    const double peak = 2.0 * memclk * 1e3 * (bus / 8) / 1e9;                           // GB/s (DDR x2)
    printf("%s: %d SMs, theoretical bandwidth %.0f GB/s (%d-bit bus)\\n", p.name, sms, peak, bus);

    const size_t n = (size_t)1 << 27;               // 128M floats = 512 MB per array: far larger than L2
    const size_t bytes = n * sizeof(float);
    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMemset(a, 0, bytes));
    CUDA_CHECK(cudaMemset(b, 0, bytes));
    CUDA_CHECK(cudaMemset(c, 0, bytes));

    const int threads = 256, blocks = sms * 16;     // hardware-sized grid + grid-stride loop
    const float q = 3.0f;

    auto report = [&](const char* name, float ms, double moved) {
        double gbs = moved / (ms * 1e-3) / 1e9;
        printf("%-10s %8.3f ms  %8.1f GB/s  (%5.1f%% of peak)\\n", name, ms, gbs, 100.0 * gbs / peak);
    };

    report("copy",  best_ms([&] { k_copy <<<blocks, threads>>>(a, c, n); }),       2.0 * bytes);
    report("scale", best_ms([&] { k_scale<<<blocks, threads>>>(c, b, q, n); }),    2.0 * bytes);
    report("add",   best_ms([&] { k_add  <<<blocks, threads>>>(a, b, c, n); }),    3.0 * bytes);
    report("triad", best_ms([&] { k_triad<<<blocks, threads>>>(a, b, c, q, n); }), 3.0 * bytes);
    report("copy f4", best_ms([&] { k_copy4<<<blocks, threads>>>((const float4*)a, (float4*)c, n / 4); }), 2.0 * bytes);

    // What to look for:
    //  * Each kernel does 0-2 FLOP per 8-12 bytes moved (arithmetic intensity < 0.2): memory-bound by construction.
    //  * Healthy streaming kernels reach ~85-95% of the theoretical bandwidth.
    //  * float4 often adds a few percent: fewer instructions, more bytes in flight per thread (Little's law).
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
`,L=`// Every CUDA memory space in one kernel.
// build:   nvcc -O3 -arch=sm_80 -Xptxas -v memory_spaces.cu -o spaces
// -Xptxas -v prints per-kernel register count, shared bytes, constant bytes and stack frame / spills.
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

__constant__ float c_coef[4];          // CONSTANT: 64 KB total, read-only in kernels, broadcast through the constant cache
__device__   unsigned int g_counter;   // GLOBAL: statically allocated, visible to every thread and kernel

#define BLOCK 256

__global__ void spaces(const float* __restrict__ in, float* __restrict__ out, const int* __restrict__ pick, int n) {
    __shared__ float tile[BLOCK];                       // SHARED: one copy per block, on-chip

    int i = blockIdx.x * blockDim.x + threadIdx.x;      // REGISTER (copied from special registers)
    float x = (i < n) ? in[i] : 0.f;                    // REGISTER, filled from GLOBAL memory (HBM -> L2 -> L1 -> reg)

    tile[threadIdx.x] = x;                              // register -> shared
    __syncthreads();
    float nb = tile[(threadIdx.x + 1) % BLOCK];         // shared -> register (a neighbour's value: reuse!)

    float y = c_coef[0] * x + c_coef[1] * nb;           // every lane reads the SAME constant address: 1 broadcast

    // Static indices after unrolling -> registers. No memory traffic at all.
    float acc[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = y * (j + 1);

    // A RUNTIME index into a per-thread array cannot live in registers (registers are not addressable).
    // The compiler puts \`scratch\` in LOCAL memory: private to the thread, physically in device memory.
    float scratch[16];
    for (int j = 0; j < 16; ++j) scratch[j] = y + j;
    float dyn = (i < n) ? scratch[pick[i] & 15] : 0.f;  // dynamic index -> local-memory load (look for "stack frame" in -Xptxas -v)

    if (i < n) out[i] = acc[0] + acc[3] + dyn;
    if (threadIdx.x == 0) atomicAdd(&g_counter, 1u);    // atomics are resolved in L2
}

// Same computation, but dynamic indexing replaced by a shared-memory slice per thread:
// the spill disappears at the price of shared memory (and potential bank conflicts, see next chapters).
__global__ void spaces_shared_scratch(const float* __restrict__ in, float* __restrict__ out, const int* __restrict__ pick, int n) {
    __shared__ float scratch[16][BLOCK + 1];            // +1 column of padding to dodge bank conflicts
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (i < n) ? in[i] : 0.f;
    float y = c_coef[0] * x + c_coef[1] * x;
    for (int j = 0; j < 16; ++j) scratch[j][threadIdx.x] = y + j;
    float dyn = (i < n) ? scratch[pick[i] & 15][threadIdx.x] : 0.f;
    if (i < n) out[i] = dyn;
}

int main() {
    const int n = 1 << 22;
    float h_c[4] = {0.5f, 0.25f, 0.f, 0.f};
    CUDA_CHECK(cudaMemcpyToSymbol(c_coef, h_c, sizeof(h_c)));

    float *d_in, *d_out; int* d_pick;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pick, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_in, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pick, 0, n * sizeof(int)));

    // Opt in to more shared memory per block / change the L1-vs-shared split (hint, in percent of the maximum shared size).
    CUDA_CHECK(cudaFuncSetAttribute(spaces, cudaFuncAttributePreferredSharedMemoryCarveout, 25));

    int blocks = (n + BLOCK - 1) / BLOCK;
    cudaEvent_t t0, t1; float ms;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    for (int variant = 0; variant < 2; ++variant) {
        for (int w = 0; w < 2; ++w) {                   // second pass is the timed one (first warms up)
            CUDA_CHECK(cudaEventRecord(t0));
            if (variant == 0) spaces<<<blocks, BLOCK>>>(d_in, d_out, d_pick, n);
            else              spaces_shared_scratch<<<blocks, BLOCK>>>(d_in, d_out, d_pick, n);
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        printf("%-22s %.3f ms\\n", variant == 0 ? "local-memory scratch" : "shared-memory scratch", ms);
    }

    unsigned int h_counter = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_counter, g_counter, sizeof(h_counter)));
    printf("blocks that incremented the global counter: %u (expected %d x 2 launches)\\n", h_counter, blocks);

    // Try:  nvcc -Xptxas -v ...                 -> compare "stack frame" of the two kernels
    //       nvcc -maxrregcount=32 -Xptxas -v ... -> force spills and watch "spill stores / loads" appear
    //       ncu --section MemoryWorkloadAnalysis ./spaces   -> local-memory traffic per level
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_pick);
    return 0;
}
`,R=`// Pointer-chasing latency microbenchmark.
// ONE thread follows a random cyclic chain of dependent loads: p = next[p].
// Every load depends on the previous one, so nothing can overlap and time/iteration IS the latency
// of whichever memory level the working set fits in.
//
// build: nvcc -O3 -arch=sm_80 pointer_chase.cu -o chase && ./chase > chase.csv
// Expect steps in latency at roughly: L1 size  ->  L2 size  ->  (TLB reach) beyond.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
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

__global__ void chase(const unsigned* __restrict__ next, unsigned start, long iters,
                      unsigned* sink, long long* cycles) {
    unsigned p = start;
    long long t0 = clock64();
    for (long i = 0; i < iters; ++i) p = next[p];      // dependent load chain
    long long t1 = clock64();
    *sink = p;                                         // keep the chain alive
    *cycles = t1 - t0;
}

int main() {
    const int LINE = 32;                               // one element per 128-byte cache line (32 x 4 B)
    int dev = 0, clk_khz = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
    const double ghz = clk_khz / 1e6;

    unsigned* d_sink; long long* d_cyc;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_cyc, sizeof(long long)));

    printf("bytes,cycles_per_load,ns_per_load\\n");
    for (size_t bytes = 4 << 10; bytes <= ((size_t)1 << 30); bytes = bytes * 3 / 2 + 1024) {
        size_t lines = bytes / (LINE * sizeof(unsigned));
        if (lines < 2) continue;

        // random cyclic permutation over cache lines (Sattolo) -> defeats hardware prefetching
        std::vector<unsigned> perm(lines);
        std::iota(perm.begin(), perm.end(), 0u);
        std::mt19937 rng(42);
        for (size_t i = lines - 1; i > 0; --i) {
            std::uniform_int_distribution<size_t> d(0, i - 1);
            std::swap(perm[i], perm[d(rng)]);
        }
        std::vector<unsigned> next(lines * LINE, 0);
        for (size_t i = 0; i < lines; ++i)
            next[(size_t)perm[i] * LINE] = perm[(i + 1) % lines] * LINE;   // element index of the next line

        unsigned* d_next;
        CUDA_CHECK(cudaMalloc(&d_next, next.size() * sizeof(unsigned)));
        CUDA_CHECK(cudaMemcpy(d_next, next.data(), next.size() * sizeof(unsigned), cudaMemcpyHostToDevice));

        long iters = (long)std::min<size_t>(lines * 4, (size_t)1 << 20);
        chase<<<1, 1>>>(d_next, perm[0] * LINE, iters, d_sink, d_cyc);      // warm-up: fills the caches
        CUDA_CHECK(cudaDeviceSynchronize());
        chase<<<1, 1>>>(d_next, perm[0] * LINE, iters, d_sink, d_cyc);      // measured pass
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        long long cyc = 0;
        CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, sizeof(cyc), cudaMemcpyDeviceToHost));
        double per = (double)cyc / iters;
        printf("%zu,%.1f,%.1f\\n", bytes, per, per / ghz);
        CUDA_CHECK(cudaFree(d_next));
    }
    // Reading the CSV: the first plateau is L1 latency, the second is L2, the third is HBM.
    // Where the curve bends tells you the effective capacity of each level for this access pattern.
    return 0;
}
`,z=`// Measure YOUR GPU's roofline.
// Each thread loads one float, performs K fused multiply-adds on it (two independent chains for ILP),
// and stores it back. K sets the arithmetic intensity:
//     FLOP/element = 2*K*2 (two chains, FMA = 2 FLOP)   bytes/element = 8 (4 read + 4 write)
//     AI = (4*K) / 8 = K/2 FLOP/B
// Sweep K from 1 to 1024 and the kernel walks from the memory roof to the compute roof.
//
// build: nvcc -O3 -std=c++17 -arch=sm_80 roofline_probe.cu -o probe && ./probe > roofline.csv
// plot:  python plot_roofline.py roofline.csv
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

template <int K>
__global__ void fma_kernel(float* __restrict__ x, size_t n, float a, float b) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v0 = x[i], v1 = v0 + 1.0f;
    #pragma unroll 16
    for (int j = 0; j < K; ++j) {              // K is a compile-time constant -> unrolled, no loop overhead
        v0 = fmaf(v0, a, b);                   // chain 0
        v1 = fmaf(v1, a, b);                   // chain 1 (independent: instruction-level parallelism)
    }
    x[i] = v0 + v1;                            // use both results so nothing is optimised away
}

template <int K>
void run(float* d, size_t n, double* gflops, double* gbs) {
    const int threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    const float a = 0.9999f, b = 1e-4f;        // |a|<1: values stay bounded for any K
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    fma_kernel<K><<<blocks, threads>>>(d, n, a, b);              // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < 10; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        fma_kernel<K><<<blocks, threads>>>(d, n, a, b);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    const double flops = (double)n * (4.0 * K + 1.0);          // 2 chains x K FMAs x 2 FLOP + the final add
    const double bytes = (double)n * 8.0;
    *gflops = flops / (best * 1e-3) / 1e9;
    *gbs = bytes / (best * 1e-3) / 1e9;
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
}

int main() {
    const size_t n = (size_t)1 << 26;           // 256 MB: larger than L2, so HBM traffic is real
    float* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));

    printf("fma_per_chain,flop_per_byte,gflops,gbs\\n");
    double gf, gb;
#define PROBE(K) run<K>(d, n, &gf, &gb); printf("%d,%.4f,%.1f,%.1f\\n", K, (4.0 * K + 1.0) / 8.0, gf, gb);
    PROBE(1) PROBE(2) PROBE(4) PROBE(8) PROBE(16) PROBE(32)
    PROBE(64) PROBE(128) PROBE(256) PROBE(512) PROBE(1024)
#undef PROBE
    cudaFree(d);
    return 0;
}
`,B=`"""Plot the CSV printed by roofline_probe.cu as a measured roofline.

usage:  python plot_roofline.py roofline.csv [--peak-gflops 19500 --peak-gbs 2039]
The optional --peak-* flags draw the datasheet roof for comparison.
"""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--peak-gflops", type=float, default=None, help="datasheet FP32 peak (GFLOP/s)")
ap.add_argument("--peak-gbs", type=float, default=None, help="datasheet bandwidth (GB/s)")
args = ap.parse_args()

with open(args.csv) as f:
    rows = [r for r in csv.DictReader(f)]
ai = np.array([float(r["flop_per_byte"]) for r in rows])
gf = np.array([float(r["gflops"]) for r in rows])
gb = np.array([float(r["gbs"]) for r in rows])

# measured roofs: bandwidth from the lowest-intensity points, compute from the highest
bw = gb[:3].max()          # GB/s sustained while memory-bound
peak = gf.max()            # GFLOP/s sustained when compute-bound
ridge = peak / bw          # FLOP/B

x = np.logspace(np.log10(ai.min() / 2), np.log10(ai.max() * 2), 200)
roof = np.minimum(peak, bw * x)

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.loglog(x, roof, "-", lw=2.5, label=f"measured roof: {bw:.0f} GB/s, {peak / 1000:.1f} TFLOP/s")
ax.loglog(ai, gf, "o", ms=6, label="probe kernel (K = 1 ... 1024)")
if args.peak_gflops and args.peak_gbs:
    ax.loglog(x, np.minimum(args.peak_gflops, args.peak_gbs * x), "--", lw=1.2, color="gray", label="datasheet roof")
ax.axvline(ridge, ls=":", color="k", lw=1)
ax.annotate(f"measured ridge ~ {ridge:.1f} FLOP/B", (ridge, peak / 3), rotation=90, va="center", ha="right", fontsize=8)
ax.set_xlabel("arithmetic intensity (FLOP / byte)")
ax.set_ylabel("performance (GFLOP/s)")
ax.set_title("Measured roofline")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("roofline.png", dpi=150)
print(f"memory roof {bw:.0f} GB/s | compute roof {peak:.0f} GFLOP/s | ridge {ridge:.2f} FLOP/B -> roofline.png")
`,V=o(),H=.46,U=[3.2,2.2,1.2,0,-1.3],W=4.4,G={smem:1,l1:2,l2:3,hbm:4},K={smem:`#22d3ee`,l1:`#76d12a`,l2:`#fbbf24`,hbm:`#fb7185`},q=(e,t)=>t===`cap`?e.capTotal:t===`lat`?e.lat:e.bw;function te(e,t){let n=e.map(e=>Math.log10(q(e,t))),r=Math.min(...n),i=Math.max(...n);return n.map(e=>1.7+5*(e-r)/(i-r||1))}function J(e,t,n){let r=[{t:0,y:e[0],lvl:-1}],i=0;for(let a=1;a<e.length;a++)i+=.28,r.push({t:i,y:e[a],lvl:n[a]}),i+=t[a],r.push({t:i,y:e[a],lvl:n[a]});for(let t=e.length-2;t>=0;t--)i+=.2,r.push({t:i,y:e[t],lvl:-1});return{k:r,total:i}}function Y(e,t){let n=e.k;for(let e=1;e<n.length;e++)if(t<=n[e].t){let r=n[e-1],i=n[e],a=(t-r.t)/(i.t-r.t||1);return{y:r.y+(i.y-r.y)*a,lvl:r.y===i.y?i.lvl:-1}}return{y:n[n.length-1].y,lvl:-1}}function ne({text:e,color:t=`#e7e9f4`,position:n,align:a=`center`,height:o=.4}){let s=(0,k.useMemo)(()=>{let n=document.createElement(`canvas`),a=n.getContext(`2d`),o=`600 44px "JetBrains Mono Variable", ui-monospace, monospace`;a.font=o,n.width=Math.ceil(a.measureText(e).width)+16,n.height=64,a.font=o,a.fillStyle=t,a.textBaseline=`middle`,a.fillText(e,8,34);let s=new r(n);return s.colorSpace=i,s.userData.aspect=n.width/n.height,s},[e,t]);(0,k.useEffect)(()=>()=>s.dispose(),[s]);let c=o*s.userData.aspect,l=a===`right`?-c/2:a===`left`?c/2:0;return(0,V.jsx)(`sprite`,{position:[n[0]+l,n[1],n[2]],scale:[c,o,1],renderOrder:10,children:(0,V.jsx)(`spriteMaterial`,{map:s,transparent:!0,depthTest:!1})})}function re({lv:e,idx:n,w:r,sel:i,onSel:o,live:s}){let c=(0,k.useRef)();return t((e,t)=>{let r=c.current;if(!r)return;let a=s.current.lvl===n?1.2:i?.5:.12;r.emissiveIntensity+=(a-r.emissiveIntensity)*Math.min(1,t*10)}),(0,V.jsxs)(`group`,{position:[0,U[n],0],children:[(0,V.jsxs)(`mesh`,{onClick:t=>{t.stopPropagation(),o(e.id)},onPointerOver:()=>document.body.style.cursor=`pointer`,onPointerOut:()=>document.body.style.cursor=``,children:[(0,V.jsx)(`boxGeometry`,{args:[r,H,3.4]}),(0,V.jsx)(`meshStandardMaterial`,{ref:c,color:e.color,emissive:e.color,emissiveIntensity:.12,roughness:.5,transparent:!0,opacity:.9}),(0,V.jsx)(a,{color:i?`#ffffff`:`#0a0c14`})]}),(0,V.jsx)(ne,{text:e.name,color:e.color,position:[-r/2-.35,0,1.7],align:`right`})]})}function ie({job:e,live:n}){let r=(0,k.useRef)(),i=(0,k.useRef)({t:0,tl:null});(0,k.useEffect)(()=>{e&&(i.current={t:0,tl:J(e.ys,e.dw,e.idxs)})},[e]),t((e,t)=>{let a=i.current,o=r.current;if(!o)return;if(!a.tl||a.t>a.tl.total){o.visible=!1,n.current={lvl:-1};return}a.t+=Math.min(t,.05);let s=Y(a.tl,a.t);o.visible=!0,o.position.y=s.y,n.current={lvl:s.lvl}});let a=e?e.color:`#ffffff`;return(0,V.jsxs)(`mesh`,{ref:r,visible:!1,children:[(0,V.jsx)(`sphereGeometry`,{args:[.2,20,20]}),(0,V.jsx)(`meshStandardMaterial`,{color:a,emissive:a,emissiveIntensity:1})]})}function ae(){let[e,t]=(0,k.useState)(`a100`),[n,r]=(0,k.useState)(`cap`),[i,a]=(0,k.useState)(`l2`),[o,c]=(0,k.useState)(60),[l,u]=(0,k.useState)(70),[d,f]=(0,k.useState)(`global`),[p,m]=(0,k.useState)(!0),[h,g]=(0,k.useState)(null),[_,S]=(0,k.useState)({n:0,cyc:0,c:{smem:0,l1:0,l2:0,hbm:0}}),D=(0,k.useRef)({lvl:-1}),P=A[e],F=(0,k.useMemo)(()=>N(P),[P]),I=(0,k.useMemo)(()=>Object.fromEntries(F.map(e=>[e.id,e])),[F]),ee=(0,k.useMemo)(()=>te(F,n),[F,n]),L=()=>d===`shared`?`smem`:Math.random()<o/100?`l1`:Math.random()<l/100?`l2`:`hbm`,R=e=>{let t=e===`smem`?[`smem`]:e===`l1`?[`l1`]:e===`l2`?[`l1`,`l2`]:[`l1`,`l2`,`hbm`],n=[W,...t.map(e=>U[G[e]])],r=[-1,...t.map(e=>G[e])],i=[0,...t.map((e,n)=>n===t.length-1?.12+.5*Math.log10(I[e].lat)/3:.08)];g({ys:n,idxs:r,dw:i,color:K[e],id:Math.random()})},z=()=>{let e=L();S(t=>({n:t.n+1,cyc:t.cyc+I[e].lat,c:{...t.c,[e]:t.c[e]+1}})),R(e)},B=()=>{let e={smem:0,l1:0,l2:0,hbm:0},t=0;for(let n=0;n<1e3;n++){let n=L();e[n]++,t+=I[n].lat}S(n=>({n:n.n+1e3,cyc:n.cyc+t,c:{smem:n.c.smem+e.smem,l1:n.c.l1+e.l1,l2:n.c.l2+e.l2,hbm:n.c.hbm+e.hbm}}))},H=()=>S({n:0,cyc:0,c:{smem:0,l1:0,l2:0,hbm:0}});C(p,1700,z),(0,k.useEffect)(H,[e,o,l,d]);let q=o/100,J=(1-q)*l/100,Y=1-q-J,ae=q*I.l1.lat+J*I.l2.lat+Y*I.hbm.lat,oe=_.n?_.cyc/_.n:null,X=I[i],se=(0,V.jsxs)(V.Fragment,{children:[(0,V.jsx)(`b`,{style:{color:X.color},children:X.name}),` · `,X.where,(0,V.jsx)(`br`,{}),`capacity `,X.capText,(0,V.jsx)(`br`,{}),`latency ≈ `,(0,V.jsx)(`b`,{children:X.lat}),` cycles (`,(X.lat/P.clk).toFixed(0),` ns) · bandwidth ≈ `,(0,V.jsx)(`b`,{children:M(X.bw)}),(0,V.jsx)(`br`,{}),`scope: `,X.scope,` · lifetime: `,X.life,(0,V.jsx)(`br`,{}),(0,V.jsx)(`span`,{style:{color:`#fbbf24`},children:X.kw})]});return(0,V.jsxs)(V.Fragment,{children:[(0,V.jsxs)(v,{height:540,camera:[9,5.2,12.5],target:[0,1.6,0],fov:42,overlay:se,hint:`drag to orbit · click a slab for its specs · packets show a load's journey`,children:[(0,V.jsxs)(`mesh`,{position:[0,(W+U[4])/2,0],children:[(0,V.jsx)(`boxGeometry`,{args:[.04,W-U[4],.04]}),(0,V.jsx)(`meshStandardMaterial`,{color:`#3a4170`})]}),F.map((e,t)=>(0,V.jsx)(re,{lv:e,idx:t,w:ee[t],sel:i===e.id,onSel:a,live:D},e.id)),(0,V.jsxs)(`mesh`,{position:[0,W,0],children:[(0,V.jsx)(`boxGeometry`,{args:[.55,.55,.55]}),(0,V.jsx)(`meshStandardMaterial`,{color:`#ffffff`,emissive:`#8b7bff`,emissiveIntensity:.4})]}),(0,V.jsx)(ne,{text:`thread`,position:[.55,W,0],align:`left`}),(0,V.jsx)(ie,{job:h,live:D})]}),(0,V.jsxs)(x,{children:[(0,V.jsx)(E,{label:`GPU`,value:e,onChange:t,options:j}),(0,V.jsx)(E,{label:`Slab width = (log)`,value:n,onChange:r,options:[[`cap`,`capacity`],[`lat`,`latency`],[`bw`,`bandwidth`]]}),(0,V.jsx)(E,{label:`Load type`,value:d,onChange:f,options:[[`global`,`global load (via L1 → L2 → HBM)`],[`shared`,`shared-memory load`]]}),d===`global`&&(0,V.jsxs)(V.Fragment,{children:[(0,V.jsx)(b,{label:`L1 hit rate`,min:0,max:100,value:o,onChange:c,fmt:e=>e+`%`}),(0,V.jsx)(b,{label:`L2 hit rate (of L1 misses)`,min:0,max:100,value:l,onChange:u,fmt:e=>e+`%`})]}),(0,V.jsx)(O,{label:`Auto-fire loads`,value:p,onChange:m}),(0,V.jsx)(T,{primary:!0,onClick:z,children:`Fire a load`}),(0,V.jsx)(T,{onClick:B,children:`Sample 1000`}),(0,V.jsx)(T,{onClick:H,children:`Reset stats`})]}),(0,V.jsx)(y,{items:[[K.l1,`served by L1`],[K.l2,`served by L2`],[K.hbm,`served by HBM`],[K.smem,`shared memory`]]}),(0,V.jsxs)(w,{children:[d===`shared`?(0,V.jsxs)(V.Fragment,{children:[`Shared memory is addressed explicitly (`,(0,V.jsx)(`b`,{children:`__shared__`}),`) so there is no hit-or-miss: every load costs ≈ `,(0,V.jsx)(`b`,{children:I.smem.lat}),` cycles — about `,(0,V.jsxs)(`b`,{children:[(I.hbm.lat/I.smem.lat).toFixed(0),`×`]}),` faster than a trip to `,I.hbm.name,`.`]}):(0,V.jsxs)(V.Fragment,{children:[`AMAT = `,o,`%·`,I.l1.lat,` + `,(100*J).toFixed(0),`%·`,I.l2.lat,` + `,(100*Y).toFixed(0),`%·`,I.hbm.lat,` = `,(0,V.jsx)(`b`,{children:ae.toFixed(0)}),` cycles`,oe!=null&&(0,V.jsxs)(V.Fragment,{children:[` · observed over `,s(_.n),` loads: `,(0,V.jsx)(`b`,{children:oe.toFixed(0)}),` cycles (L1 `,_.c.l1,` · L2 `,_.c.l2,` · HBM `,_.c.hbm,`)`]}),` `,`· `,(I.hbm.lat/ae).toFixed(1),`× better than every load going to HBM. Drag the hit rates: a kernel is only as fast as its `,(0,V.jsx)(`b`,{className:`w`,children:`miss path`}),`.`]}),` `,(0,V.jsx)(`span`,{style:{opacity:.7},children:`(packet dwell times are log-scaled so L1 stays visible; latencies are approximate.)`})]})]})}function oe(){let[e,t]=(0,k.useState)(`a100`),[n,r]=(0,k.useState)(`hbm`),[i,a]=(0,k.useState)(16),[o,s]=(0,k.useState)(4),[f,g]=(0,k.useState)(1),_=A[e],v=(0,k.useMemo)(()=>N(_),[_]),y=v.find(e=>e.id===n),S=y.bw/_.sms/_.clk,C=S*y.lat,T=i*32*o*f,O=Math.min(1,T/C),P=C/(32*o*f),[F]=D(250,(e,t)=>{let n=(t-128-24)/2-8,r=128+n+32;h(e,`latency · cycles (log)`,128,14,{size:11,color:m.mute,weight:600}),h(e,`aggregate bandwidth · GB/s (log)`,r,14,{size:11,color:m.mute,weight:600}),v.forEach((t,i)=>{let a=30+i*40;h(e,t.name,116,a+14,{align:`right`,size:12,color:t.color,weight:600});let o=(r,i,o)=>{let s=Math.max(5,i*n);e.fillStyle=d(t.color,.85),l(e,r,a,s,28,5),e.fill(),s>150?h(e,o,r+s-8,a+14,{align:`right`,size:11,color:`#0a0c14`,mono:!0,weight:700}):h(e,o,r+s+7,a+14,{size:11,color:m.ink,mono:!0})};o(128,u(Math.log10(t.lat)/3,0,1),`${t.lat} cyc · ${(t.lat/_.clk).toFixed(0)} ns`),o(r,u(Math.log10(t.bw)/6,0,1),M(t.bw))});let i=v[0],a=v[4];h(e,`HBM vs register: ${(a.lat/i.lat).toFixed(0)}× slower to reach, ${(i.bw/a.bw).toFixed(0)}× less bandwidth`,128,238,{size:11,color:m.d,mono:!0})}),[I]=D(250,(e,t,n)=>{let r=t-58-24,a=n-22-40,s=e=>58+(e-1)/63*r,l=e=>22+a-e*a;e.strokeStyle=m.grid,e.lineWidth=1,e.fillStyle=m.mute;for(let t=0;t<=1.001;t+=.25)e.beginPath(),e.moveTo(58,l(t)),e.lineTo(58+r,l(t)),e.stroke(),h(e,Math.round(t*100)+`%`,50,l(t),{align:`right`,size:10.5,color:m.mute,mono:!0});[1,8,16,32,48,64].forEach(t=>h(e,t,s(t),22+a+14,{align:`center`,size:10.5,color:m.mute,mono:!0})),h(e,`resident warps per SM →`,58+r/2,n-8,{align:`center`,size:11,color:m.mute}),h(e,`achieved ${y.name} bandwidth`,58,9,{size:11,color:m.mute,weight:600});let u=e=>Array.from({length:64},(t,n)=>[s(n+1),l(Math.min(1,(n+1)*32*o*e/C))]);f>1&&c(e,u(1),m.dim,1.5,[4,3]),c(e,u(f),y.color,2.8),P<=64?(e.strokeStyle=d(`#ffffff`,.35),e.setLineDash([3,3]),e.beginPath(),e.moveTo(s(P),22),e.lineTo(s(P),22+a),e.stroke(),e.setLineDash([]),h(e,`saturates at ${Math.ceil(P)} warps`,s(P)-6,32,{align:`right`,size:10.5,color:m.ink,mono:!0})):h(e,`needs ${P.toFixed(0)} warps — more than the SM can hold!`,58+r-6,32,{align:`right`,size:10.5,color:m.r,mono:!0}),e.fillStyle=`#fff`,e.beginPath(),e.arc(s(i),l(O),5.5,0,7),e.fill(),f>1&&h(e,`1 load / thread`,s(40),l(Math.min(1,1280*o/C))+14,{size:10.5,color:m.dim,mono:!0})});return(0,V.jsxs)(V.Fragment,{children:[(0,V.jsx)(`canvas`,{...F}),(0,V.jsx)(`canvas`,{...I}),(0,V.jsxs)(x,{children:[(0,V.jsx)(E,{label:`GPU`,value:e,onChange:t,options:j}),(0,V.jsx)(E,{label:`Level to saturate`,value:n,onChange:r,options:[[`smem`,`Shared memory / L1`],[`l2`,`L2`],[`hbm`,`HBM / global`]]}),(0,V.jsx)(b,{label:`Resident warps / SM`,min:1,max:64,value:i,onChange:a}),(0,V.jsx)(E,{label:`Bytes per load`,value:o,onChange:e=>s(+e),options:[[4,`4 B (float)`],[8,`8 B (float2)`],[16,`16 B (float4)`]]}),(0,V.jsx)(b,{label:`Independent loads in flight / thread`,min:1,max:8,value:f,onChange:g})]}),(0,V.jsxs)(w,{children:[`Little’s law: bytes in flight = bandwidth × latency = `,(0,V.jsx)(`b`,{children:S.toFixed(1)}),` B/cycle/SM × `,(0,V.jsx)(`b`,{children:y.lat}),` cycles = `,(0,V.jsx)(`b`,{children:p(C)}),` per SM. You supply `,i,` warps × 32 lanes × `,o,` B × `,f,` = `,(0,V.jsx)(`b`,{children:p(T)}),` → `,(0,V.jsxs)(`b`,{className:O>=.99?`g`:`w`,children:[(100*O).toFixed(0),`%`]}),` of peak `,y.name,` bandwidth.`,P>64&&(0,V.jsxs)(V.Fragment,{children:[` Even a full SM cannot hide this latency with `,f,` load/thread — `,(0,V.jsx)(`b`,{className:`r`,children:`raise ILP or widen the loads`}),`.`]}),P<=64&&O<.99&&(0,V.jsxs)(V.Fragment,{children:[` You need ≈ `,(0,V.jsx)(`b`,{children:Math.ceil(P)}),` warps (or more loads per thread) to saturate it.`]}),O>=.99&&(0,V.jsx)(V.Fragment,{children:` Saturated: extra warps no longer help — bandwidth is now the limit, not latency.`})]})]})}var X={fp64:8,fp32:4,fp16:2},se=[{id:`vecadd`,name:`vector add`,eff:.9,ai:e=>1/(3*e.s),how:`1 add per 3 elements moved (2 loads + 1 store)`},{id:`saxpy`,name:`SAXPY`,eff:.9,ai:e=>2/(3*e.s),how:`1 FMA = 2 FLOP per 3 elements moved`},{id:`reduce`,name:`sum reduction`,eff:.8,ai:e=>1/e.s,how:`1 add per element read`},{id:`stencil`,name:`3-pt stencil`,eff:.75,ai:e=>5/(2*e.s),how:`5 FLOP per element, neighbours reused on-chip`},{id:`spmv`,name:`SpMV (CSR)`,eff:.45,ai:e=>2/(2*e.s+4),how:`2 FLOP per non-zero; value + column index + gathered x`},{id:`lnorm`,name:`LayerNorm`,eff:.8,ai:e=>8/(2*e.s),how:`≈8 FLOP per element, one read + one write`},{id:`naive`,name:`naive GEMM`,eff:.35,ai:e=>1/e.s,how:`each output re-reads a full row and column from global memory: 2N FLOP per 2N elements`},{id:`tiled`,name:`tiled GEMM`,eff:.6,tc:!0,ai:e=>e.T/e.s,how:`T×T tiles cut global traffic by T: AI = T / bytes`},{id:`gemm`,name:`library GEMM`,eff:.9,tc:!0,ai:e=>2*e.N/(3*e.s),how:`ideal traffic: 2N³ FLOP over 3N² elements`},{id:`decode`,name:`LLM decode`,eff:.85,tc:!0,ai:e=>2*e.B/e.s,how:`2 FLOP per weight per sequence; each weight is read once per step`}],ce={fp64:`FP64`,fp32:`FP32`,tf32:`TF32 TC`,fp16:`FP16 TC`},le={fp64:`FP64`,fp32:`FP32 (CUDA cores)`,tf32:`TF32 Tensor Cores`,fp16:`FP16 Tensor Cores`},ue=(e,t,n)=>e&&n?t===`fp16`?`fp16`:t===`fp32`?`tf32`:`fp64`:t===`fp64`?`fp64`:`fp32`,de=430,Z=64,fe=118,Q=18,pe=46,$=-5,me=-1.5,he=3.3,ge=e=>e<1?`1/${Math.round(1/e)}`:e>=1024?`${e/1024}k`:String(e);function _e(){let[e,t]=(0,k.useState)(`a100`),[n,r]=(0,k.useState)(`fp32`),[i,a]=(0,k.useState)(!1),[o,s]=(0,k.useState)(!1),[l,p]=(0,k.useState)(10),[g,_]=(0,k.useState)(5),[v,S]=(0,k.useState)(0),[C,T]=(0,k.useState)(`saxpy`),[N,P]=(0,k.useState)({lx:1,lp:.2}),F=(0,k.useRef)(!1),I=A[e],ee={s:X[n],N:2**l,T:2**g,B:2**v},L=I.bw/1e3,R=ue(!0,n,i),z=Math.max(...Object.values(I.peak)),B=se.map((e,t)=>{let r=e.ai(ee),a=ue(e.tc,n,i),o=I.peak[a],s=Math.min(o,L*r),c=o/L;return{...e,n:t+1,ai:r,key:a,peak:o,att:s,ach:s*e.eff,rg:c,mem:r<c}}),H=2**N.lx,U=I.peak[R],W=Math.min(U,L*H),G=Math.min(10**N.lp,W),K={id:`my`,name:`your kernel`,n:`★`,ai:H,key:R,peak:U,att:W,ach:G,rg:U/L,mem:H<U/L,eff:G/W,how:`drag the ★ on the plot`},q=C===`my`?K:B.find(e=>e.id===C),[te]=D(de,(e,t,n)=>{let r=t-Z-fe,i=n-Q-pe,a=e=>Z+(Math.log2(e)-$)/17*r,s=e=>Q+i-(Math.log10(e)-me)/4.8*i;for(let t=-4;t<=12;t+=2)e.strokeStyle=m.grid,e.lineWidth=1,e.beginPath(),e.moveTo(a(2**t),Q),e.lineTo(a(2**t),Q+i),e.stroke(),h(e,ge(2**t),a(2**t),Q+i+14,{align:`center`,size:10.5,color:m.mute,mono:!0});for(let t=-1;t<=3;t++)e.strokeStyle=m.grid,e.beginPath(),e.moveTo(Z,s(10**t)),e.lineTo(Z+r,s(10**t)),e.stroke(),h(e,String(10**t),56,s(10**t),{align:`right`,size:10.5,color:m.mute,mono:!0});h(e,`arithmetic intensity (FLOP / byte) →`,Z+r/2,n-8,{align:`center`,size:11,color:m.mute}),e.save(),e.translate(14,Q+i/2),e.rotate(-Math.PI/2),h(e,`attainable TFLOP/s`,0,0,{align:`center`,size:11,color:m.mute}),e.restore(),e.save(),e.beginPath(),e.rect(Z,Q,r,i),e.clip();let l=I.peak[R]/L;e.fillStyle=d(m.b,.04),e.fillRect(Z,Q,a(l)-Z,i),e.fillStyle=d(m.g,.04),e.fillRect(a(l),Q,Z+r-a(l),i),o&&c(e,[[a(2**$),s(I.l2bw/1e3*2**$)],[a(z/(I.l2bw/1e3)),s(z)]],d(m.b,.8),1.8,[6,4]),c(e,[[a(2**$),s(L*2**$)],[a(z/L),s(z)]],m.a,3.2),Object.entries(I.peak).forEach(([t,n])=>{let i=t===R;c(e,[[a(n/L),s(n)],[Z+r,s(n)]],i?m.g:d(`#ffffff`,.28),i?3:1.5,i?null:[5,4])}),e.strokeStyle=d(m.g,.6),e.setLineDash([4,4]),e.beginPath(),e.moveTo(a(l),s(I.peak[R])),e.lineTo(a(l),Q+i),e.stroke(),e.setLineDash([]),h(e,`memory-bound`,74,32,{size:11,color:d(m.b,.9),weight:600}),h(e,`compute-bound`,Z+r-10,32,{size:11,color:d(m.g,.9),weight:600,align:`right`}),h(e,`ridge ${l.toFixed(+(l<10))}`,a(l)+6,Q+i-10,{size:10.5,color:m.g,mono:!0});let f=(t,n,r)=>{let i=a(u(t.ai,2**$,2**11.85)),o=s(t.ach),c=q&&q.id===t.id;e.strokeStyle=d(`#ffffff`,.35),e.setLineDash([2,3]),e.beginPath(),e.moveTo(i,o),e.lineTo(i,s(t.att)),e.stroke(),e.setLineDash([]),e.fillStyle=n,e.beginPath(),r?(e.moveTo(i,o-12),e.lineTo(i+12,o),e.lineTo(i,o+12),e.lineTo(i-12,o),e.closePath()):e.arc(i,o,10,0,7),e.fill(),c&&(e.strokeStyle=`#fff`,e.lineWidth=2.2,e.stroke()),h(e,t.n,i,o+.5,{align:`center`,size:10.5,color:`#0a0c14`,mono:!0,weight:800})};B.forEach(e=>f(e,e.mem?m.d:m.e,!1)),f(K,m.c,!0),e.restore();let p=[];Object.entries(I.peak).sort((e,t)=>t[1]-e[1]).forEach(([e,t])=>{let n=p[p.length-1];n&&Math.abs(n.p-t)/t<.01?n.names.push(ce[e]):p.push({p:t,names:[ce[e]],on:e===R}),e===R&&(p[p.length-1].on=!0)}),p.forEach(t=>{h(e,`${t.p>=100?Math.round(t.p):t.p} TF`,Z+r+8,s(t.p)-7,{size:10.5,color:t.on?m.g:m.mute,mono:!0,weight:700}),h(e,t.names.join(` / `),Z+r+8,s(t.p)+6,{size:9.5,color:t.on?m.g:m.dim,mono:!0})});let g=Math.min(Math.log2(z/L)-1.6,10);h(e,`HBM ${M(I.bw)}`,a(2**g)+10,s(L*2**g)+14,{size:10.5,color:m.a,mono:!0,weight:700})}),J=e=>{let{x:t,y:n}=f(e),r=e.currentTarget.getBoundingClientRect().width-Z-fe;P({lx:u($+(t-Z)/r*17,$,11.8),lp:u(me+(384-n)/366*4.8,me,he)})},Y=100*q.att/q.peak,ne=q.mem?q.ai<q.rg/8?(0,V.jsxs)(V.Fragment,{children:[`Far left of the ridge (`,(q.rg/q.ai).toFixed(0),`× below it): extra FLOPs are free, `,(0,V.jsx)(`b`,{children:`bytes are the cost`}),`. Fuse kernels, vectorise loads (`,(0,V.jsx)(`b`,{children:`float4`}),`), store data in fewer bits, never re-read what you can keep in registers/shared memory.`]}):(0,V.jsx)(V.Fragment,{children:`Memory-bound but near the ridge: raising reuse (tiling, register blocking, bigger tiles) pushes you right; after that you will hit the compute roof.`}):q.tc&&!i&&n!==`fp64`?(0,V.jsxs)(V.Fragment,{children:[`Compute-bound on CUDA cores — but this kernel could use `,(0,V.jsx)(`b`,{children:`Tensor Cores`}),`. Flip the toggle and watch the roof jump.`]}):(0,V.jsx)(V.Fragment,{children:`Compute-bound: memory is no longer the problem. Gains now come from instruction efficiency, ILP, and (if available) Tensor Cores / lower precision.`});return(0,V.jsxs)(V.Fragment,{children:[(0,V.jsx)(`canvas`,{...te,style:{...te.style,cursor:`crosshair`},onPointerDown:e=>{e.currentTarget.setPointerCapture(e.pointerId),F.current=!0,T(`my`),J(e)},onPointerMove:e=>{F.current&&J(e)},onPointerUp:()=>{F.current=!1}}),(0,V.jsx)(y,{items:[...B.map(e=>[e.mem?m.d:m.e,`${e.n} ${e.name}`]),[m.c,`★ your kernel (drag on the plot)`]]}),(0,V.jsxs)(x,{children:[(0,V.jsx)(E,{label:`GPU`,value:e,onChange:t,options:j}),(0,V.jsx)(E,{label:`Data type`,value:n,onChange:r,options:[[`fp64`,`FP64 (8 B)`],[`fp32`,`FP32 (4 B)`],[`fp16`,`FP16 (2 B)`]]}),(0,V.jsx)(O,{label:`Tensor Cores for GEMM-class kernels`,value:i,onChange:a}),(0,V.jsx)(O,{label:`Show L2 roof`,value:o,onChange:s}),(0,V.jsx)(E,{label:`Focus`,value:C,onChange:T,options:[...B.map(e=>[e.id,`${e.n} ${e.name}`]),[`my`,`★ your kernel`]]})]}),(0,V.jsxs)(x,{children:[(0,V.jsx)(b,{label:`GEMM size N`,min:6,max:14,value:l,onChange:p,fmt:e=>2**e}),(0,V.jsx)(b,{label:`Tile size T`,min:2,max:7,value:g,onChange:_,fmt:e=>2**e}),(0,V.jsx)(b,{label:`Decode batch`,min:0,max:9,value:v,onChange:S,fmt:e=>2**e})]}),(0,V.jsxs)(w,{children:[(0,V.jsxs)(`b`,{children:[q.n,` `,q.name]}),`: AI = `,(0,V.jsx)(`b`,{children:q.ai.toFixed(q.ai<10?3:1)}),` FLOP/B `,(0,V.jsxs)(`span`,{style:{opacity:.7},children:[`(`,q.how,`)`]}),(0,V.jsx)(`br`,{}),`Ceiling `,(0,V.jsx)(`b`,{children:le[q.key]}),` `,q.peak,` TF, ridge `,(0,V.jsx)(`b`,{children:q.rg.toFixed(+(q.rg<10))}),` FLOP/B → attainable min(`,q.peak,`, `,L.toFixed(2),`×`,q.ai.toFixed(2),`) = `,(0,V.jsx)(`b`,{children:q.att<1?(q.att*1e3).toFixed(0)+` GFLOP/s`:q.att.toFixed(1)+` TFLOP/s`}),` (`,Y.toFixed(+(Y<10)),`% of peak) →`,` `,(0,V.jsx)(`b`,{className:q.mem?`w`:`g`,children:q.mem?`memory-bound`:`compute-bound`}),`. `,ne]})]})}var ve=[[`reg`,`Register`],[`local`,`Local memory`],[`shared`,`Shared memory`],[`const`,`Constant memory`],[`global`,`Global memory`]],ye=[{ask:"Where does `x` live?",ans:`reg`,code:`__global__ void k(const float* a, float* o) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[i] * 2.0f;      // <- x
  o[i] = x;
}`,why:`A scalar automatic variable is assigned a **register**: fastest storage on the chip, private to the thread.`},{ask:"Where does `tile` live?",ans:`shared`,code:`__global__ void k(...) {
  __shared__ float tile[32][33];   // <- tile
  ...
}`,why:"`__shared__` puts one copy **per block** in on-chip SRAM (32 banks). The `33` pads rows to dodge bank conflicts (next chapters)."},{ask:"Where does `coeff` live?",ans:`const`,code:`__constant__ float coeff[16];       // <- coeff
// host: cudaMemcpyToSymbol(coeff, h_coeff, sizeof(h_coeff));
__global__ void k(...) { y = coeff[3] * x; }`,why:`**Constant memory** (64 KB total) is backed by device memory but read through a dedicated cache that **broadcasts** one address to a whole warp at register-like speed.`},{ask:"Where does `buf` live?",ans:`local`,code:`__global__ void k(const int* idx, float* o) {
  float buf[64];
  for (int j = 0; j < 64; ++j) buf[j] = j * 0.5f;
  o[threadIdx.x] = buf[idx[threadIdx.x] % 64];   // runtime index
}`,why:`Registers are not addressable, so a per-thread array indexed with a **runtime value** goes to **local memory**: private to the thread but physically in device memory (cached by L1/L2). Name says local, speed says global.`},{ask:"Where does the *data* behind `d_x` live?",ans:`global`,code:`float* d_x;
cudaMalloc(&d_x, n * sizeof(float));   // <- data behind d_x`,why:"`cudaMalloc` returns **global memory** (HBM / GDDR). Visible to every thread and kernel until `cudaFree`."},{ask:"Where does `acc` live?",ans:`reg`,code:`float acc[4] = {0, 0, 0, 0};
#pragma unroll
for (int j = 0; j < 4; ++j) acc[j] += a[i + j] * w;`,why:"After the loop is **unrolled** every index is a compile-time constant, so the compiler keeps `acc[0..3]` in four **registers**. Without the unroll (runtime index) it could fall back to local memory."},{ask:"Where does `s` live?",ans:`shared`,code:`extern __shared__ float s[];          // <- s
...
kernel<<<grid, block, 4096>>>(...);   // 4096 B of dynamic shared memory`,why:`The third launch parameter sizes **dynamic shared memory**: still one block-private copy in on-chip SRAM.`},{ask:"Where does `counter` live?",ans:`global`,code:`__device__ unsigned int counter;      // <- counter
__global__ void k() { atomicAdd(&counter, 1u); }`,why:"A `__device__` variable is statically allocated **global memory**, visible to all threads. The atomic is performed in the **L2** cache, which is why contended atomics are slow."},{ask:`A kernel needs ~300 live floats per thread but a thread may use at most 255 registers. Where does the excess go?`,ans:`local`,code:`// nvcc -Xptxas -v prints:
//   ptxas info : 168 bytes stack frame,
//                120 bytes spill stores, 132 bytes spill loads`,why:"The compiler **spills** to **local memory** (in device memory, cached in L1/L2). It also lowers occupancy. Check the spill counters in `-Xptxas -v` output."},{ask:"Where does `tid` live?",ans:`reg`,code:`int tid = threadIdx.x;   // <- tid`,why:"`threadIdx` itself is a read-only **special register**; copying it into a local gives an ordinary **register**."}];function be(){let[e,t]=(0,k.useState)(0),[n,r]=(0,k.useState)(null),[i,a]=(0,k.useState)({ok:0,n:0}),o=ye[e],s=e=>{n||(r(e),a(t=>({ok:t.ok+ +(e===o.ans),n:t.n+1})))};return(0,V.jsxs)(V.Fragment,{children:[(0,V.jsxs)(`div`,{style:{display:`grid`,gap:12},children:[(0,V.jsxs)(`div`,{style:{font:`600 11px var(--f-mono)`,letterSpacing:`.12em`,textTransform:`uppercase`,color:m.mute},children:[`Challenge `,e+1,` / `,ye.length,` · score `,i.ok,`/`,i.n]}),(0,V.jsx)(g,{lang:`cuda`,title:`Which memory space?`,src:o.code}),(0,V.jsx)(`div`,{style:{fontSize:15},children:(0,V.jsx)(_,{inline:!0,children:o.ask})})]}),(0,V.jsxs)(x,{children:[ve.map(([e,t])=>(0,V.jsxs)(T,{primary:n===e,onClick:()=>s(e),className:n?e===o.ans?`ok`:e===n?`bad`:``:``,children:[n&&e===o.ans?`✓ `:n===e?`✗ `:``,t]},e)),n&&(0,V.jsx)(T,{onClick:()=>{r(null),t(e=>(e+1)%ye.length)},children:`Next →`})]}),(0,V.jsx)(w,{children:n?(0,V.jsxs)(V.Fragment,{children:[(0,V.jsx)(`b`,{className:n===o.ans?`g`:`r`,children:n===o.ans?`Correct. `:`Not quite. `}),(0,V.jsx)(_,{inline:!0,children:o.why})]}):(0,V.jsxs)(V.Fragment,{children:[`Pick the memory space. Remember: `,(0,V.jsx)(`b`,{children:`scope`}),` (who can see it) and `,(0,V.jsx)(`b`,{children:`physical location`}),` (where the bytes are) are different questions.`]})})]})}function xe(){return(0,V.jsx)(S,{views:[{id:`h`,label:`3D: the memory hierarchy`,render:()=>(0,V.jsx)(ae,{})},{id:`l`,label:`Latency, bandwidth & Little’s law`,render:()=>(0,V.jsx)(oe,{})},{id:`r`,label:`Roofline explorer`,render:()=>(0,V.jsx)(_e,{})},{id:`s`,label:`Where does it live?`,render:()=>(0,V.jsx)(be,{})}]})}var Se={Lab:xe,vizTitle:`Descend the memory hierarchy, then find where your kernel sits on the roofline`,tryIt:[`In the 3D tab set **L1 hit = 0%** and **L2 hit = 0%**: every load pays the full HBM trip. Now raise the hit rates and watch **AMAT** collapse.`,`Click each slab and switch **slab width** between capacity, latency and bandwidth: capacity grows downward while bandwidth and speed shrink.`,`In the Little’s-law tab pick **HBM** with 4-byte loads: you need ~50 warps/SM. Raise **loads in flight** to 4 and the same bandwidth needs only a few.`,`In the Roofline tab compare **vector add** with **library GEMM** while sweeping **N**; then enable Tensor Cores with **FP16** and see the GEMM ridge move far to the right.`,`Select **LLM decode**, set FP16 and slide the **batch** from 1 to 512: it only becomes compute-bound near the ridge.`,`Drag the **★** into the compute-bound region with Tensor Cores off — see the advice change.`],theory:P,math:F,practice:I,code:[{title:`STREAM-style bandwidth benchmark: copy / scale / add / triad (+ float4)`,lang:`cuda`,note:`Measures effective HBM bandwidth and compares it with the theoretical number from the device attributes.`,src:ee},{title:`The memory spaces in one kernel: register, local, shared, constant, global`,lang:`cuda`,note:`Compile with -Xptxas -v to see register counts and spills.`,src:L},{title:`Pointer chasing: measure the latency ladder (L1 → L2 → HBM)`,lang:`cuda`,note:`Latency jumps as the working set outgrows each cache level. This is the real version of the lab ladder.`,src:R},{title:`Roofline probe: sweep arithmetic intensity and measure your own roofline`,lang:`cuda`,note:`Prints CSV: FMAs per element, FLOP/B, GFLOP/s, GB/s.`,src:z},{title:`Plot the measured roofline (matplotlib)`,lang:`python`,src:B}],quiz:[{q:`Order these from lowest to highest access latency on a modern NVIDIA GPU:`,options:[`HBM, L2, shared memory, registers`,`Registers, shared memory / L1, L2, HBM`,`Registers, L2, shared memory, HBM`,`Shared memory, registers, HBM, L2`],answer:1,why:`Registers (a few cycles) → shared memory / L1 (tens of cycles) → L2 (a couple of hundred) → HBM (several hundred cycles).`},{q:"A per-thread array `float buf[64]` indexed by a runtime value lives in:",options:[`Registers`,`Shared memory`,`Local memory (backed by device memory)`,`Constant memory`],answer:2,why:`Registers cannot be indexed dynamically. The array goes to **local memory**: private to the thread but physically in device memory, cached by L1/L2.`},{q:"SAXPY (`y = a*x + y`, FP32) on an A100 (≈19.5 TFLOP/s, ≈2 TB/s) is:",options:[`Compute-bound; it can reach ~19.5 TFLOP/s`,`Memory-bound at ≈0.17 FLOP/B; roughly 0.34 TFLOP/s attainable`,`Latency-bound at exactly 1 TFLOP/s`,`Limited by shared memory`],answer:1,why:`2 FLOP per 12 bytes = $0.167$ FLOP/B, far below the ridge ($\\approx9.6$). Attainable $\\approx 0.167\\times2.04\\text{ TB/s}\\approx0.34$ TFLOP/s ($\\approx1.7\\%$ of peak).`},{q:`The ridge point of the roofline is:`,options:[`The point where latency equals bandwidth`,`$\\pi/\\beta$: peak FLOP/s divided by peak bytes/s`,`The L2 cache size`,`The number of SMs`],answer:1,why:`Kernels with arithmetic intensity below $I^*=\\pi/\\beta$ are bandwidth-limited; above it they are limited by compute.`},{q:`Tiling a GEMM with $T\\times T$ shared-memory tiles changes its global-memory arithmetic intensity by about:`,options:[`×1 (no change)`,`×T`,`×T²`,`÷T`],answer:1,why:`Each element loaded into a tile is reused $T$ times, so global traffic drops by $T$ and AI rises by $T$.`},{q:`Why is batch-1 LLM decoding usually memory-bound?`,options:[`Softmax is slow`,`Each weight is read once but used for only ~2 FLOP, so AI ≈ 1 FLOP/B`,`Tensor Cores cannot run it`,`Kernel launch overhead dominates`],answer:1,why:`Per token every weight streams from HBM and performs one multiply-add. Batching reuses each weight across sequences and raises AI.`},{q:`Little’s law says to saturate memory bandwidth you need roughly ___ bytes in flight per SM.`,options:[`Bandwidth ÷ latency`,`Bandwidth × latency`,`Latency ÷ clock`,`The L2 size`],answer:1,why:`$\\text{in-flight}=\\text{bandwidth}\\times\\text{latency}$. That is why many resident warps, or many independent (and wide) loads per thread, are needed.`}]};export{Se as default};