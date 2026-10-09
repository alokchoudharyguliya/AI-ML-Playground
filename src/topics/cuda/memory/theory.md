## The memory wall

In the last chapter you saw that a GPU hides latency with thousands of threads. This chapter asks the follow-up: *what are those threads waiting for?* Almost always: **data**.

Compute throughput has grown much faster than memory bandwidth. From A100 to H100 the peak FP32 rate went from 19.5 to 67 TFLOP/s (**3.4×**) and the FP16 Tensor rate from 312 to 989 TFLOP/s (**3.2×**), while HBM bandwidth went from about 2.0 to 3.35 TB/s (**1.6×**). Each generation, a fixed kernel gets *more* memory-bound, not less. Moving a 32-bit word from DRAM also costs on the order of **100× more energy** than the floating-point operation you do with it.

So the central skill in GPU optimisation is not "make the math faster". It is **"move fewer bytes, and move them from closer."**

## The ladder

Every byte a thread uses lives somewhere on this ladder. Faster means smaller, closer to the ALUs, and shared by fewer threads.

| Space | Physical location | Scope | Lifetime | Approx. latency | How you get it |
|---|---|---|---|---|---|
| **Register** | register file in each SM sub-core (256 KB/SM) | one thread | thread | ~4 cycles | automatic locals |
| **Shared memory** | on-chip SRAM in the SM | one block | block | ~20–30 cycles | `__shared__` |
| **L1 cache** | *same* SRAM array as shared memory | one SM | transparent | ~30–40 cycles | automatic |
| **Constant** | device memory + small constant cache | whole grid (read-only) | application | cache hit ≈ L1-like, broadcast | `__constant__` |
| **Local** | **device memory** (cached in L1/L2) | one thread | thread | like global | dynamic-index arrays, spills |
| **L2 cache** | chip-wide SRAM (40 MB A100, 50 MB H100) | whole GPU | transparent | ~200–270 cycles | automatic |
| **Global (HBM/GDDR)** | off-chip DRAM stacks | whole GPU | until `cudaFree` | ~400–600 cycles | `cudaMalloc`, `__device__` |
| **Host over PCIe / NVLink** | CPU DRAM | CPU + GPU | allocation | microseconds | `cudaMemcpy`, pinned / managed |

Latencies are *approximate*, from public microbenchmark studies; they depend on clock, access pattern and driver. The point is the **order of magnitude**: from registers to HBM is roughly a **100× latency** gap, and about **50×** in bandwidth.

<div class="callout">

**Scope ≠ location.** *Local* memory is private to a thread, yet physically lives in slow device memory. *Constant* memory is read-only and global in scope, yet is served by a cache. Always ask two questions: *who can see it?* and *where are the bytes?*

</div>

Bandwidth also falls down the ladder, and the numbers are aggregate across the whole GPU. On an A100, registers can deliver on the order of 100 TB/s of operands, shared memory/L1 about 19 TB/s, L2 about 5 TB/s, HBM about 2 TB/s. Open the **3D** and **latency** tabs of the lab to see both axes at once, and switch to an H100 or RTX 4090.

## Anatomy of one global load

When a warp executes `y = x[i]` where `x` is in global memory:

1. The **load/store unit** collects the 32 lane addresses and merges them into the minimum number of **128-byte cache lines**, tracked as **32-byte sectors**. How well this merging works is *coalescing* (next chapters).
2. **L1** is checked. A hit returns in tens of cycles. L1 is *write-through*: stores always continue to L2.
3. On a miss the request crosses the on-chip network to the right **L2 slice**. L2 is write-back and is the point of coherence for the whole GPU: atomics execute here.
4. On an L2 miss, the **memory controller** fetches the sectors from an **HBM stack** (or GDDR chip).
5. Data returns up the chain; the warp, which was *parked* the entire time, becomes eligible again.

That 5-step round trip is the 400–600 cycles. Other warps run meanwhile. This is exactly the latency hiding from the *Execution Model* chapter, now with a price tag attached.

## Registers

The register file is the largest and fastest storage on the SM: 64K 32-bit registers per SM (256 KB), more than the L1/shared array. Each thread gets up to **255** registers.

- **Registers are not addressable.** Use `float a[4]` with only compile-time-constant indices (e.g. after `#pragma unroll`) and the compiler keeps it in registers. Index it with a runtime value and the array moves to **local memory**.
- If a kernel needs more registers than are allowed, values **spill** to local memory (slow, and visible in `nvcc -Xptxas -v` as *spill stores/loads*).
- Registers are allocated per thread for the **lifetime of the block**. More registers per thread means fewer resident warps (the *Occupancy* chapter).

## Shared memory and L1

Shared memory and L1 are carved from one physical array per SM (192 KB on A100, 256 KB on H100). You choose the split per kernel:

```cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50);          // percent hint
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);     // opt in above 48 KB
```

Shared memory is a **software-managed scratchpad**: you decide what is cached and when. It is organised into **32 banks** (4 B wide). Its payoff is *data reuse inside a block*, covered in **Tiled MatMul**; its pitfall is *bank conflicts*, covered in **Coalescing & Bank Conflicts**.

## Constant and read-only memory

`__constant__` data (64 KB total) is stored in device memory but read through a small **constant cache** that can **broadcast** one value to all 32 lanes in a single access. It is ideal for coefficients and small tables that every thread reads *uniformly*, and a poor fit when lanes read different addresses (that serialises).

For large read-only arrays, declaring pointers `const __restrict__` or loading with `__ldg()` routes loads through the read-only data path, which lets the compiler assume no aliasing.

## L2 and HBM

**L2** (40 MB on A100, 50 MB on H100, 72 MB on RTX 4090) is shared by all SMs. If your *working set* fits in L2, repeated passes run at L2 bandwidth rather than HBM bandwidth (toggle **Show L2 roof** in the roofline tab). On A100 and later you can **reserve part of L2 for persisting data** with `cudaAccessPolicyWindow`.

**HBM** is a stack of DRAM dies on a silicon interposer next to the GPU. The enormous 5,120-bit bus (A100) is what delivers terabytes per second, but each byte is still ~100× more expensive in time and energy than a register operand.

### Hopper additions

- **Thread-block clusters and distributed shared memory** let blocks on neighbouring SMs read each other's shared memory, a new rung between shared memory and L2.
- The **Tensor Memory Accelerator (TMA)** copies whole tiles global ⇄ shared asynchronously, freeing threads from issuing loads.

## Latency × bandwidth = parallelism

*Little's law* (see the Math tab) says that to keep a pipe full you need `bandwidth × latency` bytes **in flight**. For HBM on an A100 that is roughly **6–7 KB per SM**. You can supply it with:

- **many warps** (occupancy), or
- **many independent loads per thread** (instruction-level parallelism, ILP), or
- **wider loads** (`float4` moves 16 B per instruction).

The second tab of the lab lets you trade these against each other.

## The roofline model

Raw numbers do not tell you what to optimise. The **roofline model** does. For a kernel, count

- $W$: floating-point operations performed, and
- $Q$: bytes moved to/from the memory level you care about (usually HBM).

Their ratio is the **arithmetic intensity** $I = W/Q$ in FLOP/byte. A GPU has a peak compute rate $\pi$ and a peak bandwidth $\beta$. Then

$$
P \le \min\big(\pi,\ \beta\, I\big).
$$

On a log-log plot this is a **roof**: a slanted line (bandwidth-limited) meeting a flat line (compute-limited) at the **ridge point** $I^*=\pi/\beta$. Plot your kernel at its intensity:

- **Left of the ridge:** *memory-bound*. Making arithmetic faster does nothing. Only moving fewer bytes helps.
- **Right of the ridge:** *compute-bound*. Memory traffic is not the issue; you need better instruction throughput or Tensor Cores.

On the H100, the FP32 ridge is about **20 FLOP/B** but the FP16 Tensor ridge is about **300 FLOP/B**: with Tensor Cores, almost everything becomes memory-bound, which is why data movement dominates modern AI kernels.

### What to do about it

| Where you are | What helps | What does not |
|---|---|---|
| **Memory-bound**, far left | fuse kernels, avoid re-reads, vectorise (`float4`), fewer bits (FP16/FP8), better caching/reuse, compression | more FLOP/s, Tensor Cores |
| **Memory-bound**, near ridge | tiling, register blocking, larger tiles (raise $I$) | micro-optimising instructions |
| **Compute-bound** on CUDA cores | Tensor Cores, lower precision, ILP, fewer instructions per FLOP | memory tweaks |
| **Neither** (far below the roof) | latency problem: occupancy/ILP, coalescing, bank conflicts, divergence, launch overhead | changing precision |

The last row is important: the roofline gives an *upper bound*. A dot far below it means something else (latency, divergence, bad access patterns) is the limiter. That is the topic of the next chapters.

## How to measure it

- **Effective bandwidth** = bytes actually read + written ÷ kernel time (use `cudaEvent` timing). Compare it with the theoretical peak from the device attributes (see `bandwidth_stream.cu`). Well-tuned streaming kernels reach **85–95 %** of peak.
- **Nsight Compute** reports a *Speed of Light* section (percent of peak compute and memory throughput), a *Memory Workload Analysis* section (hit rates, bytes per level) and a **Roofline chart** for the kernel, so you do not have to count bytes by hand.
- **Microbenchmarks**: `pointer_chase.cu` reveals each cache level's size and latency; `roofline_probe.cu` measures your GPU's real roof.

<div class="callout tip">

**A 30-second triage.** (1) Estimate FLOPs and bytes → arithmetic intensity. (2) Compare with the ridge. (3) Measure achieved GB/s and GFLOP/s. If you are near the roof on the correct side, you are done. If you are far below it, look for latency problems. If you want to go *above* it, change the algorithm (fusion, tiling, lower precision) so that $I$ increases.

</div>
