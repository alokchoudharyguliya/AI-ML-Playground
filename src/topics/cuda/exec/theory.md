## Why GPUs look the way they do

A CPU is built to make **one thread fast**: big caches, branch prediction, out-of-order execution — a latency machine with a handful of cores. A GPU is built for **throughput**: thousands of simple lanes, small caches, and *hide* memory latency by always having other work ready. A modern data-centre GPU (H100) has 132 Streaming Multiprocessors (SMs), each able to keep up to 2048 threads resident — **~270,000 threads in flight**. Latency isn't avoided; it is overlapped.

CUDA exposes this with the **SIMT** (Single Instruction, Multiple Threads) model: you write the program for *one thread* (a **kernel**), then launch it for many.

## The thread hierarchy

```
Grid  (one kernel launch)
 └─ Thread blocks (a.k.a. CTAs)     independent, any order, any SM
     └─ Warps                       32 threads, scheduled together
         └─ Threads                 each with its own registers and indices
```

| Level | Size limits | What it shares | How it synchronises |
|---|---|---|---|
| **Thread** | — | its own registers / local memory | — |
| **Warp** | 32 threads | instruction stream; `__shfl_*` registers | `__syncwarp()`, implicit lockstep |
| **Block** | ≤ 1024 threads, 1–3-D | **shared memory**, L1 | `__syncthreads()` |
| *Cluster* (Hopper+) | ≤ 8–16 blocks | distributed shared memory | cluster barriers |
| **Grid** | up to $2^{31}-1$ blocks in x | global memory | kernel boundaries, atomics, cooperative groups |

Every thread can read the built-in variables `threadIdx`, `blockIdx`, `blockDim`, `gridDim` and derives *which data it owns*:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;   // global 1-D index
if (i < n) y[i] = a * x[i] + y[i];               // bounds check: the grid is rounded up!
```

In the 3D lab, tile = block, cubes = threads; colour = warp. Hover a cube to see that formula evaluated for it.

## Launching

```cuda
saxpy<<<gridDim, blockDim, sharedBytes, stream>>>(args…);
dim3 block(16, 16);                              // 256 threads
dim3 grid((W + 15) / 16, (H + 15) / 16);         // round up so every pixel is covered
```

Kernel launches are **asynchronous**: the CPU continues immediately; use streams/events or `cudaDeviceSynchronize()` to wait. Launch overhead is ~3–10 µs, so tiny kernels are latency-dominated.

## What happens on the hardware

1. The **block scheduler** assigns each block to an SM that has enough **registers, shared memory, warp slots and block slots** (see *Occupancy*). If none is free, the block waits in the queue.
2. Inside the SM the block's threads are split into **warps** (consecutive `threadIdx.x` first, then y, then z).
3. Four **warp schedulers** per SM pick, every cycle, a *ready* warp (operands available) and issue one instruction for all 32 lanes. A warp waiting on memory is simply skipped for another — **zero-cost context switching**, because every warp keeps its registers resident.
4. When all warps of a block finish, its resources are freed and the next queued block is placed.

Second lab tab: blocks are packed onto SMs in "waves". If the number of blocks isn't a multiple of the machine's slots, the **last wave runs partly empty** — the *tail effect*. It matters when blocks are few and long; with thousands of short blocks it vanishes.

<div class="callout">

**Rules that fall out of this design.**
1. Blocks must be **independent** — no ordering, no global barrier inside a kernel (except cooperative launches).
2. Threads in a block may **cooperate** via shared memory + `__syncthreads()`.
3. **Always** guard against out-of-range indices when `n` isn't a multiple of the block size.
4. Use block sizes that are **multiples of 32** (128–512 is the sweet spot) so no warp is partially empty.

</div>

## SM anatomy (Hopper H100, per SM)

- **4 sub-cores**, each with a warp scheduler, 32 FP32 lanes (128 FP32 per SM), 16 INT32, 1 Tensor Core, register file slice (64K × 32-bit registers per SM = 256 KB).
- **Up to 228 KB shared memory/L1** (software-managed), 64 resident warps (2048 threads), 32 resident blocks.
- Special function units, load/store units, and the **Tensor Memory Accelerator (TMA)** for async bulk copies.

## Writing grid-stride loops

A robust pattern when `n` may exceed the launched thread count, or to reuse a fixed, occupancy-sized grid:

```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = a * x[i] + y[i];
```

## Compilation pipeline

`nvcc` splits host and device code; device code compiles to **PTX** (virtual ISA) and then to **SASS** (the real machine code for a specific architecture, e.g. `sm_80`, `sm_90`). Compile for your target: `nvcc -O3 -arch=sm_90 kernel.cu`. Inspect with `cuobjdump --dump-sass`, profile with **Nsight Systems** (timeline) and **Nsight Compute** (per-kernel counters).

## Error handling you should always do

Kernel launches don't return errors directly. Check `cudaGetLastError()` right after the launch (configuration errors) and the result of `cudaDeviceSynchronize()` / the next API call (execution errors). Wrap calls in a macro like `CUDA_CHECK(...)` (see the code tab).
