## Resident warps are the latency budget

An SM does not context-switch the way a CPU does. Every resident warp keeps its registers, so switching to a ready warp is free — and a warp that is not resident cannot be switched to at all. **Occupancy** is how full that set is:

$$
\text{occupancy} = \frac{\text{resident warps per SM}}{\text{maximum warps per SM}}.
$$

The maximum is **64** on A100 and H100 (2048 threads) and **48** on the RTX 4090 (1536 threads). Four resources can stop you earlier. The lab takes the minimum:

| Resource | What one block consumes | A100 / H100 ceiling | RTX 4090 |
|---|---|---|---|
| Registers | threads × registers, rounded up | 65 536 × 32-bit | same |
| Shared memory | your allocation + 1 KB reserve | 164 KB / 228 KB | 100 KB |
| Warp slots | `ceil(threads / 32)` warps | 64 | 48 |
| Block slots | one | 32 | 24 |

A block of 1024 threads is 32 warps, so an A100 holds **two** of them even when registers and shared memory would allow more. A block of 32 threads is one warp, so the 32-block cap is what stops you, not the 64 warp slots: half the warp slots stay empty.

<div class="callout">

**Full occupancy is not the goal.** It is the budget of ready warps you can spend on stalls. A dependent FMA is ~4 cycles; sixteen resident warps already keep an A100's four schedulers busy. A dependent trip to HBM is a few hundred cycles, and 64 warps do not cover it. Past the stall you actually have, extra warps only help if they don't push the compiler into local memory.

</div>

## How the register file is actually handed out

The compiler's register count is rounded **up to a multiple of 8** per thread (256 registers per warp). The number of warps that fit in the 64K register file is then rounded **down to a multiple of 4**.

So 32 registers/thread is exact: 1024 registers/warp, 64 warps, 100 % on an A100 if nothing else binds. **37 becomes 40.** That is 1280 registers/warp, only 48 warps, **75 %**. The source did not ask for 25 % fewer warps; the allocation unit did.

`__launch_bounds__(maxThreads, minBlocks)` tells the compiler to stay inside a register budget that allows `minBlocks` resident blocks, spilling to local memory if it has to. Spills bring the occupancy back and add a round trip through the same path as global memory. `ptxas` prints both numbers: registers per thread, and spill stores/loads. Trust the spill count over the occupancy percentage.

## Shared memory and the 1 KB you didn't allocate

CUDA reserves **1 KB of shared memory per block**. That is why the per-block maximum is 163 KB on A100 (164 − 1), 227 KB on H100 and 99 KB on the 4090. Static `__shared__` is also capped at 48 KB until the kernel opts in with `cudaFuncSetAttribute`.

The carveout is the split of one SRAM array between shared memory and L1. Asking for 100 KB of shared memory on an A100 shrinks L1. A kernel that was L1-resident can get slower as its occupancy goes up.

## Waves

The block scheduler places `SMs × blocksPerSM` blocks at a time. That many blocks is one **wave**. A grid that is not a multiple of the wave leaves the last wave partly empty — the tail. It matters when the grid is small or each block runs for a long time. A persistent kernel (a fixed grid of about one wave, with a grid-stride loop) simply does not have a tail.

## Hiding the stall you actually have

Each SM has four warp schedulers. A warp that issues a load and then needs the result is not eligible again for the latency of that load, unless it has other independent instructions to issue first. With latency $L$ cycles and $I$ independent instructions before the use:

$$
\text{warps to keep every scheduler busy} \approx 4 \cdot \frac{L}{I}.
$$

| Stall | $L$ on A100 (≈) | $I = 1$ | What actually saves you |
|---|---|---|---|
| Dependent FMA | 4 | 16 warps | almost any legal block |
| Shared memory | ~20 | ~80 warps | a few independent uses, or just not depending on the load immediately |
| HBM | ~500 | ~2000 warps | impossible by occupancy alone |

Two thousand warps is not a real target. A streaming kernel does not need the schedulers busy: it needs the **memory pipe** full. Little's law, from the memory chapter, says an A100 SM must keep about **6.5 KB** in flight to hit HBM peak, which is about **52** coalesced 128-byte warp loads. Occupancy covers that. It does not cover a pointer chase, where each warp has one 4-byte miss in flight and then waits. That kernel needs more independent misses per thread, not a higher percentage.

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` applies this arithmetic to a real kernel, including its compiled register count. The calculator in the lab is the same model with the inputs in your hands.
