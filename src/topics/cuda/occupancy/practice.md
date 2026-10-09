## Exercises

**Q1.** A100, 128 threads/block, 32 registers/thread, no shared memory. How many blocks per SM, how many warps, and what is the occupancy?

<details>
<summary>Show answer</summary>

4 warps/block. 32 × 32 = 1024 registers/warp, and $65536/1024 = 64$ warps exactly. $64/4 = 16$ blocks, which is also the warp-slot cap. **16 blocks, 64 warps, 100 %.** Registers and warp slots tie.

</details>

**Q2.** The compiler reports 37 registers instead of 32. What does the SM actually allocate, and what is the occupancy for the same 128-thread block?

<details>
<summary>Show answer</summary>

37 rounds up to **40** registers/thread (the unit is 8). That is 1280 registers/warp. $4\lfloor 65536/1280/4 \rfloor = 48$ warps, so **12 blocks, 75 %.** The limiter is the register file. The source change was 5 registers; the occupancy change was a quarter of the SM.

</details>

**Q3.** Same 37 registers, but the block is 1024 threads. Blocks, warps, occupancy?

<details>
<summary>Show answer</summary>

A 1024-thread block is 32 warps. Only **one** fits in the 48 warps the register file allows, leaving 16 slots empty. Occupancy is $32/64 = 50\%$. A smaller block would have used those 16 slots.

</details>

**Q4.** 128 threads, 32 registers, 32 KB of shared memory, A100. Which resource limits, and what is the occupancy?

<details>
<summary>Show answer</summary>

Charged shared memory is $32768 + 1024 = 33792$ bytes. $\lfloor 164 \times 1024 / 33792 \rfloor = 4$ blocks. Warp slots would allow 16, so **shared memory** limits. 4 × 4 = 16 warps, occupancy **25 %.**

</details>

**Q5.** Take the kernel from Q2 (128 threads, 37 registers) and move it to an RTX 4090. Occupancy?

<details>
<summary>Show answer</summary>

The register file still allows 48 warps, and a 4090 SM holds 48. Occupancy is $48/48 = 100\%$. Same kernel, higher percentage, because the maximum shrank from 64 to 48. Check achieved bandwidth or time, not the percentage, when you compare those two chips.

</details>

**Q6.** A dependent FMA has a 4-cycle latency and the kernel has no other independent instruction ($I = 1$). How many resident warps keep four schedulers busy? Is 100 % occupancy useful here?

<details>
<summary>Show answer</summary>

$W = \lceil 4 \times 4 / 1 \rceil = 16$ warps. That is 25 % of an A100. More resident warps have nothing to hide. This is the case for cutting occupancy on purpose (fewer registers, more of the L1 left for data).

</details>

**Q7.** An A100 SM must keep about 6.5 KB in flight to fill HBM. How many coalesced 128-byte warp loads is that, and why doesn't the same count cover a 4-byte pointer chase?

<details>
<summary>Show answer</summary>

$6700 / 128 \approx 52$ warp loads. One resident warp per load covers a streaming kernel. A chase that puts a single 4-byte load in flight per warp keeps $52 \times 4 \approx 208$ bytes moving, a few percent of the pipe. That kernel needs more independent misses per thread. Occupancy cannot invent them.

</details>

**Q8.** `__launch_bounds__(256, 8)` is set on a kernel whose unconstrained compile uses 80 registers and spills nothing. What is the compiler being asked to do?

<details>
<summary>Show answer</summary>

Keep **8 blocks of 256 threads** resident. That is 64 warps, the whole A100 SM, so the register budget is about $65536 / 2048 = 32$ registers/thread before rounding. The compiler will cut from 80 toward that budget and **spill** the rest to local memory. Occupancy goes up; local-memory traffic may eat the gain. Read the `ptxas` spill lines.

</details>
