## Exercises

**Q1.** You process a $1920\times1080$ image with $16\times16$ blocks. What is the grid size, how many threads are launched, and how many do no useful work?

<details>
<summary>Show answer</summary>

Grid $=(\lceil1920/16\rceil,\lceil1080/16\rceil)=(120,68)$ = 8,160 blocks × 256 = 2,088,960 threads. Useful: $1920\cdot1080=2{,}073{,}600$. Idle: $15{,}360$ ($0.7\%$) — the bottom 8 rows of the last block-row (1088−1080 = 8 rows × 1920). They must be guarded by `if (x < W && y < H)`.

</details>

**Q2.** A kernel uses blocks of 300 threads. How many warps does each block use and what fraction of issue slots is wasted?

<details>
<summary>Show answer</summary>

$\lceil300/32\rceil=10$ warps = 320 lanes; 20 are idle $\Rightarrow6.25\%$ waste. Switching to 288 (9 warps) or 320 (10 warps) fixes it — or use 256/512, which are multiples of 32 and typical.

</details>

**Q3.** For a thread at `threadIdx=(5,3)` in a block of `dim3(16,8)`, what are its linear thread id, warp, and lane?

<details>
<summary>Show answer</summary>

$\text{tid}=5+3\cdot16=53$ → warp $\lfloor53/32\rfloor=1$, lane $53\bmod32=21$. Note a warp spans *two rows* of the 16-wide block here (rows 2 and 3): neighbouring `threadIdx.y` threads can share a warp.

</details>

**Q4.** 5,000 blocks, A100 (108 SMs), kernel allows 6 resident blocks/SM. How many waves, and what fraction of the final wave is used?

<details>
<summary>Show answer</summary>

Capacity $=108\cdot6=648$ per wave. Waves $=\lceil5000/648\rceil=8$ (7 full = 4536, last has 464 blocks) → last wave utilisation $464/648=72\%$; overall efficiency $5000/(8\cdot648)=96\%$.

</details>

**Q5.** Why can't a kernel use `__syncthreads()` to synchronise the whole grid? What are the alternatives?

<details>
<summary>Show answer</summary>

Blocks can be scheduled in waves — block 5000 may not even start until block 0 has finished — so a grid-wide barrier would deadlock. Alternatives: split into **multiple kernels** (the kernel boundary is a global barrier), **atomics** with a counter ("last block done" pattern), or a **cooperative launch** (`cudaLaunchCooperativeKernel`, `grid.sync()`) which requires the whole grid to be co-resident.

</details>

**Q6 (code).** Write a grid-stride SAXPY kernel and launch it with a grid size of `numSMs * 8` blocks. Why is this robust?

<details>
<summary>Show answer</summary>

See `vecadd.cu`. A fixed, hardware-sized grid amortises launch overhead, keeps all SMs busy, handles any `n`, and lets you tune for occupancy; each thread strides by `gridDim.x*blockDim.x`, which also keeps global accesses coalesced.

</details>

## In practice

- **Start with 256 threads/block** (or 128–512), measure, then tune. For 2-D problems use 16×16 or 32×8 (x contiguous for coalescing — see the next chapters).
- **Check errors** after every launch during development; run `compute-sanitizer` for out-of-bounds and race detection.
- **Profile before optimising**: Nsight Systems for the timeline (launch gaps, copies), Nsight Compute for per-kernel limiters.
- **Prefer libraries**: cuBLAS, cuDNN, CUTLASS, CUB, Thrust, cuFFT — hand-written kernels rarely beat them for standard operations, but fused custom kernels (Triton/CUDA) win for non-standard fusions.
- **Triton / CuTe / Warp** let you write tile-level GPU code in Python/C++ templates when full CUDA is overkill.

## Common pitfalls

- Forgetting the `if (i < n)` guard → out-of-bounds writes (silent corruption).
- Using `int` indices for arrays > 2³¹ elements (use `size_t`/`long long`).
- Assuming blocks run in order or concurrently.
- Reading results on the host without synchronising (or copying with a non-blocking stream).
- Launching with 0 blocks when `n==0` (`<<<0,…>>>` is an error).
- Block size > 1024 or `dim3` order mix-ups (`dim3(x,y)` – x is the fastest-varying thread dimension).
- Timing with the CPU clock without `cudaDeviceSynchronize()`; use `cudaEvent`s.
