## Exercises

**Q1.** Naive vs $T=16$ tiled GEMM at $N=2048$, FP32. How many bytes are loaded from global memory in each case, and what is the reuse factor?

<details>
<summary>Show answer</summary>

Naive loads $2N^3=2\cdot2048^3=17.2\times10^9$ floats $=68.7$ GB. Tiled loads $2N^3/T=17.2\times10^9/16=1.07\times10^9$ floats $=4.29$ GB. Reuse $T=16$. (Stores $N^2\cdot4=16$ MB are noise in both.)

</details>

**Q2.** A thread at `threadIdx=(3,5)` in block `(1,2)` with `TILE=16`, $N=64$. Which element of $C$ does it write, and which $A$ element does it load on k-tile $t=1$?

<details>
<summary>Show answer</summary>

`row=2·16+5=37`, `col=1·16+3=19` → writes $C_{37,19}$. On $t=1$, `As[5][3]=A[37,\ 16+3]=A_{37,19}$. (It still *reads* a whole row of `As` during the MAC: $A_{37,16\ldots31}$.)

</details>

**Q3.** Why are there **two** `__syncthreads()` in the k-loop? What goes wrong if you drop the second?

<details>
<summary>Show answer</summary>

The first makes the cooperative load visible before any thread starts the inner product. The second prevents the *next* iteration's load from overwriting `As`/`Bs` while another warp is still on the inner product. Dropping it is a write-after-read race on shared memory; `racecheck` will flag it. It often passes at TILE=8 (few warps) and fails at TILE=16.

</details>

**Q4.** Can you write `if (row < N && col < N) { load; __syncthreads(); … }` to skip padded threads?

<details>
<summary>Show answer</summary>

No — that is the deadlock rule. Threads with `row ≥ N` would skip the barrier while the others wait forever. Always barrier unconditionally; mask only the load (`?: 0`) and the store.

</details>

**Q5.** Shared memory for two FP32 $T\times T$ tiles at $T=32$? Could an A100 SM (164 KB shared, max 2048 threads, max 32 blocks) be limited by shared memory, threads, or blocks?

<details>
<summary>Show answer</summary>

$S=2\cdot32^2\cdot4=8$ KB. Threads/block $=1024$ ⇒ at most $2048/1024=2$ blocks/SM from the thread cap. Shared would allow $164/8=20$ blocks; the block cap is 32. **Limiter = threads.** (A $T=16$ kernel is also thread-limited: 256 thr ⇒ 8 blocks; shared would allow 82.)

</details>

**Q6.** A 5-point 1-D stencil (radius 2) on 256-thread blocks. What fraction of naive global traffic does a halo load achieve?

<details>
<summary>Show answer</summary>

Naive: 5 loads/point. Halo: $256+4=260$ loads for 256 outputs $\Rightarrow 260/(5\cdot256)=20.3\%$ of naive, essentially $1/5$, plus a $4/256=1.6\%$ overhead.

</details>

**Q7 (code).** Compile `tiled_gemm.cu` at `TILE=16` and `TILE=32` for the same $N$ (start with 1024, then 4096). Which is faster, and does the ratio match the $2\times$ traffic cut?

<details>
<summary>Show answer</summary>

At 1024 the $T=32$ kernel has only $(1024/32)^2=1024$ blocks — on an A100 (108 SMs) that is a handful of waves, so occupancy/tail effects compete with the traffic win. At 4096 there are 16× more blocks and $T=32$ should approach $2\times$ faster *if* HBM-bound, less if you are already near the compute roof (Math tab). Always measure both; bigger tiles are not automatically better.

</details>

**Q8 (code).** Run `syncthreads_race.cu` as written (32 threads) and then change both launches to `<<<1,256>>>` with `s[256]` and `(threadIdx.x+1)%256`. When does the unsynchronised variant go wrong? Confirm with `compute-sanitizer --tool racecheck`.

<details>
<summary>Show answer</summary>

At 32 threads, one warp: the race is often silent. At 256 threads, 8 warps: the first warps to pass the load read neighbours that later warps have not written — garbage / zeros / stale. `racecheck` reports a *Hazard* on `s` even in the 32-thread case.

</details>

## In practice

- **Start at 16×16 threads, one output per thread.** It is the picture in the lab. Then try 32×32, then (if you need more) register blocking or cuBLAS.
- **Pad the load, never the barrier.**
- **Coalesce on `threadIdx.x`:** `A[row, t+tx]` and `B[t+ty, col]` are the canonical pair. Swapping x/y in the launch (`dim3(TILE,TILE)` is `(x,y)`) is a classic slowdown.
- **Fuse after you tile.** A tiled GEMM + bias + ReLU in one kernel saves a whole extra read/write of $C$.
- **Prefer the library for plain GEMM**; steal this *pattern* for the kernels they don't ship (attention tiles, custom epilogues, stencils).

## Common pitfalls

- One `__syncthreads()` instead of two in the k-loop (race on the next load).
- Barrier inside a divergent `if` (deadlock, especially on the last blocks of a non-multiple $N$).
- `tile[tx][ty]` instead of `tile[ty][tx]` — transposes the cooperative load, usually uncoalesced.
- Assuming shared memory is "like a cache" that fills itself. Nothing lands in `__shared__` unless a thread writes it.
- Counting on a 32-thread test to prove the barrier is unnecessary.
- TILE that is not a multiple of 32: wasted warp lanes (Execution Model) *and* messier coalescing.
- Ignoring that naive $B$ is uncoalesced: a "tiled" kernel that still walks $B$ in the inner loop with a strided global index has not actually tiled $B$.
