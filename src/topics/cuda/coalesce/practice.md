## Exercises

**Q1.** A warp loads 32 `float`s at stride 1, starting at element 0. How many 32-byte sectors, and what is the efficiency?

<details>
<summary>Show answer</summary>

128 consecutive bytes = **4 sectors**. Useful = 128 B, so $\eta = 100\%$. This is the access `in[blockIdx.x * blockDim.x + threadIdx.x]` generates.

</details>

**Q2.** Same warp, but the pointer is offset by one float (`b = 1`). Sectors and efficiency?

<details>
<summary>Show answer</summary>

Bytes $[4, 132)$ cover sectors 0, 1, 2, 3 and 4: **5 sectors**, 160 B moved, $\eta = 128/160 = 80\%$. Every subsequent warp of a misaligned column pays the same extra sector.

</details>

**Q3.** Stride 4, `float`, aligned. A kernel copies $2^{22}$ outputs. How many bytes does the memory system move, against how many useful bytes?

<details>
<summary>Show answer</summary>

$\eta = 1/4$. Useful $= 2^{22}\cdot 4 = 16$ MiB. Moved $= 64$ MiB. From the table, stride 4 is 16 sectors per warp against 4 for stride 1.

</details>

**Q4.** `float s[32][32]`. A warp reads `s[threadIdx.x][5]` (each lane a different row, column 5). Conflict degree?

<details>
<summary>Show answer</summary>

Word index $= \ell \cdot 32 + 5$, bank $= 5$ for every lane, 32 different addresses. **32-way** conflict. The column index does not matter when the leading dimension is a multiple of 32.

</details>

**Q5.** You change the declaration to `s[32][33]` and read `s[threadIdx.x][5]`. Degree now? What about `s[32][34]`?

<details>
<summary>Show answer</summary>

$L = 33$: bank $= (\ell \cdot 33 + 5) \bmod 32 = (\ell + 5) \bmod 32$. All 32 banks, **degree 1**. $L = 34$: bank $= (2\ell + 5) \bmod 32$, only 16 distinct banks, **degree 2**. Padding must be odd, not merely "a bit bigger".

</details>

**Q6.** `float tile[32][16]`. A warp reads the column `tile[threadIdx.x][0]`. Conflict degree?

<details>
<summary>Show answer</summary>

Bank $= (\ell \cdot 16) \bmod 32$, which is 0 for even $\ell$ and 16 for odd $\ell$. Sixteen distinct rows on each bank: **degree 16**. Half the pain of $L = 32$, still a 16× serialisation. `tile[32][17]` (or `[16][17]`) brings it back to 1.

</details>

**Q7.** Why is `As[ty][k] * Bs[k][tx]` in the *unpadded* tiled GEMM from the previous chapter not a 32-way conflict, while `tile[tx][ty]` in the transpose is?

<details>
<summary>Show answer</summary>

In the GEMM a warp has one `ty` and consecutive `tx`. `As[ty][k]` is **one address** read by every lane: a broadcast, degree 1. `Bs[k][tx]` is a consecutive row: degree 1. The transpose reads `tile[tx][ty]` — consecutive lanes, *different rows*, same column — which is the column pattern, degree 32 when the width is 32.

</details>

**Q8 (code).** Build `transpose.cu` for $N = 2048$ and $N = 4096$. Order the three kernels by GB/s and say which limit each one hits.

<details>
<summary>Show answer</summary>

Naive is limited by uncoalesced stores (stride $N$): expect a small fraction of peak. Tiled without padding fixes global traffic but the shared column read is 32-way; it lands in between. Padded tiled should approach the STREAM copy bandwidth from the memory chapter (both kernels move $2 N^2$ bytes, coalesced). The gap between "tiled, no pad" and "padded" is the bank-conflict term with no change in global bytes.

</details>

## In practice

- **Profile the pattern, then the kernel.** A one-warp microbenchmark (`stride_copy.cu`, `smem_banks.cu`) tells you the factor before you rewrite a 200-line kernel.
- **`float4` only helps if it is aligned and stride-1.** A vectorised load of a stride-2 array just wastes a wider transaction.
- **SoA by default** for any per-thread record you stream. Convert AoS at the boundary.
- **Pad every shared tile whose second index is not `threadIdx.x`.** If you only ever read rows, padding is pure overhead.
- **Nsight Compute → Memory Workload Analysis** splits "sectors" (global) from "bank conflicts" (shared). The two fixes are different; the roofline will not tell them apart.

## Common pitfalls

- Assuming "consecutive `threadIdx.y`" coalesces. The warp's consecutive lanes are `threadIdx.x`.
- Counting a broadcast as a conflict, and "fixing" it with padding that changes nothing.
- Padding by 1 on an already-odd width, or padding by 2 (makes a 2-way conflict).
- Declaring the pad but still indexing `[TILE][TILE]` in the column read.
- Calling a kernel memory-bound because it is slow, when it is actually moving 8× more bytes than you think, or serialising 32 ways inside the SM.
- Testing bank conflicts with a single timing of 32 threads and no inner loop: launch and latency noise swamp a conflict that is obvious over a few thousand iterations.
