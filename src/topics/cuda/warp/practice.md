## Exercises

**Q1.** A warp executes `if (threadIdx.x < 16) heavy(); else heavy();` and `heavy` is 40 instructions. How many instruction issues does the warp spend inside the region, and what is the lane utilization?

<details>
<summary>Show answer</summary>

Both arms are taken, so the warp issues $40 + 40 = 80$ instructions. Each issue does useful work on only 16 lanes, so utilization is **50 %**. A uniform warp would have issued 40.

</details>

**Q2.** Same region, but the predicate is `threadIdx.x < 32` (always true). How many issues?

<details>
<summary>Show answer</summary>

Only the `if` arm is taken. The `else` is never issued. **40** instructions, utilization **100 %**. Divergence is about taken paths, not about the source text containing an `else`.

</details>

**Q3.** `switch (threadIdx.x % 4)` has four arms of 10 instructions, and every residue occurs in the warp. Issues and utilization?

<details>
<summary>Show answer</summary>

$P = 4$, so $T = 40$ issues and utilization is $1/4 = 25\%$. Eight lanes are active on each pass.

</details>

**Q4.** All-ones registers, `__shfl_down_sync` with offsets 16, 8, 4, 2, 1. What does lane 0 hold after the offset-8 step, and after the full sweep?

<details>
<summary>Show answer</summary>

After offset 16, lanes 0–15 hold 2. After offset 8, lanes 0–7 hold **4**. After all five offsets lane 0 holds **32**, the sum of the warp. Lane 31 still holds 1: nothing was ever added into it.

</details>

**Q5.** Same input, `__shfl_xor_sync` with the same offsets. What does lane 31 hold at the end, and how many lanes were idle on the last step?

<details>
<summary>Show answer</summary>

Every lane holds **32**. The partner of lane $\ell$ is $\ell \oplus 1$, which is always inside the warp, so **no lane is idle** on any step. That is the difference from the down-sweep.

</details>

**Q6.** A block of 256 threads reduces with the shuffle pattern. How many `__syncthreads()` calls, and how many shared-memory floats?

<details>
<summary>Show answer</summary>

256 / 32 = 8 warps. Each warp's lane 0 writes one partial: **8 floats**. Then **one** `__syncthreads()`, and warp 0 reduces those 8 values (lanes 8–31 of that warp add zeros). A shared-memory tree would have needed 8 barriers.

</details>

**Q7.** Why is this deadlock, even though the arithmetic looks right?

```cuda
if (tid < n)
    v += __shfl_down_sync(0xffffffff, v, 16);
```

<details>
<summary>Show answer</summary>

`0xffffffff` means "wait for all 32 lanes". Lanes with `tid >= n` never execute the shuffle, so the wait cannot complete. Every lane of the warp must call the primitive. The lanes past `n` should pass `0` and participate.

</details>

**Q8.** A ballot returns `0b0000…0101` (bits 0 and 2 set). What compaction indices do lanes 0 and 2 get, and how many outputs does the warp write?

<details>
<summary>Show answer</summary>

Lane 0 has no set bits below it, so its index is **0**. Lane 2 has one set bit below it (bit 0), so its index is **1**. `popcount` of the word is **2**, which is the number of writes.

</details>
