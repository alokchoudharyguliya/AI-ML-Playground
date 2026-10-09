## Exercises

**Q1.** SAXPY on $2^{28}$ FP32 elements runs on an A100 (19.5 TFLOP/s, 2.039 TB/s). What is its arithmetic intensity, is it compute- or memory-bound, and what is the fastest it can possibly run?

<details>
<summary>Show answer</summary>

$W=2\cdot2^{28}$ FLOP, $Q=3\cdot4\cdot2^{28}=3.22$ GB, so $I=2/12=0.167$ FLOP/B, far below the ridge ($9.6$): **memory-bound**. Time bound $t\ge Q/\beta=3.22\ \text{GB}/2.039\ \text{TB/s}=1.58$ ms, i.e. at most $0.167\times2039=340$ GFLOP/s, **1.7 %** of the FP32 peak. Even a perfect kernel cannot beat this; a *good* one reaches 85–95 % of it.

</details>

**Q2.** A copy kernel moves a 1 GiB array to another 1 GiB array in 1.2 ms on an A100. What effective bandwidth is that and what fraction of peak?

<details>
<summary>Show answer</summary>

Bytes moved $=2\times1.0737\times10^9=2.147\times10^9$ (read + write). $\text{BW}_{\text{eff}}=2.147\times10^9/1.2\times10^{-3}=1.79$ TB/s $=88\ \%$ of $2.039$ TB/s. Counting only the written bytes (a classic mistake) would report half of that.

</details>

**Q3.** Using the A100 latencies L1 = 33, L2 = 200, HBM = 500 cycles, compute the AMAT for $h_1=0.8$, $h_2=0.5$. What happens if a code change lifts $h_2$ to 0.9?

<details>
<summary>Show answer</summary>

$p_1=0.8$, $p_2=0.2\cdot0.5=0.1$, $p_3=0.1$: $\text{AMAT}=0.8(33)+0.1(200)+0.1(500)=26.4+20+50=96.4$ cycles. With $h_2=0.9$: $p_2=0.18$, $p_3=0.02$: $26.4+36+10=72.4$ cycles, a **25 %** drop from improving only the L2 hit rate. Misses to HBM dominate.

</details>

**Q4.** An H100 (132 SMs, 1.98 GHz, 3.35 TB/s, HBM latency ≈ 600 cycles). How many bytes must be in flight per SM to saturate HBM? How many warps does that require with 4-byte loads at one load per thread, and with four independent loads per thread?

<details>
<summary>Show answer</summary>

$\lambda=3350/(132\times1.98)=12.8$ B/cycle/SM. $B_{\text{inflight}}=12.8\times600\approx7.7$ KB. Per warp-load (4 B × 32) $=128$ B, so $\approx60$ warps with one load/thread (nearly the SM's 64-warp maximum!). With 4 independent loads/thread each warp holds 512 B in flight: $\approx15$ warps. **Moral: ILP and wide loads are cheaper than occupancy.**

</details>

**Q5.** At what matrix size $N$ does an *ideal* square GEMM become compute-bound on an H100 (a) in FP32 on CUDA cores, (b) in FP16 on Tensor Cores?

<details>
<summary>Show answer</summary>

FP32: $I=N/6>20\Rightarrow N>120$. FP16 Tensor: $I=2N/(3\cdot2)=N/3>295\Rightarrow N>885$. Below those sizes even a perfect GEMM is memory-bound, so batched/small GEMMs (e.g. attention heads) rarely hit Tensor Core peak.

</details>

**Q6.** A 7B-parameter model in FP16 is served at batch 1 on an RTX 4090 (1.008 TB/s, 165 TFLOP/s FP16 Tensor). What is the maximum tokens/s? What batch makes it compute-bound?

<details>
<summary>Show answer</summary>

Weights $=14$ GB. Each token streams them once: $14/1008\approx13.9$ ms $\Rightarrow\lesssim72$ tokens/s. $I=B$ FLOP/B vs ridge $165/1.008=164$, so $B\gtrsim164$ (in practice the KV cache grows too and limits the batch on a 24 GB card).

</details>

**Q7.** Where does `float t[16]; ... t[k % 16] = v;` live (with $k$ a runtime value) and how do you check it? Give two ways to fix it.

<details>
<summary>Show answer</summary>

In **local memory** (device memory, cached in L1/L2): registers cannot be indexed dynamically. Check with `nvcc -Xptxas -v` (stack frame, spill stores/loads) or Nsight Compute's local-memory counters. Fixes: make the index a compile-time constant (`#pragma unroll` plus constant indices), restructure into scalar variables / `switch`, or move the array to **shared memory** (one slice per thread, padded to avoid bank conflicts).

</details>

**Q8 (code).** Run `pointer_chase.cu`. At which working-set sizes does latency jump, and how do those sizes compare with your GPU's L1 and L2 capacity?

<details>
<summary>Show answer</summary>

You should see three plateaus: a low one (tens of cycles) while the array fits in L1 (roughly 128–256 KB), a middle one (a couple of hundred cycles) up to the L2 size (40 MB on A100), then a high one (400+ cycles) beyond it, sometimes with a further step from TLB misses on very large arrays. The jumps are fuzzy because of associativity and because L1 and shared memory share SRAM.

</details>

**Q9 (code).** Run `roofline_probe.cu`, plot the CSV with `plot_roofline.py`, and compare your measured knee with the theoretical ridge $\pi/\beta$.

<details>
<summary>Show answer</summary>

Expect the measured memory roof at ~85–92 % of theoretical bandwidth and the compute roof at 85–95 % of the FP32 peak (clocks may throttle below boost). Therefore the measured knee sits a bit away from the datasheet ridge. Always use the *measured* roof for tuning decisions.

</details>

## In practice

- **Estimate $I$ before writing code.** If a kernel is hopelessly memory-bound on paper, no instruction-level tuning will rescue it. Reduce bytes first.
- **Fuse element-wise operations** (bias + activation + residual + norm) into one kernel: each fused op saves a full read and write of the tensor.
- **Use fewer bits**: FP16/BF16/FP8 storage halves or quarters $Q$ with almost no extra work.
- **Vectorise** with `float4`/`int4` loads (align to 16 B) to raise bytes-in-flight per instruction.
- **Keep reuse in registers first**, then shared memory, then rely on L2. Each step down the ladder costs 5–10× more.
- **Measure with Nsight Compute** (Speed of Light + Roofline) rather than guessing: it reports achieved percent of peak for both compute and memory.

## Common pitfalls

- Counting only *useful* bytes: the hardware moves whole 32-byte sectors, so strided accesses move far more than you asked for (see *Coalescing*).
- Comparing against the wrong ceiling: FP32-peak ridge for a kernel that uses Tensor Cores, or HBM roof for a working set that lives in L2.
- Trusting boost-clock peaks: sustained clocks under power or thermal limits are lower, so measure.
- Assuming *local* memory is "local" in the speed sense. It is global memory with a thread-private view.
- Declaring a kernel "memory-bound" because it is slow. Memory-bound means near the *memory roof*; if you are far below it you have a latency or access-pattern problem.
- Forgetting that the L1/shared split is a per-kernel choice: a kernel that uses little shared memory can run with a larger L1.
