## Exercises

**Q1.** A100, square GEMM, $N = 256$, FP32. Intensity, and which roof does it hit?

<details>
<summary>Show answer</summary>

$I = 2 \cdot 256 / (3 \cdot 4) = 42.7$ FLOP/byte. The FP32 ridge is 9.6, so the dot is **compute-bound at 19.5 TFLOP/s**. A faster memory would not move it.

</details>

**Q2.** Same GEMM, TF32 tensor cores. Why is the answer not 156 TFLOP/s?

<details>
<summary>Show answer</summary>

Storage is still 4 bytes, so $I$ is still 42.7. The TF32 ridge is 76.5, and 42.7 is on the memory slope: $2039 \times 42.7 / 1000 \approx 87$ TFLOP/s. The 156 peak is real and this matrix does not feed it.

</details>

**Q3.** Same GEMM, FP16. Intensity and achieved rate on A100?

<details>
<summary>Show answer</summary>

$I = 2 \cdot 256 / (3 \cdot 2) = 85.3$ FLOP/byte. Ridge is 153, so still memory-bound: about **174 TFLOP/s**, not 312. FP16 needs $N \ge 459$ before this GEMM shape reaches the tensor-core roof.

</details>

**Q4.** Standard attention, FP16, $d = 64$. What does the intensity tend to as $N$ grows, and does A100's FP16 ridge (153) ever get crossed?

<details>
<summary>Show answer</summary>

$I \to d/s = 64/2 = 32$ FLOP/byte. **No.** The $N^2$ score matrix grows as fast as the FLOPs. At $N = 2048$ the finite value is already 31.

</details>

**Q5.** FlashAttention, FP16, $B_r = 128$. Asymptotic intensity? Does that clear the A100 FP16 ridge?

<details>
<summary>Show answer</summary>

$I \to 2 B_r / s = 128$ FLOP/byte. **Not quite** — 128 is under 153. $B_r = 256$ tends to 256 and does clear it, provided the tiles still fit in SRAM.

</details>

**Q6.** $N = 2048$, $d = 64$, $B_r = B_c = 128$, FP16. Standard bytes, flash bytes, and SRAM?

<details>
<summary>Show answer</summary>

Standard **33.0 MB**, flash **8.50 MB** (3.9× less). SRAM is **96 KB**: two $Q$/$O$ tiles, two $K$/$V$ tiles, and the $128 \times 128$ score tile, at 2 bytes.

</details>

**Q7.** The lab's scores start $[1, 2, 0, 1,\; 5, 1, 0, 2]$. After the second tile, what is the running max, and by what factor is the first tile's mass rescaled?

<details>
<summary>Show answer</summary>

$m = 5$. The first tile's max was 2, so the rescale is $e^{2-5} = e^{-3} \approx 0.0498$. The output is still exact; the earlier exponentials were computed against a max that later moved.

</details>

**Q8.** The backward pass needs $P$ as well. Why does FlashAttention recompute the score tiles instead of storing them from the forward pass?

<details>
<summary>Show answer</summary>

Storing $P$ writes the $N \times N$ matrix the forward pass just avoided. Recomputing a tile from $Q$ and $K$, which are $N \times d$, is cheaper in HBM than reading an $N \times N$ matrix back. The extra FLOPs land on the tensor cores; the bytes were the bottleneck.

</details>
