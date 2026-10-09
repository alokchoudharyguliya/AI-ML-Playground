## A tensor core multiplies a tile, not a scalar

A CUDA core computes one FMA per instruction. A **tensor core** computes a small matrix multiply-accumulate per instruction: a warp (or, on Hopper, a warpgroup) feeds it tiles of A, B and C and gets back C + AB. The CUDA WMMA API exposes this as a 16×16×16 tile for FP16 and BF16, accumulated in FP32:

```cuda
wmma::load_matrix_sync(a, A, lda);
wmma::load_matrix_sync(b, B, ldb);
wmma::mma_sync(c, a, b, c);          // c is FP32 even when a and b are FP16
wmma::store_matrix_sync(C, c, ldc, wmma::mem_row_major);
```

Every lane of the warp must execute those calls together. The tile is the unit of work; a scalar loop never reaches the unit.

The peaks in the memory chapter are this hardware, dense, with no 2:4 sparsity:

| GPU | FP32 CUDA cores | TF32 tensor | FP16 tensor |
|---|---|---|---|
| A100 | 19.5 TFLOP/s | 156 | 312 |
| H100 SXM | 67 | 494 | 989 |
| RTX 4090 | 82.6 | 82.6 | 165 |

On the 4090 the TF32 column equals FP32: that chip's tensor cores show up in FP16/BF16, not as a separate TF32 peak. Hopper adds FP8 (about twice FP16) and a larger **warpgroup** MMA (`wgmma`) fed by the tensor memory accelerator. The idea is the same tile, with a wider tile.

A higher peak moves the roofline's ridge to the right. A100 FP32 turns compute-bound near 10 FLOP/byte. A100 FP16 tensor cores turn compute-bound near **153 FLOP/byte**. A GEMM that was sitting on the FP32 roof is often still on the memory slope for FP16, and the tensor cores idle. The lab's N = 256 case is exactly that.

## Attention is a GEMM that writes an N×N matrix

One head:

$$
S = QK^\top / \sqrt{d}, \qquad P = \mathrm{softmax}(S), \qquad O = PV.
$$

Two GEMMs, so the FLOPs are the familiar $4N^2 d$. The trouble is $S$ and $P$. Each is $N \times N$, and the textbook implementation writes $S$, reads it back for the softmax, writes $P$, and reads $P$ for the second GEMM. That is $4N^2$ elements of HBM on top of $Q, K, V, O$.

Arithmetic intensity then tends to $d/s$ (element size $s$ in bytes), **not** to something that grows with $N$. For FP16 and $d = 64$ that is 32 FLOP/byte. A100's FP16 ridge is 153. Making the sequence longer does not cross the ridge. The kernel is memory-bound at every length, and the bytes it moves are the score matrix.

## Online softmax, then never store the tile

Softmax needs the row max and the row sum, which you do not know until the row is finished. The online algorithm keeps a running max $m$ and a running sum $\ell$, and rescales when a later tile raises the max. For a block of scores with max $m_{\text{block}}$:

$$
m' = \max(m, m_{\text{block}}),
\qquad
\ell' = e^{m - m'}\,\ell + \sum_j e^{s_j - m'},
\qquad
o' = e^{m - m'}\,o + \sum_j e^{s_j - m'}\,v_j.
$$

The factor $e^{m-m'}$ is 1 when the max did not move, and a fraction when it did. After the last block, $o/\ell$ is the exact softmax-weighted output. No approximation, and no $N$-long row resident anywhere.

FlashAttention is that recurrence applied to tiles that fit in SRAM:

- A $B_r \times d$ tile of $Q$ stays on chip while every $B_c \times d$ tile of $K$ and $V$ streams through.
- The $B_r \times B_c$ score tile is computed, softmaxed online, multiplied into $V$, and dropped.
- $O$'s tile is updated in SRAM and written once.

$K$ and $V$ are re-read once per $Q$ tile. $S$ and $P$ are not written at all. The backward pass recomputes the tiles instead of storing them; saving them would put the $N \times N$ matrix back in HBM.

<div class="callout">

**The ridge still has to be reachable.** Asymptotically the flash kernel moves $2B_r/s$ FLOP/byte. On A100 FP16 that needs $B_r$ above about 153, so a tile of 128 is close and a tile of 256 clears it — if the $Q$, $K$, $V$, $O$ and score tiles fit in the SM's SRAM together. A tile that misses the SRAM spills to HBM and the intensity collapses back toward the standard kernel.

</div>

Causal masking is free inside the tile: scores past the diagonal are set to $-\infty$ before the row max. Nothing extra is stored.
