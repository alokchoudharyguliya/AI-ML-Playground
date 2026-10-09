## GEMM intensity, three precisions

A square product $C = AB$ that reads each input once and writes $C$ once:

$$
\text{FLOPs} = 2N^3,
\qquad
\text{bytes} = 3N^2 s,
\qquad
I = \frac{2N}{3s}.
$$

$s$ is 4 for FP32 and for TF32 (the values stored are FP32; TF32 is the compute format) and 2 for FP16.

The roofline from the memory chapter, with $\pi$ in TFLOP/s and $\beta$ in GB/s:

$$
\text{achieved} = \min\big(\pi,\; \beta I / 1000\big),
\qquad
I^\star = \pi \cdot 1000 / \beta.
$$

| GPU | $I^\star$ FP32 | $I^\star$ TF32 | $I^\star$ FP16 |
|---|---|---|---|
| A100 | 9.6 | 76.5 | 153 |
| H100 SXM | 20 | 147 | 295 |
| RTX 4090 | 82 | 82 | 164 |

### N = 256 on A100

| Format | $s$ | $I = 2N/(3s)$ | Roof it hits | Achieved |
|---|---|---|---|---|
| FP32 | 4 | 42.7 | compute, 19.5 | 19.5 TFLOP/s |
| TF32 | 4 | 42.7 | memory | $2039 \times 42.7 / 1000 \approx 87$ |
| FP16 | 2 | 85.3 | memory | $2039 \times 85.3 / 1000 \approx 174$ |

Same matrix. FP32 is already on its roof, so a faster CUDA core would not help. TF32's roof is 156 and this GEMM only feeds it 87. FP16's roof is 312 and the GEMM feeds it 174. The tensor core pays off in full only once $I$ passes $I^\star$, which for A100 FP16 means

$$
\frac{2N}{3 \cdot 2} \ge 153 \implies N \ge 459.
$$

At $N = 1024$, $I = 341$ and the FP16 dot sits on the 312 TFLOP/s roof.

## Standard attention does not get there by growing N

Forward FLOPs, one head: $4N^2 d$ (the two GEMMs; softmax is lower order).

Bytes, materializing $S$ and $P$ at $s$ bytes/element:

$$
4N^2 s + 4Nds.
$$

Divide:

$$
I_{\text{std}} = \frac{4N^2 d}{s(4N^2 + 4Nd)} = \frac{d}{s}\cdot\frac{N}{N+d} \;\xrightarrow{N \gg d}\; \frac{d}{s}.
$$

FP16, $d = 64$: $I \to 32$ FLOP/byte. A100's FP16 ridge is 153. The gap does not close as $N$ grows. At $N = 2048$ the finite formula gives $31.0$, already at the limit.

## FlashAttention bytes

$T_r = \lceil N / B_r \rceil$ tiles of $Q$. Each stays in SRAM while $K$ and $V$ are read in full, so $K$ and $V$ are read $T_r$ times. $Q$ and $O$ move once. $S$ and $P$ do not move.

$$
\text{elements} = 2Nd\,(1 + T_r),
\qquad
I_{\text{flash}} = \frac{4N^2 d}{2Nd\,(1+T_r)\,s} = \frac{2N}{s(1+T_r)}.
$$

For $N \gg B_r$, $T_r \approx N/B_r$ and

$$
I_{\text{flash}} \to \frac{2 B_r}{s}.
$$

FP16 ($s = 2$): the limit is $B_r$ itself, in FLOP/byte. $B_r = 128$ tends to 128, just under A100's ridge of 153. $B_r = 256$ tends to 256 and clears it.

SRAM for the live tiles, in bytes:

$$
\big(2 B_r d + 2 B_c d + B_r B_c\big)\, s.
$$

$Q$ and $O$ contribute the two $B_r d$ terms, $K$ and $V$ the two $B_c d$ terms, and the score tile is $B_r B_c$.

Worked point, the lab's default. $N = 2048$, $d = 64$, $B_r = B_c = 128$, FP16:

| | Elements | Bytes | Intensity |
|---|---|---|---|
| Standard | $4N^2 + 4Nd$ | 33.0 MB | 31.0 |
| Flash | $2Nd(1 + 16)$ | 8.50 MB | 120 |

SRAM $= (2\cdot128\cdot64 + 2\cdot128\cdot64 + 128^2)\cdot 2 = 96$ KB, inside an A100's 164 KB and inside an H100's 228 KB. A 4090's 100 KB still holds it; $B_r = 256$ at this $d$ generally does not.

The speedup on bytes is $33.0 / 8.50 = 3.9\times$ here. It grows as $N/B_r$ grows, because the standard kernel's $N^2$ term keeps pulling ahead of the flash kernel's $N^2 d / B_r$ term.

## The rescale

Scores $[1, 2, 0, 1]$ then $[5, 1, 0, 2]$, which are the first two tiles in the lab.

After tile 0, $m = 2$. Tile 1 contains 5, so $m' = 5$ and the mass already accumulated is multiplied by

$$
e^{m - m'} = e^{2 - 5} = e^{-3} \approx 0.0498.
$$

Tiles that do not raise $m$ multiply by $e^0 = 1$. With $V = 1$ the normalized output is exactly 1: the weights are a softmax.
