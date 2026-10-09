## Sectors touched by one warp

Lane $\ell$ issues a load of $w$ bytes at byte address

$$
a_\ell = (b + \ell\, s)\, w_0,
$$

where $s$ is the stride in elements, $b$ the starting element, and $w_0$ the element size. A `float` has $w = w_0 = 4$; a `float4` has $w = 16$. The load covers the half-open range $[a_\ell,\ a_\ell + w)$.

$$
\text{sectors}
  = \left|\left\{\ \left\lfloor \frac{t}{32} \right\rfloor
    : t \in \bigcup_{\ell=0}^{31} [a_\ell,\ a_\ell+w)\ \right\}\right|.
$$

Bytes moved $= 32 \times \text{sectors}$ (the hardware always transfers a whole sector). Useful bytes $= 32 w$ only when the ranges do not overlap; they don't for $s \ge 1$.

$$
\eta = \frac{32 w}{32 \cdot \text{sectors}} = \frac{w}{\text{sectors}}.
$$

### Aligned power-of-two strides, $w = 4$

The 32 lanes span $128 s$ bytes starting on a sector boundary, so $\text{sectors} = 4s$ and $\eta = 1/s$.

| Stride $s$ | Sectors | Bytes moved | $\eta$ |
|---|---|---|---|
| 1 | 4 | 128 B | 100 % |
| 2 | 8 | 256 B | 50 % |
| 4 | 16 | 512 B | 25 % |
| 8 | 32 | 1 KiB | 12.5 % |
| 32 | 32 | 1 KiB | 12.5 % |

Stride 8 and stride 32 land on the same sector count: there are only 32 lanes, so you cannot touch more than 32 sectors with a 4-byte load that sits inside one sector. Past $s = 8$ the *efficiency* stays at $4/32$; you are already at one sector per lane.

### The misaligned float

$s = 1$, $w = 4$, $b = 1$. Bytes $[4,\ 132)$ cover sectors $\lfloor 4/32 \rfloor = 0$ through $\lfloor 131/32 \rfloor = 4$: **5 sectors**, $\eta = 4/5 = 80\%$. One element of offset costs a whole extra sector on every warp.

### float4

$w = 16$, $s = 1$, $b$ a multiple of 4 (16-byte aligned): $32 \times 16 = 512$ useful bytes $= 16$ sectors, $\eta = 1$. The same `float4` with $b = 1$ (4-byte aligned only) straddles a sector on every lane and $\eta$ drops. Alignment is part of the type, not just the pointer.

## Banks

Word index $i$ (32-bit words from the base of the array) maps to

$$
\text{bank}(i) = i \bmod 32.
$$

For a 2-D array with leading dimension $L$ (in words), element $(r, c)$ has $i = r L + c$.

A warp that reads addresses $i_0, \ldots, i_{31}$:

$$
d = \max_{0 \le b < 32} \left|\{\, i_\ell : \text{bank}(i_\ell) = b \,\}\right|.
$$

$d$ is the conflict degree. The instruction is issued in $d$ serial passes, so shared-memory throughput on that instruction scales as $1/d$. If all 32 lanes share one address, each bank's set has size 1: a **broadcast**, $d = 1$.

### Column of `tile[32][L]`

Lane $\ell$ reads row $\ell$, column $0$: $i = \ell L$.

$$
\text{bank} = (\ell L) \bmod 32.
$$

- $L = 32$: bank $= 0$ for every $\ell$. All 32 addresses differ. $d = 32$.
- $L = 16$: bank $= 0$ for even $\ell$ and $16$ for odd $\ell$. Sixteen distinct rows on each of those banks. $d = 16$.
- $L = 33$: $(\ell \cdot 33) \bmod 32 = \ell$. All different. $d = 1$.
- $L = 34$: $(\ell \cdot 34) \bmod 32 = (2\ell) \bmod 32$. Only even banks, two rows each. $d = 2$.

$d = 1$ for every column (any $c$) exactly when $L$ is **odd**: $L$ and $32$ are coprime, so $\ell \mapsto \ell L$ is a bijection on the banks. That is the whole content of "pad to `TILE+1`".

### Row

Lane $\ell$ reads $i = r L + \ell$. Those are 32 consecutive words, hence 32 consecutive banks, regardless of $L$ (a row never wraps inside 32 words). $d = 1$. Padding does nothing for pure row access, and it costs $T$ extra words of shared memory per tile.

### Cost of the pad

`tile[32][33]` instead of `tile[32][32]` is $32$ extra floats $= 128$ bytes per tile. A block that keeps one such tile spends 128 B more shared memory. Against a 32× slowdown on every column read, it is the right trade until shared memory is the occupancy limiter (the Occupancy chapter).

## What the transpose moves

$N \times N$ floats, read once and written once: $Q = 2 N^2 \cdot 4$ bytes. That number is identical for the naive and the tiled kernels — tiling a transpose does not change arithmetic intensity, because there is no reuse. The naive kernel just *fails to reach* $\beta$, because the stores touch $N/32$ times more sectors than a coalesced write (stride $N$, and $N$ is a multiple of 32 in every interesting case). The unpadded tiled kernel reaches the global roof on paper and then loses a factor of up to 32 in the shared-memory column read. The padded kernel is the one whose measured GB/s should sit near the STREAM numbers from the memory chapter.
