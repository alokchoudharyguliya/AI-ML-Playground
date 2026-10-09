## Naive GEMM traffic

$C=AB$ with $N\times N$ matrices, $s$ bytes per element, $W=2N^3$ FLOP.

Each of the $N^2$ outputs reads $N$ elements of $A$ and $N$ of $B$:

$$
Q_{\text{naive}}=2N^3 s + N^2 s,\qquad
I_{\text{naive}}=\frac{2N^3}{2N^3 s+N^2 s}\approx\frac{1}{s}.
$$

FP32: $I=1/4$ FLOP/B. On an A100 ($\pi=19.5$ TFLOP/s, $\beta=2.04$ TB/s) the ridge is $9.6$ FLOP/B, so this kernel is **~38× below the ridge** no matter how large $N$ is.

## Tiled traffic

A TILE $T$ grid: $(N/T)^2$ output tiles, $N/T$ slabs along $K$. Each slab loads $T^2$ elements of $A$ and $T^2$ of $B$:

$$
Q_{\text{tiled}}=\underbrace{\Big(\frac{N}{T}\Big)^2\cdot\frac{N}{T}\cdot 2T^2 s}_{A,B\text{ loads}} + N^2 s
=\frac{2N^3 s}{T}+N^2 s,
$$

$$
I_{\text{tiled}}=\frac{2N^3}{2N^3 s/T+N^2 s}=\frac{T}{s}\cdot\frac{1}{1+T/(2N)}\approx\frac{T}{s}.
$$

Reuse factor versus naive: $Q_{\text{naive}}/Q_{\text{tiled}}\approx T$. FP32, $T=16$: $I\approx 4$ FLOP/B (still memory-bound on A100, but 16× fewer bytes). $T=32$: $I\approx 8$, almost at the FP32 ridge.

### Worked numbers, $N=1024$, FP32

| Kernel | Loads | Stores | $Q$ | $I$ | HBM-bound time on 2 TB/s |
|---|---|---|---|---|---|
| Naive | $2\cdot1024^3$ | $1024^2$ | $8.59$ GB | $0.25$ | $4.2$ ms |
| $T=16$ | $2\cdot1024^3/16$ | $1024^2$ | $0.54$ GB | $4.0$ | $0.26$ ms |
| $T=32$ | $2\cdot1024^3/32$ | $1024^2$ | $0.27$ GB | $8.0$ | $0.13$ ms |
| Ideal (read $A,B$ once) | $2\cdot1024^2$ | $1024^2$ | $12.6$ MB | $170$ | **compute-bound** ($\approx 0.11$ ms at 19.5 TF) |

$W=2\cdot1024^3=2.15$ GFLOP. At A100 peak that is $0.11$ ms, so $T=32$ is within ~20 % of the compute roof *if* you actually hit HBM peak. In practice the naive kernel is much slower than 4.2 ms because the $B$ column is uncoalesced; tiling fixes that too.

### When does tiled GEMM become compute-bound?

Ignore the lower-order store: $T/s > \pi/\beta$. FP32 A100: $T/4>9.6\Rightarrow T>38$ — larger than a block can hold at one output per thread. That is why production kernels **register-block**: $T_{\text{effective}}=T_{\text{smem}}\cdot r$ with $r$ outputs per thread, or they switch to Tensor Cores whose ridge is ~150 FLOP/B and demand even more reuse (FlashAttention chapter).

## Shared-memory footprint and occupancy

Two FP32 tiles:

$$
S=2\,T^2 s=8T^2\text{ bytes}.
$$

$T=16\Rightarrow 2$ KB, $T=32\Rightarrow 8$ KB. An SM with 164 KB shared (A100) could in principle hold $164/2=82$ such blocks, but it is also limited to 16–32 blocks and 2048 threads: at 256 threads/block the thread cap is 8 blocks, so **shared memory is not the limiter** at $T=16$. At $T=32$ (1024 threads) the thread cap is 2 blocks and shared memory is still fine. The Occupancy chapter puts numbers on the three-way fight (registers, shared, threads).

Dynamic shared memory is just $S$ passed at launch; occupancy tools see the same $S$.

## Inner-product work per thread

Per k-tile each thread does $T$ FMAs ($2T$ FLOP) and 2 shared loads per FMA. Over $N/T$ tiles:

$$
\text{FLOP/thread}=2N,\qquad \text{shared loads/thread}=2N.
$$

The global loads per thread are only $2\cdot(N/T)$ — one $A$ element and one $B$ element per k-tile. That is the whole point.

## Stencil halo

Block of $B$ threads, radius-$r$ stencil (3-point $\Rightarrow r=1$). Cooperative load of $B+2r$ elements, then $B$ outputs:

$$
\frac{Q_{\text{smem}}}{Q_{\text{naive}}}=\frac{B+2r}{(2r+1)B}\approx\frac{1}{2r+1}\quad(B\gg r).
$$

3-point: traffic $\times 1/3$. 2-D 9-point on a $B\times B$ tile with a 1-cell apron: $(B+2)^2 / (9B^2)\approx 1/9$.

## Barrier cost

`__syncthreads()` is a few cycles when all warps of the block arrive together, and a *warp stall* for early warps when they don't. In the tile loop it is issued twice per $T$ FMAs. For $T=16$ that is one barrier per 8 FMAs — cheap next to HBM, visible if you have already hit the compute roof. Removing the *second* barrier is a data race, not an optimisation.
