## The reuse problem

The last chapter left GEMM sitting on the roofline: a *naive* kernel that streams a whole row of $A$ and column of $B$ for every output has arithmetic intensity $I=1/s$ (0.25 FLOP/B in FP32) — pinned to the memory roof, a few percent of peak. The *ideal* kernel that reads each input once has $I=2N/(3s)$, which crosses the ridge as soon as $N$ is a few hundred. The gap is **reuse**: each $A_{ik}$ is needed by $N$ outputs in its row, each $B_{kj}$ by $N$ outputs in its column.

Registers cannot hold an $N\times N$ panel. **Shared memory** can hold a $T\times T$ *tile*, visible to every thread of the block. Load the tile once, reuse it $T$ times, slide along $K$. Global traffic drops by $\sim T$. That is this chapter.

## Shared memory in one paragraph

```cuda
__shared__ float tile[16][16];          // static: size known at compile time, one copy per block
extern __shared__ float dyn[];          // dynamic: size is the 3rd launch parameter
kernel<<<grid, block, bytes>>>(...);
```

- **Scope:** the thread block. Not visible to other blocks, not to the host.
- **Lifetime:** the block. Contents are gone when the block finishes.
- **Location:** on-chip SRAM, same physical array as L1 (you pick the split). ~20–30 cycles, ~10× HBM bandwidth.
- **Layout:** 32 banks, 4 bytes wide. Two threads of a warp hitting different addresses in the *same* bank serialise — **bank conflicts**, the next chapter. Tiled GEMM with consecutive `threadIdx.x` is naturally conflict-free; we will not chase padding here.

A thread writes `tile[ty][tx] = ...`. That write is **not** guaranteed visible to other threads until `__syncthreads()`.

## `__syncthreads()` is a contract

`__syncthreads()` is a **block-wide barrier and memory fence**. After it returns:

1. Every thread of the block has reached the call.
2. Every shared *and* global write issued before it is visible to every thread of the block.

<div class="callout warn">

**The deadlock rule.** *Every* thread of the block must execute the *same* `__syncthreads()`. Putting it inside `if (threadIdx.x < 16)` (or behind `if (row < N)`) deadlocks the block: the taken threads wait forever for the skipped ones. Bounds checks go *around the memory operation*, not around the barrier.

</div>

A second barrier is needed at the *bottom* of a tile loop: otherwise the next cooperative load overwrites `As`/`Bs` while another warp is still multiplying the old tile.

On a *single warp* (32 threads) a missing barrier often *looks* fine, because the 32 lanes already issue together. The race is real as soon as two warps share the array — `compute-sanitizer --tool racecheck` is the authority, not a lucky run. The third lab tab and `syncthreads_race.cu` make this concrete.

## Cooperative load

The pattern for almost every shared-memory kernel:

1. **Each thread loads one (or a few) elements** of a tile from global memory into shared memory. Together they cover the whole tile. This is a *cooperative load*.
2. `__syncthreads()`.
3. **Every thread reads many elements** of the tile, now hitting SRAM instead of HBM.
4. `__syncthreads()` if the tile will be overwritten.
5. Repeat, or write the result back.

The load should be **coalesced**: `threadIdx.x` walks consecutive addresses so a warp's 32 loads become one 128-byte transaction (next chapter). In the GEMM kernel below, `As[ty][tx] = A[row, t+tx]` does exactly that.

## Tiled matrix multiply

$C = AB$, all $N\times N$, row-major. Launch a 2-D grid of **TILE×TILE** thread blocks. Block `(bx, by)` owns the output tile

$$
C\big[by\cdot T :\, by\cdot T+T,\ \ bx\cdot T :\, bx\cdot T+T\big].
$$

Thread `(tx, ty)` inside the block owns one output, $C[by\cdot T+ty,\ bx\cdot T+tx]$, and keeps a running product in a **register** `acc`.

Along $K$ the block slides a window of width $T$:

```
k-tile t = 0, 1, …, N/T − 1
   load  As = A[by·T : by·T+T,  t·T : t·T+T]     // T×T, cooperatively
   load  Bs = B[t·T : t·T+T,  bx·T : bx·T+T]
   __syncthreads()
   acc += row ty of As  ·  column tx of Bs       // T FMAs, all in shared memory
   __syncthreads()
write C
```

Each input element that lands in `As` is then read by **T threads** (a whole output-tile row). Same for `Bs` and a column. That is the factor-of-$T$ traffic cut.

When $N$ is not a multiple of $T$, **pad with zeros** on the load (the `?: 0` in the code tab) rather than skipping the barrier. The extra FMAs on zeros are free next to an HBM round-trip.

Open the **3D** lab: the two floating tiles *are* `As` and `Bs`; the floor is the C-tile. Step through a k-slab and watch one thread's inner product.

## How big a tile?

| TILE | Threads / block | Shared bytes (2 tiles, FP32) | Reuse $T$ | Notes |
|---|---|---|---|---|
| 8 | 64 | 0.5 KB | 8 | Occupancy-friendly, little reuse |
| 16 | 256 | 2 KB | 16 | The sweet spot to start with |
| 32 | 1024 | 8 KB | 32 | Max threads/block; fewer resident blocks |

Bigger $T$ raises arithmetic intensity ($I=T/s$) and so moves you right on the roofline, but it burns shared memory and thread slots, which can *drop occupancy* enough that you can no longer hide HBM latency (Occupancy chapter). Measure; don't assume 32 wins.

A further trick, **register blocking**, has each thread own a small $r\times r$ patch of $C$ (e.g. 4×4) so the inner kernel is $T$ FMAs into 16 accumulators. CUTLASS / cuBLAS do this; it is how you actually reach Tensor Core peak. We stay at one output per thread so the algorithm stays visible.

## A smaller cousin: stencils and halos

Not every kernel is GEMM. A 3-point stencil $y_i = (x_{i-1}+2x_i+x_{i+1})/4$ looks like 3 global loads per output. Have the block load its interior **plus one halo cell on each side** into shared memory, barrier, then every output is 3 shared reads. Halo cost is $2/\text{blockDim}$ extra loads — ~1 % at 256 threads. The same idea in 2-D is the 5-point / 9-point stencil with a 1-cell apron, and in attention it is the $B_r\times B_c$ tiles of FlashAttention.

## What this chapter is not

- **Bank conflicts and padding `As[TILE][TILE+1]`** — next chapter. The GEMM kernel here is already conflict-free for the inner product.
- **Tensor Cores / WMMA / `mma.sync`** — FlashAttention chapter. Tiling is the *software* analogue of what those units do in hardware.
- **cuBLAS.** For production GEMM, call the library. The point of writing this kernel is to *see* why tiling works, and to reuse the pattern on fusions the library cannot write for you.

<div class="callout tip">

**The checklist.** (1) Identify data reused by many threads of a block. (2) Stage it in `__shared__` with a coalesced cooperative load. (3) Barrier. (4) Compute from shared memory / registers. (5) Barrier again if you will overwrite the tile. (6) Count bytes: if $I$ is still left of the ridge, fuse, widen the tile, or drop to a smaller dtype — don't micro-optimise the FMAs.

</div>
