## A warp does not load 32 times

The hardware does not issue one memory request per thread. It looks at the 32 addresses a warp produces and **covers them with as few transactions as possible**. Two different memories, two different rules:

| Memory | Unit that gets merged or split | What you want |
|---|---|---|
| **Global** (HBM, via L1/L2) | **32-byte sectors** | many threads inside the same few sectors |
| **Shared** | **32 banks** | every thread in a different bank, or all of them on the *same address* |

The previous chapter cut *how many* bytes you load. This one is about *how the hardware actually moves them*. A tiled GEMM that reloads every tile through stride-32 accesses gives the bandwidth back.

## Sectors, not threads

Global memory is transferred in **32-byte sectors**. Four sectors make the **128-byte cache line** mentioned in the memory chapter. A fully coalesced warp load of 32 `float`s is exactly one cache line: **4 sectors, 128 bytes, nothing wasted**.

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;   // consecutive threadIdx.x
out[i] = in[i];                                  // stride 1: 4 sectors / warp
```

`threadIdx.x` is the consecutive lane (the linear thread id runs x, then y, then z). So the *fast* dimension of every array you touch from a warp must be `threadIdx.x`.

Stride breaks it. `in[i * stride]` with `stride == 32` puts each lane in its own sector: **32 sectors, 1024 bytes moved, 128 bytes used** — 12.5 % efficiency. The kernel is still "touching N floats". The memory system is moving 8× more.

<div class="callout">

**Efficiency, not sector count.** A `float4` load also touches 16 sectors per warp (32 × 16 B = 512 B). That is fine: every byte is useful, and each instruction puts more bytes in flight (Little's law). A stride-2 `float` load touches 8 sectors for only 128 useful bytes. Same "more sectors", opposite meaning. The lab reports **useful bytes / bytes moved**.

</div>

### What else splits a warp

- **Misalignment.** 32 consecutive floats starting at element 1 cover bytes 4…131, which is **5** sectors instead of 4. `cudaMalloc` is aligned; a pointer plus an odd column offset is not. `float4` loads must be 16-byte aligned or they split (and on some toolchains they fail to vectorise).
- **AoS.** `struct Point { float x, y, z, w; }` and then reading only `.x` is stride 4. Lay the same data out as four arrays (SoA) and the load is stride 1. This is the single most common coalescing bug in particle and graph code.
- **The transpose write.** `out[x * N + y] = in[y * N + x]` reads coalesced and **writes** with stride N. Stores coalesce by the same sector rule as loads. `transpose.cu` fixes it by staging a tile and writing the transpose back out along `threadIdx.x`.

## Banks

Shared memory is 32 banks wide. Successive **32-bit words** go to successive banks:

$$
\text{bank} = \text{wordIndex} \bmod 32.
$$

A warp's shared-memory instruction is one access if every lane hits a **different** bank, or if several lanes hit the **same address** (a **broadcast** — the hardware serves it once). It splits into several serialised passes when two lanes hit **different addresses in the same bank**. The number of passes is the **conflict degree**: the busiest bank's count of distinct addresses. An n-way conflict makes that instruction about n times slower. It does not change the answer.

```cuda
__shared__ float s[32][32];
float row = s[threadIdx.y][threadIdx.x];   // lanes read s[y][0..31] → banks 0..31, no conflict
float col = s[threadIdx.x][0];             // lanes read s[0..31][0] → wordIndex = tx*32, bank 0 for every lane
```

The column is a **32-way conflict**. Every lane wants bank 0, and each wants a different row.

### The +1 padding

```cuda
__shared__ float s[32][33];
float col = s[threadIdx.x][0];             // wordIndex = tx*33, bank = tx
```

A leading dimension that is **odd** (coprime with 32) rotates the column across the banks. `33` is the usual choice: one extra column, and `tile[32][33]` is the line the tiled-matmul chapter pointed at. `34` is not a fix — `tx*34` only lands on even banks, a 2-way conflict.

Broadcast is not a conflict. In the plain tiled GEMM, `As[ty][k]` is the *same* address for every lane of a warp (one row, one k). That is a broadcast, which is why that kernel did not need padding. The conflict appears when you **transpose on the way through shared memory**, because the read back is a column. `transpose.cu` times both.

64-bit words (`double`, `float2`) occupy two consecutive banks. Two threads whose 64-bit values start on the same bank still conflict. The lab and the rule of thumb below are the 32-bit model; widen the word and the same padding still works.

## Putting both on one kernel

A tiled transpose does four memory operations:

1. **Global load** `in[y, x]` with `x = threadIdx.x` — coalesced.
2. **Shared store** `tile[ty][tx]` — a row, no bank conflict. `__syncthreads()`.
3. **Shared load** `tile[tx][ty]` — a column. Conflict degree 32 unless the row stride is 33.
4. **Global store** `out[y2, x2]` with `x2 = threadIdx.x` — coalesced again.

Step 3 is why "I coalesced the global accesses" can still leave the kernel slow. Nsight Compute's Memory Workload Analysis shows shared-memory bank conflicts separately from global sector counts; the Speed-of-Light section will not tell you which of the two you hit.

<div class="callout tip">

**Rules that fall out of the hardware.**
1. Make `threadIdx.x` the contiguous index of every global array you read or write.
2. Count sectors when you stride, gather, or start at an odd offset. Efficiency is useful bytes over sectors × 32.
3. Shared rows are free. Shared columns of a multiple-of-32 width are not. Pad the leading dimension to an odd number.
4. Same address many times is a broadcast, not a conflict. Different addresses, same bank, is a conflict.
5. Fix the access pattern before you touch instruction scheduling. A 32-way conflict or an 8× overfetch dominates.

</div>
