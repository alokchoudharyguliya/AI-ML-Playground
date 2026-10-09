## The warp is the real instruction

A CUDA thread is the program you write. The hardware runs a **warp**: 32 consecutive threads that share one instruction pointer. Lanes have their own registers and their own predicates, but they are issued together. That is SIMT — single instruction, multiple threads.

Threads of a block are grouped with `threadIdx.x` varying fastest, then `y`, then `z`. Lane `ℓ` of a 1-D block is `threadIdx.x & 31`. The warp id is `threadIdx.x >> 5`.

On Volta and later the scheduler *can* move lanes independently (**independent thread scheduling**). The compiler still reconverges them, but you may not assume lockstep. Every warp primitive takes an explicit mask and ends in `_sync`. The examples below use the full mask `0xffffffff`, which is correct only when all 32 lanes actually reach the call.

## Divergence

```cuda
if (threadIdx.x < 16)  a();   // lanes 0–15
else                   b();   // lanes 16–31
```

Both arms run, one after the other. While `a()` is issued the odd half of the warp is masked off; then the mask flips for `b()`. A uniform warp (every lane takes the same arm) issues only that arm.

The cost is the **sum of the taken arms**, not the max. Sixteen lanes in a long `else` still make the other sixteen wait. Nested data-dependent branches multiply the number of passes. A four-way `switch (lane % 4)` issues four times and each pass is one quarter full.

<div class="callout">

**Two rules that fall out of the mask.**

1. `__syncthreads()` inside a branch that not every thread of the block takes is undefined. The missing threads never arrive.
2. A `_sync` primitive waits for every lane in its mask. `if (valid) __shfl_sync(0xffffffff, …)` deadlocks when some lanes skip the call. Invalid lanes must still execute the shuffle; give them a zero and let them participate.

</div>

Very short bodies are often **predicated** instead of branched: both sides become ordinary instructions with a per-lane flag. The issue slot is still spent, but there is no reconvergence point to get wrong.

## Shuffles

A shuffle reads a register from another lane of the same warp. No shared memory, no barrier.

| Intrinsic | Source lane | Typical use |
|---|---|---|
| `__shfl_sync(mask, v, src)` | absolute lane `src` | broadcast one lane |
| `__shfl_up_sync(mask, v, Δ)` | `lane − Δ` | inclusive scan |
| `__shfl_down_sync(mask, v, Δ)` | `lane + Δ` | reduction into lane 0 |
| `__shfl_xor_sync(mask, v, maskBit)` | `lane XOR maskBit` | butterfly; every lane gets the result |

`__shfl_down_sync` returns the caller's own value when `lane + Δ` falls outside the warp. The reduction below relies on that.

## Reduction

Five instructions reduce 32 values. With `__shfl_down_sync` and offsets 16, 8, 4, 2, 1, **lane 0** holds the sum. With `__shfl_xor_sync` and the same offsets, **every lane** holds it, because each step is a perfect pairing.

A block reduction is that warp reduction plus one barrier:

1. Each warp reduces its own registers. Lane 0 writes one partial to shared memory (at most 32 floats per block).
2. `__syncthreads()`.
3. The first warp loads those partials and reduces again. Thread 0 writes the block result.

That is one barrier per block, not the five barriers of a shared-memory tree. The grid then finishes with an atomic add or a second small kernel. `warp_reduce.cu` is the pattern; production sums call CUB.

## Ballot

`__ballot_sync(mask, pred)` returns a 32-bit word whose bit `ℓ` is lane `ℓ`'s predicate. `__popc` of the bits below you is your index among the lanes that passed — a warp-local stream compaction, done entirely in registers. `ballot.cu` uses it to pack the hits from one warp into a dense list.
