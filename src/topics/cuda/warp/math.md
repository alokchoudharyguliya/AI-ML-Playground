## What one divergent region costs

A warp has 32 lanes and a branch region with arms $A_1, \ldots, A_k$. Arm $j$ has length $T_j$ (issue slots) and is **taken** when at least one lane enters it. The hardware issues each taken arm once, with the other lanes masked off:

$$
T = \sum_{j : \text{arm } j \text{ taken}} T_j.
$$

A uniform warp pays only the arm it actually runs. If every arm has the same length $T_0$ and $P$ arms are taken,

$$
T = P\, T_0, \qquad \text{lane utilization} = \frac{1}{P}.
$$

Each lane does useful work in exactly one arm, but every arm occupies all 32 lanes of the issue slot.

| Predicate | Taken arms $P$ | Utilization |
|---|---|---|
| same arm for every lane | 1 | 100 % |
| `threadIdx.x < 16` | 2 | 50 % |
| `threadIdx.x & 1` | 2 | 50 % |
| `threadIdx.x % 4` | 4 | 25 % |

Unequal arms do not change the rule, only the weights. Sixteen lanes taking a body of length 1 and sixteen taking a body of length 20 cost $21$ slots. The short arm is almost free; the long one is paid in full.

Predication of a tiny body is the same arithmetic with $T_j$ equal to a few instructions and no join point. It does not make the inactive lanes free.

## Warp sum, five steps

Let $v_\ell^{(0)}$ be lane $\ell$'s register. The down-sweep with offset $\Delta \in \{16, 8, 4, 2, 1\}$ is

$$
v_\ell \leftarrow v_\ell + v_{\ell+\Delta}
\quad \text{when } \ell+\Delta < 32,
$$

and $v_\ell$ unchanged otherwise (that is what `__shfl_down_sync` returns past the end of the warp). After all five offsets, lane 0 holds $\sum_{\ell=0}^{31} v_\ell^{(0)}$. Lane $k$ holds the sum of the original values from $k$ through $31$.

The xor (butterfly) sweep is a permutation at every step:

$$
v_\ell \leftarrow v_\ell + v_{\ell \oplus \Delta}.
$$

$\ell \oplus \Delta$ is always a lane in $0 \ldots 31$, so nobody is idle. Each step adds a disjoint partner, and after $\Delta = 1$ **every** lane holds the full sum. Same five instructions, a different final distribution.

Check on the all-ones input: the down-sweep writes $2, 4, 8, 16, 32$ into lane 0, and the butterfly writes that same sequence into every lane.

## From a warp to a block

A block of $B$ threads has $W = B/32$ warps (use a multiple of 32). Each warp reduces independently, then one lane per warp stores a partial:

$$
s_w = \sum_{\ell=0}^{31} v_{32w+\ell}, \qquad w = 0 \ldots W-1.
$$

Shared memory holds $W$ floats — 128 bytes at $B = 1024$. After one `__syncthreads()`, warp 0 reduces $(s_0, \ldots, s_{W-1})$, padding with zeros so the shuffle mask stays a full warp. Thread 0 holds the block sum.

A shared-memory tree over the same block needs $\log_2 B$ barriers (10 at $B = 1024$). The shuffle version needs **one**.

## The mask is part of the call

`__shfl_*_sync(mask, …)` and `__ballot_sync(mask, …)` wait until every lane whose bit is set in `mask` has executed the instruction. Two consequences:

- All 32 lanes must *reach* the call when `mask = 0xffffffff`. A divergent `if` around the shuffle leaves some lanes out, and the warp never completes the wait.
- A partial last warp (fewer than 32 live elements) still has 32 hardware lanes. The live lanes and the padding lanes both execute the primitive. Padding contributes $0$. The mask stays `0xffffffff` as long as the whole warp is converged, which it is when the surrounding kernel has no divergent branch around the reduction.

`__ballot_sync` packs 32 predicates into one unsigned int. The compaction index of lane $\ell$ is the population count of the bits strictly below it:

$$
d_\ell = \operatorname{popcount}\big( \textit{ballot} \mathbin{\&} (2^\ell - 1) \big).
$$

Lanes with a false predicate do not write. The number of outputs from the warp is `popcount(ballot)`.
