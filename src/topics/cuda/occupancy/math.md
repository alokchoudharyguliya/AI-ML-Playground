## The four ceilings

A block of $T$ threads uses

$$
W_b = \left\lceil \frac{T}{32} \right\rceil
$$

warps. The compiler's register count $R$ is charged as

$$
R' = 8\left\lceil \frac{R}{8} \right\rceil,
\qquad
\text{registers per warp} = 32 R' = 256\left\lceil \frac{R}{8} \right\rceil.
$$

Warps the register file can hold, on a 65 536-register SM:

$$
W_{\text{regs}} = 4\left\lfloor \frac{65536 / (32 R')}{4} \right\rfloor.
$$

Blocks that fit:

$$
B_{\text{regs}} = \left\lfloor W_{\text{regs}} / W_b \right\rfloor.
$$

Shared memory charged per block is $0$ when the kernel allocates none. Otherwise, with the 128-byte allocation unit and the 1 KB reserve,

$$
S' = 128\left\lceil \frac{S}{128} \right\rceil + 1024,
\qquad
B_{\text{smem}} = \left\lfloor \frac{S_{\text{SM}}}{S'} \right\rfloor.
$$

$S$ above $S_{\text{SM}} - 1024$ does not launch. The other two ceilings need no rounding:

$$
B_{\text{warps}} = \left\lfloor W_{\max} / W_b \right\rfloor,
\qquad
B_{\text{blocks}} = B_{\max}.
$$

$$
B = \min(B_{\text{regs}}, B_{\text{smem}}, B_{\text{warps}}, B_{\text{blocks}}),
\qquad
\text{occupancy} = \frac{B \cdot W_b}{W_{\max}}.
$$

The limiter is whichever ceiling equals $B$. Two of them often tie when the register file divides evenly.

### Worked A100 numbers ($W_{\max} = 64$, $S_{\text{SM}} = 164$ KB)

| Block | $R$ | $S$ | $R'$ | $B$ | Warps | Occupancy | Limiter |
|---|---|---|---|---|---|---|---|
| 128 | 32 | 0 | 32 | 16 | 64 | 100 % | registers and warp slots |
| 128 | 37 | 0 | 40 | 12 | 48 | 75 % | registers |
| 1024 | 37 | 0 | 40 | 1 | 32 | 50 % | registers (16 slots left over) |
| 256 | 64 | 0 | 64 | 4 | 32 | 50 % | registers |
| 128 | 32 | 32 KB | 32 | 4 | 16 | 25 % | shared memory |

The 37-register row is the allocation unit, not the source. $32 \times 40 = 1280$ registers/warp, and $4\lfloor 65536/1280/4 \rfloor = 48$ warps. Twelve blocks of 4 warps use all 48. A 1024-thread block is 32 warps, so only one fits, and $48 - 32 = 16$ warp slots stay empty.

On an RTX 4090 the same 128-thread, 37-register kernel gets $48/48 = 100\%$, because that SM's ceiling is 48 warps. The kernel did not get faster; the denominator shrank.

Shared memory: $32 \times 1024 + 1024 = 33792$ bytes/block. $\lfloor 164 \times 1024 / 33792 \rfloor = 4$ blocks.

## Waves

With $N_{\text{SM}}$ SMs and $B$ resident blocks per SM, one wave is $N_{\text{SM}} B$ blocks. A grid of $G$ blocks runs

$$
\left\lceil \frac{G}{N_{\text{SM}} B} \right\rceil
$$

waves, and the last wave occupies $G \bmod (N_{\text{SM}} B)$ block slots (or a full wave, when that remainder is 0).

## Warps required to hide a stall

Four schedulers, each wanting one eligible warp per cycle. A warp that has $I$ independent instructions and then waits out a latency of $L$ cycles is eligible a fraction $\min(1, I/L)$ of the time. Warps that keep every scheduler busy:

$$
W_{\text{hide}} = \left\lceil \frac{4}{\min(1, I/L)} \right\rceil = \left\lceil 4 \cdot \frac{L}{I} \right\rceil
\quad\text{when } I < L.
$$

Dependent FMA, $L = 4$, $I = 1$: $W_{\text{hide}} = 16$. That is 25 % occupancy on an A100. Raising occupancy from 25 % to 100 % does not speed a pure FMA loop.

Dependent HBM load, $L \approx 500$, $I = 1$: $W_{\text{hide}} = 2000$. The SM has 64 warp slots, so the schedulers stay idle $\tfrac{64}{2000}$ of the time no matter how you launch. The fix is a larger $I$ (more independent misses per thread), not a higher percentage.

## When the pipe, not the scheduler, is the limit

Bytes one SM must keep in flight to hit a link of bandwidth $\beta$ (bytes/s) and latency $t$ (seconds):

$$
\text{bytes} = \frac{\beta}{N_{\text{SM}}} \cdot t,
\qquad
t = \frac{L}{f}.
$$

A100 HBM: $\beta = 2039$ GB/s, $N_{\text{SM}} = 108$, $L = 500$, $f = 1.41$ GHz.

$$
t = 355\text{ ns},
\qquad
\text{bytes} \approx 6.5\text{ KB per SM}.
$$

A coalesced warp load is 128 bytes, so about $6.5 \times 1024 / 128 \approx 52$ such loads in flight fill the pipe. Fifty-two resident warps are enough for a streaming kernel. They are not enough for a chase of 4-byte dependent loads, which keeps $52 \times 4 = 208$ bytes in flight — about 3 % of the pipe.
