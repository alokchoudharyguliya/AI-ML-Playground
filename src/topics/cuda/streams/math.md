## Serial baseline

One stream, $N$ chunks, stages of length $H$ (host to device), $K$ (kernel) and $D$ (device to host):

$$
T_1 = N\,(H + K + D).
$$

Nothing overlaps, pinned or not. The same total is what you get from pageable `cudaMemcpyAsync`: the host cannot stay ahead of the copies, so the streams never have two operations ready at once.

## Two copy engines, enough streams

Call the bottleneck $B = \max(H, K, D)$. With pinned memory, separate H2D and D2H engines, and enough streams to keep every unit fed, the timeline is one pass of the short stages plus $N$ passes of the long one:

$$
T = N B + (H + K + D - B).
$$

The added term is the fill and the drain: the stages that are not the bottleneck run once beside the ends, instead of once per chunk.

For three stages of similar length, "enough" is **three streams**. Two streams cannot hold a copy-in, a kernel and a copy-out at the same time, so they land above this line. A fourth stream has nothing left to overlap and lands on it.

### Worked timelines

Stages are in the same time unit. Two engines unless noted. Pinned unless noted.

| $N$ | Streams | $H, K, D$ | Engines | $T$ | $T_1$ |
|---|---|---|---|---|---|
| 8 | 1 | 2, 2, 2 | 2 | 48 | 48 |
| 8 | 2 | 2, 2, 2 | 2 | 26 | 48 |
| 8 | 3 | 2, 2, 2 | 2 | 20 | 48 |
| 8 | 4 | 2, 2, 2 | 2 | 20 | 48 |
| 8 | 3 | 2, 2, 2 | 1 | 32 | 48 |
| 6 | 2 | 4, 2, 1 | 2 | 27 | 42 |
| 8 | 3 | 2, 2, 2 | 2, pageable | 48 | 48 |

Check the formula on the equal row: $B = 2$, $T = 8 \cdot 2 + (6 - 2) = 20$. Three streams and four streams both hit it. Two streams finish at 26, because one of the three units is idle whenever the other two are busy.

The copy-bound row: $B = H = 4$, $T = 6 \cdot 4 + (4 + 2 + 1 - 4) = 27$. The last chunk's kernel and copy-out are the only extra. A fourth stream still finishes at 27. The copy engine is already full.

One copy engine, equal stages: H2D and D2H never run together, so the copies alone cost $N(H + D) = 32$. Each kernel is the same length as one copy and hides beside the other direction. $T = 32$, not 20. Half the overlap disappeared with the second engine.

Pageable memory: $T = T_1$ at any stream count.

## Speedup

$$
\frac{T_1}{T} = \frac{H + K + D}{B + (H + K + D - B)/N}
\;\xrightarrow{N \to \infty}\;
\frac{H + K + D}{B}.
$$

Equal stages, two engines: the ceiling is $3\times$, and $N = 8$ is already at $48/20 = 2.4\times$. You do not get the ceiling at small $N$, because the fill and drain are still a visible fraction. One engine caps the same problem at $(H+K+D)/(H+D) = 1.5\times$, which is exactly $48/32$.

## A dependency without a host stall

`cudaStreamWaitEvent` orders one operation in stream B behind one event in stream A. The host thread is not in that wait, so the enqueue of the rest of the pipeline continues. The GPU timeline gains a single edge. `cudaDeviceSynchronize` is the opposite edge: every stream, and the host, stop together.
