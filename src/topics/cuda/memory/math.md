## Effective bandwidth

For a kernel that reads $R$ bytes and writes $W$ bytes in time $t$:

$$
\text{BW}_{\text{eff}}=\frac{R+W}{t}.
$$

Theoretical peak from the device attributes (DDR doubles the effective rate):

$$
\text{BW}_{\text{peak}}=2\times f_{\text{mem}}\times\frac{\text{bus width}}{8}.
$$

A100 80GB: $2\times1.593\,\text{GHz}\times\frac{5120}{8}=2.04$ TB/s. STREAM *triad* $a_i=b_i+q\,c_i$ moves $3\,n\,s$ bytes ($s$ = bytes per element) and does $2n$ FLOP.

## Average memory access time (AMAT)

Let a load be served by L1 with probability $p_1$, by L2 with $p_2$ and by HBM with $p_3$, where $p_1=h_1$, $p_2=(1-h_1)h_2$, $p_3=(1-h_1)(1-h_2)$. With *end-to-end* latencies $L_1<L_2<L_3$:

$$
\text{AMAT}=p_1L_1+p_2L_2+p_3L_3 .
$$

Example (A100: $L_1=33$, $L_2=200$, $L_3=500$ cycles) with $h_1=0.5,\ h_2=0.6$:

$$
0.5(33)+0.3(200)+0.2(500)=16.5+60+100=176.5\ \text{cycles},
$$

about $2.8\times$ better than sending everything to HBM. Note that the *misses* dominate: the 20 % of loads that reach HBM contribute 57 % of the average.

## Little's law

For a pipeline with latency $\ell$ cycles and throughput $\lambda$ bytes/cycle, the bytes that must be in flight to keep it full are

$$
B_{\text{inflight}}=\lambda\,\ell .
$$

Per SM, with aggregate bandwidth $\beta$ (B/s), $N_{SM}$ SMs and clock $f$: $\lambda=\dfrac{\beta}{N_{SM}\,f}$.

**A100 / HBM:**

$$
\lambda=\frac{2.039\times10^{12}}{108\times1.41\times10^{9}}\approx13.4\ \text{B/cycle/SM},\qquad
B_{\text{inflight}}\approx13.4\times500\approx6.7\ \text{KB per SM}.
$$

Your supply: $B_{\text{supply}}=N_{\text{warps}}\cdot32\cdot b\cdot k$ for loads of $b$ bytes with $k$ independent loads in flight per thread. Achievable fraction of peak:

$$
\eta=\min\!\Big(1,\ \frac{N_{\text{warps}}\cdot32\cdot b\cdot k}{\lambda\,\ell}\Big),\qquad
N_{\text{warps}}^{\text{needed}}=\frac{\lambda\,\ell}{32\,b\,k}.
$$

With 4-byte loads and $k=1$: $6{,}700/128\approx52$ warps (of 64 possible). With `float4` ($b=16$): 13 warps. With $b=4,\ k=4$: 13 warps. **Wider loads and ILP substitute for occupancy.**

## Arithmetic intensity

$$
I=\frac{W}{Q}\ \ \text{[FLOP/byte]},\qquad W=\text{FLOPs},\quad Q=\text{bytes moved at the level of interest.}
$$

With $s$ bytes per element (FP64 8, FP32 4, FP16 2):

| Kernel | $W$ per element | $Q$ per element | $I$ |
|---|---|---|---|
| vector add $c=a+b$ | 1 | $3s$ | $\dfrac{1}{3s}$ ($=\tfrac1{12}$ in FP32) |
| SAXPY $y=ax+y$ | 2 | $3s$ | $\dfrac{2}{3s}$ ($=\tfrac16$) |
| sum reduction | 1 | $s$ | $\dfrac1s$ |
| 3-point stencil (cached neighbours) | 5 | $2s$ | $\dfrac{5}{2s}$ |
| LayerNorm (read + write) | ≈8 | $2s$ | $\approx\dfrac{4}{s}$ |

### GEMM, naive vs tiled vs ideal

$C=AB$ with $N\times N$ matrices does $W=2N^3$ FLOP.

- **Naive** (every output re-reads a row of $A$ and a column of $B$ from global memory): $Q\approx 2N^3 s$, so $I=\dfrac{1}{s}$, independent of $N$.
- **Tiled** with $T\times T$ shared-memory tiles: each tile load is reused $T$ times, so $Q\approx\dfrac{2N^3s}{T}$ and
$$
I_{\text{tiled}}=\frac{T}{s}.
$$
- **Ideal** (each of $A,B$ read once, $C$ written once, $Q=3N^2s$):
$$
I_{\text{ideal}}=\frac{2N^3}{3N^2s}=\frac{2N}{3s}\quad\Big(=\frac N6\ \text{in FP32}\Big).
$$

## The roofline

$$
P_{\text{attainable}}(I)=\min\big(\pi,\ \beta I\big),\qquad I^*=\frac{\pi}{\beta},
$$

with $\pi$ in FLOP/s and $\beta$ in B/s. Equivalently the **time model**

$$
t\ \ge\ \max\Big(\frac{W}{\pi},\ \frac{Q}{\beta}\Big),
$$

whose first term is compute time and second is memory time; whichever is larger is the bound.

### Ridge points

| GPU | $\pi$ (TFLOP/s) | $\beta$ (TB/s) | $I^*=\pi/\beta$ |
|---|---|---|---|
| A100, FP32 | 19.5 | 2.04 | **9.6** FLOP/B |
| A100, FP16 Tensor | 312 | 2.04 | **153** |
| H100, FP32 | 67 | 3.35 | **20** |
| H100, FP16 Tensor | 989 | 3.35 | **295** |
| RTX 4090, FP32 | 82.6 | 1.01 | **82** |

Unit trick: $1\ \text{TB/s}\times1\ \text{FLOP/B}=1$ TFLOP/s, so on the plot the memory roof is just $P=\beta_{\text{TB/s}}\cdot I$.

### Worked examples (A100)

**Vector add (FP32):** $I=\tfrac1{12}=0.083$. Attainable $=0.083\times2039\ \text{GB/s}=170$ GFLOP/s, only **0.87 %** of the 19.5 TFLOP/s peak, yet that is the *best possible*. For $2^{26}$ elements: $Q=12\cdot2^{26}=805$ MB, so $t\ge805\ \text{MB}/2.039\ \text{TB/s}=0.40$ ms.

**GEMM becomes compute-bound at** $I>I^*$. FP32 ideal: $N/6>9.6\Rightarrow N>58$. FP16 Tensor ideal ($s=2$, $I=N/3$): $N/3>153\Rightarrow N>459$. On H100 the same test gives $N>885$. Small GEMMs are memory-bound even with perfect kernels.

**LLM decode:** each weight (2 B in FP16) is used for 2 FLOP per sequence in the batch, so

$$
I_{\text{decode}}=\frac{2B}{s}=B\quad(\text{FP16}),
$$

and the batch needed to reach the H100 FP16 Tensor ridge is $B\approx295$. At $B=1$ a 7B-parameter FP16 model streams 14 GB per token: $14\ \text{GB}/3.35\ \text{TB/s}\approx4.2$ ms, or $\lesssim240$ tokens/s no matter how many TFLOPs the GPU has.

## Multiple ceilings

Real GPUs have a roof per memory level (L1, L2, HBM) and per instruction type (FP64, FP32, TF32/FP16 Tensor). The *cache-aware* roofline draws all of them. The binding one is the lowest roof at your kernel's intensity *for the traffic at that level*. If the working set fits in L2, the relevant slope is $\beta_{L2}$, not $\beta_{HBM}$.

## Efficiency against the roof

$$
\text{efficiency}=\frac{P_{\text{achieved}}}{P_{\text{attainable}}(I)}.
$$

An efficiency near 1 means the kernel is on the roof: only an algorithmic change (raising $I$) can help. A low efficiency means a latency or inefficiency problem the roofline cannot see: low occupancy, uncoalesced access, bank conflicts, divergence, launch overhead.
