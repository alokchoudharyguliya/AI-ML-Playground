## Index arithmetic

**1-D:**

$$
i=\text{blockIdx.x}\cdot\text{blockDim.x}+\text{threadIdx.x},\qquad
\text{gridDim.x}=\Big\lceil\frac{N}{\text{blockDim.x}}\Big\rceil .
$$

**2-D** (row-major image, width $W$):

$$
x=\text{blockIdx.x}\cdot\text{blockDim.x}+\text{threadIdx.x},\quad
y=\text{blockIdx.y}\cdot\text{blockDim.y}+\text{threadIdx.y},\quad
\text{idx}=y\,W+x .
$$

**Linear thread id inside a block** (determines the warp):

$$
\text{tid}=\text{threadIdx.x}+\text{threadIdx.y}\cdot\text{blockDim.x}+\text{threadIdx.z}\cdot\text{blockDim.x}\,\text{blockDim.y},
$$

$$
\text{warp}=\lfloor\text{tid}/32\rfloor,\qquad\text{lane}=\text{tid}\bmod32 .
$$

Warps per block: $\lceil T_b/32\rceil$ with $T_b$ threads per block. Wasted lanes: $32\lceil T_b/32\rceil-T_b$.

## Counting work

Threads launched $=\text{gridDim}\times\text{blockDim}\ge N$. Wasted (guarded-off) threads $\le\text{blockDim}-1$. For $N=10^6$ and 256-thread blocks: 3907 blocks $\Rightarrow$ 1,000,192 threads, 192 idle.

## Waves and the tail effect

With $S$ SMs, $R$ resident blocks per SM (from occupancy), $B$ blocks:

$$
\text{waves}=\Big\lceil\frac{B}{S\,R}\Big\rceil,\qquad
\text{tail utilisation}=\frac{B}{S\,R\cdot\text{waves}} .
$$

Example: $B=1000$, $S=108$, $R=8$ ⇒ capacity 864/wave ⇒ 2 waves, utilisation $1000/1728=58\%$; but with $B=1728$ it is 100%. Tail waste is $<1/\text{waves}$, so more, smaller blocks (or persistent kernels) reduce it.

## Peak throughput from the hardware spec

$$
\text{FP32 peak}=\underbrace{\#SM}_{}\times\underbrace{\#\text{FP32 lanes/SM}}_{}\times\underbrace{2}_{\text{FMA}}\times f_{clk}.
$$

- A100: $108\times64\times2\times1.41\,\mathrm{GHz}=19.5$ TFLOPS.
- H100 SXM: $132\times128\times2\times1.98\,\mathrm{GHz}\approx67$ TFLOPS.

An FMA (fused multiply-add) counts as 2 FLOPs.

## Latency hiding (Little's law)

To sustain throughput $\lambda$ (operations/cycle) when each takes latency $L$ cycles you need $\lambda L$ operations in flight:

$$
\text{in-flight}=\text{latency}\times\text{throughput}.
$$

Memory: A100 HBM $\approx2\,\mathrm{TB/s}$ with $\approx500$-cycle latency at 1.4 GHz ⇒ bytes in flight $\approx2\times10^{12}\times\tfrac{500}{1.4\times10^9}\approx7\times10^{5}$ B $\approx700$ KB ⇒ across 108 SMs ≈ 6.5 KB per SM outstanding — e.g. 1,600 independent 4-byte loads per SM. That is why you want **many resident warps** (or many loads in flight per thread, "ILP").

## Instruction issue

Each SM sub-core issues one warp-instruction per cycle. FP32 FMA throughput per SM per cycle: $4\times32=128$ lanes (H100). A warp stalled on a dependency simply doesn't issue; the scheduler picks another eligible warp. Fully hiding a 4-cycle ALU latency needs ≥ 4 independent warps per scheduler, i.e. ≥ 16 warps per SM; hiding a 400-cycle memory latency needs far more parallelism or independent loads.

## Launch limits (compute capability 8.x / 9.0)

| Resource | Limit |
|---|---|
| Threads per block | 1024 |
| Block dims (x, y, z) | (1024, 1024, 64) with product ≤ 1024 |
| Grid dims (x, y, z) | ($2^{31}-1$, 65535, 65535) |
| Resident threads / SM | 2048 (1536 on Ada, 2048 on Hopper) |
| Resident blocks / SM | 32 (24 on Ada) |
| Resident warps / SM | 64 |
| Registers / thread | ≤ 255 |
| Registers / SM | 65,536 |
