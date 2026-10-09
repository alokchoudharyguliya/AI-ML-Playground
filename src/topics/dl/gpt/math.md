## Language-model loss, perplexity, bits

$$
\mathcal L=-\frac1N\sum_{t=1}^{N}\log p_\theta(x_t\mid x_{<t}),\qquad
\mathrm{PPL}=e^{\mathcal L},\qquad
\text{bits/token}=\mathcal L/\ln2 .
$$

Perplexity is the effective branching factor: a uniform guess over $V$ tokens has PPL $=V$. Bits-per-*byte* normalises across tokenisers: $\mathrm{BPB}=\mathcal L\cdot\frac{\#\text{tokens}}{\#\text{bytes}\cdot\ln2}$.

## Temperature

$$
p_i(T)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}.
$$

$T\to0$: argmax; $T=1$: model distribution; $T\to\infty$: uniform. Entropy increases monotonically with $T$. The ratio of two probabilities changes as $p_i/p_j=\exp\big((z_i-z_j)/T\big)$.

## Truncation samplers

Sort tokens by probability $p_{(1)}\ge p_{(2)}\ge\cdots$.

**Top-k**: keep $\{(1),\dots,(k)\}$, renormalise.

**Top-p** (nucleus): keep the smallest $m$ with $\sum_{i\le m}p_{(i)}\ge p$.

**Min-p**: keep tokens with $p_i\ge\alpha\,p_{(1)}$.

All then renormalise: $\tilde p_i=p_i\,\mathbb 1[i\in S]\big/\sum_{j\in S}p_j$. The lab applies temperature → top-k → top-p in that order (as Hugging Face does).

## Penalties

Repetition penalty $\theta>1$ on tokens already generated (CTRL-style): $z_i\leftarrow z_i/\theta$ if $z_i>0$ else $z_i\theta$. Frequency / presence penalties (OpenAI): $z_i\leftarrow z_i-\alpha_f\,c_i-\alpha_p\,\mathbb 1[c_i>0]$, with $c_i$ the count so far.

## Beam search

Keep the $B$ highest-scoring partial sequences $y_{1:t}$ with score $\sum\log p(y_t\mid y_{<t})$, optionally divided by $\mathrm{len}^{\alpha}$ to avoid short-sequence bias.

## KV-cache memory

$$
\text{KV bytes}=2\;\underbrace{L}_{\text{layers}}\;\underbrace{n_{kv}}_{\text{KV heads}}\;\underbrace{d_{head}}_{\text{head dim}}\;\underbrace{b}_{\text{bytes}}\;\times\;\underbrace{n}_{\text{tokens}}\times\underbrace{B}_{\text{batch}} .
$$

Llama-3-8B ($L=32,n_{kv}=8,d_{head}=128$, fp16): $2\cdot32\cdot8\cdot128\cdot2=131{,}072$ B $=128$ KiB/token → $1$ GiB at 8k context, $16$ GiB at 128k, per sequence.

**Cost without cache.** Generating $N$ tokens by re-running the prefix each step costs $\sum_{t=1}^N t=\tfrac{N(N+1)}2$ token-forward passes; with a cache, $N$.

## Decode is memory-bound

For batch size $B$, one decode step performs about $2NB$ FLOPs on weights (with $N$ parameters) but must read $2N$ bytes of fp16 weights (plus the KV cache) from HBM:

$$
\text{arithmetic intensity}\approx\frac{2NB}{2N+\text{KV}}\approx B\ \text{FLOP/byte}.
$$

A GPU with bandwidth $\mathrm{BW}$ and peak $\Pi$ has ridge point $\Pi/\mathrm{BW}$ (≈ 150–300 FLOP/B for H100 tensor cores). Below it the speed limit is

$$
\text{tokens/s per stream}\le\frac{\mathrm{BW}}{2N+\text{KV bytes}},
$$

e.g. an 8B fp16 model on 3.35 TB/s: $3.35\times10^{12}/16\times10^{9}\approx210$ tokens/s upper bound at batch 1. Batching raises throughput almost linearly until the ridge point.

## Speculative decoding

With draft length $\gamma$ and per-token acceptance rate $\alpha$ (i.i.d.), the expected tokens produced per target-model pass is

$$
\mathbb E[\text{tokens}]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}.
$$

For $\alpha=0.8,\gamma=4$: $\frac{1-0.8^5}{0.2}=3.36$ tokens per big-model step. Accept/reject uses $\min\big(1,p_{target}(x)/p_{draft}(x)\big)$, preserving the target distribution exactly.

## Scaling of FLOPs

Training: $\approx6ND$ FLOPs for $N$ parameters and $D$ tokens. Inference: $\approx2N$ FLOPs/token plus $\approx4Lnd$ for attention over $n$ cached tokens.
