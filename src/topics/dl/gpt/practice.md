## Exercises

**Q1.** Logits for three tokens are $(4,2,0)$. Compute the probabilities at $T=1$ and $T=0.5$.

<details>
<summary>Show answer</summary>

$T=1$: $e^{(4,2,0)}=(54.6,7.39,1)\Rightarrow(0.867,0.117,0.016)$. $T=0.5$: logits $(8,4,0)$ → $(2981,54.6,1)\Rightarrow(0.982,0.018,0.0003)$. Lower temperature concentrates mass on the top token.

</details>

**Q2.** With probabilities $(0.5,0.3,0.1,0.05,0.03,0.02)$, which tokens survive top-k=3 and top-p=0.85? What are the renormalised probabilities for top-p?

<details>
<summary>Show answer</summary>

Top-k=3 → tokens 1–3 (mass 0.9). Top-p=0.85: cumulative $0.5,0.8,0.9\ge0.85$ → keep tokens 1–3 as well; renormalised by 0.9: $(0.556,0.333,0.111)$. With $p=0.75$ only tokens 1–2 would stay ($0.5,0.8$) → $(0.625,0.375)$.

</details>

**Q3.** Compute the KV-cache size for Mistral-7B ($L=32$, 8 KV heads, $d_{head}=128$) at 32k context, batch 16, fp16.

<details>
<summary>Show answer</summary>

Per token: $2\cdot32\cdot8\cdot128\cdot2=131{,}072$ B. Total: $131{,}072\cdot32{,}768\cdot16=6.87\times10^{10}$ B $\approx64$ GiB — more than four times the 14.5 GB of weights. Quantising KV to fp8 halves it; paging eliminates waste.

</details>

**Q4.** Estimate the max decode throughput of a 70B fp16 model at batch 1 on an H100 (3.35 TB/s). Why does it not fit on one GPU, and what changes with 8-way tensor parallelism?

<details>
<summary>Show answer</summary>

Weights: 140 GB > 80 GB, so it needs ≥2 GPUs (or 4-bit quantisation: 35 GB). Bandwidth bound: $3.35\times10^{12}/1.4\times10^{11}\approx24$ tokens/s per GPU-equivalent. With TP=8 each GPU reads 1/8 of the weights concurrently: ideal $\approx190$ tokens/s, reduced by all-reduce latency.

</details>

**Q5.** Speculative decoding with $\alpha=0.7$, $\gamma=5$, draft cost 5% of the target. Estimate the speed-up.

<details>
<summary>Show answer</summary>

Tokens per cycle: $\frac{1-0.7^6}{0.3}=2.94$. Cost per cycle: 1 target pass + 5 draft passes $=1+5\cdot0.05=1.25$ target-equivalents. Speed-up $\approx2.94/1.25\approx2.4\times$.

</details>

**Q6 (code).** Prove numerically that incremental decoding with a KV cache gives the same logits as recomputing the whole prefix.

<details>
<summary>Show answer</summary>

`kv_cache.py` implements both paths for a single attention layer and asserts `torch.allclose` on the last-token output at every step; the cache path touches one token per step.

</details>

## In practice

- **Sampling presets**: factual/coding → $T\in[0,0.3]$; creative writing → $T\approx0.8$–$1.0$ with top-p 0.9–0.95; always cap `max_tokens`.
- **Serving stacks**: vLLM, TensorRT-LLM, SGLang, llama.cpp (CPU/Metal, GGUF quantisation), TGI. Choose by hardware, latency vs throughput goals.
- **Quantisation**: weights int8/int4 (GPTQ, AWQ), KV fp8 — 2–4× less memory traffic with small quality loss.
- **Evaluation**: perplexity is necessary but insufficient — use task evals, human preference and regression suites; watch for contamination.
- **Cost model**: price ≈ GPU-hour cost ÷ (tokens/s × 3600); batching and caching dominate economics.
- **RAG**: retrieve passages → put in context; use prefix caching for static instructions.

## Common pitfalls

- Using temperature 0 with sampling code that divides by zero (special-case greedy).
- Applying top-p before temperature (order matters).
- Forgetting to apply the *same* tokenizer chat template used in training.
- Position-id bugs when decoding with a cache (off-by-one) or with left-padded batches.
- Comparing perplexities across different tokenisers.
- Assuming longer context = better recall (needle-in-a-haystack ≠ reasoning over long documents).
