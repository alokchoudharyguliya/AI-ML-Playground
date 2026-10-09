## Exercises

**Q1.** A decoder-only model has $d=2048$, $L=24$, vocabulary $50{,}000$ (tied embeddings). Estimate the parameter count.

<details>
<summary>Show answer</summary>

Blocks: $12\,L\,d^2=12\cdot24\cdot2048^2=1.208\times10^9$. Embeddings: $50{,}000\cdot2048=1.02\times10^8$ (tied, counted once). Total $\approx1.31$ B.

</details>

**Q2.** Estimate the attention-score matrix memory (fp16) for one layer with 16 heads at sequence length 8192, batch 4, if materialised naively. What does FlashAttention store instead?

<details>
<summary>Show answer</summary>

$4\cdot16\cdot8192^2\cdot2$ bytes $=8.6$ GB per layer (before the softmax copy!). FlashAttention keeps only $O(n)$ statistics per row (running max and sum — the logsumexp) and recomputes tiles in the backward pass, so memory is linear in $n$.

</details>

**Q3.** You remove the positional encoding from a Transformer encoder and feed the sentence "dog bites man" vs "man bites dog". What can the model tell apart?

<details>
<summary>Show answer</summary>

Nothing about order: self-attention is permutation-equivariant, so the pooled/bag representation is identical. (Decoder models with causal masks can still infer *some* order implicitly, because token $i$ sees exactly $i$ predecessors — "NoPE" works to a degree — but explicit position is far better.)

</details>

**Q4.** Show that the FFN has twice the parameters of the attention block.

<details>
<summary>Show answer</summary>

Attention: $W_Q,W_K,W_V,W_O\Rightarrow4d^2$. FFN: $d\cdot4d+4d\cdot d=8d^2$. So FFN : attention $=2:1$, i.e. the FFN holds two-thirds of a block's weights.

</details>

**Q5.** With $d_k=64$ unscaled scores have std ≈8. If the largest score exceeds the others by 16, what is the maximum softmax weight roughly, and why does it hurt training?

<details>
<summary>Show answer</summary>

$e^{16}\approx8.9\times10^{6}$ vs $(n-1)$ other terms of order 1 → weight $\approx1-10^{-5}$. The Jacobian $p_i(1-p_i)\approx10^{-5}$: gradients to the scores vanish, so the model cannot learn to *change* its attention.

</details>

**Q6 (code).** Implement causal masking via `-inf` and check that `F.scaled_dot_product_attention(..., is_causal=True)` matches.

<details>
<summary>Show answer</summary>

`sdpa_check.py` in the Code tab builds the mask with `torch.triu(torch.ones(T,T),1).bool()`, applies `masked_fill(-inf)` before softmax and asserts `allclose` against the fused kernel.

</details>

## In practice

- **Use the fused kernels.** `F.scaled_dot_product_attention` dispatches to FlashAttention / memory-efficient kernels; hand-written softmax(QKᵀ)V is 2–4× slower and memory hungry.
- **Shapes to remember**: `(B, H, T, d_head)` for Q/K/V; keep `d_head ∈ {64, 128}` for Tensor-Core efficiency.
- **Pre-LN + residual scaling** (or μP-style init) for stable deep training; QK-norm for very large models.
- **Inference**: cache keys/values (next chapters), use GQA, batch with continuous batching (vLLM), quantise weights (INT8/INT4).
- **Vision**: ViT splits images into 16×16 patches → linear embedding → Transformer; needs more data than CNNs unless pre-trained or heavily augmented.
- **Probing a model**: attention rollout, logit-lens, activation patching; remember that attention weights alone are not explanations.

## Common pitfalls

- Applying softmax over the wrong dimension.
- Forgetting the causal mask during training (the model "cheats" and perplexity looks implausibly good).
- Mixing up batch-first and sequence-first conventions in `nn.MultiheadAttention`.
- Position indices off by one at inference when using a KV cache.
- fp16 overflow in attention logits — compute softmax in fp32 or use bf16.
- Sequence lengths beyond the trained context with learned absolute positions.
