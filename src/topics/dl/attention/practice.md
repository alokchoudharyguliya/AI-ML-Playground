## Exercises

**Q1.** An encoder produces $S=5$ states in $\mathbb R^2$: $h_1=(1,0)$, $h_2=(0,1)$, $h_3=(1,1)$, $h_4=(-1,0)$, $h_5=(0,-1)$. The decoder query is $s=(2,0)$ (dot-product scores, $T=1$). Compute the weights approximately and the context vector.

<details>
<summary>Show answer</summary>

Scores $e=s\cdot h=(2,0,2,-2,0)$. $\exp$: $(7.39,1,7.39,0.135,1)$, sum $=16.92$. $\alpha\approx(0.437,0.059,0.437,0.008,0.059)$. Context $c=\sum\alpha_ih_i\approx(0.437+0.437-0.008,\ 0.059+0.437-0.059)=(0.866,\,0.437)$. It sits between $h_1$ and $h_3$, the two best matches.

</details>

**Q2.** What happens to the attention weights as $T\to0$ and $T\to\infty$? What does the second lab view show for large "sharpness"?

<details>
<summary>Show answer</summary>

$T\to0$: one-hot on the argmax — hard attention, zero entropy, vanishing gradients to the other positions. $T\to\infty$: uniform average, entropy $\log_2S$ — the model loses selectivity. Sharpness in the lab is $1/T$: at large values the star (context) snaps to the nearest key.

</details>

**Q3.** Why must attention scores of padding tokens be set to $-\infty$ *before* the softmax rather than zeroing the weights afterwards?

<details>
<summary>Show answer</summary>

Zeroing after the softmax leaves the remaining weights unnormalised (they no longer sum to one) and padding would still have stolen probability mass. Masking before the softmax renormalises over the real tokens only. Use a large negative number (e.g. $-10^9$ or `finfo.min`) rather than literal $-\infty$ if an entire row can be masked, to avoid `NaN`.

</details>

**Q4.** The attention matrix for a source of 100 tokens and a target of 100 tokens at batch 32 with fp16 weights: how much memory?

<details>
<summary>Show answer</summary>

$32\cdot100\cdot100\cdot2$ bytes $=640$ KB per head per layer — trivial here. For 32k-token self-attention it becomes $32\cdot32768^2\cdot2\approx68$ GB per head — which is why FlashAttention (CUDA track) avoids materialising it.

</details>

**Q5.** Name two ways attention weights can mislead as an explanation of model behaviour.

<details>
<summary>Show answer</summary>

(1) Different attention distributions can yield the same prediction (weights are not unique explanations). (2) Information flows through values and later layers too; a token with low weight can still matter and a high weight can be irrelevant. Use gradient-based or perturbation methods for attribution.

</details>

**Q6 (code).** Extend `attention_numpy.py` to add a causal mask and verify that row $t$ of the weight matrix has zeros for all positions $>t$.

<details>
<summary>Show answer</summary>

Add `mask = np.triu(np.ones((T, T), bool), 1)`; set `scores[mask] = -1e9` before the softmax; assert `np.allclose(weights[mask], 0)`. This is exactly the decoder-side mask in GPT-style models.

</details>

## In practice

- Pre-2017 production MT systems (GNMT, early Google Translate NMT) used LSTM encoder–decoders with attention; today's translation models are Transformers, but the *cross-attention* layer of an encoder-decoder Transformer is this same mechanism.
- **Cross-attention is everywhere**: text-conditioned diffusion models attend from image latents to prompt tokens; Whisper attends from text decoder to audio encoder; retrieval-augmented models attend to retrieved passages.
- **Debugging**: plot attention heat-maps for a few validation sentences — a diagonal band is healthy for monotonic tasks like speech recognition; smeared or collapsed maps signal training problems.
- **Monotonic / location-aware attention** helps speech synthesis (Tacotron) where alignment must progress left to right.

## Common pitfalls

- Forgetting the source padding mask (the model learns to attend to padding).
- Computing the softmax over the wrong axis (over targets instead of sources).
- Leaking future target tokens during training (no causal mask in the decoder).
- Using attention weights as an "explanation" without ablation.
- Beam search without length normalisation → overly short outputs.
