## "Attention is all you need"

Vaswani et al. (2017) asked: if attention is what makes seq2seq work, why keep the recurrence? The **Transformer** drops RNNs entirely. Every token looks at every other token in a *single parallel step*, so

- training parallelises over the whole sequence (RNNs are sequential in time),
- any two positions are connected by a **path of length 1** (RNN: up to $n$), easing long-range credit assignment,
- compute is dominated by large matrix multiplications — exactly what GPUs and Tensor Cores are built for.

This match between the model and the hardware is a large part of why Transformers scaled to GPT-4-class models.

## Self-attention in one sentence

Every token emits a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a **value** ("what I will hand over"). Token $i$'s output is the average of all values, weighted by how well its query matches each key:

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)V$$

In the first tab of the lab each row of the heat-map is one token's attention distribution, computed exactly from random $q,k$ vectors. Notice what the **$1/\sqrt{d_k}$** does: without it, the scores grow like $\sqrt{d_k}$, the softmax collapses to a one-hot, and its gradient dies.

<div class="callout">

**Mental model.** Attention is a *content-addressable memory read*: the token composes a query, the sequence is the memory, and the result is a similarity-weighted blend of memory contents. Stack many layers and many heads and the network can route, copy, compare and compose information between positions.

</div>

## Multi-head attention

One softmax can only express one "pattern" of dependence per token. **Multi-head attention** runs $h$ attentions in parallel in lower-dimensional subspaces ($d_{head}=d/h$), then concatenates and mixes with $W_O$. Different heads specialise: local syntax, previous-token copying, coreference, delimiter "sinks", induction heads (the core of in-context learning). Total cost and parameters are the same as one big head.

## The Transformer block

A modern (pre-LN) block is two residual sub-layers:

$$x\leftarrow x+\mathrm{MHA}(\mathrm{LN}(x)),\qquad x\leftarrow x+\mathrm{FFN}(\mathrm{LN}(x))$$

- **Residual stream**: the vector $x$ per token is a shared communication channel; each layer reads from it and *adds* to it. This is the same skip-connection idea as ResNet and the LSTM cell state.
- **LayerNorm** stabilises scale per token (RMSNorm in modern LLMs). *Pre-LN* trains stably at depth; the original *post-LN* needs careful warmup.
- **Feed-forward network**: $\mathrm{FFN}(x)=W_2\,\sigma(W_1x)$ with hidden size $4d$ (or SwiGLU with $\tfrac83d$). It holds ~2/3 of the parameters and acts like a key–value *memory* of facts and features; attention moves information *between* tokens, the FFN transforms it *within* each token.

## Three architectures from the same parts

| Family | Mask | Examples | Use |
|---|---|---|---|
| **Encoder-only** | bidirectional | BERT, RoBERTa, DeBERTa | classification, retrieval, embeddings |
| **Decoder-only** | causal | GPT, Llama, Claude-style LLMs | generation, chat, code |
| **Encoder–decoder** | bidirectional enc, causal dec + **cross-attention** | T5, BART, Whisper, original Transformer | translation, summarisation, speech |

**Cross-attention** takes queries from the decoder and keys/values from the encoder — exactly the Bahdanau attention of the previous chapter, now multi-head.

## Position information

Self-attention treats its input as a *set*: shuffle the tokens and the output merely shuffles. Order must be injected:

- **Sinusoidal** (original): fixed sines/cosines at geometric wavelengths added to embeddings (third lab tab).
- **Learned absolute** (BERT, GPT-2): a trainable table; cannot extrapolate beyond training length.
- **Relative** (T5 bias, Transformer-XL) and **ALiBi**: add a distance-dependent bias to scores.
- **RoPE** (Llama, most modern LLMs): rotate $q,k$ by position-dependent angles — see *Modern LLM Toolkit*.

## Cost and the long-context problem

Self-attention builds an $n\times n$ score matrix: **$O(n^2)$** compute *and* (naively) memory. At $n=32$k and fp16 a single head's matrix is 2 GB. Remedies: **FlashAttention** (exact, IO-aware tiling — see the CUDA track), sparse/sliding-window attention, linear attention, KV-cache tricks (GQA/MQA), and state-space hybrids.

## Training recipe (what actually matters)

- **Warmup + cosine/linear decay**, AdamW, $\beta_2=0.95$–$0.999$, weight decay 0.1, gradient clip 1.0.
- **Tokenisation** (BPE / SentencePiece) and **data quality/quantity** dominate results.
- **Label smoothing** (translation), **dropout** 0–0.1 (often 0 for LLMs), **mixed precision** (bf16).
- **Scaling**: loss follows power laws in parameters, data and compute (see the last DL chapter).

## Beyond text

The same block, with different tokenisation, powers **ViT** (image patches), **Whisper/audio**, **AlphaFold's** Evoformer, diffusion Transformers (**DiT**), video models and multimodal LLMs. "Token" is just a vector.
