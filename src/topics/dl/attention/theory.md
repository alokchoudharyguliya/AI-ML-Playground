## The encoder–decoder bottleneck

Sequence-to-sequence (Sutskever et al., 2014; Cho et al., 2014) solved variable-length input → variable-length output (translation, summarisation, speech → text) with two RNNs:

1. an **encoder** reads the source and produces a final state $h_T$;
2. a **decoder** is initialised from $h_T$ and emits the target one token at a time.

Everything the decoder knows about the source must pass through that **one fixed-size vector**. For a 40-word sentence that is hopeless: early words are overwritten, and translation quality collapses as sentences get longer. Toggle *bottleneck* in the lab to see the shape of the problem.

## The attention fix

Bahdanau, Cho & Bengio (2014) let the decoder look back at **all** encoder states $h_1,\dots,h_S$ and build a *fresh* context vector for every output word:

$$c_t=\sum_{i=1}^{S}\alpha_{ti}\,h_i,\qquad\alpha_t=\mathrm{softmax}\big(\mathrm{score}(s_{t-1},h_i)\big)$$

The weights $\alpha_{ti}$ — "how much should target word $t$ care about source word $i$?" — are computed on the fly and **learned end-to-end** with no alignment supervision. The result (the heat-map in the lab) is a soft **word alignment**: “chat” → *cat*, “noir” → *black* even though the order is swapped.

<div class="callout">

**Key idea — attention is a differentiable dictionary lookup.** There is a *query* (what the decoder needs right now), a set of *keys* (what each source position offers) and *values* (the content to retrieve). Similarity of query and key gives weights; the output is the weighted average of the values. In the second lab tab you can literally drag the query and watch the weighted average move.

</div>

## Score functions

| Name | Score $e_{ti}$ | Notes |
|---|---|---|
| Additive (Bahdanau) | $v^{\top}\tanh(W s_{t-1}+Uh_i)$ | small MLP; works at any dimension |
| Dot product (Luong) | $s_t^{\top}h_i$ | cheapest; needs equal dims |
| General (bilinear) | $s_t^{\top}Wh_i$ | learned similarity |
| **Scaled dot product** | $s_t^{\top}h_i/\sqrt{d}$ | the Transformer's choice |

Dot-product variants map onto a matrix multiplication — hugely GPU-friendly — which is a quiet reason they won.

## Design variations

- **Global vs local attention** (Luong 2015): attend to all source positions, or only a window around a predicted alignment point.
- **Input feeding**: feed the previous attentional vector back into the decoder input so it knows what it looked at.
- **Coverage** penalties discourage repeating or ignoring source words.
- **Pointer / copy networks**: attention weights directly define a distribution over *copying* source tokens (names, numbers, rare words).
- **Attention over images** (Show, Attend and Tell): attend to CNN grid cells while captioning; **over audio** (Listen, Attend and Spell).
- **Hard attention** samples a single position (needs REINFORCE); **soft attention** is differentiable and standard.

## What attention gave us

1. **No bottleneck** — long sentences work.
2. **Shorter gradient paths** — the loss at target step $t$ reaches source step $i$ in one hop instead of $|t-i|$ recurrent steps (a cousin of the LSTM gating solution to vanishing gradients).
3. **Interpretability (partial)** — alignments are visible, though attention weights are not guaranteed faithful explanations.
4. **A building block** — if attention is so good at *routing information*, why keep the recurrence? The answer, *Attention Is All You Need* (2017), is the next chapter.

## Decoding and training notes

- **Teacher forcing** at train time (ground-truth previous token), **autoregressive generation** at inference — mismatch known as *exposure bias*.
- **Beam search** keeps the $k$ best partial hypotheses; length-normalise scores to avoid favouring short outputs.
- **Padding masks** in the attention softmax are mandatory (set padded scores to $-\infty$).
- **Label smoothing** and **dropout** are standard regularisers; BLEU/chrF/COMET evaluate translation quality.
- **Complexity**: attention adds $O(S\cdot T)$ memory/compute per sentence pair — negligible for sentences, but it foreshadows the quadratic cost of self-attention on long contexts.
