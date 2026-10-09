## The idea: a protected memory with gates

The vanilla RNN rewrites its whole state at every step through a matrix multiply and a squashing non-linearity — a recipe for forgetting. Hochreiter & Schmidhuber (1997) added a separate **cell state** $c_t$ and **gates** that decide what to forget, what to write and what to reveal:

| Gate | Question it answers | Range |
|---|---|---|
| **Forget** $f_t$ | How much of the old memory do I keep? | $(0,1)$ per unit |
| **Input** $i_t$ | How much of the new candidate do I write? | $(0,1)$ |
| **Candidate** $\tilde g_t$ | What would I write? | $(-1,1)$ |
| **Output** $o_t$ | How much of the memory do I expose as $h_t$? | $(0,1)$ |

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde g_t,\qquad h_t=o_t\odot\tanh(c_t)$$

<div class="callout">

**Key idea — the constant error carousel.** The cell state is updated **additively**. When $f\approx1$ and $i\approx0$, $c_t=c_{t-1}$ and the gradient flows back unchanged: $\partial c_t/\partial c_{t-1}=\mathrm{diag}(f_t)$. No repeated weight-matrix products, so no exponential decay.

</div>

## Reading the lab

The top panel is the cell at one time step; line thickness encodes how open each gate is. The bottom panel is the whole sequence:

- A pulse of 1.0 arrives at $t=3$. The **input gate** opens, the candidate $\tilde g\approx\tanh(2)$ is written into $c$.
- With a **high forget bias** the cell state stays flat afterwards — a long-term memory. With a **low** one it decays like the vanilla RNN (dashed).
- With **noise**, an input gate that opens too easily pollutes memory; learned gating must stay selective.

## Why the details work

- **Gates are soft switches** (sigmoid), hence differentiable; the network *learns* when to remember or erase.
- **Forget-bias initialisation** at ≈1 starts the model in "remember" mode, which makes early training dramatically easier.
- **Separate $c$ and $h$**: $c$ is internal memory (unbounded), $h$ is a squashed, gated read-out. This lets the model keep information that is not currently needed.
- Four gates ⇒ **4× the parameters** of a vanilla RNN of the same width.

## Variants

- **Peephole LSTM** — gates also see $c$.
- **Bidirectional LSTM** — forward and backward passes concatenated; the workhorse of pre-Transformer NLP (ELMo, NER, speech).
- **Stacked / residual LSTMs** — depth for capacity; residual links beyond 4–8 layers.
- **Projected LSTM (LSTMP)**, **LayerNorm-LSTM**, **xLSTM** (2024) with exponential gating and matrix memory — a modern revival.

## What LSTMs were used for

Machine translation (GNMT), speech recognition, handwriting, language modelling (AWD-LSTM), time-series forecasting, anomaly detection, reinforcement-learning policies with memory (A3C-LSTM, OpenAI Five). Today Transformers dominate language, but LSTMs remain sensible for **small data, streaming, low-latency and edge** settings, and are still common in forecasting.

## Practical knobs

- Gradient clipping (norm 0.25–5) is still needed — gates fix vanishing, not exploding gradients.
- Dropout between layers, **variational** dropout across time, weight-drop on $U$.
- Use cuDNN fused kernels (`nn.LSTM`) — a hand-written cell loop is 5–10× slower.
- Hidden size 256–1024; layers 1–4; embedding tying for language models.
