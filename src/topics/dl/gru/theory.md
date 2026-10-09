## A leaner gated cell

The Gated Recurrent Unit (Cho et al., 2014) keeps the key insight of the LSTM — **additive, gated state updates give gradients a highway** — but trims the machinery:

- **one** state vector $h_t$ (no separate cell state),
- **two** gates instead of three,
- **no output gate** (the whole state is exposed).

$$
\begin{aligned}
z_t&=\sigma(W_zx_t+U_zh_{t-1}+b_z) &&\text{update gate}\\
r_t&=\sigma(W_rx_t+U_rh_{t-1}+b_r) &&\text{reset gate}\\
\tilde h_t&=\tanh\!\big(W_hx_t+U_h(r_t\odot h_{t-1})+b_h\big) &&\text{candidate}\\
h_t&=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t &&\text{interpolation}
\end{aligned}
$$

<div class="callout">

**Key idea.** $h_t$ is a *convex combination* of the old state and a new candidate, with the mixing weight $z_t$ learned per unit. Keep ($z\to0$) = long-term memory; overwrite ($z\to1$) = fast adaptation. A single gate does the job of both the LSTM's forget and input gates (they are *coupled*: what you write replaces what you drop).

</div>

## What each gate means

**Update gate $z$** — "how much new information?" Low values copy the state forward and let gradients pass unchanged. Units with persistently low $z$ become long-term memory; units with high $z$ track fast-changing features. Different units learn **different time scales**.

**Reset gate $r$** — "how much of the past matters for the *next candidate*?" With $r\to0$ the candidate ignores history and computes purely from $x_t$ (useful at sentence/segment boundaries). The reset gate acts *inside* the candidate only; it never erases $h_{t-1}$ itself — only the update gate decides that.

Play with the lab: the cyan state sits on the line between the grey old state and the violet candidate, at fraction $z$.

## GRU vs LSTM

| | LSTM | GRU |
|---|---|---|
| States | $h_t$, $c_t$ | $h_t$ |
| Gates | forget, input, output | update, reset |
| Parameters (width $n$) | $4(\cdot)$ | $3(\cdot)$ — 25% fewer |
| Output exposure | gated by $o_t$ | full state |
| Typical outcome | slightly better with lots of data / very long dependencies | equal or better on smaller data, faster |

Large empirical studies (Chung 2014, Jozefowicz 2015, Greff 2017) found **no consistent winner**: tune forget/update biases and regularisation first; pick GRU for speed and simplicity, LSTM when you have headroom.

## Why gating works (shared with LSTM)

Gradients through $h_t=(1-z_t)h_{t-1}+\dots$ contain a term $\mathrm{diag}(1-z_t)$ — an element-wise scaling, not a matrix product. When the network drives $z\approx0$ on a unit, the gradient to earlier steps stays near 1. This is the same *constant error carousel* mechanism as in the LSTM, and the same one resurfacing in residual connections: $y=x+F(x)$ is a GRU/LSTM-style skip in depth.

## Practical guidance

- **Initialisation**: bias the update gate **negative** (e.g. −1…−2) so $z$ starts low and memory is preserved, the GRU analogue of the LSTM forget-bias trick.
- **Speed**: cuDNN GRU kernels are fused; ~25% fewer FLOPs/params than LSTM.
- **Stacking and bidirectionality** work identically to LSTMs.
- **Regularisation**: variational dropout on inputs/hidden, weight decay, layer norm inside the cell for long sequences.
- **Where used**: speech (RNN-T encoders), on-device keyword spotting, time series, music generation, early neural machine translation (the original Bahdanau attention model used a GRU).
