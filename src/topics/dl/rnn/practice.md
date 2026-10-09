## Exercises

**Q1.** A vanilla RNN has hidden size 256 and input size 100. How many recurrent-cell parameters does it have? How do they change if the sequence length goes from 50 to 5000?

<details>
<summary>Show answer</summary>

$n^2+nd+n=256^2+256\cdot100+256=65{,}536+25{,}600+256=91{,}392$. They do **not** change with length — only compute and activation memory grow ($O(T)$).

</details>

**Q2.** $W_{hh}$ has spectral radius 0.9 and $\tanh'\approx0.5$ on average. Estimate the gradient magnitude reaching 30 steps back.

<details>
<summary>Show answer</summary>

Per step the factor is $\approx0.9\times0.5=0.45$, so after 30 steps $\approx0.45^{30}\approx4\times10^{-11}$. Even a "stable" $\rho<1$ network cannot assign credit 30 steps away — which is what the gated cells fix.

</details>

**Q3.** Your RNN loss suddenly becomes `NaN` mid-training. Give a three-step diagnosis.

<details>
<summary>Show answer</summary>

1. Log the global gradient norm — look for a spike before the NaN (exploding gradients).
2. Add `clip_grad_norm_(…, 1.0)` (or 5.0) and lower the LR.
3. Check initial $W_{hh}$ (orthogonal init), input scale/normalization, and (for mixed precision) loss scaling.

</details>

**Q4.** Why does truncated BPTT still let the model use information older than the window?

<details>
<summary>Show answer</summary>

The hidden state is carried forward across window boundaries (detached from the graph), so earlier information can influence predictions — it just cannot receive gradient from them. The model can *use* long context but only *learn* dependencies up to $k$ steps.

</details>

**Q5.** Show that for the scalar linear RNN $h_t=wh_{t-1}+x_t$, a perturbation $\varepsilon$ at $t=0$ changes $h_T$ by $w^{T}\varepsilon$. For what $w$ is the memory "perfect"?

<details>
<summary>Show answer</summary>

By induction $h_T=w^Th_0+\sum_{k=1}^{T}w^{T-k}x_k$, so $\partial h_T/\partial h_0=w^T$. Perfect memory needs $|w|=1$ — a knife-edge; learning must keep the dynamics precisely there (or use gates to *switch* retention on and off).

</details>

**Q6 (code).** Verify your BPTT implementation with a finite-difference gradient check.

<details>
<summary>Show answer</summary>

`rnn_bptt.py` in the **Code** tab does exactly that: perturb each parameter by $\pm10^{-5}$, compute $(L_+-L_-)/(2\cdot10^{-5})$, and compare with the analytic gradient; relative error should be $\lesssim10^{-6}$.

</details>

## In practice

- **Where plain RNNs still appear**: tiny on-device models, streaming keyword spotting, low-latency control; most others use GRU/LSTM, 1-D CNNs/TCNs, or Transformers.
- **Packed sequences** (`pack_padded_sequence`) avoid wasted compute on padding and prevent padded steps from contaminating the final state.
- **Stateful inference**: for streaming, keep $h_t$ between chunks; reset it at sequence boundaries.
- **Optimizer**: Adam with gradient clipping; learning-rate ≈ $10^{-3}$.
- **Modern relevance**: understanding the RNN is the prerequisite for state-space models (Mamba), RWKV and linear attention.

## Common pitfalls

- Forgetting to `detach()` the hidden state between truncated windows → memory grows without bound.
- Applying dropout to the recurrent connection with a fresh mask each step (destroys memory).
- Evaluating in `train()` mode.
- Feeding unsorted, unmasked padded batches.
- Using sigmoid/tanh with huge inputs → saturation → zero gradients.
