## Why we need recurrence

Text, speech, sensor streams, time series, DNA — data whose **length varies** and whose **order matters**. A CNN sees a fixed window; an MLP needs a fixed input size. A recurrent network processes one element at a time and carries a **hidden state** $h_t$ — a learned summary of everything seen so far:

$$h_t=\phi(W_{hh}h_{t-1}+W_{xh}x_t+b)$$

The same weights are reused at every step (weight sharing across **time**, the sibling of weight sharing across space in CNNs), so the model handles any length with a constant number of parameters.

## Unrolling and the computation graph

"Unroll" the loop and an RNN becomes a very deep feed-forward network with $T$ layers that all share parameters. The lab's top strip is exactly that: each column is $h_t$ for one time step. Training with **backpropagation through time (BPTT)** is ordinary backprop on this unrolled graph, with gradients for the shared weights summed over all steps.

Common input/output patterns:

| Pattern | Example | How |
|---|---|---|
| many → one | sentiment classification | use last $h_T$ (or pooled states) |
| one → many | image captioning | feed image encoding once, generate tokens |
| many → many (aligned) | tagging, speech frames | output at every step |
| many → many (seq2seq) | translation | encoder RNN → decoder RNN (next chapters) |

Variants: **stacked** RNNs (the hidden sequence of layer $\ell$ is the input to $\ell+1$) and **bidirectional** RNNs (a forward and a backward pass, states concatenated; only for non-causal tasks).

## The long-range problem

Credit assignment from step $T$ back to step $k$ passes through $T-k$ Jacobians:

$$\frac{\partial h_T}{\partial h_k}=\prod_{t=k+1}^{T}\mathrm{diag}\big(\phi'(a_t)\big)\,W_{hh}$$

Multiply many matrices and the norm shrinks or grows **exponentially**. Open the lab: with spectral radius $\rho<1$ the backward bars decay like $\rho^{T-t}$; with $\rho>1$ they explode. This is the **vanishing / exploding gradient problem** (Hochreiter 1991, Bengio 1994).

- **Exploding** → loss spikes, NaNs. Cure: **gradient clipping** (rescale if $\|g\|>c$).
- **Vanishing** → the net cannot learn dependencies beyond ~10–20 steps. Cures: **gating** (LSTM, GRU — next two chapters), orthogonal/identity initialisation, skip connections, or abandoning recurrence for attention.

<div class="callout warn">

**Both views matter.** The forward view (second lab tab) shows the same phenomenon: with $\rho<1$ information about early inputs is forgotten; with $\rho>1$ the state is chaotic. Useful memory needs dynamics near the **edge of stability** — which gated cells learn to maintain.

</div>

## Training details that matter

- **Teacher forcing**: during training feed the *true* previous token, not the model's own prediction — fast and stable but creates *exposure bias* at inference. Scheduled sampling and sequence-level losses mitigate it.
- **Truncated BPTT**: backprop only through the last $k$ steps (e.g. 128) while still passing the hidden state forward. Bounds memory; limits learnable dependency length.
- **Initialisation**: orthogonal $W_{hh}$ and (for ReLU RNNs) identity initialisation keep $\rho\approx1$ at the start.
- **Regularisation**: dropout only on non-recurrent connections or with *variational* (same mask each step) dropout; layer-norm RNNs; zoneout.
- **Batching variable-length sequences**: sort/bucket by length, use packed sequences and masks.

## Limitations — and why recurrence is back

1. **Sequential compute**: $h_t$ needs $h_{t-1}$, so training can't be parallelised over time — poor GPU utilisation versus Transformers.
2. **Memory bottleneck**: everything must squeeze into a fixed-size $h_t$.
3. **Hard long-range credit assignment** even with gates.

Yet recurrence has returned in a new form: **state-space models** (S4, Mamba), RWKV, RetNet and linear-attention variants have *linear* (or parallel-scan) recurrences that train in parallel and run inference in $O(1)$ memory per token — attractive for very long contexts. Understanding the RNN is understanding the foundation of these models.
