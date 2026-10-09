## Exercises

**Q1.** Count the parameters of a single-layer LSTM with input size 128 and hidden size 512 (PyTorch convention with two biases).

<details>
<summary>Show answer</summary>

$4\,(nd+n^2+2n)=4\,(512\cdot128+512^2+1024)=4\,(65{,}536+262{,}144+1{,}024)=1{,}314{,}816$. A vanilla RNN of the same size has a quarter of that.

</details>

**Q2.** A memory unit has a constant forget gate $f=0.95$ and input gate 0. After how many steps does the stored value fall to 10%?

<details>
<summary>Show answer</summary>

$0.95^k=0.1\Rightarrow k=\ln0.1/\ln0.95\approx45$ steps. Time constant $\tau=-1/\ln0.95\approx19.5$.

</details>

**Q3.** Why is setting the forget-gate bias to about 1 (or higher) recommended at initialisation?

<details>
<summary>Show answer</summary>

With $b_f=0$ the initial $f\approx0.5$, so memory halves every step and gradients vanish before learning begins. $b_f=1$–$3$ gives $f=0.73$–$0.95$, so information (and gradient) survives long enough for the network to discover that it is useful.

</details>

**Q4.** Design a 1-bit memory: using the gate equations, describe how an LSTM can "write" a bit on a trigger and keep it indefinitely.

<details>
<summary>Show answer</summary>

Make $f\approx1$ when no trigger (bias high), $i\approx0$ (bias low). On the trigger input, a large input weight pushes $i\to1$ and $f\to0$ so the old value is erased and the candidate $\tilde g=\pm1$ (set by another input weight) is written. After the trigger, gates return to their defaults and $c$ persists unchanged. This is essentially what the lab with the second pulse does.

</details>

**Q5.** When would you still choose an LSTM over a Transformer today?

<details>
<summary>Show answer</summary>

Small data; streaming inference with O(1) memory per step; tight latency or memory budgets on-device; short noisy time series where recurrence is a good inductive bias; online learning. For large-scale language/vision pre-training, Transformers (or modern SSMs) win.

</details>

**Q6 (code).** Compare `nn.LSTM` and a Python loop over `LSTMCell` for speed.

<details>
<summary>Show answer</summary>

The fused cuDNN kernel in `nn.LSTM` runs the whole sequence in one launch and is typically 5–20× faster than a Python loop on GPU — see the kernel-launch overhead discussion in the CUDA track.

</details>

## In practice

- **Forecasting**: LSTM encoder over a window → linear head; standardise per series; predict multiple horizons directly rather than recursively when possible.
- **Anomaly detection**: train an LSTM autoencoder / predictor on normal data and flag high reconstruction or prediction error.
- **Speech/handwriting**: bidirectional LSTMs with CTC loss.
- **Text classification**: embeddings → BiLSTM → max/mean pooling → linear; still a strong small-data baseline.
- **Deployment**: fuse to ONNX/TensorRT; int8 quantisation works well for LSTMs but keep the cell state in higher precision.

## Common pitfalls

- Using the last time-step output of a *padded* batch (take the output at the true length or use packed sequences).
- Forgetting to reset hidden state between independent sequences; or resetting it every batch in stateful training.
- Not clipping gradients.
- Initialising the forget bias to zero.
- Applying batch norm across time (use layer norm).
