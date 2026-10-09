## Exercises

**Q1.** A GRU unit has a constant update gate $z=0.05$. What fraction of the old state survives 20 steps (no new input), and what is the time constant?

<details>
<summary>Show answer</summary>

$(1-z)^{20}=0.95^{20}\approx0.36$. $\tau=-1/\ln0.95\approx19.5$ steps.

</details>

**Q2.** Compute the parameters of a single-layer GRU with input 300, hidden 256 using the PyTorch convention (two bias vectors).

<details>
<summary>Show answer</summary>

$3\,(nd+n^2+2n)=3\,(256\cdot300+256^2+512)=3\,(76{,}800+65{,}536+512)=428{,}544$.

</details>

**Q3.** Show that if $r_t=1$ and $z_t=1$ for all $t$ the GRU reduces to a vanilla RNN.

<details>
<summary>Show answer</summary>

Then $h_t=\tilde h_t=\tanh(W_hx_t+U_hh_{t-1}+b_h)$ — exactly the Elman RNN.

</details>

**Q4.** Which gate would you expect to be near 0 at the start of a new sentence when modelling text, and why?

<details>
<summary>Show answer</summary>

The **reset gate**: the candidate state should ignore the previous sentence, so $r\to0$ lets $\tilde h$ depend only on the new token. (The update gate remains free to carry slowly varying context such as topic or speaker.)

</details>

**Q5.** In the lab, why does the vanilla RNN (dashed) forget the pulse while the GRU with $b_z=-4$ does not?

<details>
<summary>Show answer</summary>

The RNN rewrites its state every step with factor $\tanh$ and recurrence weight 0.9 — a contraction (<1) that decays the pulse exponentially. The GRU with a very negative $b_z$ has $z\approx0.02$, so $h_t\approx h_{t-1}$: the identity path preserves it.

</details>

**Q6 (code).** Show that a GRU's per-step cost is $\approx75\%$ of an LSTM's. Measure it.

<details>
<summary>Show answer</summary>

`compare_cells.py` times `nn.RNN`, `nn.GRU`, `nn.LSTM` for forward+backward. Expect roughly RNN : GRU : LSTM ≈ 1 : 3 : 4 in FLOPs, though measured wall-clock ratios are compressed by memory-bound pointwise ops and kernel-launch overhead.

</details>

## In practice

- **Default choice for small/medium sequence tasks**: a 1–2 layer (Bi)GRU with hidden 128–512 is a strong, cheap baseline.
- **Mobile / embedded**: fewer parameters and no extra cell state mean a smaller memory footprint — popular for keyword spotting and speech enhancement.
- **Quantisation**: weights int8; keep gate pre-activations in fp16/fp32 to avoid saturating the sigmoids.
- **Debugging**: log the mean and histogram of $z$ and $r$ per layer — collapsed gates (all ≈0 or ≈1) indicate a problem.

## Common pitfalls

- Mixing PyTorch's GRU gate order (`r, z, n`) with other implementations when porting weights.
- Using the final hidden state of padded batches.
- Assuming the GRU is "worse" for long dependencies without tuning biases/regularisation first.
- Forgetting to detach states across truncated windows.
