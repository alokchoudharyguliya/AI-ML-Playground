## Exercises

**Q1.** For $f(x,y)=\tfrac12(a x^2+b y^2)$ with $a=100$, $b=1$, what is the largest stable plain-GD learning rate, and how many iterations does GD need (order of magnitude) to shrink the error by $10^{-6}$ with the optimal $\eta$?

<details>
<summary>Show answer</summary>

$\lambda_{max}=100$ so $\eta<2/100=0.02$. With $\kappa=100$ and $\eta^\star=2/101$, $\rho=99/101\approx0.980$; iterations $\approx\ln(10^{6})/(-\ln\rho)\approx13.8/0.0200\approx690$. With momentum the rate is $\tfrac{\sqrt{100}-1}{\sqrt{100}+1}=0.82$, i.e. $\approx70$ iterations — a ~10× speed-up ($\sqrt\kappa$).

</details>

**Q2.** You use SGD with momentum $\beta=0.9$ and $\eta=0.01$. What is the effective step size on a constant gradient, and what happens if you raise $\beta$ to $0.99$ without changing $\eta$?

<details>
<summary>Show answer</summary>

Effective LR $=\eta/(1-\beta)=0.1$. With $\beta=0.99$ it becomes $1.0$ — a **10× larger effective step**, which will likely diverge. If you tune $\beta$, rescale $\eta$ so $\eta/(1-\beta)$ stays comparable.

</details>

**Q3.** The first Adam update uses bias-corrected moments. What is its size per parameter, and why does this matter for the first few iterations?

<details>
<summary>Show answer</summary>

$\hat m_1/\sqrt{\hat v_1}=g/|g|=\pm1$, so every parameter moves by exactly $\eta$ regardless of gradient scale. Early updates are therefore *large and sign-like*, which is a reason Transformers need **warmup** — a full-size first step on noisy, uncalibrated moments can derail training.

</details>

**Q4.** You scale pre-training from batch size 256 to 2048 on 8× more GPUs. How do you adjust the hyper-parameters?

<details>
<summary>Show answer</summary>

With SGD: multiply LR by 8 (linear rule) and lengthen warmup (5 epochs in the Goyal et al. ImageNet recipe). With AdamW: start from $\sqrt8\approx2.8\times$ and sweep, keep weight decay, and check for loss spikes. Verify against the **critical batch size** — beyond it you burn compute without fewer steps.

</details>

**Q5.** Why is weight decay applied differently in AdamW, and which parameters should usually be excluded?

<details>
<summary>Show answer</summary>

Decay is *decoupled* from the adaptive scaling, so every weight shrinks at the same relative rate. Exclude **biases and normalization gains/shifts** (and often embeddings) — decaying them hurts and has little regularization value.

</details>

**Q6 (code).** Implement an *LR range test*: ramp the LR exponentially from $10^{-7}$ to $1$ over 200 steps, record the loss, and choose the LR.

<details>
<summary>Show answer</summary>

Train for 200 steps with `lr = 1e-7 * (1e7) ** (step/200)`; plot loss vs LR on a log axis. Pick an LR about **10× below the point where the loss is lowest** (before it blows up), or the steepest-descent region. This takes a minute and often saves hours of sweeps.

</details>

## In practice

**Defaults that rarely fail.** AdamW, peak LR $3\times10^{-4}$ (small models) down to $10^{-5}$–$10^{-4}$ (large models), $\beta=(0.9,0.95)$, weight decay 0.1, linear warmup for ~1% of steps, cosine decay to 10% of peak, gradient clip at 1.0, BF16.

**Fine-tuning.** Use 10–100× smaller LRs than pre-training; LoRA adapters tolerate larger ones ($10^{-4}$–$10^{-3}$).

**Debug order when loss is bad:** (1) overfit one batch, (2) LR range test, (3) check data/labels/normalization, (4) add clipping, (5) then change optimizers.

**Monitoring.** Log grad-norm, update-to-weight ratio ($\|\Delta\theta\|/\|\theta\|\approx10^{-3}$ is healthy), LR and loss per step; spikes tell you more than epoch averages.

**Memory.** Adam keeps two extra fp32 states per parameter (8 bytes) — for a 7B model that is ~56 GB on top of weights and gradients; this is why ZeRO/FSDP shard optimizer state and 8-bit optimizers exist.

## Common pitfalls

- Forgetting to zero gradients (or accumulating them unintentionally).
- Applying the schedule per-epoch when it was designed per-step.
- Decaying biases/LayerNorm parameters.
- Changing batch size without touching LR/warmup.
- Comparing optimizers at a single LR — each needs its own sweep.
- Using FP16 without loss scaling (silent underflow of gradients).
