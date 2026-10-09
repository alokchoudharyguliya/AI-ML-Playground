## Training is geometry

A network defines a loss surface $L(\theta)$ over millions or billions of parameters. Training is a walk across it. The 3D lab above shows 2-parameter toys, but the intuition transfers: the shape of the surface — long narrow valleys, flat plateaus, saddles, multiple basins — decides which optimizer behaves well.

<div class="callout">

**Key idea.** In high dimensions, bad *local minima* are rare. The real obstacles are **saddle points**, **ill-conditioning** (steep in some directions, flat in others) and **noise**. Every optimizer below is a different answer to one of those.

</div>

## Stochastic gradient descent

SGD estimates the gradient on a mini-batch: $\theta\leftarrow\theta-\eta\,\hat g$. The estimate is noisy, which is both a cost (jitter) and a feature (it helps escape sharp basins and acts as implicit regularization). Batch size trades gradient noise for hardware efficiency; the **learning rate $\eta$ is the single most important hyper-parameter** — too small crawls, too large diverges, and the usable range is bounded by the sharpest curvature direction ($\eta<2/\lambda_{max}$).

**The ravine problem.** On the ravine surface the gradient is huge across the valley and tiny along it. SGD must use a small $\eta$ to stay stable across, so it crawls along — the classic ill-conditioning picture. The ratio of largest to smallest curvature is the **condition number** $\kappa$.

## Momentum and Nesterov

Momentum keeps a velocity $v\leftarrow\beta v+g$ and steps along it. Oscillating components cancel; consistent components accumulate, giving up to $1/(1-\beta)$ times the step along the valley. Convergence on quadratics improves from $O(\kappa)$ to $O(\sqrt\kappa)$ iterations. **Nesterov** momentum evaluates the gradient at the look-ahead point, damping overshoot slightly.

## Adaptive methods

Rather than one global step size, give each parameter its own, scaled by recent gradient magnitude:

- **AdaGrad** divides by the root of the *sum* of squared gradients — great for sparse features, but the step eventually shrinks to zero.
- **RMSProp** uses an exponential *moving average* instead, fixing the decay.
- **Adam** = momentum on the first moment + RMSProp on the second moment + **bias correction** for the zero-initialised averages. Steps are roughly $\eta\cdot\text{sign}$-like, so $\eta\approx10^{-3}$–$10^{-4}$ works across wildly different gradient scales.
- **AdamW** *decouples* weight decay from the adaptive step. It is the default for Transformers and most modern training.

Rules of thumb: **SGD + momentum** (often with large LR and long schedules) is still strong for CNNs and generalizes well; **AdamW** is the safe default for Transformers, embeddings and anything with heterogeneous gradient scales. Newer optimizers (Lion, Sophia, Shampoo, Muon) trade memory/compute for faster convergence and see use in large-scale training.

## Learning-rate schedules and warmup

A constant LR rarely works best. Typical recipes:

| Schedule | Used for |
|---|---|
| Step decay (×0.1 at milestones) | Classic ImageNet ResNet training |
| **Cosine decay** | Default for vision and LLM pre-training |
| **Linear warmup + cosine/linear decay** | Transformers, LLMs |
| Inverse-sqrt ("Noam") | Original Transformer |
| One-cycle / super-convergence | Fast CNN training with large peak LR |
| WSD (warmup–stable–decay) | LLM pre-training where the final length is unknown |

**Warmup** matters because Adam's variance estimates are unreliable at step 1 and Transformers are fragile early on; ramping LR from ~0 over the first 0.5–5% of steps avoids early divergence.

## Batch size, noise and scaling

Gradient noise scale $\propto \eta/B$. When you multiply the batch size by $k$, multiply the LR by $k$ (SGD, the *linear scaling rule*) or by $\sqrt k$ (Adam-style) and add warmup. Beyond the **critical batch size** larger batches stop reducing the number of steps you need.

## Practical stabilizers

- **Gradient clipping** (global norm $\le1$) is nearly universal in sequence models and LLMs.
- **Mixed precision** needs loss scaling (FP16) or uses BF16, which does not.
- **Gradient accumulation** simulates large batches under memory limits.
- **Weight decay** ($0.01$–$0.1$ with AdamW) and **EMA of weights** improve generalization.
- **Loss spikes** in huge runs are mitigated by lower $\beta_2$ (0.95), QK-norm, z-loss and skipping bad batches.

## Flat vs sharp minima

Solutions found by small-batch SGD tend to be *flatter* (low curvature) and generalize better; sharpness-aware minimization (**SAM**) optimizes for this explicitly. The idea is why "train longer with noise" often beats "converge fast".
