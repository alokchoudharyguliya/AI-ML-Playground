## Gradient descent on a quadratic

For $L(\theta)=\tfrac12\theta^\top H\theta$ with eigenvalues $\lambda_i$ of $H$, gradient descent $\theta\leftarrow\theta-\eta H\theta$ acts independently on each eigen-direction:

$$
\theta_i^{(t+1)}=(1-\eta\lambda_i)\,\theta_i^{(t)} .
$$

Stable iff $|1-\eta\lambda_i|<1$ for all $i$, i.e.

$$
\eta<\frac{2}{\lambda_{max}} .
$$

With the best step $\eta^\star=2/(\lambda_{max}+\lambda_{min})$ the contraction factor per step is

$$
\rho=\frac{\kappa-1}{\kappa+1},\qquad\kappa=\frac{\lambda_{max}}{\lambda_{min}} ,
$$

so reaching accuracy $\varepsilon$ needs $\approx\tfrac{\kappa}{2}\ln(1/\varepsilon)$ iterations.

## Momentum (heavy ball)

$$
v_{t+1}=\beta v_t+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_{t+1} .
$$

In the steady state with a constant gradient, $v=g/(1-\beta)$: the *effective* learning rate is $\eta/(1-\beta)$ (10× for $\beta=0.9$). With optimal $(\eta,\beta)$ the rate becomes

$$
\rho_{mom}=\frac{\sqrt\kappa-1}{\sqrt\kappa+1},
$$

needing $O(\sqrt\kappa)$ iterations — a quadratic improvement. **Nesterov**:

$$
v_{t+1}=\beta v_t+\nabla L(\theta_t-\eta\beta v_t),\qquad\theta_{t+1}=\theta_t-\eta v_{t+1}.
$$

## RMSProp

$$
s_t=\beta\, s_{t-1}+(1-\beta)\,g_t^2,\qquad
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{s_t}+\epsilon}\,g_t
$$

(all operations element-wise). Each coordinate's step is normalised to $\approx\eta$ regardless of gradient scale.

## Adam

$$
\begin{aligned}
m_t&=\beta_1 m_{t-1}+(1-\beta_1)g_t, &\qquad v_t&=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,\\
\hat m_t&=\frac{m_t}{1-\beta_1^t}, &\qquad \hat v_t&=\frac{v_t}{1-\beta_2^t},\\
\theta_{t+1}&=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}. &&
\end{aligned}
$$

**Why bias correction?** With $m_0=0$, $\mathbb E[m_t]=(1-\beta_1^t)\,\mathbb E[g]$ when gradients are stationary, so dividing by $1-\beta_1^t$ restores an unbiased estimate. At $t=1$: $\hat m_1=g_1$, $\hat v_1=g_1^2$, so the first step is $\eta\,g_1/|g_1|=\eta\,\mathrm{sign}(g_1)$.

Defaults: $\beta_1=0.9$, $\beta_2=0.999$ (LLMs often $0.95$), $\epsilon=10^{-8}$.

## L2 regularization vs decoupled weight decay

Adding $\tfrac\lambda2\|\theta\|^2$ to the loss adds $\lambda\theta$ to $g_t$. In Adam that term is then divided by $\sqrt{\hat v_t}$, so parameters with large gradient history are decayed *less*. **AdamW** applies decay outside the adaptive rescaling:

$$
\theta_{t+1}=\theta_t-\eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\,\theta_t\right).
$$

## SGD noise and the scaling rules

With batch size $B$, the gradient estimate has covariance $\Sigma/B$. The SGD "temperature" scales like $\eta/B$ (in the SDE view, noise scale $g\approx\eta N/B$). Keeping $\eta/B$ constant — the **linear scaling rule** — preserves the dynamics for moderate $B$. For Adam-style methods a square-root rule, $\eta\propto\sqrt B$, is often better.

## Schedules

With peak rate $\eta_{max}$, total steps $T$ and warmup $T_w$:

$$
\eta_t=\begin{cases}
\eta_{max}\,\dfrac{t}{T_w} & t<T_w\\[2mm]
\eta_{min}+\tfrac12(\eta_{max}-\eta_{min})\left(1+\cos\dfrac{\pi\,(t-T_w)}{T-T_w}\right) & t\ge T_w
\end{cases}
$$

**Inverse-sqrt (Noam):** $\eta_t=\eta_{max}\min\!\left(\dfrac{t}{T_w},\sqrt{\dfrac{T_w}{t}}\right)$.

## Gradient clipping

Global-norm clipping rescales when the norm exceeds $c$:

$$
g\leftarrow g\cdot\min\!\left(1,\frac{c}{\|g\|_2}\right).
$$

It preserves the gradient *direction* and bounds the step, which tames exploding gradients and loss spikes.
