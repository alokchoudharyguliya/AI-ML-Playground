## Encoder

A bidirectional RNN produces annotations $h_i=[\overrightarrow h_i;\overleftarrow h_i]$ for source tokens $x_1,\dots,x_S$.

## Attention step (Bahdanau)

For decoder step $t$ with previous state $s_{t-1}$:

$$
e_{ti}=v^{\top}\tanh\!\big(W_a s_{t-1}+U_a h_i\big),\qquad
\alpha_{ti}=\frac{\exp(e_{ti})}{\sum_{k=1}^{S}\exp(e_{tk})},\qquad
c_t=\sum_{i=1}^{S}\alpha_{ti}\,h_i .
$$

The decoder then updates and predicts:

$$
s_t=f\big(s_{t-1},\,y_{t-1},\,c_t\big),\qquad
p(y_t\mid y_{<t},x)=\mathrm{softmax}\big(W_o[\,s_t;\,c_t;\,E y_{t-1}]\big).
$$

## Luong variants

$$
\mathrm{score}(s_t,h_i)=\begin{cases}
s_t^{\top}h_i & \text{dot}\\
s_t^{\top}W_ah_i & \text{general}\\
v^{\top}\tanh(W_a[s_t;h_i]) & \text{concat}
\end{cases}
\qquad
\tilde s_t=\tanh\!\big(W_c[c_t;s_t]\big).
$$

Luong attention uses the *current* decoder state $s_t$ (compute attention after the RNN step); Bahdanau uses $s_{t-1}$ (before it).

## Matrix form

Stack $H\in\mathbb R^{S\times d}$ and queries $Q\in\mathbb R^{T\times d}$:

$$
\alpha=\mathrm{softmax}\!\Big(\frac{QH^{\top}}{\sqrt d}\Big)\in\mathbb R^{T\times S},\qquad C=\alpha H\in\mathbb R^{T\times d}.
$$

Cost: $O(TSd)$ time and $O(TS)$ memory for the weights.

## Masking

With padding mask $m_i\in\{0,1\}$:

$$
\alpha_{ti}=\frac{m_i\,\exp(e_{ti})}{\sum_k m_k\exp(e_{tk})}\;=\;\mathrm{softmax}(e_{ti}-\infty\cdot(1-m_i)).
$$

## Softmax derivative

For $\alpha=\mathrm{softmax}(e)$,

$$
\frac{\partial\alpha_i}{\partial e_j}=\alpha_i(\delta_{ij}-\alpha_j),
$$

so the gradient flowing into the scores is $\partial L/\partial e_j=\alpha_j\big(g_j-\sum_i\alpha_ig_i\big)$ with $g_i=\partial L/\partial\alpha_i$. When $\alpha$ is nearly one-hot the Jacobian vanishes — the saturation problem that motivates the $1/\sqrt d$ scaling (see *The Transformer*).

## Temperature view

Scaling scores by $1/T$ interpolates between a uniform average ($T\to\infty$) and hard argmax ($T\to0$):

$$
\alpha_i(T)=\frac{e^{e_i/T}}{\sum_ke^{e_k/T}} .
$$

The lab's entropy readout is $H(\alpha)=-\sum_i\alpha_i\log_2\alpha_i$, between $0$ (one-hot) and $\log_2S$ (uniform).

## Beam search score

Length-normalised log-probability with penalty $\lambda$ (GNMT):

$$
s(Y)=\frac{\log p(Y\mid X)}{lp(Y)},\qquad lp(Y)=\frac{(5+|Y|)^{\lambda}}{(5+1)^{\lambda}} .
$$

## Training objective

Teacher-forced maximum likelihood:

$$
\mathcal L=-\sum_{t=1}^{T}\log p\big(y_t^{*}\mid y^{*}_{<t},x\big),
$$

optionally with label smoothing $\epsilon$: replace the one-hot target by $(1-\epsilon)\,\mathbb 1_{y^*}+\epsilon/V$.
