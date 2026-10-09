## Forward equations

For input $x_t\in\mathbb R^{d}$ and hidden size $n$:

$$
a_t=W_{hh}h_{t-1}+W_{xh}x_t+b,\qquad h_t=\tanh(a_t),\qquad \hat y_t=\mathrm{softmax}(W_{hy}h_t+c)
$$

Parameters: $n^2+nd+n$ for the recurrent cell (plus the output layer). They do **not** depend on $T$.

Loss over a sequence: $L=\sum_t \ell_t(\hat y_t,y_t)$.

## Backpropagation through time

Let $\delta_t=\partial L/\partial a_t$. Working backwards from $t=T$:

$$
\delta_t=\Big(\mathrm{diag}\big(1-h_t^2\big)\Big)\Big(W_{hy}^{\!\top}(\hat y_t-y_t)+W_{hh}^{\!\top}\,\delta_{t+1}\Big)
$$

Because the parameters are shared, their gradients sum over time:

$$
\frac{\partial L}{\partial W_{hh}}=\sum_{t=1}^{T}\delta_t\,h_{t-1}^{\!\top},\qquad
\frac{\partial L}{\partial W_{xh}}=\sum_{t=1}^{T}\delta_t\,x_t^{\!\top},\qquad
\frac{\partial L}{\partial b}=\sum_{t}\delta_t .
$$

## Why gradients vanish or explode

The influence of step $k$ on step $T$ is

$$
\frac{\partial h_T}{\partial h_k}=\prod_{t=k+1}^{T}D_t\,W_{hh},\qquad D_t=\mathrm{diag}\big(1-h_t^2\big),\ \ \|D_t\|\le\gamma=1 .
$$

Taking norms,

$$
\Big\|\frac{\partial h_T}{\partial h_k}\Big\|\le\big(\gamma\,\sigma_{max}(W_{hh})\big)^{T-k}.
$$

- If $\gamma\,\sigma_{max}<1$ the gradient **vanishes** exponentially — a *sufficient* condition for vanishing.
- If the spectral radius $\rho(W_{hh})>1/\gamma$ the product **can** explode (*necessary* condition for explosion; Pascanu et al., 2013).

For an orthogonal matrix scaled by $\rho$ (what the lab uses) every singular value equals $\rho$, so the norm is *exactly* $\rho^{T-k}$ in the linear regime; $\tanh'$ makes it smaller still.

For the **linear** RNN $h_t=Wh_{t-1}$, $h_T=W^{T}h_0$; with eigendecomposition $W=V\Lambda V^{-1}$ the state is $\sum_i\lambda_i^{T}c_i v_i$ — eigenvalues $|\lambda|<1$ decay, $|\lambda|>1$ blow up, and only $|\lambda|\approx1$ preserve information.

## Gradient clipping

$$
g\leftarrow\begin{cases}g&\|g\|\le c\\ c\,g/\|g\|&\text{otherwise}\end{cases}
$$

Direction is preserved, magnitude capped; this is precisely the "clip" toggle in the lab.

## Truncated BPTT

Split the sequence into windows of length $k$. For window $w$ set $h_{\text{start}}=\mathrm{stopgrad}(h_{\text{end of }w-1})$. Memory is $O(k)$; gradients cross at most $k$ steps.

## Linear recurrences are parallelisable (link to SSMs)

The linear recurrence $h_t=a_th_{t-1}+b_t$ composes **associatively**:

$$
(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\;a_2b_1+b_2),
$$

so all $h_t$ can be computed in $O(\log T)$ parallel depth with a prefix scan. Removing the nonlinearity from the *recurrence* (while keeping it elsewhere) is the trick behind S4/Mamba/linear attention — and the reason they train as fast as Transformers while inferring like RNNs.
