## Equations

With $\sigma$ the logistic sigmoid and $[h_{t-1};x_t]$ the concatenated input:

$$
\begin{aligned}
f_t&=\sigma\!\big(W_f x_t+U_f h_{t-1}+b_f\big)\\
i_t&=\sigma\!\big(W_i x_t+U_i h_{t-1}+b_i\big)\\
o_t&=\sigma\!\big(W_o x_t+U_o h_{t-1}+b_o\big)\\
\tilde g_t&=\tanh\!\big(W_g x_t+U_g h_{t-1}+b_g\big)\\
c_t&=f_t\odot c_{t-1}+i_t\odot\tilde g_t\\
h_t&=o_t\odot\tanh(c_t)
\end{aligned}
$$

Parameters for hidden size $n$ and input size $d$:

$$
4\,\big(n\,d+n^2+n\big).
$$

(PyTorch stores two bias vectors per gate, giving $4(nd+n^2+2n)$.)

## Gradient flow along the cell state

Differentiating $c_t=f_t\odot c_{t-1}+i_t\odot\tilde g_t$ with respect to $c_{t-1}$ (treating gates as functions of $h_{t-1}$ gives extra terms, which are second-order and small in practice):

$$
\frac{\partial c_t}{\partial c_{t-1}}=\mathrm{diag}(f_t)+\underbrace{\cdots}_{\text{via }h_{t-1}} .
$$

Therefore

$$
\frac{\partial c_T}{\partial c_k}\approx\prod_{t=k+1}^{T}\mathrm{diag}(f_t)=\mathrm{diag}\Big(\prod_t f_t\Big).
$$

If the forget gate for a unit sits at $f=0.99$, the signal after 100 steps is $0.99^{100}\approx0.37$ — versus $\approx10^{-30}$ for a vanilla RNN with factor $0.5$ per step. If $f=0.9$ the memory half-life is $\ln2/\ln(1/0.9)\approx6.6$ steps. **Time-scale of a memory unit** $\tau=-1/\ln f$.

## Backward pass (BPTT through an LSTM)

Given $dh_t$ and $dc_t$ from the future:

$$
\begin{aligned}
do_t&=dh_t\odot\tanh(c_t), & dc_t&\mathrel{+}=dh_t\odot o_t\odot(1-\tanh^2c_t)\\
df_t&=dc_t\odot c_{t-1}, & di_t&=dc_t\odot\tilde g_t,\quad d\tilde g_t=dc_t\odot i_t\\
dc_{t-1}&=dc_t\odot f_t &&
\end{aligned}
$$

and each gate pre-activation gradient is multiplied by its local derivative: $\sigma'(a)=\sigma(a)(1-\sigma(a))$ for $f,i,o$ and $1-\tilde g^2$ for the candidate. The weight gradients are sums over time of $d a_t\,[x_t;h_{t-1}]^{\top}$.

## Information-theoretic view of gates

A gate value $g\in(0,1)$ interpolates "keep" and "replace". For the forget gate, $b_f$ sets the *prior* probability of retention $\sigma(b_f)$: $b_f=1\Rightarrow0.73$, $b_f=3\Rightarrow0.95$, $b_f=5\Rightarrow0.993$.

## Chrono initialisation

To give units a spread of memory time scales up to $T_{max}$ (Tallec & Ollivier, 2018), draw

$$
b_f\sim\log\big(\mathcal U[1,T_{max}-1]\big),\qquad b_i=-b_f .
$$

so $\sigma(b_f)\approx1-1/T$ and the cell begins with "leaky integrator" dynamics across a range of horizons.
