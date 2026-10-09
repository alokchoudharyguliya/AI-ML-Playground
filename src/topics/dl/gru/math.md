## Full equations

$$
\begin{aligned}
z_t&=\sigma\big(W_zx_t+U_zh_{t-1}+b_z\big)\\
r_t&=\sigma\big(W_rx_t+U_rh_{t-1}+b_r\big)\\
\tilde h_t&=\tanh\big(W_hx_t+U_h\,(r_t\odot h_{t-1})+b_h\big)\\
h_t&=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t
\end{aligned}
$$

(Some implementations — including cuDNN/PyTorch — compute the candidate as $\tanh(W_hx_t+b_{ih}+r_t\odot(U_hh_{t-1}+b_{hh}))$, applying the reset *after* the matrix multiply. It is cheaper because $U_hh_{t-1}$ can be batched with the other gates.)

**Convention warning.** PyTorch defines the interpolation as $h_t=(1-z_t)\odot\tilde h_t+z_t\odot h_{t-1}$, i.e. its $z$ is $1-z$ of the original paper. Keep this in mind when porting weights or reading gate plots.

Parameters (width $n$, input $d$): $\;3\,(nd+n^2+n)$.

## Gradient through time

Differentiating the interpolation with respect to the previous state:

$$
\frac{\partial h_t}{\partial h_{t-1}}=\mathrm{diag}(1-z_t)+\underbrace{\mathrm{diag}(z_t)\,\frac{\partial\tilde h_t}{\partial h_{t-1}}+\mathrm{diag}\big(\tilde h_t-h_{t-1}\big)\frac{\partial z_t}{\partial h_{t-1}}}_{\text{paths through the gates and candidate}} .
$$

For units with $z_t\approx0$ the Jacobian is $\approx I$, so the product over many steps stays $\approx I$ instead of decaying as $\rho^k$. Quantitatively, if a unit sits at $z=0.02$ for $k$ steps the retained signal is $(1-z)^k=0.98^k$: after 100 steps ≈ 13%.

## Memory time scale

With constant $z$ and no input the state decays toward the candidate with time constant

$$
\tau=-\frac{1}{\ln(1-z)}\;\approx\;\frac1z\quad(z\ll1).
$$

Initialising $b_z\sim-\log\mathcal U[1,T_{max}]$ produces a spectrum of time scales between 1 and $T_{max}$ steps (the same chrono-initialisation idea used for LSTMs).

## Relation to the LSTM

Set the LSTM's input gate to $i_t=1-f_t$ (coupled input-forget gate, CIFG), drop the output gate and peepholes, and merge $c_t$ with $h_t$: you recover the GRU update $h_t=(1-z_t)h_{t-1}+z_t\tilde h_t$. Greff et al. showed the CIFG variant performs on par with the full LSTM, which explains why the GRU is competitive.

## Parameter-count table (single layer, $d=n$)

| Cell | Formula | $n=512$ |
|---|---|---|
| Vanilla RNN | $n(n+d)+n$ | 524,800 |
| GRU | $3\,(n(n+d)+n)$ | 1,574,400 |
| LSTM | $4\,(n(n+d)+n)$ | 2,099,200 |

## FLOPs per step

Dominated by the matrix–vector products: $2\cdot g\cdot n(n+d)$ FLOPs with $g=1,3,4$ gate blocks. For training on GPUs, all gate matmuls are fused into a single GEMM of shape $(B\times(n+d))\cdot((n+d)\times gn)$ — wide enough to use tensor cores efficiently when $B\cdot gn$ is large.
