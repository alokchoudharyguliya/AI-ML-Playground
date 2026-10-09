## The operation

Deep-learning "convolution" is really **cross-correlation** (no kernel flip). For input $x\in\mathbb R^{C_{in}\times H\times W}$, kernel $w\in\mathbb R^{C_{out}\times C_{in}\times k\times k}$ and bias $b$:

$$
y_{o,i,j}=b_o+\sum_{c=1}^{C_{in}}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} w_{o,c,u,v}\; x_{c,\;si+du-p,\;sj+dv-p}
$$

with stride $s$, dilation $d$ and zero padding $p$ (reads outside the image return $0$).

## Output size

$$
H'=\left\lfloor \frac{H+2p-d\,(k-1)-1}{s}\right\rfloor+1
$$

For $d=1$: $H'=\lfloor (H+2p-k)/s\rfloor+1$. "Same" padding at stride 1 needs $p=(k-1)/2$.

## Parameters and cost

$$
\text{params}=C_{out}\,(C_{in}k^2+1),\qquad
\text{MACs}=H'W'\,C_{out}\,C_{in}\,k^2 .
$$

Note that params do **not** depend on $H,W$, but compute does. Doubling resolution quadruples MACs.

**Depthwise-separable** factorization (a $k\times k$ depthwise conv per channel followed by a $1\times1$ pointwise conv):

$$
\frac{\text{MACs}_{sep}}{\text{MACs}_{std}}=\frac{H'W'\,C_{in}(k^2+C_{out})}{H'W'\,C_{out}C_{in}k^2}=\frac1{C_{out}}+\frac1{k^2}.
$$

## Convolution is a matrix multiply (im2col)

Unfold every $k\times k\times C_{in}$ patch into a row of a matrix $X_{col}\in\mathbb R^{(H'W')\times(C_{in}k^2)}$ and flatten the filters into $W_{col}\in\mathbb R^{(C_{in}k^2)\times C_{out}}$:

$$
Y = X_{col}\,W_{col} + b .
$$

This is how GEMM-based libraries (and Tensor Cores) execute convolutions.

## Backpropagation through a convolution

Let $L$ be the loss and $\delta=\partial L/\partial y$.

**Gradient w.r.t. the kernel** — a correlation of the input with the upstream gradient:

$$
\frac{\partial L}{\partial w_{o,c,u,v}}=\sum_{i,j}\delta_{o,i,j}\;x_{c,\,si+du-p,\,sj+dv-p}
$$

**Gradient w.r.t. the input** — a *full* convolution of $\delta$ with the 180°-rotated kernel (a transposed convolution):

$$
\frac{\partial L}{\partial x_{c,m,n}}=\sum_{o}\sum_{u,v}w_{o,c,u,v}\;\delta_{o,\,(m+p-du)/s,\,(n+p-dv)/s}
$$

where terms with non-integer indices are dropped. Because the same $w$ is reused at every position, its gradient **accumulates** over all positions — that is weight sharing in the backward pass.

## Receptive-field recurrence

Track the receptive field $r_\ell$ and the "jump" $j_\ell$ (input pixels between adjacent units):

$$
r_\ell=r_{\ell-1}+d_\ell\,(k_\ell-1)\,j_{\ell-1},\qquad j_\ell=j_{\ell-1}\,s_\ell,\qquad r_0=j_0=1 .
$$

Stride-1 $3\times3$ stacks: $r_L=2L+1$ (linear). Dilations $1,2,4,\dots$: $r_L=2^{L+1}-1$ (exponential).

## Normalization and residuals

**BatchNorm** (per channel, mini-batch $\mathcal B$):

$$
\hat x=\frac{x-\mu_{\mathcal B}}{\sqrt{\sigma^2_{\mathcal B}+\epsilon}},\qquad y=\gamma\hat x+\beta .
$$

At inference $\mu,\sigma^2$ are running averages, so BN can be **folded** into the preceding conv's weights.

**Residual block** $y=x+F(x)$ gives the Jacobian

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x},
$$

so across $L$ blocks the gradient contains the term $\prod_\ell (I+J_\ell)=I+\sum_\ell J_\ell+\dots$ — a path of **exactly the identity** that cannot vanish.

## He initialization

To keep activation variance constant through ReLU layers, initialize with

$$
w\sim\mathcal N\!\left(0,\;\frac{2}{C_{in}k^2}\right).
$$
