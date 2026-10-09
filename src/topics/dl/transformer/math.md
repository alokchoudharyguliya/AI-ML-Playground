## Scaled dot-product attention

For a sequence of $n$ tokens with embeddings $X\in\mathbb R^{n\times d}$:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad
\mathrm{Attn}(X)=\mathrm{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}+M\Big)V
$$

where $M_{ij}=-\infty$ if position $j$ must be hidden from $i$ (causal: $j>i$; padding), else $0$. Row $i$ of the softmax is a probability distribution over keys.

## Why $\sqrt{d_k}$?

Take $q,k\in\mathbb R^{d_k}$ with independent zero-mean, unit-variance entries. Then

$$
\mathbb E[q\cdot k]=0,\qquad \mathrm{Var}(q\cdot k)=\sum_{m=1}^{d_k}\mathrm{Var}(q_mk_m)=d_k .
$$

The standard deviation $\sqrt{d_k}$ grows with dimension; large-magnitude logits push the softmax into saturation where the Jacobian $\mathrm{diag}(p)-pp^{\top}\to0$ and gradients vanish. Dividing by $\sqrt{d_k}$ restores unit variance (verify with the lab's readout).

## Multi-head attention

$$
\mathrm{head}_i=\mathrm{Attn}\big(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)}\big),\qquad
\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W_O ,
$$

with $W^{(i)}_{Q,K,V}\in\mathbb R^{d\times d/h}$ and $W_O\in\mathbb R^{d\times d}$. Parameters: $4d^2$ regardless of $h$.

## Position-wise feed-forward network

$$
\mathrm{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2,\quad W_1\in\mathbb R^{d_{ff}\times d},\ d_{ff}=4d,
$$

giving $2\cdot d\cdot4d=8d^2$ parameters. With **SwiGLU** ($W_2(\mathrm{SiLU}(W_gx)\odot W_1x)$ with $d_{ff}\approx\tfrac83d$) it is $3\cdot d\cdot\tfrac83d=8d^2$ as well.

## Layer normalisation

$$
\mathrm{LN}(x)=\gamma\odot\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}+\beta,\qquad
\mathrm{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\tfrac1d\sum_ix_i^2+\epsilon}} .
$$

## Sinusoidal positional encoding

$$
PE_{(pos,2i)}=\sin\!\Big(\frac{pos}{10000^{2i/d}}\Big),\qquad
PE_{(pos,2i+1)}=\cos\!\Big(\frac{pos}{10000^{2i/d}}\Big).
$$

For each frequency $\omega_i=10000^{-2i/d}$, the pair at position $pos+k$ is a rotation of the pair at $pos$:

$$
\begin{pmatrix}\sin\omega(p+k)\\ \cos\omega(p+k)\end{pmatrix}=
\begin{pmatrix}\cos\omega k&\sin\omega k\\-\sin\omega k&\cos\omega k\end{pmatrix}
\begin{pmatrix}\sin\omega p\\ \cos\omega p\end{pmatrix},
$$

a **linear function of $PE_p$** that depends only on the offset $k$ — the justification for relative-position generalisation. The similarity $PE_p\cdot PE_{p+k}=\sum_i\cos(\omega_ik)$ decays with $|k|$ (second panel of the lab).

## Parameter and FLOP counts

For $L$ layers, width $d$, vocabulary $V$, and ignoring biases/norms:

$$
N\approx\underbrace{12\,L\,d^2}_{\text{blocks}}+\underbrace{V\,d}_{\text{embeddings}} .
$$

Training FLOPs per token $\approx6N$ (2N forward + 4N backward); inference $\approx2N$ plus attention's $\approx 4\,L\,n\,d$ per token which grows with context. Example: $d=4096,\ L=32$ → $12\cdot32\cdot4096^2\approx6.4$ B block parameters (+ embeddings → ~7 B).

| Operation | FLOPs (forward, per layer) | Memory |
|---|---|---|
| QKV + output projections | $8nd^2$ | $O(nd)$ |
| Attention scores + weighted sum | $4n^2d$ | $O(n^2h)$ (naive) |
| FFN | $16nd^2$ | $O(nd_{ff})$ |

Attention dominates once $n\gtrsim6d$ (since $4n^2d>24nd^2\Leftrightarrow n>6d$) — tens of thousands of tokens for typical widths.

## Softmax gradient and saturation

$$
\frac{\partial p_i}{\partial s_j}=p_i(\delta_{ij}-p_j),\qquad \Big\|\frac{\partial p}{\partial s}\Big\|_F\to0\ \text{ as }p\to\text{one-hot}.
$$

## Cross-attention

$$
\mathrm{CrossAttn}(Y,X_{enc})=\mathrm{softmax}\!\Big(\frac{(YW_Q)(X_{enc}W_K)^{\top}}{\sqrt{d_k}}\Big)X_{enc}W_V,
$$

queries from the decoder $Y$, keys and values from the encoder output.

## Language-model objective

$$
\mathcal L=-\frac1n\sum_{i=1}^{n}\log p_\theta(x_i\mid x_{<i}),\qquad p_\theta=\mathrm{softmax}(h_i\,E^{\top}),
$$

with output projection often **tied** to the input embedding matrix $E$.
