## Masked language modelling objective

Let $\mathcal M$ be the set of selected positions ($|\mathcal M|\approx0.15\,n$) and $\tilde x$ the corrupted input. The loss is cross-entropy **only over $\mathcal M$**:

$$
\mathcal L_{MLM}=-\sum_{i\in\mathcal M}\log p_\theta\big(x_i\mid\tilde x\big),\qquad
p_\theta(\cdot\mid\tilde x)=\mathrm{softmax}\big(E\,\mathrm{LN}(\mathrm{GELU}(W h_i))+b\big),
$$

with the decoder matrix tied to the input embedding $E$.

**Why 80/10/10?** For a selected position the input is `[MASK]` w.p. $0.8$, a uniformly random token w.p. $0.1$, the original w.p. $0.1$. Per token, the expected corruption is $0.15\cdot0.9=13.5\%$ non-original, and the model sees an unchanged-but-supervised token in $1.5\%$ of positions, so it must keep contextual information for every token instead of copying.

Expected number of tokens carrying loss per sequence $=0.15\,n$; for $n=512$ that's $\approx77$ targets — compared with $512$ for a causal LM.

## Input representation

$$
h^{(0)}_i=\mathrm{LN}\big(E_{tok}[x_i]+E_{seg}[s_i]+E_{pos}[i]\big),\qquad
h^{(\ell)}=\mathrm{EncoderLayer}(h^{(\ell-1)}) .
$$

Post-LN encoder layer (original BERT):

$$
u=\mathrm{LN}\big(h+\mathrm{MHA}(h)\big),\qquad h'=\mathrm{LN}\big(u+\mathrm{FFN}(u)\big),\quad
\mathrm{FFN}(u)=W_2\,\mathrm{GELU}(W_1u).
$$

## Parameter count (exact)

For vocabulary $V$, positions $P$, hidden $H$, FFN size $F=rH$, layers $L$:

$$
\begin{aligned}
\text{embeddings}&=VH+PH+2H+2H\quad(\text{token, position, 2 segments, LN})\\
\text{per layer}&=\underbrace{4(H^2+H)}_{QKVO}+\underbrace{2rH^2+rH+H}_{\text{FFN}}+\underbrace{4H}_{2\text{ LNs}}\\
\text{pooler}&=H^2+H
\end{aligned}
$$

BERT-base ($V=30522,P=512,H=768,F=3072,L=12$): embeddings $23{,}837{,}184$; layers $12\times7{,}087{,}872=85{,}054{,}464$; pooler $590{,}592$ → **109,482,240** ✓.

FLOPs per token (forward) $\approx2\cdot N_{non\text{-}emb}+4LnH$, where the second term is attention over a context of $n$ tokens.

## Fine-tuning heads

**Classification** with $K$ classes:

$$
p(y\mid x)=\mathrm{softmax}\big(W_c\tanh(W_ph_{[CLS]}+b_p)+b_c\big),\qquad \mathcal L=-\log p(y^*\mid x).
$$

**Span extraction** (QA): learn start and end vectors $S,T\in\mathbb R^H$:

$$
P_i^{start}=\frac{e^{S\cdot h_i}}{\sum_je^{S\cdot h_j}},\quad P_i^{end}=\frac{e^{T\cdot h_i}}{\sum_je^{T\cdot h_j}},\quad
\mathcal L=-\log P^{start}_{a}-\log P^{end}_{b},
$$

predict the span $(i,j)$, $i\le j$, maximising $S\cdot h_i+T\cdot h_j$.

## ELECTRA: replaced-token detection

A small generator proposes replacements; the discriminator labels each token as original/replaced:

$$
\mathcal L_{disc}=-\sum_{i=1}^{n}\Big[\mathbb 1[\hat x_i=x_i]\log D_i+\mathbb 1[\hat x_i\neq x_i]\log(1-D_i)\Big],
$$

giving a gradient from **all $n$** positions rather than 15%.

## Sentence embeddings (contrastive)

For a batch of positive pairs $(a_i,b_i)$ with embeddings $u_i,v_i$ and temperature $\tau$ (InfoNCE):

$$
\mathcal L=-\frac1B\sum_{i}\log\frac{\exp\big(\cos(u_i,v_i)/\tau\big)}{\sum_{j}\exp\big(\cos(u_i,v_j)/\tau\big)} .
$$

## Distillation

$$
\mathcal L=\alpha\,\mathrm{CE}(y,\sigma(z_s))+(1-\alpha)\,\tau^2\,\mathrm{KL}\big(\sigma(z_t/\tau)\,\|\,\sigma(z_s/\tau)\big),
$$

with teacher logits $z_t$, student logits $z_s$ and temperature $\tau$ (softened targets carry "dark knowledge" about class similarity).

## Layer-wise learning-rate decay

$$
\eta_\ell=\eta_{top}\cdot\gamma^{\,L-\ell},\qquad\gamma\in[0.8,0.95],
$$

so lower layers (general features) change less than upper layers (task-specific).
