## RMSNorm and SwiGLU

$$
\mathrm{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}},\qquad
\mathrm{SwiGLU}(x)=W_2\big(\mathrm{SiLU}(W_gx)\odot W_ux\big),\quad\mathrm{SiLU}(z)=z\,\sigma(z).
$$

With hidden size $d_{ff}=\tfrac83d$ (rounded) the SwiGLU FFN has $3\,d\,d_{ff}\approx8d^2$ parameters — the same as a ReLU FFN with $4d$.

## Rotary position embedding (RoPE)

Group the $d_{head}$ dimensions into pairs $(x_{2i},x_{2i+1})$ and rotate pair $i$ at position $m$ by angle $m\theta_i$:

$$
R_m^{(i)}=\begin{pmatrix}\cos m\theta_i&-\sin m\theta_i\\ \sin m\theta_i&\cos m\theta_i\end{pmatrix},\qquad
\theta_i=b^{-2i/d_{head}},\ \ b=10^4\ (\text{or larger}).
$$

Then, because rotations compose, $R_m^{\top}R_n=R_{n-m}$:

$$
\langle R_mq,\,R_nk\rangle=q^{\top}R_m^{\top}R_nk=q^{\top}R_{n-m}k .
$$

The attention logit depends on **position only through $n-m$**. Per pair, the contribution is $\|q_i\|\|k_i\|\cos\big((n-m)\theta_i+\varphi_i\big)$ with $\varphi_i$ the angle between $q_i$ and $k_i$. Low $i$ ⇒ high frequency (local); high $i$ ⇒ slow rotation (long range). **Position interpolation** to extend a context from $L$ to $L'$ uses $m\to m\,L/L'$; **NTK-aware** scaling instead enlarges the base $b$.

## Grouped-query attention

With $H$ query heads and $G$ KV heads ($G\mid H$), query head $h$ attends using KV head $\lfloor hG/H\rfloor$. KV-cache size scales by $G/H$; $G=1$ is **MQA**, $G=H$ is MHA.

## Mixture of Experts

Router logits $r=W_rx\in\mathbb R^E$; select $\mathcal T=\mathrm{top\text{-}k}(r)$:

$$
y=\sum_{e\in\mathcal T}g_e\,\mathrm{FFN}_e(x),\qquad g_e=\frac{\exp r_e}{\sum_{j\in\mathcal T}\exp r_j}.
$$

**Switch-style load-balancing loss.** With $f_e$ the fraction of tokens whose top-1 choice is expert $e$ and $P_e$ the mean router probability for $e$ over the batch:

$$
\mathcal L_{aux}=\alpha\,E\sum_{e=1}^{E}f_e\,P_e,\qquad\min=\alpha\ \text{ at } f_e=P_e=1/E .
$$

Capacity per expert: $\mathrm{cap}=\Big\lceil\kappa\,\dfrac{k\,T}{E}\Big\rceil$ with capacity factor $\kappa\ge1$ for $T$ tokens in the batch.

**Active vs total parameters.** With $N_{dense}$ parameters shared by all tokens (attention, embeddings) and $N_{exp}$ per expert:

$$
N_{total}=N_{dense}+E\,N_{exp},\qquad N_{active}=N_{dense}+k\,N_{exp}.
$$

## LoRA

For a frozen $W_0\in\mathbb R^{d_{out}\times d_{in}}$ add $\Delta W=\dfrac{\alpha}{r}BA$, $B\in\mathbb R^{d_{out}\times r}$ (init 0), $A\in\mathbb R^{r\times d_{in}}$ (init Gaussian):

$$
h=W_0x+\frac{\alpha}{r}B(Ax),\qquad\#\text{trainable}=r\,(d_{in}+d_{out}).
$$

Because $B=0$ at start, $\Delta W=0$ and training starts exactly from the base model. After training, merge: $W=W_0+\tfrac\alpha rBA$ (no inference overhead).

**Memory accounting** with Adam and mixed precision: weights (2 B) + grads (2 B) + fp32 master copy (4 B) + Adam $m,v$ (8 B) $=16$ B/param for trainable parameters; frozen base weights cost 2 B (fp16) or ≈0.5 B (NF4).

## Scaling laws

$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},\qquad C\approx6ND .
$$

With $D=C/(6N)$ minimise over $N$: $\;\partial L/\partial N=0$ gives

$$
N^{*}=G\Big(\frac C6\Big)^{\frac{\beta}{\alpha+\beta}},\quad
D^{*}=G^{-1}\Big(\frac C6\Big)^{\frac{\alpha}{\alpha+\beta}},\quad
G=\Big(\frac{\alpha A}{\beta B}\Big)^{\frac1{\alpha+\beta}} .
$$

With $\alpha\approx0.34,\beta\approx0.28$: $N^*\propto C^{0.45}$, $D^*\propto C^{0.55}$ — double compute → multiply both $N$ and $D$ by ≈1.4–1.5. Irreducible loss $E$ is the entropy of natural text.

## Direct Preference Optimization

Given prompt $x$, preferred $y_w$, rejected $y_l$, policy $\pi_\theta$ and frozen reference $\pi_{ref}$:

$$
\mathcal L_{DPO}=-\,\mathbb E\Big[\log\sigma\Big(\beta\Big(\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big)\Big].
$$

It is the closed-form solution of the KL-regularised RLHF objective $\max_\pi\mathbb E[r(x,y)]-\beta\,\mathrm{KL}(\pi\|\pi_{ref})$ with the reward implicitly $r=\beta\log(\pi/\pi_{ref})$ — no reward model or PPO loop needed.

## Compute accounting for training

$$
\text{FLOPs}\approx6ND,\qquad \text{MFU}=\frac{6ND/t}{n_{GPU}\cdot\text{peak FLOPS}} .
$$

Training a 7B model on 2T tokens: $6\cdot7\!\times\!10^9\cdot2\!\times\!10^{12}=8.4\times10^{22}$ FLOPs; on 256 H100s (≈400 TFLOPS achievable bf16 each at ~40% MFU) ≈ $8.4\times10^{22}/(256\cdot4\times10^{14})\approx8.2\times10^{5}$ s ≈ 9.5 days.
