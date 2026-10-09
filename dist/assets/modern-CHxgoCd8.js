import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{c as r,f as i,g as a,h as o,i as s,m as c,n as l,s as u,t as d,v as f,y as p}from"./viz-CPys2405.js";import{c as m,i as h,l as g,o as _,s as v,t as y}from"./hooks-Dw7oo1m3.js";var b=e(t(),1),x=`## The recipe converged — here are the ingredients

Open any 2024–25 open-weights LLM (Llama 3, Mistral, Qwen, DeepSeek, Gemma) and the architecture looks like the 2017 Transformer *after a decade of ablations*. The differences are individually small and collectively decisive:

| Component | Original (2017) | Modern default | Why |
|---|---|---|---|
| Norm | post-LayerNorm | **pre-RMSNorm** | stable at depth, cheaper (no mean/bias) |
| Activation / FFN | ReLU, $4d$ | **SwiGLU**, $\\tfrac83d$ | gated units train better at equal params |
| Positions | sinusoidal / learned | **RoPE** | relative, extrapolates with scaling tricks |
| Attention | MHA | **GQA** (or MLA) | much smaller KV cache |
| Biases | everywhere | **none** | simpler, no quality loss |
| Embeddings | tied | tied (small) / untied (large) | capacity vs params |
| Attention kernel | naive | **FlashAttention** | IO-aware, exact, long context |
| Optimiser | Adam | **AdamW**, β₂≈0.95, cosine/WSD | stability at scale |
| Precision | fp32 | **bf16** (fp8 emerging) | 2–4× throughput |

## RoPE and long context

**Rotary Position Embedding** rotates each 2-D slice of $q$ and $k$ by an angle proportional to position, with geometrically spaced frequencies. The dot product then depends only on the *offset* — verified in the first lab tab. To extend context beyond training length, rescale positions or frequencies: **Position Interpolation**, **NTK-aware** scaling, **YaRN**, or train with a larger RoPE base. Further long-context tools: sliding-window / local–global attention, attention sinks, ring attention for sequence parallelism, and KV-cache compression.

## Mixture of Experts (MoE)

Replace the dense FFN with $E$ expert FFNs and a **router**; each token runs through only its top-$k$ experts, so **parameters ≫ active parameters**. Mixtral-8×7B has 47B parameters but ~13B active per token; DeepSeek-V3 has 671B total / 37B active.

Engineering realities (second lab tab):

- **Load balancing**: unconstrained routers collapse onto a few experts. An auxiliary loss ($E\\sum_i f_iP_i$) or bias-based balancing keeps experts used equally.
- **Capacity & dropped tokens**: each expert has a fixed buffer; overflow tokens skip the layer (or are rerouted).
- **Expert parallelism**: experts live on different GPUs; tokens are exchanged with all-to-all communication — often the bottleneck.
- **Fine-grained + shared experts** (DeepSeek): many small experts plus always-on shared experts improve specialisation.
- MoE trades **memory for compute**: cheaper per token, but you must store every expert.

## Parameter-efficient fine-tuning

Full fine-tuning a 7B model needs ~16 bytes/parameter ≈ 112 GB for weights + optimiser state. **LoRA** freezes $W$ and trains a low-rank update $\\Delta W=BA$ ($r\\ll d$): 0.1–1% of the parameters, near-identical quality on many tasks, tiny adapter files that can be hot-swapped per customer. **QLoRA** also stores the frozen base in 4-bit NF4, fitting a 65B model on one 48 GB GPU. Related: adapters, prefix/prompt tuning, DoRA, LoRA+, GaLore.

## Quantisation and efficient inference

Post-training weight quantisation (GPTQ, AWQ, SmoothQuant) to int8/int4, FP8 training/inference on Hopper+, KV-cache quantisation, speculative decoding, continuous batching and paged KV memory. Rule of thumb: **int4 weights ≈ 3–4× less memory, <1–2% quality loss** for ≥7B models.

## Scaling laws

Loss falls as a power law in parameters $N$, data $D$ and compute $C\\approx6ND$. **Kaplan et al. (2020)** suggested parameters matter most; **Chinchilla (Hoffmann et al., 2022)** corrected this: for compute-optimal training, scale $N$ and $D$ *together* (~20 tokens/parameter in the original fits). Modern labs **over-train** smaller models on 100–1000+ tokens/parameter because inference cost, not training cost, dominates a product's lifetime. Scaling laws also guide hyper-parameters (μP/μTransfer), data mixtures and when *not* to scale (data walls, synthetic data).

## Alignment and post-training

- **SFT** on demonstrations, then **preference optimisation**: **RLHF** (reward model + PPO) or the simpler offline **DPO** family.
- **RL with verifiable rewards** (GRPO and relatives) for math/code reasoning; **test-time compute** (long chain-of-thought, self-verification, best-of-$n$) became a second scaling axis.
- **Distillation** from strong models; **constitutional / RLAIF** methods; **safety training** and red-teaming.

## Training at scale

Thousands of GPUs combine **data parallel** (FSDP/ZeRO shards weights, grads, optimiser state), **tensor parallel** (split matmuls within a node), **pipeline parallel** (split layers) and **sequence/context parallel**, plus activation checkpointing, fused kernels and overlapping communication with compute. Model FLOPs utilisation (**MFU**) of 40–55% is considered excellent. This is where the CUDA track becomes essential.

## Where this is heading

Multimodal tokens (vision/audio/video), hybrid attention–SSM stacks (Jamba, Mamba-2), longer context, reasoning models, agentic tool use and retrieval — all built from the same ingredients you have now seen: attention, gating, residual streams, careful optimisation, and hardware-aware kernels.
`,S=`## RMSNorm and SwiGLU

$$
\\mathrm{RMSNorm}(x)=\\gamma\\odot\\frac{x}{\\sqrt{\\frac1d\\sum_i x_i^2+\\epsilon}},\\qquad
\\mathrm{SwiGLU}(x)=W_2\\big(\\mathrm{SiLU}(W_gx)\\odot W_ux\\big),\\quad\\mathrm{SiLU}(z)=z\\,\\sigma(z).
$$

With hidden size $d_{ff}=\\tfrac83d$ (rounded) the SwiGLU FFN has $3\\,d\\,d_{ff}\\approx8d^2$ parameters — the same as a ReLU FFN with $4d$.

## Rotary position embedding (RoPE)

Group the $d_{head}$ dimensions into pairs $(x_{2i},x_{2i+1})$ and rotate pair $i$ at position $m$ by angle $m\\theta_i$:

$$
R_m^{(i)}=\\begin{pmatrix}\\cos m\\theta_i&-\\sin m\\theta_i\\\\ \\sin m\\theta_i&\\cos m\\theta_i\\end{pmatrix},\\qquad
\\theta_i=b^{-2i/d_{head}},\\ \\ b=10^4\\ (\\text{or larger}).
$$

Then, because rotations compose, $R_m^{\\top}R_n=R_{n-m}$:

$$
\\langle R_mq,\\,R_nk\\rangle=q^{\\top}R_m^{\\top}R_nk=q^{\\top}R_{n-m}k .
$$

The attention logit depends on **position only through $n-m$**. Per pair, the contribution is $\\|q_i\\|\\|k_i\\|\\cos\\big((n-m)\\theta_i+\\varphi_i\\big)$ with $\\varphi_i$ the angle between $q_i$ and $k_i$. Low $i$ ⇒ high frequency (local); high $i$ ⇒ slow rotation (long range). **Position interpolation** to extend a context from $L$ to $L'$ uses $m\\to m\\,L/L'$; **NTK-aware** scaling instead enlarges the base $b$.

## Grouped-query attention

With $H$ query heads and $G$ KV heads ($G\\mid H$), query head $h$ attends using KV head $\\lfloor hG/H\\rfloor$. KV-cache size scales by $G/H$; $G=1$ is **MQA**, $G=H$ is MHA.

## Mixture of Experts

Router logits $r=W_rx\\in\\mathbb R^E$; select $\\mathcal T=\\mathrm{top\\text{-}k}(r)$:

$$
y=\\sum_{e\\in\\mathcal T}g_e\\,\\mathrm{FFN}_e(x),\\qquad g_e=\\frac{\\exp r_e}{\\sum_{j\\in\\mathcal T}\\exp r_j}.
$$

**Switch-style load-balancing loss.** With $f_e$ the fraction of tokens whose top-1 choice is expert $e$ and $P_e$ the mean router probability for $e$ over the batch:

$$
\\mathcal L_{aux}=\\alpha\\,E\\sum_{e=1}^{E}f_e\\,P_e,\\qquad\\min=\\alpha\\ \\text{ at } f_e=P_e=1/E .
$$

Capacity per expert: $\\mathrm{cap}=\\Big\\lceil\\kappa\\,\\dfrac{k\\,T}{E}\\Big\\rceil$ with capacity factor $\\kappa\\ge1$ for $T$ tokens in the batch.

**Active vs total parameters.** With $N_{dense}$ parameters shared by all tokens (attention, embeddings) and $N_{exp}$ per expert:

$$
N_{total}=N_{dense}+E\\,N_{exp},\\qquad N_{active}=N_{dense}+k\\,N_{exp}.
$$

## LoRA

For a frozen $W_0\\in\\mathbb R^{d_{out}\\times d_{in}}$ add $\\Delta W=\\dfrac{\\alpha}{r}BA$, $B\\in\\mathbb R^{d_{out}\\times r}$ (init 0), $A\\in\\mathbb R^{r\\times d_{in}}$ (init Gaussian):

$$
h=W_0x+\\frac{\\alpha}{r}B(Ax),\\qquad\\#\\text{trainable}=r\\,(d_{in}+d_{out}).
$$

Because $B=0$ at start, $\\Delta W=0$ and training starts exactly from the base model. After training, merge: $W=W_0+\\tfrac\\alpha rBA$ (no inference overhead).

**Memory accounting** with Adam and mixed precision: weights (2 B) + grads (2 B) + fp32 master copy (4 B) + Adam $m,v$ (8 B) $=16$ B/param for trainable parameters; frozen base weights cost 2 B (fp16) or ≈0.5 B (NF4).

## Scaling laws

$$
L(N,D)=E+\\frac{A}{N^{\\alpha}}+\\frac{B}{D^{\\beta}},\\qquad C\\approx6ND .
$$

With $D=C/(6N)$ minimise over $N$: $\\;\\partial L/\\partial N=0$ gives

$$
N^{*}=G\\Big(\\frac C6\\Big)^{\\frac{\\beta}{\\alpha+\\beta}},\\quad
D^{*}=G^{-1}\\Big(\\frac C6\\Big)^{\\frac{\\alpha}{\\alpha+\\beta}},\\quad
G=\\Big(\\frac{\\alpha A}{\\beta B}\\Big)^{\\frac1{\\alpha+\\beta}} .
$$

With $\\alpha\\approx0.34,\\beta\\approx0.28$: $N^*\\propto C^{0.45}$, $D^*\\propto C^{0.55}$ — double compute → multiply both $N$ and $D$ by ≈1.4–1.5. Irreducible loss $E$ is the entropy of natural text.

## Direct Preference Optimization

Given prompt $x$, preferred $y_w$, rejected $y_l$, policy $\\pi_\\theta$ and frozen reference $\\pi_{ref}$:

$$
\\mathcal L_{DPO}=-\\,\\mathbb E\\Big[\\log\\sigma\\Big(\\beta\\Big(\\log\\tfrac{\\pi_\\theta(y_w|x)}{\\pi_{ref}(y_w|x)}-\\log\\tfrac{\\pi_\\theta(y_l|x)}{\\pi_{ref}(y_l|x)}\\Big)\\Big)\\Big].
$$

It is the closed-form solution of the KL-regularised RLHF objective $\\max_\\pi\\mathbb E[r(x,y)]-\\beta\\,\\mathrm{KL}(\\pi\\|\\pi_{ref})$ with the reward implicitly $r=\\beta\\log(\\pi/\\pi_{ref})$ — no reward model or PPO loop needed.

## Compute accounting for training

$$
\\text{FLOPs}\\approx6ND,\\qquad \\text{MFU}=\\frac{6ND/t}{n_{GPU}\\cdot\\text{peak FLOPS}} .
$$

Training a 7B model on 2T tokens: $6\\cdot7\\!\\times\\!10^9\\cdot2\\!\\times\\!10^{12}=8.4\\times10^{22}$ FLOPs; on 256 H100s (≈400 TFLOPS achievable bf16 each at ~40% MFU) ≈ $8.4\\times10^{22}/(256\\cdot4\\times10^{14})\\approx8.2\\times10^{5}$ s ≈ 9.5 days.
`,C=`## Exercises

**Q1.** A GQA model has 32 query heads, 8 KV heads, $d_{head}=128$, 32 layers. By what factor does GQA shrink the KV cache versus MHA, and what is the per-token cache in fp16?

<details>
<summary>Show answer</summary>

Factor $32/8=4$. Per token: $2\\cdot32\\cdot8\\cdot128\\cdot2=131{,}072$ B $=128$ KiB (MHA would be 512 KiB).

</details>

**Q2.** An MoE layer has 64 experts each with 100M parameters, top-2 routing, plus 2B shared (attention/embedding) parameters. Compute total and active parameters.

<details>
<summary>Show answer</summary>

Total $=2\\text{B}+64\\cdot0.1\\text{B}=8.4$ B. Active $=2\\text{B}+2\\cdot0.1\\text{B}=2.2$ B. Per-token FLOPs resemble a 2.2B dense model while holding 8.4B of knowledge — but you pay memory for all 8.4B and all-to-all communication.

</details>

**Q3.** For a $4096\\times4096$ matrix, LoRA rank 16: trainable parameters and compression ratio?

<details>
<summary>Show answer</summary>

$2\\cdot4096\\cdot16=131{,}072$ vs $16{,}777{,}216$ → **0.78%** (128× fewer). Doing this on $q,v$ in all 32 layers: $32\\cdot2\\cdot131{,}072\\approx8.4$ M parameters, a 33 MB fp32 adapter.

</details>

**Q4.** You have $10^{24}$ FLOPs. Using $C=6ND$ and a 20-tokens/parameter rule, what model and dataset size would you train? How would you change this if the model will be served to millions of users?

<details>
<summary>Show answer</summary>

$D=20N$, so $C=120N^2\\Rightarrow N=\\sqrt{10^{24}/120}\\approx9.1\\times10^{10}$ (91B), $D\\approx1.8$ T tokens. For heavy inference, train a **smaller model on far more tokens** (e.g. 30B on 6–8T): training loss is slightly worse than optimal for the compute, but every future query is cheaper.

</details>

**Q5.** Why does DPO not need a reward model, and what is a typical failure mode?

<details>
<summary>Show answer</summary>

The reward is implicit in the log-ratio $\\beta\\log(\\pi_\\theta/\\pi_{ref})$, and the Bradley–Terry preference likelihood becomes a simple logistic loss. Failure modes: both chosen and rejected likelihoods can *decrease* (probability mass drifts to unseen text), length exploitation, and sensitivity to $\\beta$ and to the quality/off-policyness of preference data.

</details>

**Q6 (code).** Verify the RoPE relative-position property numerically.

<details>
<summary>Show answer</summary>

\`modern_blocks.py\` rotates random $q,k$ at $(m,n)$ and $(m+s,n+s)$ and asserts equal dot products to $10^{-10}$, then checks that $(m,n)$ vs $(m,n+1)$ differ.

</details>

## In practice

**Fine-tuning recipe.** Start from an instruct model; LoRA $r=16$–64 on all linear layers, $\\alpha=2r$ (or $\\alpha=16$), LR $10^{-4}$–$2\\times10^{-4}$, 1–3 epochs, 1–10k *high-quality* examples beat 100k noisy ones. Mask the loss to the response tokens only. Evaluate on held-out tasks + regression tests to detect forgetting.

**Serving tips.** Merge LoRA into weights for single-tenant, or use multi-LoRA serving (S-LoRA, vLLM) for many tenants on one base. Quantise to AWQ/GPTQ int4 for 2–3× cheaper serving.

**MoE in practice.** Great for throughput at scale, painful on small clusters (memory, all-to-all). For fine-tuning, LoRA on attention + shared experts is typical; watch router collapse.

**Choosing a base model.** Match context length, license, multilingual coverage and tokenizer efficiency to your use case; always benchmark on your own data.

**Reading a model card.** Parameters (total/active), tokens trained, context, tokenizer size, attention type (GQA/MLA), norm/activation, RoPE base — you can now estimate memory, FLOPs and KV cost from these alone.

## Common pitfalls

- Forgetting the chat template during fine-tuning (train/serve mismatch).
- High LoRA rank with high LR → instability; low rank on tasks requiring new knowledge → underfit.
- Training on the prompt tokens (loss not masked).
- Merging LoRA into a *quantised* base without dequantising correctly.
- Assuming MoE "active parameters" equals memory footprint.
- Extending context with RoPE scaling but not fine-tuning on long sequences.
- Evaluating alignment only with a reward model that was also used for training (reward hacking).
`,w=`import numpy as np


def rmsnorm(x, gamma, eps=1e-6):
    return gamma * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(z):
    return z / (1 + np.exp(-z))


def swiglu(x, w_gate, w_up, w_down):
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


def rope_angles(positions, d_head, base=10000.0):
    inv_freq = base ** (-np.arange(0, d_head, 2) / d_head)       # theta_i
    return np.outer(positions, inv_freq)                          # (T, d/2)


def apply_rope(x, positions, base=10000.0):
    """x: (T, d_head). Rotate each (x[2i], x[2i+1]) pair by position * theta_i."""
    ang = rope_angles(positions, x.shape[-1], base)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def gqa_attention(q, k, v, n_heads, n_kv_heads):
    """q: (T, H*d)  k,v: (T, G*d). Causal GQA, returns (T, H*d)."""
    T = q.shape[0]
    d = q.shape[1] // n_heads
    q = q.reshape(T, n_heads, d)
    k = np.repeat(k.reshape(T, n_kv_heads, d), n_heads // n_kv_heads, axis=1)   # share KV heads
    v = np.repeat(v.reshape(T, n_kv_heads, d), n_heads // n_kv_heads, axis=1)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(d)
    scores = np.where(np.tril(np.ones((T, T), bool)), scores, -1e9)
    w = np.exp(scores - scores.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", w, v).reshape(T, -1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 64
    q, k = rng.standard_normal(d), rng.standard_normal(d)

    def score(m, n):
        return apply_rope(q[None], [m])[0] @ apply_rope(k[None], [n])[0]

    s1, s2 = score(5, 12), score(5 + 100, 12 + 100)
    print("score(5,12) =", round(s1, 10), " score(105,112) =", round(s2, 10), " |diff| =", abs(s1 - s2))
    assert abs(s1 - s2) < 1e-9, "RoPE logit must depend only on n - m"
    assert abs(score(5, 12) - score(5, 13)) > 1e-6

    x = rng.standard_normal((4, 16))
    print("RMSNorm rms per row:", np.sqrt((rmsnorm(x, np.ones(16)) ** 2).mean(-1)).round(3))

    T, H, G = 6, 8, 2
    out = gqa_attention(rng.standard_normal((T, H * 16)), rng.standard_normal((T, G * 16)), rng.standard_normal((T, G * 16)), H, G)
    print("GQA output:", out.shape, "-> KV cache is", H // G, "x smaller than MHA")
`,T=`import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, d_ff, bias=False), nn.Linear(d, d_ff, bias=False), nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))        # SwiGLU expert


class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts=8, top_k=2, aux_weight=0.01):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d, d_ff) for _ in range(n_experts))
        self.E, self.k, self.aux_weight = n_experts, top_k, aux_weight

    def forward(self, x):                                  # x: (B, T, d)
        B, T, d = x.shape
        x = x.reshape(-1, d)                               # (N, d) flatten tokens
        logits = self.router(x)                            # (N, E)
        probs = logits.softmax(-1)
        top_p, top_i = logits.topk(self.k, dim=-1)         # choose k experts per token
        gates = top_p.softmax(-1)                          # renormalise over the chosen experts

        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):          # (production: grouped GEMM / all-to-all)
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, gates[tok, slot, None] * expert(x[tok]))

        # Switch-Transformer load-balancing loss: E * sum_e f_e * P_e
        f = F.one_hot(top_i[:, 0], self.E).float().mean(0)     # fraction of tokens whose top-1 is e
        P = probs.mean(0)                                      # mean router probability
        aux = self.aux_weight * self.E * (f * P).sum()
        return out.view(B, T, d), aux


if __name__ == "__main__":
    torch.manual_seed(0)
    moe = MoE(d=64, d_ff=128)
    y, aux = moe(torch.randn(4, 16, 64))
    total = sum(p.numel() for p in moe.parameters())
    active = sum(p.numel() for p in moe.router.parameters()) + moe.k * sum(p.numel() for p in moe.experts[0].parameters())
    print(y.shape, "aux loss:", float(aux), f"(min {moe.aux_weight})", f"| total params {total:,}, active per token {active:,}")
`,E=`import copy
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W0 x + (alpha / r) * B (A x)   with W0 frozen."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                                   # freeze pretrained weights
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))      # zero init -> starts as the base model
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.A.T @ self.B.T)

    @torch.no_grad()
    def merge(self):
        """Fold the adapter into W0 for zero-overhead inference."""
        self.base.weight += self.scale * (self.B @ self.A)
        return self.base


def add_lora(model, target=("q_proj", "v_proj"), r=16, alpha=32):
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha))
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(256, 256)
    lora = LoRALinear(base, r=8, alpha=16)
    x = torch.randn(4, 256)
    assert torch.allclose(lora(x), base(x), atol=1e-6), "B=0 => identical to base at init"

    nn.init.normal_(lora.B, std=0.02)                                 # pretend we trained
    y = lora(x)
    merged = copy.deepcopy(lora).merge()                              # fold BA into W0
    print("max |adapter - merged| =", (y - merged(x)).abs().max().item())

    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    print(f"trainable {trainable:,} vs full {256 * 256 + 256:,}  ({trainable / (256 * 256):.1%})")
`,D=`import torch
import torch.nn.functional as F


def sequence_logprob(logits, labels, mask):
    """Sum of token log-probs of \`labels\` under \`logits\`. mask=1 on response tokens only."""
    logp = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)


def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=0.1):
    """All inputs are (B,) sequence log-probs. Returns loss and implicit-reward accuracy."""
    chosen_reward = beta * (pol_chosen - ref_chosen)          # implicit reward  beta * log(pi / pi_ref)
    rejected_reward = beta * (pol_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    return loss, (margin > 0).float().mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8
    ref_c, ref_r = torch.randn(B) - 20, torch.randn(B) - 20
    # a policy equal to the reference gives margin 0 -> loss = log 2
    loss, acc = dpo_loss(ref_c, ref_r, ref_c, ref_r)
    print("loss at init:", loss.item(), "(= ln 2 =", torch.log(torch.tensor(2.0)).item(), ")")
    # a policy that raises chosen / lowers rejected reduces the loss
    loss, acc = dpo_loss(ref_c + 3, ref_r - 3, ref_c, ref_r)
    print("improved policy loss:", loss.item(), " reward accuracy:", acc.item())
`,O=n(),k=8,A=(()=>{let e=o(4);return Array.from({length:k},()=>[c(e),c(e)])})(),j=(()=>{let e=o(9);return Array.from({length:k},()=>[c(e),c(e)])})(),M=([e,t],n)=>[e*Math.cos(n)-t*Math.sin(n),e*Math.sin(n)+t*Math.cos(n)];function N(e,t,n){let r=0;for(let i=0;i<k;i++){let a=n**+(-i/k),o=M(A[i],e*a),s=M(j[i],t*a);r+=o[0]*s[0]+o[1]*s[1]}return r}function P(){let[e,t]=(0,b.useState)(20),[n,r]=(0,b.useState)(12),[a,o]=(0,b.useState)(1e4),[s,c]=(0,b.useState)(1),[l,u]=(0,b.useState)(30),f=a**+(-s/k),g=N(e,n,a),v=N(e+l,n+l,a),[x]=y(340,(t,r,o)=>{let c=Math.min(o/2-30,r*.2),l=r*.22,u=o/2+6;p(t,`Frequency pair #${s}:  θ = base^(−${s}/${k}) = ${f.toFixed(4)} rad/token`,20,18,{size:12,color:d.mute,weight:600}),t.strokeStyle=d.grid,t.beginPath(),t.arc(l,u,c,0,7),t.stroke(),t.beginPath(),t.moveTo(l-c,u),t.lineTo(l+c,u),t.moveTo(l,u-c),t.lineTo(l,u+c),t.stroke();let m=c/2.4,h=M(A[s],e*f),g=M(j[s],n*f),_=(e,n,r)=>{let i=l+e[0]*m,a=u-e[1]*m;t.strokeStyle=n,t.lineWidth=3,t.beginPath(),t.moveTo(l,u),t.lineTo(i,a),t.stroke(),t.fillStyle=n,t.beginPath(),t.arc(i,a,6,0,7),t.fill(),p(t,r,i+10,a-10,{size:12,color:n,mono:!0,weight:700})};_(h,d.b,`q@${e}`),_(g,d.c,`k@${n}`),Math.atan2(g[1],g[0])-Math.atan2(h[1],h[0]),p(t,`angle between = (n−m)·θ = ${((n-e)*f*180/Math.PI%360).toFixed(0)}°`,l,u+c+18,{size:11.5,align:`center`,color:d.d,mono:!0});let v=r*.45,y=r-v-20,b=o-90;p(t,`attention logit  q·k  vs relative offset (n − m)   — sum over ${k} frequency pairs`,v,18,{size:12,color:d.mute,weight:600});let x=e=>v+(e+80)/160*y,S=[];for(let e=-80;e<=80;e++)S.push([e,N(0,e,a)]);let C=Math.max(...S.map(e=>Math.abs(e[1])),1),w=e=>40+b/2-e/C*(b/2);t.strokeStyle=d.grid,t.beginPath(),t.moveTo(v,w(0)),t.lineTo(v+y,w(0)),t.stroke(),i(t,S.map(([e,t])=>[x(e),w(t)]),d.a,2.4);let T=n-e;Math.abs(T)<=80&&(t.fillStyle=d.d,t.beginPath(),t.arc(x(T),w(N(0,T,a)),6,0,7),t.fill()),p(t,`← keys before query`,v,40+b+18,{size:11,color:d.mute}),p(t,`keys after query →`,v+y,40+b+18,{size:11,align:`right`,color:d.mute}),[-80,-40,0,40,80].forEach(e=>p(t,e,x(e),40+b+4,{size:9.5,align:`center`,color:d.dim,mono:!0}))});return(0,O.jsxs)(O.Fragment,{children:[(0,O.jsx)(`canvas`,{...x}),(0,O.jsxs)(h,{children:[(0,O.jsx)(m,{label:`Query position m`,min:0,max:80,value:e,onChange:t}),(0,O.jsx)(m,{label:`Key position n`,min:0,max:80,value:n,onChange:r}),(0,O.jsx)(m,{label:`Frequency pair`,min:0,max:7,value:s,onChange:c}),(0,O.jsx)(m,{label:`RoPE base`,min:100,max:1e6,step:100,value:a,onChange:o,fmt:e=>e>=1e6?`1M`:e>=1e3?Math.round(e/1e3)+`k`:e}),(0,O.jsx)(m,{label:`Shift both by`,min:0,max:200,value:l,onChange:u})]}),(0,O.jsxs)(_,{children:[`score(m=`,e,`, n=`,n,`) = `,(0,O.jsx)(`b`,{children:g.toFixed(4)}),` · score(m+`,l,`, n+`,l,`) = `,(0,O.jsx)(`b`,{children:v.toFixed(4)}),` · difference `,(0,O.jsx)(`b`,{className:`g`,children:Math.abs(g-v).toExponential(1)}),` → the logit depends `,(0,O.jsx)(`b`,{children:`only on n−m`}),`, even though absolute positions changed. RoPE encodes `,(0,O.jsx)(`i`,{children:`relative`}),` position inside the dot product, with high-frequency pairs resolving nearby tokens and low-frequency pairs long range.`]})]})}var F=28,I=Array.from({length:F},(e,t)=>(t*7+(t>>2))%4),L=[d.c,d.b,d.d,d.e];function R(){let[e,t]=(0,b.useState)(8),[n,r]=(0,b.useState)(2),[i,s]=(0,b.useState)(1.25),[u,g]=(0,b.useState)(1.5),v=(0,b.useMemo)(()=>{let t=o(31),r=Array.from({length:16},()=>c(t)),a=I.map(n=>Array.from({length:e},(e,i)=>(i%4===n?2.2:0)+u*r[i]*.8+(t()-.5)*1.2)),s=a.map(e=>f(e)),l=a.map(e=>e.map((e,t)=>[e,t]).sort((e,t)=>t[0]-e[0]).slice(0,n).map(e=>e[1])),d=Math.ceil(i*F*n/e),p=Array(e).fill(0),m=Array.from({length:e},()=>[]),h=[];l.forEach((e,t)=>e.forEach(e=>{p[e]<d?(p[e]++,m[e].push(t)):h.push([t,e])}));let g=Array(e).fill(0);l.forEach(e=>{g[e[0]]+=1/F});let _=Array(e).fill(0);return s.forEach(e=>e.forEach((e,t)=>{_[t]+=e/F})),{picks:l,C_:d,bins:m,dropped:h,aux:e*g.reduce((e,t,n)=>e+t*_[n],0),load:p}},[e,n,i,u]),[x]=y(380,(t,n,r)=>{let i=(n-40)/F;p(t,`tokens (colour = what the token is about)  →  router picks top-k experts`,20,18,{size:12,color:d.mute,weight:600});let o=t=>20+(t+.5)*((n-40)/e),s=r-150;v.picks.forEach((e,n)=>e.forEach((e,r)=>{let a=20+(n+.5)*i,c=v.dropped.some(([t,r])=>t===n&&r===e);t.strokeStyle=c?l(d.r,.6):l(L[I[n]],r?.28:.55),t.lineWidth=r?1:1.6,c&&t.setLineDash([3,3]),t.beginPath(),t.moveTo(a,52),t.bezierCurveTo(a,130,o(e),s-90,o(e),s),t.stroke(),t.setLineDash([])})),I.forEach((e,n)=>{t.fillStyle=L[e],t.beginPath(),t.arc(20+(n+.5)*i,40,Math.min(8,i/2-1),0,7),t.fill()});let c=Math.min(70,(n-40)/e-8);for(let n=0;n<e;n++){let e=o(n),r=v.load[n]>=v.C_;t.fillStyle=`#10131f`,a(t,e-c/2,s,c,118,8),t.fill(),t.strokeStyle=r?d.d:d.line,t.lineWidth=r?2:1,t.stroke(),v.bins[n].forEach((n,r)=>{t.fillStyle=L[I[n]],t.fillRect(e-c/2+5+r%5*((c-10)/5),s+8+Math.floor(r/5)*13,(c-10)/5-2,12)}),p(t,`E${n}`,e,s+132,{size:11,align:`center`,color:d.ink,weight:700}),p(t,`${v.load[n]}/${v.C_}`,e,s+146,{size:10,align:`center`,color:r?d.d:d.mute,mono:!0})}}),S=v.picks.length*n,C=v.aux<1.15;return(0,O.jsxs)(O.Fragment,{children:[(0,O.jsx)(`canvas`,{...x}),(0,O.jsxs)(h,{children:[(0,O.jsx)(m,{label:`Experts E`,min:4,max:16,step:2,value:e,onChange:t}),(0,O.jsx)(m,{label:`Top-k`,min:1,max:4,value:n,onChange:r}),(0,O.jsx)(m,{label:`Capacity factor`,min:.8,max:2.5,step:.05,value:i,onChange:s,fmt:e=>e.toFixed(2)}),(0,O.jsx)(m,{label:`Router imbalance`,min:0,max:3,step:.1,value:u,onChange:g,fmt:e=>e.toFixed(1)})]}),(0,O.jsxs)(_,{children:[`Capacity per expert = ⌈`,i.toFixed(2),` × `,F,`·`,n,` / `,e,`⌉ = `,(0,O.jsx)(`b`,{children:v.C_}),` slots · dropped assignments: `,(0,O.jsx)(`b`,{className:v.dropped.length?`r`:`g`,children:v.dropped.length}),` of `,S,` · load-balance loss E·Σfᵢ·Pᵢ = `,(0,O.jsx)(`b`,{className:C?`g`:`w`,children:v.aux.toFixed(2)}),` (1.0 = perfectly balanced). Only `,(0,O.jsxs)(`b`,{children:[n,`/`,e]}),` of the expert parameters run per token — `,(0,O.jsxs)(`b`,{children:[Math.round(100*n/e),`% active compute`]}),`, 100% of the memory.`]})]})}function z(){let[e,t]=(0,b.useState)(4096),[n,i]=(0,b.useState)(16),[a,o]=(0,b.useState)(32),[s,c]=(0,b.useState)(`qv`),f=Math.round(3.5*e),g=(s===`qv`?2*n*(e+e):s===`attn`?4*n*(e+e):4*n*(e+e)+3*n*(e+f))*a,x=a*(4*e*e+3*e*f),S={full:x*16,lora:x*2+g*16,qlora:x*.5+g*16},[C]=y(330,(t,i,a)=>{let o=Math.min(a-80,i*.22);p(t,`ΔW = (α/r) · B · A`,40,22,{size:13,color:`#fff`,weight:700}),t.fillStyle=l(d.a,.35),t.fillRect(40,50,o,o),t.strokeStyle=d.a,t.lineWidth=1.5,t.strokeRect(40,50,o,o),p(t,`W  (${e}×${e}) frozen`,40+o/2,50+o/2,{size:12,align:`center`,color:`#fff`,weight:600}),p(t,`${r(e*e)} params`,40+o/2,50+o/2+18,{size:10.5,align:`center`,color:d.mute,mono:!0});let s=Math.max(3,n/e*o*6),c=40+o+40;t.fillStyle=l(d.c,.8),t.fillRect(c,50,s,o),p(t,`B (${e}×${n})`,c+s/2,50+o+14,{size:11,align:`center`,color:d.c,mono:!0});let f=c+s+34;t.fillStyle=l(d.e,.8),t.fillRect(f,50,o,s),p(t,`A (${n}×${e})`,f+o/2,50+s+14,{size:11,align:`center`,color:d.e,mono:!0}),p(t,`×`,c+s+17,50+o/2,{size:22,align:`center`,color:d.mute,weight:700}),p(t,`trainable per matrix: 2·d·r = ${r(2*e*n)}  (${(200*e*n/(e*e)).toFixed(2)}% of W)`,c,50+o+44,{size:11.5,color:d.ink,mono:!0});let m=i*.62,h=i-m-30;p(t,`Training memory — ${(x/1e9).toFixed(1)}B-param model`,m,22,{size:12,color:d.mute,weight:600}),[[`Full fine-tune (Adam, mixed prec.)`,S.full,d.r],[`LoRA (fp16 base)`,S.lora,d.a],[`QLoRA (4-bit base)`,S.qlora,d.e]].forEach(([e,n,r],i)=>{let a=56+i*70;p(t,e,m,a,{size:12,color:`#fff`,weight:600}),t.fillStyle=r,t.fillRect(m,a+12,n/S.full*h*.72,24),p(t,u(n),m+n/S.full*h*.72+8,a+24,{size:12,mono:!0,color:d.ink})}),p(t,`activations & KV excluded`,m,260,{size:10,color:d.dim}),p(t,`full FT: 2 wt + 2 grad + 4 master + 8 Adam = 16 B/param`,m,274,{size:10,color:d.dim})});return(0,O.jsxs)(O.Fragment,{children:[(0,O.jsx)(`canvas`,{...C}),(0,O.jsxs)(h,{children:[(0,O.jsx)(m,{label:`Hidden size d`,min:1024,max:8192,step:512,value:e,onChange:t}),(0,O.jsx)(m,{label:`Layers`,min:12,max:96,value:a,onChange:o}),(0,O.jsx)(m,{label:`Rank r`,min:1,max:256,value:n,onChange:i}),(0,O.jsx)(v,{label:`Adapt`,value:s,onChange:c,options:[[`qv`,`q, v only`],[`attn`,`q, k, v, o`],[`all`,`all linear layers`]]})]}),(0,O.jsxs)(_,{children:[`Trainable parameters: `,(0,O.jsx)(`b`,{children:r(g)}),` = `,(0,O.jsxs)(`b`,{className:`g`,children:[(100*g/x).toFixed(3),`%`]}),` of the model · adapter file ≈ `,(0,O.jsx)(`b`,{children:u(g*2)}),` (fp16) — swap per task without storing a full copy. The frozen base weights can be `,(0,O.jsx)(`b`,{children:`quantised to 4-bit`}),` (QLoRA) because gradients flow only through the small A, B.`]})]})}var B={E:1.69,A:406.4,B:410.7,a:.34,b:.28},V=(e,t)=>B.E+B.A/e**+B.a+B.B/t**+B.b;function H(){let[e,t]=(0,b.useState)(23),[n,r]=(0,b.useState)(10.2),a=10**e,o=10**n,c=a/(6*o),l=(B.a*B.A/(B.b*B.B))**(1/(B.a+B.b))*(a/6)**(B.b/(B.a+B.b)),u=a/(6*l),[f]=y(340,(t,r,f)=>{let m=r-70-24,h=f-90,g=e=>70+(e-7)/6.5*m,_=[],v=1e9,y=0;for(let e=7;e<=13.5;e+=.05){let t=V(10**e,a/(6*10**e));_.push([e,t]),v=Math.min(v,t),y=Math.max(y,t)}y=Math.min(y,v+2.5);let b=e=>36+h-(s(e,v-.05,y)-v+.05)/(y-v+.05)*h;p(t,`Loss at fixed compute C = 10^${e.toFixed(1)} FLOPs  (D = C / 6N)`,70,18,{size:12,color:d.mute,weight:600}),t.strokeStyle=d.grid,t.lineWidth=1;for(let e=7;e<=13;e++)t.beginPath(),t.moveTo(g(e),36),t.lineTo(g(e),36+h),t.stroke(),p(t,`1e${e}`,g(e),36+h+14,{size:10,align:`center`,color:d.dim,mono:!0});p(t,`model parameters N →`,70+m/2,f-10,{size:11,align:`center`,color:d.mute}),i(t,_.map(([e,t])=>[g(e),b(t)]),d.a,2.6);let x=Math.log10(l);t.fillStyle=d.e,t.beginPath(),t.arc(g(x),b(V(l,u)),7,0,7),t.fill(),p(t,`compute-optimal`,g(x),b(V(l,u))-16,{size:11.5,align:`center`,color:d.e,weight:700}),t.fillStyle=d.d,t.beginPath(),t.arc(g(n),b(V(o,c)),6,0,7),t.fill(),p(t,`your model`,g(n),b(V(o,c))+18,{size:11.5,align:`center`,color:d.d,weight:700}),p(t,`too small / too many tokens`,74,48,{size:10.5,color:d.dim}),p(t,`too big / too few tokens`,70+m-4,48,{size:10.5,align:`right`,color:d.dim})}),g=e=>e>=0xe8d4a51000?(e/0xe8d4a51000).toFixed(1)+`T`:e>=1e9?(e/1e9).toFixed(1)+`B`:(e/1e6).toFixed(0)+`M`;return(0,O.jsxs)(O.Fragment,{children:[(0,O.jsx)(`canvas`,{...f}),(0,O.jsxs)(h,{children:[(0,O.jsx)(m,{label:`Compute log₁₀ FLOPs`,min:19,max:26,step:.1,value:e,onChange:t,fmt:e=>`10^`+e.toFixed(1)}),(0,O.jsx)(m,{label:`Your model log₁₀ N`,min:7,max:13,step:.05,value:n,onChange:r,fmt:e=>g(10**e)})]}),(0,O.jsxs)(_,{children:[`Compute-optimal: `,(0,O.jsxs)(`b`,{children:[`N* = `,g(l)]}),` params on `,(0,O.jsxs)(`b`,{children:[`D* = `,g(u)]}),` tokens (`,(0,O.jsx)(`b`,{children:(u/l).toFixed(0)}),` tokens/param), predicted loss `,(0,O.jsx)(`b`,{children:V(l,u).toFixed(3)}),`. Your choice: N = `,g(o),`, D = `,g(c),` → loss `,(0,O.jsx)(`b`,{className:V(o,c)-V(l,u)>.05?`w`:`g`,children:V(o,c).toFixed(3)}),` (+`,(V(o,c)-V(l,u)).toFixed(3),`). Parametric fit L = E + A/N^α + B/D^β with the constants of Hoffmann et al. (2022); alternative fits give tokens/param ratios nearer 20 — the shape of the trade-off is the lesson. `,(0,O.jsx)(`i`,{children:`Inference-aware`}),` labs deliberately `,(0,O.jsx)(`b`,{children:`over-train`}),` smaller models (Llama-3-8B saw ~15T tokens) to cut serving cost.`]})]})}function U(){return(0,O.jsx)(g,{views:[{id:`rope`,label:`RoPE (rotary positions)`,render:()=>(0,O.jsx)(P,{})},{id:`moe`,label:`Mixture of Experts`,render:()=>(0,O.jsx)(R,{})},{id:`lora`,label:`LoRA / QLoRA`,render:()=>(0,O.jsx)(z,{})},{id:`scale`,label:`Scaling laws`,render:()=>(0,O.jsx)(H,{})}]})}var W={Lab:U,vizTitle:`RoPE, expert routing, low-rank adapters and compute-optimal scaling`,tryIt:[`In **RoPE**, change both positions by the same shift — the logit does not change. Change the *difference* and it does.`,`In **MoE**, crank *router imbalance* up: some experts overflow (yellow border), tokens get dropped, and the load-balancing loss rises above 1.`,`In **LoRA**, compare full fine-tuning memory for a 7B-class model with QLoRA at rank 16.`,`In **Scaling laws**, move your model size away from the optimum at fixed compute and watch the loss penalty.`],theory:x,math:S,practice:C,code:[{title:`RoPE, RMSNorm, SwiGLU, GQA — with a relative-position test (NumPy)`,lang:`python`,src:w},{title:`Top-k Mixture-of-Experts layer with load-balancing loss (PyTorch)`,lang:`python`,src:T},{title:`LoRA linear layer with merge (PyTorch)`,lang:`python`,src:E},{title:`Direct Preference Optimization (DPO) loss`,lang:`python`,src:D}],quiz:[{q:`What property makes RoPE attractive?`,options:[`It adds no compute`,`The attention logit depends only on relative position $n-m$`,`It removes the softmax`,`It makes keys orthogonal`],answer:1,why:`Rotating $q$ and $k$ by $m\\theta$ and $n\\theta$ makes $\\langle R_m q,R_nk\\rangle$ a function of $n-m$ only.`},{q:`In a Mixture-of-Experts layer with $E$ experts and top-$k$ routing, per-token compute scales with:`,options:[`$E$`,`$k$`,`$E\\cdot k$`,`$1/E$`],answer:1,why:`Only $k$ experts execute per token; memory still holds all $E$.`},{q:`LoRA fine-tunes $W$ by learning:`,options:[`A sparse mask`,`A low-rank update $\\Delta W=BA$ while $W$ stays frozen`,`A smaller vocabulary`,`Larger heads`],answer:1,why:`Trainable parameters drop from $d^2$ to $2dr$ per matrix.`},{q:`Chinchilla's main message for a fixed training-compute budget:`,options:[`Always make the model larger`,`Scale parameters and tokens roughly in proportion`,`More data is useless`,`Use bigger batches`],answer:1,why:`Earlier models were under-trained: they had too many parameters for their token count.`},{q:`Grouped-query attention reduces:`,options:[`Training FLOPs by 50%`,`KV-cache size by sharing K/V heads across query heads`,`Vocabulary size`,`Layer count`],answer:1,why:`$n_{kv}<n_{heads}$ shrinks the cache by $n_{heads}/n_{kv}$ with small quality loss.`}]};export{W as default};