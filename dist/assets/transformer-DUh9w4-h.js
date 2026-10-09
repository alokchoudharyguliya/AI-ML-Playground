import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{a as r,c as i,f as a,h as o,i as s,l as c,m as l,o as u,p as d,t as f,v as p,y as m}from"./viz-CPys2405.js";import{c as h,i as g,l as _,o as v,s as y,t as b,u as x}from"./hooks-Dw7oo1m3.js";var S=e(t(),1),C=`## "Attention is all you need"

Vaswani et al. (2017) asked: if attention is what makes seq2seq work, why keep the recurrence? The **Transformer** drops RNNs entirely. Every token looks at every other token in a *single parallel step*, so

- training parallelises over the whole sequence (RNNs are sequential in time),
- any two positions are connected by a **path of length 1** (RNN: up to $n$), easing long-range credit assignment,
- compute is dominated by large matrix multiplications — exactly what GPUs and Tensor Cores are built for.

This match between the model and the hardware is a large part of why Transformers scaled to GPT-4-class models.

## Self-attention in one sentence

Every token emits a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a **value** ("what I will hand over"). Token $i$'s output is the average of all values, weighted by how well its query matches each key:

$$\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\!\\Big(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\Big)V$$

In the first tab of the lab each row of the heat-map is one token's attention distribution, computed exactly from random $q,k$ vectors. Notice what the **$1/\\sqrt{d_k}$** does: without it, the scores grow like $\\sqrt{d_k}$, the softmax collapses to a one-hot, and its gradient dies.

<div class="callout">

**Mental model.** Attention is a *content-addressable memory read*: the token composes a query, the sequence is the memory, and the result is a similarity-weighted blend of memory contents. Stack many layers and many heads and the network can route, copy, compare and compose information between positions.

</div>

## Multi-head attention

One softmax can only express one "pattern" of dependence per token. **Multi-head attention** runs $h$ attentions in parallel in lower-dimensional subspaces ($d_{head}=d/h$), then concatenates and mixes with $W_O$. Different heads specialise: local syntax, previous-token copying, coreference, delimiter "sinks", induction heads (the core of in-context learning). Total cost and parameters are the same as one big head.

## The Transformer block

A modern (pre-LN) block is two residual sub-layers:

$$x\\leftarrow x+\\mathrm{MHA}(\\mathrm{LN}(x)),\\qquad x\\leftarrow x+\\mathrm{FFN}(\\mathrm{LN}(x))$$

- **Residual stream**: the vector $x$ per token is a shared communication channel; each layer reads from it and *adds* to it. This is the same skip-connection idea as ResNet and the LSTM cell state.
- **LayerNorm** stabilises scale per token (RMSNorm in modern LLMs). *Pre-LN* trains stably at depth; the original *post-LN* needs careful warmup.
- **Feed-forward network**: $\\mathrm{FFN}(x)=W_2\\,\\sigma(W_1x)$ with hidden size $4d$ (or SwiGLU with $\\tfrac83d$). It holds ~2/3 of the parameters and acts like a key–value *memory* of facts and features; attention moves information *between* tokens, the FFN transforms it *within* each token.

## Three architectures from the same parts

| Family | Mask | Examples | Use |
|---|---|---|---|
| **Encoder-only** | bidirectional | BERT, RoBERTa, DeBERTa | classification, retrieval, embeddings |
| **Decoder-only** | causal | GPT, Llama, Claude-style LLMs | generation, chat, code |
| **Encoder–decoder** | bidirectional enc, causal dec + **cross-attention** | T5, BART, Whisper, original Transformer | translation, summarisation, speech |

**Cross-attention** takes queries from the decoder and keys/values from the encoder — exactly the Bahdanau attention of the previous chapter, now multi-head.

## Position information

Self-attention treats its input as a *set*: shuffle the tokens and the output merely shuffles. Order must be injected:

- **Sinusoidal** (original): fixed sines/cosines at geometric wavelengths added to embeddings (third lab tab).
- **Learned absolute** (BERT, GPT-2): a trainable table; cannot extrapolate beyond training length.
- **Relative** (T5 bias, Transformer-XL) and **ALiBi**: add a distance-dependent bias to scores.
- **RoPE** (Llama, most modern LLMs): rotate $q,k$ by position-dependent angles — see *Modern LLM Toolkit*.

## Cost and the long-context problem

Self-attention builds an $n\\times n$ score matrix: **$O(n^2)$** compute *and* (naively) memory. At $n=32$k and fp16 a single head's matrix is 2 GB. Remedies: **FlashAttention** (exact, IO-aware tiling — see the CUDA track), sparse/sliding-window attention, linear attention, KV-cache tricks (GQA/MQA), and state-space hybrids.

## Training recipe (what actually matters)

- **Warmup + cosine/linear decay**, AdamW, $\\beta_2=0.95$–$0.999$, weight decay 0.1, gradient clip 1.0.
- **Tokenisation** (BPE / SentencePiece) and **data quality/quantity** dominate results.
- **Label smoothing** (translation), **dropout** 0–0.1 (often 0 for LLMs), **mixed precision** (bf16).
- **Scaling**: loss follows power laws in parameters, data and compute (see the last DL chapter).

## Beyond text

The same block, with different tokenisation, powers **ViT** (image patches), **Whisper/audio**, **AlphaFold's** Evoformer, diffusion Transformers (**DiT**), video models and multimodal LLMs. "Token" is just a vector.
`,w=`## Scaled dot-product attention

For a sequence of $n$ tokens with embeddings $X\\in\\mathbb R^{n\\times d}$:

$$
Q=XW_Q,\\quad K=XW_K,\\quad V=XW_V,\\qquad
\\mathrm{Attn}(X)=\\mathrm{softmax}\\!\\Big(\\frac{QK^{\\top}}{\\sqrt{d_k}}+M\\Big)V
$$

where $M_{ij}=-\\infty$ if position $j$ must be hidden from $i$ (causal: $j>i$; padding), else $0$. Row $i$ of the softmax is a probability distribution over keys.

## Why $\\sqrt{d_k}$?

Take $q,k\\in\\mathbb R^{d_k}$ with independent zero-mean, unit-variance entries. Then

$$
\\mathbb E[q\\cdot k]=0,\\qquad \\mathrm{Var}(q\\cdot k)=\\sum_{m=1}^{d_k}\\mathrm{Var}(q_mk_m)=d_k .
$$

The standard deviation $\\sqrt{d_k}$ grows with dimension; large-magnitude logits push the softmax into saturation where the Jacobian $\\mathrm{diag}(p)-pp^{\\top}\\to0$ and gradients vanish. Dividing by $\\sqrt{d_k}$ restores unit variance (verify with the lab's readout).

## Multi-head attention

$$
\\mathrm{head}_i=\\mathrm{Attn}\\big(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)}\\big),\\qquad
\\mathrm{MHA}(X)=\\mathrm{Concat}(\\mathrm{head}_1,\\dots,\\mathrm{head}_h)\\,W_O ,
$$

with $W^{(i)}_{Q,K,V}\\in\\mathbb R^{d\\times d/h}$ and $W_O\\in\\mathbb R^{d\\times d}$. Parameters: $4d^2$ regardless of $h$.

## Position-wise feed-forward network

$$
\\mathrm{FFN}(x)=W_2\\,\\sigma(W_1x+b_1)+b_2,\\quad W_1\\in\\mathbb R^{d_{ff}\\times d},\\ d_{ff}=4d,
$$

giving $2\\cdot d\\cdot4d=8d^2$ parameters. With **SwiGLU** ($W_2(\\mathrm{SiLU}(W_gx)\\odot W_1x)$ with $d_{ff}\\approx\\tfrac83d$) it is $3\\cdot d\\cdot\\tfrac83d=8d^2$ as well.

## Layer normalisation

$$
\\mathrm{LN}(x)=\\gamma\\odot\\frac{x-\\mu(x)}{\\sqrt{\\sigma^2(x)+\\epsilon}}+\\beta,\\qquad
\\mathrm{RMSNorm}(x)=\\gamma\\odot\\frac{x}{\\sqrt{\\tfrac1d\\sum_ix_i^2+\\epsilon}} .
$$

## Sinusoidal positional encoding

$$
PE_{(pos,2i)}=\\sin\\!\\Big(\\frac{pos}{10000^{2i/d}}\\Big),\\qquad
PE_{(pos,2i+1)}=\\cos\\!\\Big(\\frac{pos}{10000^{2i/d}}\\Big).
$$

For each frequency $\\omega_i=10000^{-2i/d}$, the pair at position $pos+k$ is a rotation of the pair at $pos$:

$$
\\begin{pmatrix}\\sin\\omega(p+k)\\\\ \\cos\\omega(p+k)\\end{pmatrix}=
\\begin{pmatrix}\\cos\\omega k&\\sin\\omega k\\\\-\\sin\\omega k&\\cos\\omega k\\end{pmatrix}
\\begin{pmatrix}\\sin\\omega p\\\\ \\cos\\omega p\\end{pmatrix},
$$

a **linear function of $PE_p$** that depends only on the offset $k$ — the justification for relative-position generalisation. The similarity $PE_p\\cdot PE_{p+k}=\\sum_i\\cos(\\omega_ik)$ decays with $|k|$ (second panel of the lab).

## Parameter and FLOP counts

For $L$ layers, width $d$, vocabulary $V$, and ignoring biases/norms:

$$
N\\approx\\underbrace{12\\,L\\,d^2}_{\\text{blocks}}+\\underbrace{V\\,d}_{\\text{embeddings}} .
$$

Training FLOPs per token $\\approx6N$ (2N forward + 4N backward); inference $\\approx2N$ plus attention's $\\approx 4\\,L\\,n\\,d$ per token which grows with context. Example: $d=4096,\\ L=32$ → $12\\cdot32\\cdot4096^2\\approx6.4$ B block parameters (+ embeddings → ~7 B).

| Operation | FLOPs (forward, per layer) | Memory |
|---|---|---|
| QKV + output projections | $8nd^2$ | $O(nd)$ |
| Attention scores + weighted sum | $4n^2d$ | $O(n^2h)$ (naive) |
| FFN | $16nd^2$ | $O(nd_{ff})$ |

Attention dominates once $n\\gtrsim6d$ (since $4n^2d>24nd^2\\Leftrightarrow n>6d$) — tens of thousands of tokens for typical widths.

## Softmax gradient and saturation

$$
\\frac{\\partial p_i}{\\partial s_j}=p_i(\\delta_{ij}-p_j),\\qquad \\Big\\|\\frac{\\partial p}{\\partial s}\\Big\\|_F\\to0\\ \\text{ as }p\\to\\text{one-hot}.
$$

## Cross-attention

$$
\\mathrm{CrossAttn}(Y,X_{enc})=\\mathrm{softmax}\\!\\Big(\\frac{(YW_Q)(X_{enc}W_K)^{\\top}}{\\sqrt{d_k}}\\Big)X_{enc}W_V,
$$

queries from the decoder $Y$, keys and values from the encoder output.

## Language-model objective

$$
\\mathcal L=-\\frac1n\\sum_{i=1}^{n}\\log p_\\theta(x_i\\mid x_{<i}),\\qquad p_\\theta=\\mathrm{softmax}(h_i\\,E^{\\top}),
$$

with output projection often **tied** to the input embedding matrix $E$.
`,T=`## Exercises

**Q1.** A decoder-only model has $d=2048$, $L=24$, vocabulary $50{,}000$ (tied embeddings). Estimate the parameter count.

<details>
<summary>Show answer</summary>

Blocks: $12\\,L\\,d^2=12\\cdot24\\cdot2048^2=1.208\\times10^9$. Embeddings: $50{,}000\\cdot2048=1.02\\times10^8$ (tied, counted once). Total $\\approx1.31$ B.

</details>

**Q2.** Estimate the attention-score matrix memory (fp16) for one layer with 16 heads at sequence length 8192, batch 4, if materialised naively. What does FlashAttention store instead?

<details>
<summary>Show answer</summary>

$4\\cdot16\\cdot8192^2\\cdot2$ bytes $=8.6$ GB per layer (before the softmax copy!). FlashAttention keeps only $O(n)$ statistics per row (running max and sum — the logsumexp) and recomputes tiles in the backward pass, so memory is linear in $n$.

</details>

**Q3.** You remove the positional encoding from a Transformer encoder and feed the sentence "dog bites man" vs "man bites dog". What can the model tell apart?

<details>
<summary>Show answer</summary>

Nothing about order: self-attention is permutation-equivariant, so the pooled/bag representation is identical. (Decoder models with causal masks can still infer *some* order implicitly, because token $i$ sees exactly $i$ predecessors — "NoPE" works to a degree — but explicit position is far better.)

</details>

**Q4.** Show that the FFN has twice the parameters of the attention block.

<details>
<summary>Show answer</summary>

Attention: $W_Q,W_K,W_V,W_O\\Rightarrow4d^2$. FFN: $d\\cdot4d+4d\\cdot d=8d^2$. So FFN : attention $=2:1$, i.e. the FFN holds two-thirds of a block's weights.

</details>

**Q5.** With $d_k=64$ unscaled scores have std ≈8. If the largest score exceeds the others by 16, what is the maximum softmax weight roughly, and why does it hurt training?

<details>
<summary>Show answer</summary>

$e^{16}\\approx8.9\\times10^{6}$ vs $(n-1)$ other terms of order 1 → weight $\\approx1-10^{-5}$. The Jacobian $p_i(1-p_i)\\approx10^{-5}$: gradients to the scores vanish, so the model cannot learn to *change* its attention.

</details>

**Q6 (code).** Implement causal masking via \`-inf\` and check that \`F.scaled_dot_product_attention(..., is_causal=True)\` matches.

<details>
<summary>Show answer</summary>

\`sdpa_check.py\` in the Code tab builds the mask with \`torch.triu(torch.ones(T,T),1).bool()\`, applies \`masked_fill(-inf)\` before softmax and asserts \`allclose\` against the fused kernel.

</details>

## In practice

- **Use the fused kernels.** \`F.scaled_dot_product_attention\` dispatches to FlashAttention / memory-efficient kernels; hand-written softmax(QKᵀ)V is 2–4× slower and memory hungry.
- **Shapes to remember**: \`(B, H, T, d_head)\` for Q/K/V; keep \`d_head ∈ {64, 128}\` for Tensor-Core efficiency.
- **Pre-LN + residual scaling** (or μP-style init) for stable deep training; QK-norm for very large models.
- **Inference**: cache keys/values (next chapters), use GQA, batch with continuous batching (vLLM), quantise weights (INT8/INT4).
- **Vision**: ViT splits images into 16×16 patches → linear embedding → Transformer; needs more data than CNNs unless pre-trained or heavily augmented.
- **Probing a model**: attention rollout, logit-lens, activation patching; remember that attention weights alone are not explanations.

## Common pitfalls

- Applying softmax over the wrong dimension.
- Forgetting the causal mask during training (the model "cheats" and perplexity looks implausibly good).
- Mixing up batch-first and sequence-first conventions in \`nn.MultiheadAttention\`.
- Position indices off by one at inference when using a KV cache.
- fp16 overflow in attention logits — compute softmax in fp32 or use bf16.
- Sequence lengths beyond the trained context with learned absolute positions.
`,E=`import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)   # fused W_Q, W_K, W_V
        self.out = nn.Linear(d_model, d_model, bias=False)       # W_O
        self.drop = dropout

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, H, T, d_head)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))

        # scores = q k^T / sqrt(d_head), causal mask, softmax, weighted sum of v
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        att = F.dropout(scores.softmax(-1), self.drop, self.training)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)             # concat heads
        return self.out(y)


class Block(nn.Module):
    """Pre-LN residual block: x + MHA(LN(x)),  x + FFN(LN(x))."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d_model=256, n_heads=4, n_layers=4, max_len=512, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)                  # learned absolute positions
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight                         # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=50, temperature=1.0, top_k=None):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.pos.num_embeddings:])
            logits = logits[:, -1] / temperature
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], 1)
        return idx


if __name__ == "__main__":
    model = GPT(vocab=1000)
    x = torch.randint(0, 1000, (2, 16))
    logits, loss = model(x, x)
    print(logits.shape, float(loss), "(expected ~ln(1000) =", round(math.log(1000), 2), ")")
    print("parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
`,D=`import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, T, D = 2, 4, 128, 64
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))


def manual(q, k, v, causal=True):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        mask = torch.triu(torch.ones(q.size(-2), k.size(-2), dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(-1) @ v


ref = manual(q, k, v)
fused = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # FlashAttention / mem-efficient on GPU
print("max |manual - fused| =", (ref - fused).abs().max().item())

# why the sqrt(d) matters: score std with and without scaling
raw = (q @ k.transpose(-2, -1))
print("score std unscaled:", raw.std().item(), "  scaled:", (raw / math.sqrt(D)).std().item())

# select a backend explicitly (PyTorch >= 2.3)
if torch.cuda.is_available():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    qc, kc, vc = (t.half().cuda() for t in (q, k, v))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(qc, kc, vc, is_causal=True)
    print("flash backend ok:", out.shape)
`,O=n(),k=[`The`,`animal`,`didn't`,`cross`,`the`,`street`,`because`,`it`,`was`,`tired`],A=k.length;function j(e,t,n,r,i,a,o,s,c){t.forEach((t,o)=>t.forEach((t,s)=>{e.fillStyle=c&&c[o][s]?`#0b0d16`:a(t),e.fillRect(n+s*i,r+o*i,i-1,i-1)})),o&&(o.forEach((t,a)=>m(e,t,n+a*i+i/2,r-10,{size:9.5,align:`center`,color:f.mute})),o.forEach((t,a)=>m(e,t,n-6,r+a*i+i/2,{size:9.5,align:`right`,color:a===s?f.b:f.mute}))),s!=null&&(e.strokeStyle=f.b,e.lineWidth=2,e.strokeRect(n-1,r+s*i-1,t[0].length*i+1,i))}var M=256,N=(()=>{let e=o(42);return{q:k.map(()=>Array.from({length:M},()=>l(e))),k:k.map(()=>Array.from({length:M},()=>l(e)))}})();function P(){let[e,t]=(0,S.useState)(64),[n,r]=(0,S.useState)(!0),[i,a]=(0,S.useState)(!1),[o,s]=(0,S.useState)(7),{raw:l,P:_,mask:C}=(0,S.useMemo)(()=>{let t=k.map((t,r)=>k.map((t,i)=>{let a=0;for(let t=0;t<e;t++)a+=N.q[r][t]*N.k[i][t];return n?a/Math.sqrt(e):a})),r=k.map((e,t)=>k.map((e,n)=>i&&n>t));return{raw:t,P:t.map((e,t)=>{let n=e.map((e,n)=>r[t][n]?-1e9:e);return p(n)}),mask:r}},[e,n,i]),w=(0,S.useMemo)(()=>{let e=l.flat(),t=e.reduce((e,t)=>e+t,0)/e.length;return Math.sqrt(e.reduce((e,n)=>e+(n-t)**2,0)/e.length)},[l]),T=_[o],E=(0,S.useMemo)(()=>{let e=0;for(let t=0;t<A;t++)for(let n=0;n<A;n++){let r=(t===n?T[t]:0)-T[t]*T[n];e+=r*r}return Math.sqrt(e)},[T]),D=_.reduce((e,t)=>e+Math.max(...t),0)/A,[M]=b(420,(e,t,r)=>{let i=Math.min(34,(t*.5-90)/A,(r-66-12)/A);m(e,`Attention weights  softmax(QKᵀ`+(n?`/√d_k`:``)+`)`,90,22,{size:12,color:f.mute,weight:600}),j(e,_,90,66,i,e=>c(e),k,o,C);let a=90+A*i+50,s=t-a-24,u=Math.min(24,(r-66-10)/A);m(e,`Query “${k[o]}” — raw score  q·k${n?`/√d_k`:``}   |   weight`,a,22,{size:12,color:f.mute,weight:600});let d=Math.max(1,...l[o].map(Math.abs));k.forEach((t,n)=>{let r=66+n*u;m(e,t,a,r+u/2-1,{size:11,color:C[o][n]?f.dim:`#fff`,weight:600});let i=(s*.5-70)/2,c=a+66+i;e.strokeStyle=f.grid,e.beginPath(),e.moveTo(c,r),e.lineTo(c,r+u-2),e.stroke();let p=l[o][n]/d;e.fillStyle=C[o][n]?`#1a1e30`:p>=0?f.c:f.b,e.fillRect(Math.min(c,c+p*i),r+3,Math.abs(p*i)||1,u-8);let h=a+s*.5+40,g=s*.5-90;e.fillStyle=f.panel2,e.fillRect(h,r+3,g,u-8),e.fillStyle=f.e,e.fillRect(h,r+3,Math.max(1,g*T[n]),u-8),m(e,T[n].toFixed(2),h+g+6,r+u/2-1,{size:10.5,mono:!0,color:f.mute})})});return(0,O.jsxs)(O.Fragment,{children:[(0,O.jsx)(`canvas`,{...M,style:{...M.style,cursor:`pointer`},onPointerDown:e=>{let t=d(e),n=e.currentTarget.getBoundingClientRect(),r=Math.min(34,(n.width*.5-90)/A,(n.height-66-12)/A),i=Math.floor((t.y-66)/r);t.x>90&&t.x<90+A*r+60&&i>=0&&i<A&&s(i)}}),(0,O.jsxs)(g,{children:[(0,O.jsx)(h,{label:`Key dimension d_k`,min:4,max:256,step:4,value:e,onChange:t}),(0,O.jsx)(x,{label:`Scale by 1/√d_k`,value:n,onChange:r}),(0,O.jsx)(x,{label:`Causal mask (decoder)`,value:i,onChange:a}),(0,O.jsx)(y,{label:`Query token`,value:o,onChange:e=>s(+e),options:k.map((e,t)=>[t,`${t}: ${e}`])})]}),(0,O.jsxs)(v,{children:[`std of scores = `,(0,O.jsx)(`b`,{className:w>3?`r`:`g`,children:w.toFixed(2)}),` `,n?`(≈1 by design)`:(0,O.jsxs)(O.Fragment,{children:[`(≈√d_k = `,Math.sqrt(e).toFixed(1),`)`]}),` · mean max weight = `,(0,O.jsx)(`b`,{children:D.toFixed(2)}),` · entropy of this row = `,(0,O.jsx)(`b`,{children:u(T).toFixed(2)}),` bits · softmax Jacobian norm = `,(0,O.jsx)(`b`,{className:E<.05?`r`:`g`,children:E.toFixed(3)}),` `,E<.05&&`← saturated: gradients vanish`]})]})}function F(){let[e,t]=(0,S.useState)(64),[n,i]=(0,S.useState)(4),o=(0,S.useMemo)(()=>Array.from({length:64},(t,r)=>Array.from({length:e},(t,i)=>{let a=r/10**(2*Math.floor(i/2)*n/e);return i%2?Math.cos(a):Math.sin(a)})),[e,n]),c=(0,S.useMemo)(()=>Array.from({length:64},(t,n)=>o[32].reduce((e,t,r)=>e+t*o[n][r],0)/(e/2)),[o,e]),[l]=b(340,(t,n,i)=>{let l=n*.4,u=i-70;m(t,`PE[pos, dim]  (rows = position 0…63, columns = dimension)`,56,18,{size:12,color:f.mute,weight:600});let d=l/e,p=u/64;o.forEach((e,n)=>e.forEach((e,i)=>{t.fillStyle=r(e),t.fillRect(56+i*d,40+n*p,d+.5,p+.5)})),m(t,`pos →`,48,46,{size:10,align:`right`,color:f.dim,mono:!0}),m(t,`low freq ←  dimension  → high freq`,56+l/2,40+u+14,{size:10.5,align:`center`,color:f.dim});let h=56+l+70,g=n-h-20,_=u*.5;m(t,`similarity  PE(32)·PE(q) / (d/2)`,h,18,{size:12,color:f.mute,weight:600});let v=e=>h+e/63*g,y=e=>40+_/2-s(e,-1.1,1.1)/1.1*(_/2);t.strokeStyle=f.grid,t.beginPath(),t.moveTo(h,y(0)),t.lineTo(h+g,y(0)),t.stroke(),a(t,c.map((e,t)=>[v(t),y(e)]),f.b,2.4),t.fillStyle=f.c,t.beginPath(),t.arc(v(32),y(c[32]),4,0,7),t.fill(),m(t,`position q →`,h+g/2,40+_+14,{size:10.5,align:`center`,color:f.dim});let b=40+_+50,x=u-_-50;m(t,`single dimensions over position`,h,b-14,{size:12,color:f.mute,weight:600}),[[0,f.a],[Math.floor(e/4)*2,f.e],[Math.floor(e/2)*2-2<e?Math.min(e-2,Math.floor(e*.75/2)*2):0,f.d]].forEach(([e,n],r)=>{a(t,o.map((t,n)=>[v(n),b+x/2-t[e]*(x/2)*.9]),n,1.8),m(t,`dim ${e}`,h+g-2,b+8+r*13,{size:10.5,align:`right`,color:n,mono:!0})})});return(0,O.jsxs)(O.Fragment,{children:[(0,O.jsx)(`canvas`,{...l}),(0,O.jsxs)(g,{children:[(0,O.jsx)(h,{label:`Model dimension d`,min:16,max:128,step:16,value:e,onChange:t}),(0,O.jsx)(h,{label:`Wavelength base (10^x)`,min:2,max:5,step:.25,value:n,onChange:i,fmt:e=>`10^`+e})]}),(0,O.jsxs)(v,{children:[`Each pair of dimensions is a sine/cosine at a geometrically spaced wavelength (from 2π up to ~2π·10^4). Nearby positions have `,(0,O.jsx)(`b`,{children:`high dot-product similarity`}),` that decays with distance, and the encoding of position `,(0,O.jsx)(`i`,{children:`p+k`}),` is a `,(0,O.jsx)(`b`,{children:`fixed rotation`}),` of position `,(0,O.jsx)(`i`,{children:`p`}),` — so relative offsets are easy to learn. The Transformer adds this to the token embeddings.`]})]})}var I=32,L=(()=>{let e=o(5);return k.map(()=>Array.from({length:I},()=>l(e)))})(),R=(()=>{let e=o(77);return Array.from({length:8},()=>({wq:Array.from({length:I},()=>Array.from({length:I},()=>l(e)/Math.sqrt(I))),wk:Array.from({length:I},()=>Array.from({length:I},()=>l(e)/Math.sqrt(I)))}))})(),z=[`content`,`local window`,`previous token`,`first-token sink`,`content`,`wide local`,`previous token`,`content`];function B(e,t,n){return e===`local window`?-Math.abs(t-n)*1.2:e===`previous token`?n===t-1?4:0:e===`first-token sink`?n===0?3.5:0:e===`wide local`?-Math.abs(t-n)*.45:0}function V(){let[e,t]=(0,S.useState)(4),[n,r]=(0,S.useState)(1),[a,o]=(0,S.useState)(!0),l=I/e,d=(0,S.useMemo)(()=>Array.from({length:e},(e,t)=>{let{wq:r,wk:i}=R[t],o=L.map(e=>Array.from({length:l},(t,n)=>e.reduce((e,t,i)=>e+t*r[i][n],0))),s=L.map(e=>Array.from({length:l},(t,n)=>e.reduce((e,t,r)=>e+t*i[r][n],0)));return k.map((e,r)=>p(k.map((e,i)=>a&&i>r?-1e9:o[r].reduce((e,t,n)=>e+t*s[i][n],0)/Math.sqrt(l)*2+n*B(z[t],r,i))))}),[e,l,n,a]),[_]=b(e>4?380:225,(t,n,r)=>{let i=Math.min(e,4),a=n/i,o=Math.min((a-54)/A,(e>4?130:150)/A);d.forEach((e,n)=>{let r=n%i,l=Math.floor(n/i),d=r*a+(a-A*o)/2,p=42+l*150;m(t,`head ${n+1} · ${z[n]}`,d+A*o/2,p-24,{size:11,align:`center`,color:f.mute,weight:600}),j(t,e,d,p,o,e=>c(s(e*1.4,0,1)),null,null,null),m(t,`H=${(e.reduce((e,t)=>e+u(t),0)/A).toFixed(1)}b`,d+A*o/2,p+A*o+12,{size:10,align:`center`,color:f.dim,mono:!0})})});return(0,O.jsxs)(O.Fragment,{children:[(0,O.jsx)(`canvas`,{..._}),(0,O.jsxs)(g,{children:[(0,O.jsx)(h,{label:`Heads h`,min:1,max:8,value:e,onChange:t}),(0,O.jsx)(h,{label:`Pattern strength`,min:0,max:2,step:.1,value:n,onChange:r,fmt:e=>e.toFixed(1)}),(0,O.jsx)(x,{label:`Causal mask`,value:a,onChange:o})]}),(0,O.jsxs)(v,{children:[`d_model = `,I,` → each head works in `,(0,O.jsxs)(`b`,{children:[`d_head = `,l]}),` dims. Parameters of the attention block: 4·d_model² = `,(0,O.jsx)(`b`,{children:i(4096)}),` — `,(0,O.jsx)(`i`,{children:`independent of h`}),`. Heads are random projections plus the structural patterns trained models are known to learn (local, previous-token, first-token “sink”); the point is that `,(0,O.jsx)(`b`,{children:`different heads can specialise`}),` in different relations.`]})]})}function H(){return(0,O.jsx)(_,{views:[{id:`attn`,label:`Scaled dot-product attention`,render:()=>(0,O.jsx)(P,{})},{id:`heads`,label:`Multi-head attention`,render:()=>(0,O.jsx)(V,{})},{id:`pe`,label:`Positional encoding`,render:()=>(0,O.jsx)(F,{})}]})}var U={Lab:H,vizTitle:`Dissect self-attention, heads and positional encodings`,tryIt:[`Turn **off** the $1/\\sqrt{d_k}$ scaling and drag $d_k$ to 256: the softmax saturates (Jacobian norm → 0) — the reason the scale factor exists.`,`Enable the **causal mask**: the upper triangle goes dark — this is what makes a decoder autoregressive.`,`Click different rows of the heat-map to inspect any query token’s raw scores and weights.`,`In *multi-head*, raise heads from 1 to 8: parameter count stays fixed while patterns diversify.`,`In *positional encoding*, shrink the base to 10² and watch the long wavelengths disappear.`],theory:C,math:w,practice:T,code:[{title:`A GPT-style Transformer from scratch (PyTorch)`,lang:`python`,note:`Causal multi-head attention, pre-LN blocks, learned positions, weight-tied LM head.`,src:E},{title:`Check against F.scaled_dot_product_attention (FlashAttention backend)`,lang:`python`,src:D}],quiz:[{q:`Why divide attention scores by $\\sqrt{d_k}$?`,options:[`To save memory`,`For unit-variance scores, preventing softmax saturation and vanishing gradients`,`To make the matrix symmetric`,`To enforce causality`],answer:1,why:`If $q,k$ have unit-variance entries, $q\\cdot k$ has variance $d_k$; scaling restores variance 1.`},{q:`Self-attention is permutation-equivariant. What gives the Transformer a notion of order?`,options:[`The softmax`,`Positional encodings added to (or applied inside) attention`,`Layer norm`,`Residual connections`],answer:1,why:`Without position information, shuffling the tokens would merely shuffle the outputs.`},{q:`Parameters in the attention sublayer of a block with width $d$ (ignoring biases):`,options:[`$d^2$`,`$2d^2$`,`$4d^2$`,`$12d^2$`],answer:2,why:`$W_Q,W_K,W_V,W_O$, each $d\\times d$ — independent of the number of heads. (The FFN adds $8d^2$.)`},{q:`The compute/memory of vanilla self-attention scales with sequence length $n$ as:`,options:[`$O(n)$`,`$O(n\\log n)$`,`$O(n^2)$`,`$O(n^3)$`],answer:2,why:`Every token attends to every other: an $n\\times n$ score matrix.`},{q:`What does the causal mask do in a decoder?`,options:[`Hides padding`,`Prevents a position from attending to later positions`,`Drops 15% of tokens`,`Normalizes weights`],answer:1,why:`It sets scores for $j>i$ to $-\\infty$, so training with teacher forcing cannot "peek" at the future.`}]};export{U as default};