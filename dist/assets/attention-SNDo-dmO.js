import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{g as r,h as i,i as a,l as o,n as s,o as c,p as l,t as u,v as d,y as f}from"./viz-CPys2405.js";import{c as p,i as m,l as h,o as g,s as _,t as v,u as y}from"./hooks-Dw7oo1m3.js";var b=e(t(),1),x=`## The encoder–decoder bottleneck

Sequence-to-sequence (Sutskever et al., 2014; Cho et al., 2014) solved variable-length input → variable-length output (translation, summarisation, speech → text) with two RNNs:

1. an **encoder** reads the source and produces a final state $h_T$;
2. a **decoder** is initialised from $h_T$ and emits the target one token at a time.

Everything the decoder knows about the source must pass through that **one fixed-size vector**. For a 40-word sentence that is hopeless: early words are overwritten, and translation quality collapses as sentences get longer. Toggle *bottleneck* in the lab to see the shape of the problem.

## The attention fix

Bahdanau, Cho & Bengio (2014) let the decoder look back at **all** encoder states $h_1,\\dots,h_S$ and build a *fresh* context vector for every output word:

$$c_t=\\sum_{i=1}^{S}\\alpha_{ti}\\,h_i,\\qquad\\alpha_t=\\mathrm{softmax}\\big(\\mathrm{score}(s_{t-1},h_i)\\big)$$

The weights $\\alpha_{ti}$ — "how much should target word $t$ care about source word $i$?" — are computed on the fly and **learned end-to-end** with no alignment supervision. The result (the heat-map in the lab) is a soft **word alignment**: “chat” → *cat*, “noir” → *black* even though the order is swapped.

<div class="callout">

**Key idea — attention is a differentiable dictionary lookup.** There is a *query* (what the decoder needs right now), a set of *keys* (what each source position offers) and *values* (the content to retrieve). Similarity of query and key gives weights; the output is the weighted average of the values. In the second lab tab you can literally drag the query and watch the weighted average move.

</div>

## Score functions

| Name | Score $e_{ti}$ | Notes |
|---|---|---|
| Additive (Bahdanau) | $v^{\\top}\\tanh(W s_{t-1}+Uh_i)$ | small MLP; works at any dimension |
| Dot product (Luong) | $s_t^{\\top}h_i$ | cheapest; needs equal dims |
| General (bilinear) | $s_t^{\\top}Wh_i$ | learned similarity |
| **Scaled dot product** | $s_t^{\\top}h_i/\\sqrt{d}$ | the Transformer's choice |

Dot-product variants map onto a matrix multiplication — hugely GPU-friendly — which is a quiet reason they won.

## Design variations

- **Global vs local attention** (Luong 2015): attend to all source positions, or only a window around a predicted alignment point.
- **Input feeding**: feed the previous attentional vector back into the decoder input so it knows what it looked at.
- **Coverage** penalties discourage repeating or ignoring source words.
- **Pointer / copy networks**: attention weights directly define a distribution over *copying* source tokens (names, numbers, rare words).
- **Attention over images** (Show, Attend and Tell): attend to CNN grid cells while captioning; **over audio** (Listen, Attend and Spell).
- **Hard attention** samples a single position (needs REINFORCE); **soft attention** is differentiable and standard.

## What attention gave us

1. **No bottleneck** — long sentences work.
2. **Shorter gradient paths** — the loss at target step $t$ reaches source step $i$ in one hop instead of $|t-i|$ recurrent steps (a cousin of the LSTM gating solution to vanishing gradients).
3. **Interpretability (partial)** — alignments are visible, though attention weights are not guaranteed faithful explanations.
4. **A building block** — if attention is so good at *routing information*, why keep the recurrence? The answer, *Attention Is All You Need* (2017), is the next chapter.

## Decoding and training notes

- **Teacher forcing** at train time (ground-truth previous token), **autoregressive generation** at inference — mismatch known as *exposure bias*.
- **Beam search** keeps the $k$ best partial hypotheses; length-normalise scores to avoid favouring short outputs.
- **Padding masks** in the attention softmax are mandatory (set padded scores to $-\\infty$).
- **Label smoothing** and **dropout** are standard regularisers; BLEU/chrF/COMET evaluate translation quality.
- **Complexity**: attention adds $O(S\\cdot T)$ memory/compute per sentence pair — negligible for sentences, but it foreshadows the quadratic cost of self-attention on long contexts.
`,S=`## Encoder

A bidirectional RNN produces annotations $h_i=[\\overrightarrow h_i;\\overleftarrow h_i]$ for source tokens $x_1,\\dots,x_S$.

## Attention step (Bahdanau)

For decoder step $t$ with previous state $s_{t-1}$:

$$
e_{ti}=v^{\\top}\\tanh\\!\\big(W_a s_{t-1}+U_a h_i\\big),\\qquad
\\alpha_{ti}=\\frac{\\exp(e_{ti})}{\\sum_{k=1}^{S}\\exp(e_{tk})},\\qquad
c_t=\\sum_{i=1}^{S}\\alpha_{ti}\\,h_i .
$$

The decoder then updates and predicts:

$$
s_t=f\\big(s_{t-1},\\,y_{t-1},\\,c_t\\big),\\qquad
p(y_t\\mid y_{<t},x)=\\mathrm{softmax}\\big(W_o[\\,s_t;\\,c_t;\\,E y_{t-1}]\\big).
$$

## Luong variants

$$
\\mathrm{score}(s_t,h_i)=\\begin{cases}
s_t^{\\top}h_i & \\text{dot}\\\\
s_t^{\\top}W_ah_i & \\text{general}\\\\
v^{\\top}\\tanh(W_a[s_t;h_i]) & \\text{concat}
\\end{cases}
\\qquad
\\tilde s_t=\\tanh\\!\\big(W_c[c_t;s_t]\\big).
$$

Luong attention uses the *current* decoder state $s_t$ (compute attention after the RNN step); Bahdanau uses $s_{t-1}$ (before it).

## Matrix form

Stack $H\\in\\mathbb R^{S\\times d}$ and queries $Q\\in\\mathbb R^{T\\times d}$:

$$
\\alpha=\\mathrm{softmax}\\!\\Big(\\frac{QH^{\\top}}{\\sqrt d}\\Big)\\in\\mathbb R^{T\\times S},\\qquad C=\\alpha H\\in\\mathbb R^{T\\times d}.
$$

Cost: $O(TSd)$ time and $O(TS)$ memory for the weights.

## Masking

With padding mask $m_i\\in\\{0,1\\}$:

$$
\\alpha_{ti}=\\frac{m_i\\,\\exp(e_{ti})}{\\sum_k m_k\\exp(e_{tk})}\\;=\\;\\mathrm{softmax}(e_{ti}-\\infty\\cdot(1-m_i)).
$$

## Softmax derivative

For $\\alpha=\\mathrm{softmax}(e)$,

$$
\\frac{\\partial\\alpha_i}{\\partial e_j}=\\alpha_i(\\delta_{ij}-\\alpha_j),
$$

so the gradient flowing into the scores is $\\partial L/\\partial e_j=\\alpha_j\\big(g_j-\\sum_i\\alpha_ig_i\\big)$ with $g_i=\\partial L/\\partial\\alpha_i$. When $\\alpha$ is nearly one-hot the Jacobian vanishes — the saturation problem that motivates the $1/\\sqrt d$ scaling (see *The Transformer*).

## Temperature view

Scaling scores by $1/T$ interpolates between a uniform average ($T\\to\\infty$) and hard argmax ($T\\to0$):

$$
\\alpha_i(T)=\\frac{e^{e_i/T}}{\\sum_ke^{e_k/T}} .
$$

The lab's entropy readout is $H(\\alpha)=-\\sum_i\\alpha_i\\log_2\\alpha_i$, between $0$ (one-hot) and $\\log_2S$ (uniform).

## Beam search score

Length-normalised log-probability with penalty $\\lambda$ (GNMT):

$$
s(Y)=\\frac{\\log p(Y\\mid X)}{lp(Y)},\\qquad lp(Y)=\\frac{(5+|Y|)^{\\lambda}}{(5+1)^{\\lambda}} .
$$

## Training objective

Teacher-forced maximum likelihood:

$$
\\mathcal L=-\\sum_{t=1}^{T}\\log p\\big(y_t^{*}\\mid y^{*}_{<t},x\\big),
$$

optionally with label smoothing $\\epsilon$: replace the one-hot target by $(1-\\epsilon)\\,\\mathbb 1_{y^*}+\\epsilon/V$.
`,C=`## Exercises

**Q1.** An encoder produces $S=5$ states in $\\mathbb R^2$: $h_1=(1,0)$, $h_2=(0,1)$, $h_3=(1,1)$, $h_4=(-1,0)$, $h_5=(0,-1)$. The decoder query is $s=(2,0)$ (dot-product scores, $T=1$). Compute the weights approximately and the context vector.

<details>
<summary>Show answer</summary>

Scores $e=s\\cdot h=(2,0,2,-2,0)$. $\\exp$: $(7.39,1,7.39,0.135,1)$, sum $=16.92$. $\\alpha\\approx(0.437,0.059,0.437,0.008,0.059)$. Context $c=\\sum\\alpha_ih_i\\approx(0.437+0.437-0.008,\\ 0.059+0.437-0.059)=(0.866,\\,0.437)$. It sits between $h_1$ and $h_3$, the two best matches.

</details>

**Q2.** What happens to the attention weights as $T\\to0$ and $T\\to\\infty$? What does the second lab view show for large "sharpness"?

<details>
<summary>Show answer</summary>

$T\\to0$: one-hot on the argmax — hard attention, zero entropy, vanishing gradients to the other positions. $T\\to\\infty$: uniform average, entropy $\\log_2S$ — the model loses selectivity. Sharpness in the lab is $1/T$: at large values the star (context) snaps to the nearest key.

</details>

**Q3.** Why must attention scores of padding tokens be set to $-\\infty$ *before* the softmax rather than zeroing the weights afterwards?

<details>
<summary>Show answer</summary>

Zeroing after the softmax leaves the remaining weights unnormalised (they no longer sum to one) and padding would still have stolen probability mass. Masking before the softmax renormalises over the real tokens only. Use a large negative number (e.g. $-10^9$ or \`finfo.min\`) rather than literal $-\\infty$ if an entire row can be masked, to avoid \`NaN\`.

</details>

**Q4.** The attention matrix for a source of 100 tokens and a target of 100 tokens at batch 32 with fp16 weights: how much memory?

<details>
<summary>Show answer</summary>

$32\\cdot100\\cdot100\\cdot2$ bytes $=640$ KB per head per layer — trivial here. For 32k-token self-attention it becomes $32\\cdot32768^2\\cdot2\\approx68$ GB per head — which is why FlashAttention (CUDA track) avoids materialising it.

</details>

**Q5.** Name two ways attention weights can mislead as an explanation of model behaviour.

<details>
<summary>Show answer</summary>

(1) Different attention distributions can yield the same prediction (weights are not unique explanations). (2) Information flows through values and later layers too; a token with low weight can still matter and a high weight can be irrelevant. Use gradient-based or perturbation methods for attribution.

</details>

**Q6 (code).** Extend \`attention_numpy.py\` to add a causal mask and verify that row $t$ of the weight matrix has zeros for all positions $>t$.

<details>
<summary>Show answer</summary>

Add \`mask = np.triu(np.ones((T, T), bool), 1)\`; set \`scores[mask] = -1e9\` before the softmax; assert \`np.allclose(weights[mask], 0)\`. This is exactly the decoder-side mask in GPT-style models.

</details>

## In practice

- Pre-2017 production MT systems (GNMT, early Google Translate NMT) used LSTM encoder–decoders with attention; today's translation models are Transformers, but the *cross-attention* layer of an encoder-decoder Transformer is this same mechanism.
- **Cross-attention is everywhere**: text-conditioned diffusion models attend from image latents to prompt tokens; Whisper attends from text decoder to audio encoder; retrieval-augmented models attend to retrieved passages.
- **Debugging**: plot attention heat-maps for a few validation sentences — a diagonal band is healthy for monotonic tasks like speech recognition; smeared or collapsed maps signal training problems.
- **Monotonic / location-aware attention** helps speech synthesis (Tacotron) where alignment must progress left to right.

## Common pitfalls

- Forgetting the source padding mask (the model learns to attend to padding).
- Computing the softmax over the wrong axis (over targets instead of sources).
- Leaking future target tokens during training (no causal mask in the decoder).
- Using attention weights as an "explanation" without ablation.
- Beam search without length normalisation → overly short outputs.
`,w=`import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab, emb, hid, pad=0):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=pad)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hid, hid)

    def forward(self, src):
        out, h = self.rnn(self.embed(src))               # out: (B, S, 2H)
        h0 = torch.tanh(self.fc(torch.cat([h[0], h[1]], -1)))   # initial decoder state
        return out, h0


class BahdanauAttention(nn.Module):
    """e_ti = v^T tanh(W s + U h_i)"""

    def __init__(self, hid):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=False)
        self.U = nn.Linear(2 * hid, hid, bias=False)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, s, enc, mask):
        # s: (B, H)  enc: (B, S, 2H)  mask: (B, S) True for real tokens
        e = self.v(torch.tanh(self.W(s).unsqueeze(1) + self.U(enc))).squeeze(-1)   # (B, S)
        e = e.masked_fill(~mask, float("-inf"))                                     # ignore padding
        alpha = F.softmax(e, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), enc).squeeze(1)                     # (B, 2H)
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab, emb, hid, pad=0):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=pad)
        self.attn = BahdanauAttention(hid)
        self.cell = nn.GRUCell(emb + 2 * hid, hid)
        self.out = nn.Linear(hid + 2 * hid + emb, vocab)

    def step(self, y_prev, s, enc, mask):
        e = self.embed(y_prev)
        context, alpha = self.attn(s, enc, mask)
        s = self.cell(torch.cat([e, context], -1), s)
        logits = self.out(torch.cat([s, context, e], -1))
        return logits, s, alpha


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb=256, hid=512, pad=0):
        super().__init__()
        self.pad = pad
        self.enc, self.dec = Encoder(src_vocab, emb, hid, pad), Decoder(tgt_vocab, emb, hid, pad)

    def forward(self, src, tgt):                  # teacher forcing
        mask = src != self.pad
        enc, s = self.enc(src)
        logits = []
        for t in range(tgt.size(1) - 1):          # predict tgt[t+1] from tgt[t]
            lg, s, _ = self.dec.step(tgt[:, t], s, enc, mask)
            logits.append(lg)
        return torch.stack(logits, 1)             # (B, T-1, V)

    @torch.no_grad()
    def greedy(self, src, bos, eos, max_len=50):
        mask = src != self.pad
        enc, s = self.enc(src)
        y = torch.full((src.size(0),), bos, device=src.device)
        out, attn = [], []
        for _ in range(max_len):
            lg, s, a = self.dec.step(y, s, enc, mask)
            y = lg.argmax(-1)
            out.append(y); attn.append(a)
            if (y == eos).all():
                break
        return torch.stack(out, 1), torch.stack(attn, 1)     # tokens, alignments (B, T, S)


if __name__ == "__main__":
    model = Seq2Seq(1000, 1000, emb=64, hid=128)
    src = torch.randint(1, 1000, (4, 9)); src[:, 7:] = 0     # padded
    tgt = torch.randint(1, 1000, (4, 11))
    logits = model(src, tgt)
    loss = F.cross_entropy(logits.reshape(-1, 1000), tgt[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.1)
    print(logits.shape, float(loss))
`,T=`import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None, scale=True):
    """Q: (T, d)  K: (S, d)  V: (S, dv)  mask: (T, S) True = may attend."""
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V, weights


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = S = 6
    Q, K = rng.standard_normal((T, 8)), rng.standard_normal((S, 8))
    V = rng.standard_normal((S, 4))

    out, w = attention(Q, K, V)
    print("rows sum to 1:", np.allclose(w.sum(-1), 1), " output shape:", out.shape)

    causal = np.tril(np.ones((T, S), bool))          # position t sees only <= t
    out_c, w_c = attention(Q, K, V, mask=causal)
    print("causal: weights above the diagonal are zero:", np.allclose(w_c[~causal], 0))

    # attention is a soft dictionary lookup: a very sharp query returns (almost) one value
    q = K[2:3] * 20
    out_q, w_q = attention(q, K, V, scale=False)
    print("sharp query selects key 2:", w_q.argmax(), " weight:", round(float(w_q.max()), 4))
`,E=n(),D=[`The`,`black`,`cat`,`sat`,`on`,`the`,`mat`,`.`],O=[`Le`,`chat`,`noir`,`s’est`,`assis`,`sur`,`le`,`tapis`,`.`],k=[0,2,1,3,3,4,5,6,7],A=(()=>{let e=i(21);return O.map((t,n)=>D.map((t,r)=>(r===k[n]?6:Math.abs(r-k[n])===1?1.8:0)+(e()-.5)))})();A[1][1]+=2.2;function j(){let[e,t]=(0,b.useState)(2),[n,i]=(0,b.useState)(1),[a,h]=(0,b.useState)(!1),_=(0,b.useMemo)(()=>A.map(e=>d(e,n)),[n]),x=(0,b.useRef)([]),[S]=v(380,(t,n,i)=>{let c=n*.6,l=Math.min(56,c/D.length-6),d=e=>12+(e+.5)*((c-12)/D.length),p=e=>12+(e+.5)*((c-12)/O.length),m=i-96;f(t,`ENCODER — source (English)`,12,18,{size:11,color:u.mute,weight:600}),f(t,`DECODER — target (French), click a word`,12,i-18,{size:11,color:u.mute,weight:600});let h=_[e];if(x.current=[],a){let e=c/2,n=(70+m)/2;D.forEach((r,i)=>{t.strokeStyle=`rgba(251,191,36,.35)`,t.lineWidth=1.2,t.beginPath(),t.moveTo(d(i),86),t.lineTo(e,n-16),t.stroke()}),O.forEach((r,i)=>{t.strokeStyle=`rgba(251,191,36,.35)`,t.beginPath(),t.moveTo(e,n+16),t.lineTo(p(i),m-16),t.stroke()}),t.fillStyle=s(u.d,.2),r(t,e-70,n-16,140,32,8),t.fill(),t.strokeStyle=u.d,t.lineWidth=1.5,t.stroke(),f(t,`one fixed vector`,e,n,{size:12,align:`center`,color:u.d,weight:600})}else D.forEach((n,r)=>{t.strokeStyle=s(u.c,.18+.82*h[r]),t.lineWidth=.8+9*h[r],t.beginPath(),t.moveTo(p(e),m-16),t.bezierCurveTo(p(e),(m+70)/2,d(r),(m+70)/2,d(r),86),t.stroke()});D.forEach((e,n)=>{let i=a?.25:.12+.88*h[n];t.fillStyle=s(u.c,i),r(t,d(n)-l/2,54,l,32,8),t.fill(),t.strokeStyle=u.line,t.lineWidth=1,t.stroke(),f(t,e,d(n),70,{size:13,align:`center`,color:`#fff`,weight:600}),a||f(t,h[n].toFixed(2),d(n),42,{size:11,align:`center`,mono:!0,color:h[n]>.15?u.c:u.dim})}),O.forEach((n,i)=>{let a=i===e;t.fillStyle=a?s(u.b,.35):u.panel2,r(t,p(i)-l/2,m-16,l,32,8),t.fill(),t.strokeStyle=a?u.b:u.line,t.lineWidth=a?2:1,t.stroke(),f(t,n,p(i),m,{size:13,align:`center`,color:`#fff`,weight:600}),x.current.push([p(i)-l/2,m-16,l,32,i])});let g=c+36,v=n-g-14,y=Math.min(v/D.length,(i-60-50)/O.length);f(t,`Attention matrix α[target, source]`,g,24,{size:11,color:u.mute,weight:600});for(let e=0;e<O.length;e++)for(let n=0;n<D.length;n++)t.fillStyle=a?`#141830`:o(_[e][n]),t.fillRect(g+n*y,60+e*y,y-1,y-1);t.strokeStyle=u.b,t.lineWidth=2,t.strokeRect(g-1,60+e*y-1,D.length*y+1,y),D.forEach((e,n)=>f(t,e,g+n*y+y/2,50,{size:9.5,align:`center`,color:u.mute})),O.forEach((n,r)=>f(t,n,g-6,60+r*y+y/2,{size:9.5,align:`right`,color:r===e?u.b:u.mute}))}),C=_[e],w=C.indexOf(Math.max(...C));return(0,E.jsxs)(E.Fragment,{children:[(0,E.jsx)(`canvas`,{...S,onPointerDown:e=>{let n=l(e),r=x.current.find(([e,t,r,i])=>n.x>=e&&n.x<=e+r&&n.y>=t&&n.y<=t+i);r&&t(r[4])},style:{...S.style,cursor:`pointer`}}),(0,E.jsxs)(m,{children:[(0,E.jsx)(p,{label:`Softmax temperature`,min:.3,max:4,step:.1,value:n,onChange:i,fmt:e=>e.toFixed(1)}),(0,E.jsx)(y,{label:`No attention (fixed-vector bottleneck)`,value:a,onChange:h})]}),(0,E.jsx)(g,{children:a?(0,E.jsxs)(E.Fragment,{children:[`Without attention the decoder must reconstruct `,(0,E.jsx)(`b`,{children:`every`}),` word from a single fixed-size vector — information about early words is crushed as sentences get longer.`]}):(0,E.jsxs)(E.Fragment,{children:[`Generating “`,(0,E.jsx)(`b`,{children:O[e]}),`”: context c = Σ αᵢ·hᵢ is dominated by “`,(0,E.jsx)(`b`,{children:D[w]}),`” (α = `,C[w].toFixed(2),`). Entropy `,(0,E.jsx)(`b`,{children:c(C).toFixed(2)}),` bits — a low temperature focuses, a high one blurs toward a uniform average.`]})})]})}var M=[[`the`,-.75,.45],[`black`,.1,.85],[`cat`,.75,.45],[`sat`,.65,-.45],[`on`,-.1,-.8],[`mat`,-.7,-.45],[`.`,0,0]];function N(){let[e,t]=(0,b.useState)([.55,.3]),[n,i]=(0,b.useState)(`dot`),[o,c]=(0,b.useState)(4),h=(0,b.useRef)(!1),y=M.map(([,t,r])=>n===`dot`?o*(e[0]*t+e[1]*r):-o*((e[0]-t)**2+(e[1]-r)**2)),x=d(y),S=[x.reduce((e,t,n)=>e+t*M[n][1],0),x.reduce((e,t,n)=>e+t*M[n][2],0)],[C]=v(380,(t,i,a)=>{let c=Math.min(i*.34,a/2-20),l=i*.3,d=a/2,p=(e,t)=>[l+e*c,d-t*c];t.strokeStyle=u.grid,t.lineWidth=1,t.beginPath(),t.moveTo(l-c*1.1,d),t.lineTo(l+c*1.1,d),t.moveTo(l,d-c*1.1),t.lineTo(l,d+c*1.1),t.stroke(),t.beginPath(),t.arc(l,d,c,0,7),t.stroke();let[m,h]=p(e[0],e[1]);M.forEach(([e,n,r],i)=>{let[a,o]=p(n,r);t.strokeStyle=s(u.c,.1+.9*x[i]),t.lineWidth=.6+8*x[i],t.beginPath(),t.moveTo(m,h),t.lineTo(a,o),t.stroke()}),M.forEach(([e,n,r],i)=>{let[a,o]=p(n,r);t.fillStyle=u.a,t.beginPath(),t.arc(a,o,6+8*x[i],0,7),t.fill(),f(t,e,a,o-18-6*x[i],{size:12,align:`center`,color:`#fff`,weight:600})});let[g,_]=p(S[0],S[1]);t.fillStyle=u.d,t.beginPath();for(let e=0;e<10;e++){let n=e%2?5:11,r=e*Math.PI/5-Math.PI/2;t.lineTo(g+n*Math.cos(r),_+n*Math.sin(r))}t.closePath(),t.fill(),f(t,`context c`,g+14,_+4,{size:11,color:u.d,mono:!0}),t.fillStyle=u.b,t.beginPath(),t.moveTo(m,h-10),t.lineTo(m+10,h),t.lineTo(m,h+10),t.lineTo(m-10,h),t.closePath(),t.fill(),f(t,`query q (drag me)`,m+14,h-14,{size:11,color:u.b,mono:!0});let v=i*.62,y=i-v-60;f(t,`α = softmax(${n===`dot`?`q·k`:`−‖q−k‖²`} × ${o})`,v,24,{size:12,color:u.mute,weight:600}),M.forEach(([e],n)=>{let i=50+n*36;f(t,e,v,i+13,{size:12,color:`#fff`,weight:600}),t.fillStyle=u.panel2,r(t,v+48,i,y-40,26,6),t.fill(),t.fillStyle=u.c,r(t,v+48,i,Math.max(2,(y-40)*x[n]),26,6),t.fill(),f(t,x[n].toFixed(2),v+48+y-34,i+13,{size:11,mono:!0,color:u.ink})})}),w=e=>{if(!h.current)return;let n=C.ref.current.getBoundingClientRect(),r=Math.min(n.width*.34,n.height/2-20),i=l(e);t([a((i.x-n.width*.3)/r,-1.1,1.1),a(-(i.y-n.height/2)/r,-1.1,1.1)])};return(0,E.jsxs)(E.Fragment,{children:[(0,E.jsx)(`canvas`,{...C,style:{...C.style,cursor:`grab`,touchAction:`none`},onPointerDown:e=>{h.current=!0,e.currentTarget.setPointerCapture(e.pointerId),w(e)},onPointerMove:w,onPointerUp:()=>{h.current=!1}}),(0,E.jsxs)(m,{children:[(0,E.jsx)(_,{label:`Score function`,value:n,onChange:i,options:[[`dot`,`Dot product (Luong / Transformer)`],[`dist`,`Negative squared distance (RBF-like)`]]}),(0,E.jsx)(p,{label:`Sharpness`,min:1,max:14,step:.5,value:o,onChange:c})]}),(0,E.jsxs)(g,{children:[`Attention is a `,(0,E.jsx)(`b`,{children:`soft nearest-neighbour lookup`}),`: move the query and the weights re-distribute; the output (★) is the weighted average of the values. Higher sharpness → closer to a hard `,(0,E.jsx)(`i`,{children:`argmax`}),` lookup.`]})]})}function P(){return(0,E.jsx)(h,{views:[{id:`align`,label:`Alignment (translation)`,render:()=>(0,E.jsx)(j,{})},{id:`geo`,label:`Attention as soft lookup`,render:()=>(0,E.jsx)(N,{})}]})}var F={Lab:P,vizTitle:`See which source words the decoder looks at`,tryIt:[`Click **“noir”** (black) — attention jumps backwards to “black”, showing word-order reordering between languages.`,`Click **“s’est assis”** words: both point to “sat” — one source word aligned to several target words.`,`Lower the temperature to sharpen the alignment; raise it to watch attention blur into an average.`,`Toggle the **bottleneck** to see the encoder–decoder design attention was invented to fix.`,`In the second view, drag the query across the plane to see the weighted average (★) follow it.`],theory:x,math:S,practice:C,code:[{title:`Seq2Seq with Bahdanau attention (PyTorch)`,lang:`python`,note:`GRU encoder, additive attention, GRU decoder with teacher forcing and padding masks.`,src:w},{title:`Attention in 25 lines of NumPy (with masking)`,lang:`python`,src:T}],quiz:[{q:`What problem did attention solve in encoder–decoder RNNs?`,options:[`Slow training`,`Compressing the entire source into a single fixed-size vector`,`Overfitting`,`Vocabulary size`],answer:1,why:`Attention lets the decoder access *all* encoder states at every step instead of one summary vector.`},{q:`In $c_t=\\sum_i\\alpha_{ti}h_i$ the weights $\\alpha_{ti}$:`,options:[`Are learned parameters fixed after training`,`Are computed on the fly from the decoder state and encoder states and sum to 1`,`Are one-hot`,`Are independent of the input`],answer:1,why:`$\\alpha_t=\\mathrm{softmax}(e_t)$ with scores that depend on the current decoder state and each $h_i$.`},{q:`Bahdanau (additive) vs Luong (multiplicative) attention differ mainly in:`,options:[`The score function`,`The loss`,`The optimizer`,`The tokenizer`],answer:0,why:`Additive: $v^\\top\\tanh(Ws+Uh)$; multiplicative: $s^\\top Wh$ or $s^\\top h$.`},{q:`What do padding masks do in attention?`,options:[`Speed up softmax`,`Set scores of padded positions to $-\\infty$ so they receive zero weight`,`Normalize embeddings`,`Add noise`],answer:1,why:`Otherwise the model would attend to meaningless padding tokens.`},{q:`Why is attention called "interpretable-ish"?`,options:[`It proves causality`,`The weights give a visible alignment between target and source, but are not guaranteed explanations`,`It has no parameters`,`It uses rules`],answer:1,why:`Alignments are informative but attention weights alone are not faithful explanations (Jain & Wallace, 2019).`}]};export{F as default};