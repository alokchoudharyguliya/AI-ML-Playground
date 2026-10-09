import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{_ as r,f as i,g as a,h as o,i as s,m as c,n as l,r as u,t as d,y as f}from"./viz-CPys2405.js";import{a as p,c as m,i as h,o as g,t as _,u as v}from"./hooks-Dw7oo1m3.js";var y=e(t(),1),b=`## The idea: a protected memory with gates

The vanilla RNN rewrites its whole state at every step through a matrix multiply and a squashing non-linearity — a recipe for forgetting. Hochreiter & Schmidhuber (1997) added a separate **cell state** $c_t$ and **gates** that decide what to forget, what to write and what to reveal:

| Gate | Question it answers | Range |
|---|---|---|
| **Forget** $f_t$ | How much of the old memory do I keep? | $(0,1)$ per unit |
| **Input** $i_t$ | How much of the new candidate do I write? | $(0,1)$ |
| **Candidate** $\\tilde g_t$ | What would I write? | $(-1,1)$ |
| **Output** $o_t$ | How much of the memory do I expose as $h_t$? | $(0,1)$ |

$$c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde g_t,\\qquad h_t=o_t\\odot\\tanh(c_t)$$

<div class="callout">

**Key idea — the constant error carousel.** The cell state is updated **additively**. When $f\\approx1$ and $i\\approx0$, $c_t=c_{t-1}$ and the gradient flows back unchanged: $\\partial c_t/\\partial c_{t-1}=\\mathrm{diag}(f_t)$. No repeated weight-matrix products, so no exponential decay.

</div>

## Reading the lab

The top panel is the cell at one time step; line thickness encodes how open each gate is. The bottom panel is the whole sequence:

- A pulse of 1.0 arrives at $t=3$. The **input gate** opens, the candidate $\\tilde g\\approx\\tanh(2)$ is written into $c$.
- With a **high forget bias** the cell state stays flat afterwards — a long-term memory. With a **low** one it decays like the vanilla RNN (dashed).
- With **noise**, an input gate that opens too easily pollutes memory; learned gating must stay selective.

## Why the details work

- **Gates are soft switches** (sigmoid), hence differentiable; the network *learns* when to remember or erase.
- **Forget-bias initialisation** at ≈1 starts the model in "remember" mode, which makes early training dramatically easier.
- **Separate $c$ and $h$**: $c$ is internal memory (unbounded), $h$ is a squashed, gated read-out. This lets the model keep information that is not currently needed.
- Four gates ⇒ **4× the parameters** of a vanilla RNN of the same width.

## Variants

- **Peephole LSTM** — gates also see $c$.
- **Bidirectional LSTM** — forward and backward passes concatenated; the workhorse of pre-Transformer NLP (ELMo, NER, speech).
- **Stacked / residual LSTMs** — depth for capacity; residual links beyond 4–8 layers.
- **Projected LSTM (LSTMP)**, **LayerNorm-LSTM**, **xLSTM** (2024) with exponential gating and matrix memory — a modern revival.

## What LSTMs were used for

Machine translation (GNMT), speech recognition, handwriting, language modelling (AWD-LSTM), time-series forecasting, anomaly detection, reinforcement-learning policies with memory (A3C-LSTM, OpenAI Five). Today Transformers dominate language, but LSTMs remain sensible for **small data, streaming, low-latency and edge** settings, and are still common in forecasting.

## Practical knobs

- Gradient clipping (norm 0.25–5) is still needed — gates fix vanishing, not exploding gradients.
- Dropout between layers, **variational** dropout across time, weight-drop on $U$.
- Use cuDNN fused kernels (\`nn.LSTM\`) — a hand-written cell loop is 5–10× slower.
- Hidden size 256–1024; layers 1–4; embedding tying for language models.
`,x=`## Equations

With $\\sigma$ the logistic sigmoid and $[h_{t-1};x_t]$ the concatenated input:

$$
\\begin{aligned}
f_t&=\\sigma\\!\\big(W_f x_t+U_f h_{t-1}+b_f\\big)\\\\
i_t&=\\sigma\\!\\big(W_i x_t+U_i h_{t-1}+b_i\\big)\\\\
o_t&=\\sigma\\!\\big(W_o x_t+U_o h_{t-1}+b_o\\big)\\\\
\\tilde g_t&=\\tanh\\!\\big(W_g x_t+U_g h_{t-1}+b_g\\big)\\\\
c_t&=f_t\\odot c_{t-1}+i_t\\odot\\tilde g_t\\\\
h_t&=o_t\\odot\\tanh(c_t)
\\end{aligned}
$$

Parameters for hidden size $n$ and input size $d$:

$$
4\\,\\big(n\\,d+n^2+n\\big).
$$

(PyTorch stores two bias vectors per gate, giving $4(nd+n^2+2n)$.)

## Gradient flow along the cell state

Differentiating $c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde g_t$ with respect to $c_{t-1}$ (treating gates as functions of $h_{t-1}$ gives extra terms, which are second-order and small in practice):

$$
\\frac{\\partial c_t}{\\partial c_{t-1}}=\\mathrm{diag}(f_t)+\\underbrace{\\cdots}_{\\text{via }h_{t-1}} .
$$

Therefore

$$
\\frac{\\partial c_T}{\\partial c_k}\\approx\\prod_{t=k+1}^{T}\\mathrm{diag}(f_t)=\\mathrm{diag}\\Big(\\prod_t f_t\\Big).
$$

If the forget gate for a unit sits at $f=0.99$, the signal after 100 steps is $0.99^{100}\\approx0.37$ — versus $\\approx10^{-30}$ for a vanilla RNN with factor $0.5$ per step. If $f=0.9$ the memory half-life is $\\ln2/\\ln(1/0.9)\\approx6.6$ steps. **Time-scale of a memory unit** $\\tau=-1/\\ln f$.

## Backward pass (BPTT through an LSTM)

Given $dh_t$ and $dc_t$ from the future:

$$
\\begin{aligned}
do_t&=dh_t\\odot\\tanh(c_t), & dc_t&\\mathrel{+}=dh_t\\odot o_t\\odot(1-\\tanh^2c_t)\\\\
df_t&=dc_t\\odot c_{t-1}, & di_t&=dc_t\\odot\\tilde g_t,\\quad d\\tilde g_t=dc_t\\odot i_t\\\\
dc_{t-1}&=dc_t\\odot f_t &&
\\end{aligned}
$$

and each gate pre-activation gradient is multiplied by its local derivative: $\\sigma'(a)=\\sigma(a)(1-\\sigma(a))$ for $f,i,o$ and $1-\\tilde g^2$ for the candidate. The weight gradients are sums over time of $d a_t\\,[x_t;h_{t-1}]^{\\top}$.

## Information-theoretic view of gates

A gate value $g\\in(0,1)$ interpolates "keep" and "replace". For the forget gate, $b_f$ sets the *prior* probability of retention $\\sigma(b_f)$: $b_f=1\\Rightarrow0.73$, $b_f=3\\Rightarrow0.95$, $b_f=5\\Rightarrow0.993$.

## Chrono initialisation

To give units a spread of memory time scales up to $T_{max}$ (Tallec & Ollivier, 2018), draw

$$
b_f\\sim\\log\\big(\\mathcal U[1,T_{max}-1]\\big),\\qquad b_i=-b_f .
$$

so $\\sigma(b_f)\\approx1-1/T$ and the cell begins with "leaky integrator" dynamics across a range of horizons.
`,S=`## Exercises

**Q1.** Count the parameters of a single-layer LSTM with input size 128 and hidden size 512 (PyTorch convention with two biases).

<details>
<summary>Show answer</summary>

$4\\,(nd+n^2+2n)=4\\,(512\\cdot128+512^2+1024)=4\\,(65{,}536+262{,}144+1{,}024)=1{,}314{,}816$. A vanilla RNN of the same size has a quarter of that.

</details>

**Q2.** A memory unit has a constant forget gate $f=0.95$ and input gate 0. After how many steps does the stored value fall to 10%?

<details>
<summary>Show answer</summary>

$0.95^k=0.1\\Rightarrow k=\\ln0.1/\\ln0.95\\approx45$ steps. Time constant $\\tau=-1/\\ln0.95\\approx19.5$.

</details>

**Q3.** Why is setting the forget-gate bias to about 1 (or higher) recommended at initialisation?

<details>
<summary>Show answer</summary>

With $b_f=0$ the initial $f\\approx0.5$, so memory halves every step and gradients vanish before learning begins. $b_f=1$–$3$ gives $f=0.73$–$0.95$, so information (and gradient) survives long enough for the network to discover that it is useful.

</details>

**Q4.** Design a 1-bit memory: using the gate equations, describe how an LSTM can "write" a bit on a trigger and keep it indefinitely.

<details>
<summary>Show answer</summary>

Make $f\\approx1$ when no trigger (bias high), $i\\approx0$ (bias low). On the trigger input, a large input weight pushes $i\\to1$ and $f\\to0$ so the old value is erased and the candidate $\\tilde g=\\pm1$ (set by another input weight) is written. After the trigger, gates return to their defaults and $c$ persists unchanged. This is essentially what the lab with the second pulse does.

</details>

**Q5.** When would you still choose an LSTM over a Transformer today?

<details>
<summary>Show answer</summary>

Small data; streaming inference with O(1) memory per step; tight latency or memory budgets on-device; short noisy time series where recurrence is a good inductive bias; online learning. For large-scale language/vision pre-training, Transformers (or modern SSMs) win.

</details>

**Q6 (code).** Compare \`nn.LSTM\` and a Python loop over \`LSTMCell\` for speed.

<details>
<summary>Show answer</summary>

The fused cuDNN kernel in \`nn.LSTM\` runs the whole sequence in one launch and is typically 5–20× faster than a Python loop on GPU — see the kernel-launch overhead discussion in the CUDA track.

</details>

## In practice

- **Forecasting**: LSTM encoder over a window → linear head; standardise per series; predict multiple horizons directly rather than recursively when possible.
- **Anomaly detection**: train an LSTM autoencoder / predictor on normal data and flag high reconstruction or prediction error.
- **Speech/handwriting**: bidirectional LSTMs with CTC loss.
- **Text classification**: embeddings → BiLSTM → max/mean pooling → linear; still a strong small-data baseline.
- **Deployment**: fuse to ONNX/TensorRT; int8 quantisation works well for LSTMs but keep the cell state in higher precision.

## Common pitfalls

- Using the last time-step output of a *padded* batch (take the output at the true length or use packed sequences).
- Forgetting to reset hidden state between independent sequences; or resetting it every batch in stateful training.
- Not clipping gradients.
- Initialising the forget bias to zero.
- Applying batch norm across time (use layer norm).
`,C=`import torch
import torch.nn as nn


class LSTMCellScratch(nn.Module):
    """One fused matmul computes all four gates (this is how cuDNN does it)."""

    def __init__(self, d, n):
        super().__init__()
        self.n = n
        self.W = nn.Linear(d, 4 * n)                 # x -> [f, i, g, o]
        self.U = nn.Linear(n, 4 * n, bias=False)     # h -> [f, i, g, o]
        with torch.no_grad():
            self.W.bias.zero_()
            self.W.bias[:n] = 1.0                    # forget-gate bias = 1

    def forward(self, x, state):
        h, c = state
        z = self.W(x) + self.U(h)
        f, i, g, o = z.chunk(4, dim=-1)
        f, i, o = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g                            # additive memory update
        h = o * torch.tanh(c)
        return h, (h, c)


def run(cell, xs):
    B, T, _ = xs.shape
    h = c = xs.new_zeros(B, cell.n)
    outs = []
    for t in range(T):
        y, (h, c) = cell(xs[:, t], (h, c))
        outs.append(y)
    return torch.stack(outs, 1), (h, c)


if __name__ == "__main__":
    torch.manual_seed(0)
    d, n, B, T = 8, 16, 4, 12
    ours = LSTMCellScratch(d, n)
    ref = nn.LSTM(d, n, batch_first=True)

    # copy weights: PyTorch orders the gates (i, f, g, o); ours is (f, i, g, o)
    def reorder(w):
        i, f, g, o = w.chunk(4, 0)
        return torch.cat([f, i, g, o], 0)

    with torch.no_grad():
        ref.weight_ih_l0.copy_(reorder(ours.W.weight))
        ref.weight_hh_l0.copy_(reorder(ours.U.weight))
        ref.bias_ih_l0.copy_(reorder(ours.W.bias))
        ref.bias_hh_l0.zero_()

    x = torch.randn(B, T, d)
    y1, _ = run(ours, x)
    y2, _ = ref(x)
    print("max |scratch - nn.LSTM| =", (y1 - y2).abs().max().item())
`,w=`import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, vocab, emb=128, hidden=256, classes=2, layers=2, bidirectional=True, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=0.3, bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out, classes)

    def forward(self, tokens, lengths):
        x = self.drop(self.embed(tokens))
        # pack so padding never touches the recurrent state
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)          # (B, T, H)

        # masked mean + max pooling over valid time steps
        mask = (torch.arange(out.size(1), device=out.device)[None] < lengths[:, None]).unsqueeze(-1)
        mean = (out * mask).sum(1) / lengths[:, None]
        mx = out.masked_fill(~mask, float("-inf")).max(1).values
        return self.fc(self.drop(mean + mx))


if __name__ == "__main__":
    model = LSTMClassifier(vocab=5000)
    tokens = torch.randint(1, 5000, (4, 20))
    lengths = torch.tensor([20, 15, 9, 5])
    for i, L in enumerate(lengths):
        tokens[i, L:] = 0                                           # padding
    print(model(tokens, lengths).shape)                              # (4, 2)
`,T=n(),E=32;function D(e,t){let n=o(5);return Array.from({length:E},(r,i)=>+!!t.includes(i)+(e?.25*c(n):0))}function O(e,t,n,i){let a=0,o=0,s=0,c=[];for(let l=0;l<E;l++){let u=e[l],d=r(.5*u+.5*a+t),f=r(3*u+.5*a+n),p=r(1*u+.5*a+i),m=Math.tanh(2*u+.5*a),h=o,g=a;o=d*o+f*m,a=p*Math.tanh(o),s=Math.tanh(2*u+.9*s),c.push({x:u,f:d,i:f,o:p,g:m,c:o,h:a,hr:s,cPrev:h,hPrev:g})}return c}function k(){let[e,t]=(0,y.useState)(3),[n,r]=(0,y.useState)(-2),[o,c]=(0,y.useState)(0),[b,x]=(0,y.useState)(!1),[S,C]=(0,y.useState)(!1),[w,k]=(0,y.useState)(8),A=(0,y.useMemo)(()=>D(b,S?[3,16]:[3]),[b,S]),j=(0,y.useMemo)(()=>O(A,e,n,o),[A,e,n,o]),M=j[w],[N]=_(300,(e,t,n)=>{let r=t-150,i=n-24,o=150+(r-150)*.18,s=150+(r-150)*.42,c=150+(r-150)*.58,p=150+(r-150)*.86;e.strokeStyle=d.line,e.lineWidth=1.5,e.setLineDash([5,4]),a(e,120,20,r-150+60,n-40,16),e.stroke(),e.setLineDash([]),2+Math.min(6,Math.abs(M.cPrev)*3),u(e,20,56,r+90,56,d.b,3,9),f(e,`c(t−1) = ${M.cPrev.toFixed(2)}`,26,40,{size:12,mono:!0,color:d.b}),f(e,`c(t) = ${M.c.toFixed(2)}`,t-20,40,{size:12,mono:!0,color:d.b,align:`right`});let m=(t,n,r,i,o=`σ`)=>{e.fillStyle=l(i,.12+.45*Math.abs(r)),a(e,t-34,168,68,44,10),e.fill(),e.strokeStyle=i,e.lineWidth=1.5,e.stroke(),f(e,`${o} ${n}`,t,183,{size:11,align:`center`,color:d.mute}),f(e,r.toFixed(2),t,198,{size:14,align:`center`,mono:!0,color:`#fff`,weight:700})};m(o,`forget`,M.f,d.d),m(s,`input`,M.i,d.c),m(c,`cand.`,M.g,d.a,`tanh`),m(p,`output`,M.o,d.e);let h=(t,n,r,i)=>{e.fillStyle=d.panel2,e.strokeStyle=i,e.lineWidth=2,e.beginPath(),e.arc(t,n,14,0,7),e.fill(),e.stroke(),f(e,r,t,n+1,{size:17,align:`center`,color:`#fff`,weight:700})};u(e,o,168,o,71,d.d,1+4*M.f,7),h(o,56,`×`,d.d);let g=(s+c)/2;u(e,s,168,s,124,d.c,1+4*M.i,6),u(e,c,168,c,124,d.a,1+4*Math.abs(M.g),6),e.strokeStyle=d.mute,e.lineWidth=1.2,e.beginPath(),e.moveTo(s,124),e.lineTo(s,112),e.lineTo(c,112),e.lineTo(c,124),e.stroke(),u(e,g,112,g,71,d.mute,2,7),h(g,112,`×`,d.c),h((c+p)/2-20,56,`+`,d.b);let _=p;u(e,_,168,_,108,d.e,1+4*M.o,7),h(_,94,`×`,d.e),e.strokeStyle=d.b,e.lineWidth=1.4,e.beginPath(),e.moveTo(_-52,56),e.lineTo(_-52,94),e.lineTo(_-14,94),e.stroke(),f(e,`tanh`,_-52,78,{size:10.5,align:`center`,color:d.mute}),u(e,_+14,94,t-24,94,d.a,2.4,8),f(e,`h(t) = ${M.h.toFixed(2)}`,t-20,78,{size:12,mono:!0,color:d.a,align:`right`}),e.strokeStyle=d.dim,e.lineWidth=1.5,e.beginPath(),e.moveTo(40,i),e.lineTo(p+40,i),e.stroke(),[o,s,c,p].forEach(t=>{e.beginPath(),e.moveTo(t,i),e.lineTo(t,212),e.stroke()}),f(e,`h(t−1) = ${M.hPrev.toFixed(2)}   x(t) = ${M.x.toFixed(2)}`,40,i-12,{size:12,mono:!0,color:d.ink}),f(e,`t = ${w}`,t-20,n-26,{size:12,mono:!0,color:d.mute,align:`right`})}),[P]=_(280,(e,t,n)=>{let r=t-40-16,a=n-48,o=e=>40+e/31*r,c=e=>16+a/2-s(e,-1.5,1.5)/1.5*(a/2);e.strokeStyle=d.grid,e.lineWidth=1,[-1,0,1].forEach(t=>{e.beginPath(),e.moveTo(40,c(t)),e.lineTo(40+r,c(t)),e.stroke(),f(e,t,32,c(t),{size:10.5,align:`right`,color:d.dim,mono:!0})});for(let t=0;t<E;t+=4)f(e,t,o(t),16+a+14,{size:10.5,align:`center`,color:d.dim,mono:!0});j.forEach((t,n)=>{e.fillStyle=`rgba(255,255,255,.18)`;let r=s(t.x,-1.5,1.5)/1.5*(a/2);e.fillRect(o(n)-3,r>0?c(0)-r:c(0),6,Math.abs(r))});let l=(t,n,r,a)=>i(e,j.map((e,n)=>[o(n),c(e[t])]),n,r,a);l(`hr`,`rgba(255,255,255,.75)`,1.6,[4,4]),l(`f`,d.d,1.4),l(`i`,d.c,1.4),l(`o`,d.e,1.4),l(`h`,d.a,2.2),l(`c`,d.b,3),e.strokeStyle=`rgba(255,255,255,.35)`,e.setLineDash([3,3]),e.beginPath(),e.moveTo(o(w),16),e.lineTo(o(w),16+a),e.stroke(),e.setLineDash([])}),F=j[3].c,I=j[28].c;return(0,T.jsxs)(T.Fragment,{children:[(0,T.jsx)(`canvas`,{...N}),(0,T.jsx)(`canvas`,{...P}),(0,T.jsx)(p,{items:[[d.b,`cell state c`],[d.a,`hidden h`],[d.d,`forget f`],[d.c,`input i`],[d.e,`output o`],[`rgba(255,255,255,.7)`,`vanilla RNN h`],[`rgba(255,255,255,.3)`,`input x`]]}),(0,T.jsxs)(h,{children:[(0,T.jsx)(m,{label:`Forget bias b_f`,min:-4,max:6,step:.25,value:e,onChange:t,fmt:e=>e.toFixed(2)}),(0,T.jsx)(m,{label:`Input bias b_i`,min:-5,max:4,step:.25,value:n,onChange:r,fmt:e=>e.toFixed(2)}),(0,T.jsx)(m,{label:`Output bias b_o`,min:-4,max:4,step:.25,value:o,onChange:c,fmt:e=>e.toFixed(2)}),(0,T.jsx)(m,{label:`Inspect step t`,min:0,max:31,value:w,onChange:k}),(0,T.jsx)(v,{label:`Noisy input`,value:b,onChange:x}),(0,T.jsx)(v,{label:`Second pulse at t=16`,value:S,onChange:C})]}),(0,T.jsxs)(g,{children:[`Cell state after the pulse (t=3): `,(0,T.jsx)(`b`,{children:F.toFixed(2)}),` → at t=28: `,(0,T.jsx)(`b`,{className:I>.3?`g`:`r`,children:I.toFixed(2)}),`. The vanilla RNN (dashed) forgets within a few steps; the LSTM keeps what it wrote as long as `,(0,T.jsx)(`span`,{className:`w`,children:`f ≈ 1`}),` and `,(0,T.jsx)(`span`,{className:`p`,children:`i ≈ 0`}),` in between.`]})]})}var A={Lab:k,vizTitle:`Inside the cell: gates, the cell-state highway and memory over time`,tryIt:[`Set **forget bias = −4**: the cell forgets almost instantly, like a vanilla RNN. Set it to **+5**: it holds the pulse for the whole sequence.`,`Raise the **input bias** and enable the **noisy input**: now junk gets written into memory — selective writing needs a *closed* input gate by default.`,`Turn on the **second pulse** to watch a new value overwrite the old one when the forget gate is low.`,`Slide the inspect step and read the four gate values driving the diagram.`],theory:b,math:x,practice:S,code:[{title:`An LSTM cell from scratch (and checked against nn.LSTM)`,lang:`python`,src:C},{title:`Sequence classifier with LSTM: packing, bidirectionality, pooling`,lang:`python`,src:w}],quiz:[{q:`Why is the LSTM cell-state update $c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde g_t$ good for gradients?`,options:[`It uses ReLU`,`The path through $c$ is additive, so $\\partial c_t/\\partial c_{t-1}=\\mathrm{diag}(f_t)$ — no repeated weight-matrix multiplication`,`It removes the need for training`,`It is linear in the input`],answer:1,why:`Gradients flow along the cell state multiplied only by forget-gate values, which the network can keep near 1.`},{q:`Common LSTM initialization trick:`,options:[`Zero forget bias`,`Forget-gate bias ≈ 1`,`All biases −10`,`Orthogonal biases`],answer:1,why:`A positive $b_f$ starts the network remembering by default (Jozefowicz et al., 2015).`},{q:`How many weight matrices pairs ($W$, $U$) does an LSTM layer have relative to a vanilla RNN?`,options:[`1×`,`2×`,`3×`,`4×`],answer:3,why:`Four transformations: forget, input, output gates and the candidate.`},{q:`Which gate controls what is exposed to the next layer?`,options:[`Forget`,`Input`,`Output`,`Candidate`],answer:2,why:`$h_t=o_t\\odot\\tanh(c_t)$ — the output gate filters the cell state.`},{q:`Peephole connections let gates see:`,options:[`The next input`,`The cell state`,`The loss`,`The batch`],answer:1,why:`Gers & Schmidhuber added $c_{t-1}$ (or $c_t$ for the output gate) as an input to the gates.`}]};export{A as default};