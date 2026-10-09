import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{_ as r,c as i,f as a,g as o,h as s,i as c,m as l,t as u,y as d}from"./viz-CPys2405.js";import{a as f,c as p,i as m,o as h,t as g,u as _}from"./hooks-Dw7oo1m3.js";var v=e(t(),1),y=`## A leaner gated cell

The Gated Recurrent Unit (Cho et al., 2014) keeps the key insight of the LSTM — **additive, gated state updates give gradients a highway** — but trims the machinery:

- **one** state vector $h_t$ (no separate cell state),
- **two** gates instead of three,
- **no output gate** (the whole state is exposed).

$$
\\begin{aligned}
z_t&=\\sigma(W_zx_t+U_zh_{t-1}+b_z) &&\\text{update gate}\\\\
r_t&=\\sigma(W_rx_t+U_rh_{t-1}+b_r) &&\\text{reset gate}\\\\
\\tilde h_t&=\\tanh\\!\\big(W_hx_t+U_h(r_t\\odot h_{t-1})+b_h\\big) &&\\text{candidate}\\\\
h_t&=(1-z_t)\\odot h_{t-1}+z_t\\odot\\tilde h_t &&\\text{interpolation}
\\end{aligned}
$$

<div class="callout">

**Key idea.** $h_t$ is a *convex combination* of the old state and a new candidate, with the mixing weight $z_t$ learned per unit. Keep ($z\\to0$) = long-term memory; overwrite ($z\\to1$) = fast adaptation. A single gate does the job of both the LSTM's forget and input gates (they are *coupled*: what you write replaces what you drop).

</div>

## What each gate means

**Update gate $z$** — "how much new information?" Low values copy the state forward and let gradients pass unchanged. Units with persistently low $z$ become long-term memory; units with high $z$ track fast-changing features. Different units learn **different time scales**.

**Reset gate $r$** — "how much of the past matters for the *next candidate*?" With $r\\to0$ the candidate ignores history and computes purely from $x_t$ (useful at sentence/segment boundaries). The reset gate acts *inside* the candidate only; it never erases $h_{t-1}$ itself — only the update gate decides that.

Play with the lab: the cyan state sits on the line between the grey old state and the violet candidate, at fraction $z$.

## GRU vs LSTM

| | LSTM | GRU |
|---|---|---|
| States | $h_t$, $c_t$ | $h_t$ |
| Gates | forget, input, output | update, reset |
| Parameters (width $n$) | $4(\\cdot)$ | $3(\\cdot)$ — 25% fewer |
| Output exposure | gated by $o_t$ | full state |
| Typical outcome | slightly better with lots of data / very long dependencies | equal or better on smaller data, faster |

Large empirical studies (Chung 2014, Jozefowicz 2015, Greff 2017) found **no consistent winner**: tune forget/update biases and regularisation first; pick GRU for speed and simplicity, LSTM when you have headroom.

## Why gating works (shared with LSTM)

Gradients through $h_t=(1-z_t)h_{t-1}+\\dots$ contain a term $\\mathrm{diag}(1-z_t)$ — an element-wise scaling, not a matrix product. When the network drives $z\\approx0$ on a unit, the gradient to earlier steps stays near 1. This is the same *constant error carousel* mechanism as in the LSTM, and the same one resurfacing in residual connections: $y=x+F(x)$ is a GRU/LSTM-style skip in depth.

## Practical guidance

- **Initialisation**: bias the update gate **negative** (e.g. −1…−2) so $z$ starts low and memory is preserved, the GRU analogue of the LSTM forget-bias trick.
- **Speed**: cuDNN GRU kernels are fused; ~25% fewer FLOPs/params than LSTM.
- **Stacking and bidirectionality** work identically to LSTMs.
- **Regularisation**: variational dropout on inputs/hidden, weight decay, layer norm inside the cell for long sequences.
- **Where used**: speech (RNN-T encoders), on-device keyword spotting, time series, music generation, early neural machine translation (the original Bahdanau attention model used a GRU).
`,b=`## Full equations

$$
\\begin{aligned}
z_t&=\\sigma\\big(W_zx_t+U_zh_{t-1}+b_z\\big)\\\\
r_t&=\\sigma\\big(W_rx_t+U_rh_{t-1}+b_r\\big)\\\\
\\tilde h_t&=\\tanh\\big(W_hx_t+U_h\\,(r_t\\odot h_{t-1})+b_h\\big)\\\\
h_t&=(1-z_t)\\odot h_{t-1}+z_t\\odot\\tilde h_t
\\end{aligned}
$$

(Some implementations — including cuDNN/PyTorch — compute the candidate as $\\tanh(W_hx_t+b_{ih}+r_t\\odot(U_hh_{t-1}+b_{hh}))$, applying the reset *after* the matrix multiply. It is cheaper because $U_hh_{t-1}$ can be batched with the other gates.)

**Convention warning.** PyTorch defines the interpolation as $h_t=(1-z_t)\\odot\\tilde h_t+z_t\\odot h_{t-1}$, i.e. its $z$ is $1-z$ of the original paper. Keep this in mind when porting weights or reading gate plots.

Parameters (width $n$, input $d$): $\\;3\\,(nd+n^2+n)$.

## Gradient through time

Differentiating the interpolation with respect to the previous state:

$$
\\frac{\\partial h_t}{\\partial h_{t-1}}=\\mathrm{diag}(1-z_t)+\\underbrace{\\mathrm{diag}(z_t)\\,\\frac{\\partial\\tilde h_t}{\\partial h_{t-1}}+\\mathrm{diag}\\big(\\tilde h_t-h_{t-1}\\big)\\frac{\\partial z_t}{\\partial h_{t-1}}}_{\\text{paths through the gates and candidate}} .
$$

For units with $z_t\\approx0$ the Jacobian is $\\approx I$, so the product over many steps stays $\\approx I$ instead of decaying as $\\rho^k$. Quantitatively, if a unit sits at $z=0.02$ for $k$ steps the retained signal is $(1-z)^k=0.98^k$: after 100 steps ≈ 13%.

## Memory time scale

With constant $z$ and no input the state decays toward the candidate with time constant

$$
\\tau=-\\frac{1}{\\ln(1-z)}\\;\\approx\\;\\frac1z\\quad(z\\ll1).
$$

Initialising $b_z\\sim-\\log\\mathcal U[1,T_{max}]$ produces a spectrum of time scales between 1 and $T_{max}$ steps (the same chrono-initialisation idea used for LSTMs).

## Relation to the LSTM

Set the LSTM's input gate to $i_t=1-f_t$ (coupled input-forget gate, CIFG), drop the output gate and peepholes, and merge $c_t$ with $h_t$: you recover the GRU update $h_t=(1-z_t)h_{t-1}+z_t\\tilde h_t$. Greff et al. showed the CIFG variant performs on par with the full LSTM, which explains why the GRU is competitive.

## Parameter-count table (single layer, $d=n$)

| Cell | Formula | $n=512$ |
|---|---|---|
| Vanilla RNN | $n(n+d)+n$ | 524,800 |
| GRU | $3\\,(n(n+d)+n)$ | 1,574,400 |
| LSTM | $4\\,(n(n+d)+n)$ | 2,099,200 |

## FLOPs per step

Dominated by the matrix–vector products: $2\\cdot g\\cdot n(n+d)$ FLOPs with $g=1,3,4$ gate blocks. For training on GPUs, all gate matmuls are fused into a single GEMM of shape $(B\\times(n+d))\\cdot((n+d)\\times gn)$ — wide enough to use tensor cores efficiently when $B\\cdot gn$ is large.
`,x=`## Exercises

**Q1.** A GRU unit has a constant update gate $z=0.05$. What fraction of the old state survives 20 steps (no new input), and what is the time constant?

<details>
<summary>Show answer</summary>

$(1-z)^{20}=0.95^{20}\\approx0.36$. $\\tau=-1/\\ln0.95\\approx19.5$ steps.

</details>

**Q2.** Compute the parameters of a single-layer GRU with input 300, hidden 256 using the PyTorch convention (two bias vectors).

<details>
<summary>Show answer</summary>

$3\\,(nd+n^2+2n)=3\\,(256\\cdot300+256^2+512)=3\\,(76{,}800+65{,}536+512)=428{,}544$.

</details>

**Q3.** Show that if $r_t=1$ and $z_t=1$ for all $t$ the GRU reduces to a vanilla RNN.

<details>
<summary>Show answer</summary>

Then $h_t=\\tilde h_t=\\tanh(W_hx_t+U_hh_{t-1}+b_h)$ — exactly the Elman RNN.

</details>

**Q4.** Which gate would you expect to be near 0 at the start of a new sentence when modelling text, and why?

<details>
<summary>Show answer</summary>

The **reset gate**: the candidate state should ignore the previous sentence, so $r\\to0$ lets $\\tilde h$ depend only on the new token. (The update gate remains free to carry slowly varying context such as topic or speaker.)

</details>

**Q5.** In the lab, why does the vanilla RNN (dashed) forget the pulse while the GRU with $b_z=-4$ does not?

<details>
<summary>Show answer</summary>

The RNN rewrites its state every step with factor $\\tanh$ and recurrence weight 0.9 — a contraction (<1) that decays the pulse exponentially. The GRU with a very negative $b_z$ has $z\\approx0.02$, so $h_t\\approx h_{t-1}$: the identity path preserves it.

</details>

**Q6 (code).** Show that a GRU's per-step cost is $\\approx75\\%$ of an LSTM's. Measure it.

<details>
<summary>Show answer</summary>

\`compare_cells.py\` times \`nn.RNN\`, \`nn.GRU\`, \`nn.LSTM\` for forward+backward. Expect roughly RNN : GRU : LSTM ≈ 1 : 3 : 4 in FLOPs, though measured wall-clock ratios are compressed by memory-bound pointwise ops and kernel-launch overhead.

</details>

## In practice

- **Default choice for small/medium sequence tasks**: a 1–2 layer (Bi)GRU with hidden 128–512 is a strong, cheap baseline.
- **Mobile / embedded**: fewer parameters and no extra cell state mean a smaller memory footprint — popular for keyword spotting and speech enhancement.
- **Quantisation**: weights int8; keep gate pre-activations in fp16/fp32 to avoid saturating the sigmoids.
- **Debugging**: log the mean and histogram of $z$ and $r$ per layer — collapsed gates (all ≈0 or ≈1) indicate a problem.

## Common pitfalls

- Mixing PyTorch's GRU gate order (\`r, z, n\`) with other implementations when porting weights.
- Using the final hidden state of padded batches.
- Assuming the GRU is "worse" for long dependencies without tuning biases/regularisation first.
- Forgetting to detach states across truncated windows.
`,S=`import torch
import torch.nn as nn


class GRUCellScratch(nn.Module):
    """PyTorch-compatible GRU: gate order (r, z, n), reset applied AFTER U_h h."""

    def __init__(self, d, n):
        super().__init__()
        self.n = n
        self.W = nn.Linear(d, 3 * n)     # x -> [r, z, n]
        self.U = nn.Linear(n, 3 * n)     # h -> [r, z, n]

    def forward(self, x, h):
        xr, xz, xn = self.W(x).chunk(3, -1)
        hr, hz, hn = self.U(h).chunk(3, -1)
        r = torch.sigmoid(xr + hr)               # reset gate
        z = torch.sigmoid(xz + hz)               # update gate
        n = torch.tanh(xn + r * hn)              # candidate
        return (1 - z) * n + z * h               # NOTE: PyTorch's z weights the OLD state


def run(cell, xs):
    B, T, _ = xs.shape
    h = xs.new_zeros(B, cell.n)
    ys = []
    for t in range(T):
        h = cell(xs[:, t], h)
        ys.append(h)
    return torch.stack(ys, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    d, n = 6, 10
    ours, ref = GRUCellScratch(d, n), nn.GRU(d, n, batch_first=True)
    with torch.no_grad():
        ref.weight_ih_l0.copy_(ours.W.weight); ref.bias_ih_l0.copy_(ours.W.bias)
        ref.weight_hh_l0.copy_(ours.U.weight); ref.bias_hh_l0.copy_(ours.U.bias)
    x = torch.randn(3, 9, d)
    print("max |scratch - nn.GRU| =", (run(ours, x) - ref(x)[0]).abs().max().item())
`,C=`import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(B=128, T=60, d=8):
    """Copy-first-token task: output the sign of x[:,0] after T steps of distractors."""
    x = torch.randn(B, T, d, device=device)
    y = (x[:, 0, 0] > 0).long()
    return x, y


class Net(nn.Module):
    def __init__(self, kind, d=8, n=64):
        super().__init__()
        self.rnn = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind](d, n, batch_first=True)
        self.fc = nn.Linear(n, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


for kind in ["rnn", "gru", "lstm"]:
    torch.manual_seed(0)
    net = Net(kind).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    n_params = sum(p.numel() for p in net.parameters())
    t0 = time.time()
    for step in range(400):
        x, y = make_batch()
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    x, y = make_batch(2048)
    acc = (net(x).argmax(1) == y).float().mean().item()
    print(f"{kind:5s} params={n_params:6d}  time={time.time() - t0:5.1f}s  accuracy={acc:.2%}")
# Expect: the vanilla RNN stays near 50% (it cannot carry the bit across 60 steps);
# GRU and LSTM learn the task.
`,w=n(),T=32;function E(e,t,n){let i=0,a=0;return e.map(e=>{let o=r(2*e+.5*i+t),s=r(1*e+.5*i+n),c=Math.tanh(2.2*e+1.5*s*i),l=i;return i=(1-o)*i+o*c,a=Math.tanh(2*e+.9*a),{x:e,z:o,r:s,ht:c,h:i,hp:l,hr:a}})}function D(){let[e,t]=(0,v.useState)(-2),[n,r]=(0,v.useState)(0),[y,b]=(0,v.useState)(!1),[x,S]=(0,v.useState)(3),[C,D]=(0,v.useState)(512),O=(0,v.useMemo)(()=>{let e=s(9);return Array.from({length:T},(t,n)=>(n===3?1:n===18?-.8:0)+(y?.25*l(e):0))},[y]),k=(0,v.useMemo)(()=>E(O,e,n),[O,e,n]),A=k[x],[j]=g(290,(e,t,n)=>{let r=t-40-16,i=n-46,o=e=>40+e/31*r,s=e=>14+i/2-c(e,-1.1,1.1)/1.1*(i/2);e.strokeStyle=u.grid,e.lineWidth=1,[-1,0,1].forEach(t=>{e.beginPath(),e.moveTo(40,s(t)),e.lineTo(40+r,s(t)),e.stroke(),d(e,t,32,s(t),{size:10.5,align:`right`,color:u.dim,mono:!0})});for(let t=0;t<T;t+=4)d(e,t,o(t),14+i+14,{size:10.5,align:`center`,color:u.dim,mono:!0});k.forEach((t,n)=>{e.fillStyle=`rgba(255,255,255,.18)`;let r=c(t.x,-1.1,1.1)/1.1*(i/2);e.fillRect(o(n)-3,r>0?s(0)-r:s(0),6,Math.abs(r))});let l=(t,n,r,i)=>a(e,k.map((e,n)=>[o(n),s(e[t])]),n,r,i);l(`hr`,`rgba(255,255,255,.75)`,1.6,[4,4]),l(`z`,u.d,1.4),l(`r`,u.c,1.4),l(`ht`,`rgba(139,123,255,.55)`,1.4),l(`h`,u.b,3),e.strokeStyle=`rgba(255,255,255,.35)`,e.setLineDash([3,3]),e.beginPath(),e.moveTo(o(x),14),e.lineTo(o(x),14+i),e.stroke(),e.setLineDash([])}),[M]=g(190,(e,t,n)=>{let r=t/2;d(e,`Update gate at t=${x}: h(t) = (1−z)·h(t−1) + z·h̃`,18,18,{size:12,color:u.mute,weight:600});let a=r-40,s=e=>40+(c(e,-1.1,1.1)+1.1)/2.2*(a-40);e.strokeStyle=u.line,e.lineWidth=2,e.beginPath(),e.moveTo(s(A.hp),92),e.lineTo(s(A.ht),92),e.stroke(),e.strokeStyle=u.d,e.lineWidth=5,e.beginPath(),e.moveTo(s(A.hp),92),e.lineTo(s(A.h),92),e.stroke();let l=(t,n,r,i)=>{e.fillStyle=n,e.beginPath(),e.arc(s(t),92,7,0,7),e.fill(),d(e,`${r} ${t.toFixed(2)}`,s(t),92+i,{size:11,align:`center`,mono:!0,color:n})};l(A.hp,u.mute,`h(t−1)`,26),l(A.ht,u.a,`h̃`,26),l(A.h,u.b,`h(t)`,-22),d(e,`z = ${A.z.toFixed(2)}  (${A.z<.2?`copy old state`:A.z>.8?`overwrite with candidate`:`blend`})   r = ${A.r.toFixed(2)}`,18,n-22,{size:12,mono:!0,color:u.ink});let f=r+30,p=t-f-30;d(e,`Parameters, hidden = input = ${C}`,f,18,{size:12,color:u.mute,weight:600});let m=C*(C+C)+C;[[`RNN`,m,u.mute],[`GRU`,3*m,u.b],[`LSTM`,4*m,u.a]].forEach(([t,n,r],a)=>{let s=n/(4*m)*(p-120);e.fillStyle=r,o(e,f+46,38+a*38,s,26,6),e.fill(),d(e,t,f,38+a*38+13,{size:12,color:u.ink,weight:600}),d(e,i(n),f+46+s+8,38+a*38+13,{size:11,mono:!0,color:u.mute})}),d(e,`GRU ≈ 75% of LSTM`,f,158,{size:12,color:u.b,mono:!0})});return(0,w.jsxs)(w.Fragment,{children:[(0,w.jsx)(`canvas`,{...j}),(0,w.jsx)(`canvas`,{...M}),(0,w.jsx)(f,{items:[[u.b,`hidden h`],[u.d,`update z`],[u.c,`reset r`],[`rgba(139,123,255,.8)`,`candidate h̃`],[`rgba(255,255,255,.7)`,`vanilla RNN`],[`rgba(255,255,255,.3)`,`input x`]]}),(0,w.jsxs)(m,{children:[(0,w.jsx)(p,{label:`Update bias b_z`,min:-5,max:4,step:.25,value:e,onChange:t,fmt:e=>e.toFixed(2)}),(0,w.jsx)(p,{label:`Reset bias b_r`,min:-4,max:4,step:.25,value:n,onChange:r,fmt:e=>e.toFixed(2)}),(0,w.jsx)(p,{label:`Inspect step t`,min:0,max:31,value:x,onChange:S}),(0,w.jsx)(p,{label:`Hidden size n`,min:64,max:2048,step:64,value:C,onChange:D}),(0,w.jsx)(_,{label:`Noisy input`,value:y,onChange:b})]}),(0,w.jsxs)(h,{children:[`The GRU has `,(0,w.jsx)(`b`,{children:`one`}),` state and `,(0,w.jsx)(`b`,{children:`two`}),` gates. A low update gate `,(0,w.jsx)(`b`,{children:`z≈0`}),` copies `,(0,w.jsx)(`i`,{children:`h(t−1)`}),` forward (memory); a high one overwrites it with the candidate. The reset gate `,(0,w.jsx)(`b`,{children:`r`}),` decides how much of the past the candidate may look at.`]})]})}var O={Lab:D,vizTitle:`Update gate as a dial between "keep" and "overwrite"`,tryIt:[`Set the **update bias very negative (−4)**: z≈0, the state is copied and the pulse is remembered. Set it to **+3**: the GRU behaves like a fast vanilla RNN.`,`Inspect step t=3 and t=18: the blue dot lands between the old state and the candidate at a position set by z.`,`Slide the **hidden size** and compare parameter counts of RNN, GRU and LSTM.`],theory:y,math:b,practice:x,code:[{title:`GRU cell from scratch, verified against nn.GRU`,lang:`python`,src:S},{title:`Benchmark RNN / GRU / LSTM: parameters, speed, accuracy on a toy memory task`,lang:`python`,src:C}],quiz:[{q:`In the GRU, $h_t=(1-z_t)\\odot h_{t-1}+z_t\\odot\\tilde h_t$. If $z_t\\to0$:`,options:[`The state is overwritten`,`The previous state is copied unchanged`,`The reset gate closes`,`The gradient is zero`],answer:1,why:`A small $z$ means "keep the old state", giving an identity path for gradients like the LSTM's constant error carousel.`},{q:`What does the reset gate $r_t$ do?`,options:[`Resets the weights`,`Controls how much of $h_{t-1}$ is used to compute the candidate`,`Sets the output to zero`,`Clips the gradient`],answer:1,why:`$\\tilde h_t=\\tanh(Wx_t+U(r_t\\odot h_{t-1}))$ — it lets the candidate ignore irrelevant history.`},{q:`Relative to an LSTM of the same width, a GRU has about:`,options:[`Half the parameters`,`3/4 of the parameters`,`The same`,`2× the parameters`],answer:1,why:`Three weight blocks instead of four: $3/4$.`},{q:`Which statement about GRU vs LSTM is most accurate?`,options:[`GRU always wins`,`LSTM always wins`,`They perform similarly on most tasks; GRU is cheaper, LSTM has an extra memory/output decoupling`,`GRUs cannot be stacked`],answer:2,why:`Empirical comparisons (Chung et al. 2014, Greff et al. 2017) show task-dependent, small differences.`},{q:`The GRU merges which two LSTM components?`,options:[`Input and output gates`,`Cell state and hidden state`,`Forget and candidate`,`Weights and biases`],answer:1,why:`There is no separate cell state; $h$ is both memory and output.`}]};export{O as default};