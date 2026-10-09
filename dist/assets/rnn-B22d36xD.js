import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{a as r,f as i,h as a,i as o,m as s,t as c,y as l}from"./viz-CPys2405.js";import{c as u,i as d,l as f,o as p,t as m,u as h}from"./hooks-Dw7oo1m3.js";var g=e(t(),1),_=`## Why we need recurrence

Text, speech, sensor streams, time series, DNA — data whose **length varies** and whose **order matters**. A CNN sees a fixed window; an MLP needs a fixed input size. A recurrent network processes one element at a time and carries a **hidden state** $h_t$ — a learned summary of everything seen so far:

$$h_t=\\phi(W_{hh}h_{t-1}+W_{xh}x_t+b)$$

The same weights are reused at every step (weight sharing across **time**, the sibling of weight sharing across space in CNNs), so the model handles any length with a constant number of parameters.

## Unrolling and the computation graph

"Unroll" the loop and an RNN becomes a very deep feed-forward network with $T$ layers that all share parameters. The lab's top strip is exactly that: each column is $h_t$ for one time step. Training with **backpropagation through time (BPTT)** is ordinary backprop on this unrolled graph, with gradients for the shared weights summed over all steps.

Common input/output patterns:

| Pattern | Example | How |
|---|---|---|
| many → one | sentiment classification | use last $h_T$ (or pooled states) |
| one → many | image captioning | feed image encoding once, generate tokens |
| many → many (aligned) | tagging, speech frames | output at every step |
| many → many (seq2seq) | translation | encoder RNN → decoder RNN (next chapters) |

Variants: **stacked** RNNs (the hidden sequence of layer $\\ell$ is the input to $\\ell+1$) and **bidirectional** RNNs (a forward and a backward pass, states concatenated; only for non-causal tasks).

## The long-range problem

Credit assignment from step $T$ back to step $k$ passes through $T-k$ Jacobians:

$$\\frac{\\partial h_T}{\\partial h_k}=\\prod_{t=k+1}^{T}\\mathrm{diag}\\big(\\phi'(a_t)\\big)\\,W_{hh}$$

Multiply many matrices and the norm shrinks or grows **exponentially**. Open the lab: with spectral radius $\\rho<1$ the backward bars decay like $\\rho^{T-t}$; with $\\rho>1$ they explode. This is the **vanishing / exploding gradient problem** (Hochreiter 1991, Bengio 1994).

- **Exploding** → loss spikes, NaNs. Cure: **gradient clipping** (rescale if $\\|g\\|>c$).
- **Vanishing** → the net cannot learn dependencies beyond ~10–20 steps. Cures: **gating** (LSTM, GRU — next two chapters), orthogonal/identity initialisation, skip connections, or abandoning recurrence for attention.

<div class="callout warn">

**Both views matter.** The forward view (second lab tab) shows the same phenomenon: with $\\rho<1$ information about early inputs is forgotten; with $\\rho>1$ the state is chaotic. Useful memory needs dynamics near the **edge of stability** — which gated cells learn to maintain.

</div>

## Training details that matter

- **Teacher forcing**: during training feed the *true* previous token, not the model's own prediction — fast and stable but creates *exposure bias* at inference. Scheduled sampling and sequence-level losses mitigate it.
- **Truncated BPTT**: backprop only through the last $k$ steps (e.g. 128) while still passing the hidden state forward. Bounds memory; limits learnable dependency length.
- **Initialisation**: orthogonal $W_{hh}$ and (for ReLU RNNs) identity initialisation keep $\\rho\\approx1$ at the start.
- **Regularisation**: dropout only on non-recurrent connections or with *variational* (same mask each step) dropout; layer-norm RNNs; zoneout.
- **Batching variable-length sequences**: sort/bucket by length, use packed sequences and masks.

## Limitations — and why recurrence is back

1. **Sequential compute**: $h_t$ needs $h_{t-1}$, so training can't be parallelised over time — poor GPU utilisation versus Transformers.
2. **Memory bottleneck**: everything must squeeze into a fixed-size $h_t$.
3. **Hard long-range credit assignment** even with gates.

Yet recurrence has returned in a new form: **state-space models** (S4, Mamba), RWKV, RetNet and linear-attention variants have *linear* (or parallel-scan) recurrences that train in parallel and run inference in $O(1)$ memory per token — attractive for very long contexts. Understanding the RNN is understanding the foundation of these models.
`,v=`## Forward equations

For input $x_t\\in\\mathbb R^{d}$ and hidden size $n$:

$$
a_t=W_{hh}h_{t-1}+W_{xh}x_t+b,\\qquad h_t=\\tanh(a_t),\\qquad \\hat y_t=\\mathrm{softmax}(W_{hy}h_t+c)
$$

Parameters: $n^2+nd+n$ for the recurrent cell (plus the output layer). They do **not** depend on $T$.

Loss over a sequence: $L=\\sum_t \\ell_t(\\hat y_t,y_t)$.

## Backpropagation through time

Let $\\delta_t=\\partial L/\\partial a_t$. Working backwards from $t=T$:

$$
\\delta_t=\\Big(\\mathrm{diag}\\big(1-h_t^2\\big)\\Big)\\Big(W_{hy}^{\\!\\top}(\\hat y_t-y_t)+W_{hh}^{\\!\\top}\\,\\delta_{t+1}\\Big)
$$

Because the parameters are shared, their gradients sum over time:

$$
\\frac{\\partial L}{\\partial W_{hh}}=\\sum_{t=1}^{T}\\delta_t\\,h_{t-1}^{\\!\\top},\\qquad
\\frac{\\partial L}{\\partial W_{xh}}=\\sum_{t=1}^{T}\\delta_t\\,x_t^{\\!\\top},\\qquad
\\frac{\\partial L}{\\partial b}=\\sum_{t}\\delta_t .
$$

## Why gradients vanish or explode

The influence of step $k$ on step $T$ is

$$
\\frac{\\partial h_T}{\\partial h_k}=\\prod_{t=k+1}^{T}D_t\\,W_{hh},\\qquad D_t=\\mathrm{diag}\\big(1-h_t^2\\big),\\ \\ \\|D_t\\|\\le\\gamma=1 .
$$

Taking norms,

$$
\\Big\\|\\frac{\\partial h_T}{\\partial h_k}\\Big\\|\\le\\big(\\gamma\\,\\sigma_{max}(W_{hh})\\big)^{T-k}.
$$

- If $\\gamma\\,\\sigma_{max}<1$ the gradient **vanishes** exponentially — a *sufficient* condition for vanishing.
- If the spectral radius $\\rho(W_{hh})>1/\\gamma$ the product **can** explode (*necessary* condition for explosion; Pascanu et al., 2013).

For an orthogonal matrix scaled by $\\rho$ (what the lab uses) every singular value equals $\\rho$, so the norm is *exactly* $\\rho^{T-k}$ in the linear regime; $\\tanh'$ makes it smaller still.

For the **linear** RNN $h_t=Wh_{t-1}$, $h_T=W^{T}h_0$; with eigendecomposition $W=V\\Lambda V^{-1}$ the state is $\\sum_i\\lambda_i^{T}c_i v_i$ — eigenvalues $|\\lambda|<1$ decay, $|\\lambda|>1$ blow up, and only $|\\lambda|\\approx1$ preserve information.

## Gradient clipping

$$
g\\leftarrow\\begin{cases}g&\\|g\\|\\le c\\\\ c\\,g/\\|g\\|&\\text{otherwise}\\end{cases}
$$

Direction is preserved, magnitude capped; this is precisely the "clip" toggle in the lab.

## Truncated BPTT

Split the sequence into windows of length $k$. For window $w$ set $h_{\\text{start}}=\\mathrm{stopgrad}(h_{\\text{end of }w-1})$. Memory is $O(k)$; gradients cross at most $k$ steps.

## Linear recurrences are parallelisable (link to SSMs)

The linear recurrence $h_t=a_th_{t-1}+b_t$ composes **associatively**:

$$
(a_2,b_2)\\circ(a_1,b_1)=(a_2a_1,\\;a_2b_1+b_2),
$$

so all $h_t$ can be computed in $O(\\log T)$ parallel depth with a prefix scan. Removing the nonlinearity from the *recurrence* (while keeping it elsewhere) is the trick behind S4/Mamba/linear attention — and the reason they train as fast as Transformers while inferring like RNNs.
`,y=`## Exercises

**Q1.** A vanilla RNN has hidden size 256 and input size 100. How many recurrent-cell parameters does it have? How do they change if the sequence length goes from 50 to 5000?

<details>
<summary>Show answer</summary>

$n^2+nd+n=256^2+256\\cdot100+256=65{,}536+25{,}600+256=91{,}392$. They do **not** change with length — only compute and activation memory grow ($O(T)$).

</details>

**Q2.** $W_{hh}$ has spectral radius 0.9 and $\\tanh'\\approx0.5$ on average. Estimate the gradient magnitude reaching 30 steps back.

<details>
<summary>Show answer</summary>

Per step the factor is $\\approx0.9\\times0.5=0.45$, so after 30 steps $\\approx0.45^{30}\\approx4\\times10^{-11}$. Even a "stable" $\\rho<1$ network cannot assign credit 30 steps away — which is what the gated cells fix.

</details>

**Q3.** Your RNN loss suddenly becomes \`NaN\` mid-training. Give a three-step diagnosis.

<details>
<summary>Show answer</summary>

1. Log the global gradient norm — look for a spike before the NaN (exploding gradients).
2. Add \`clip_grad_norm_(…, 1.0)\` (or 5.0) and lower the LR.
3. Check initial $W_{hh}$ (orthogonal init), input scale/normalization, and (for mixed precision) loss scaling.

</details>

**Q4.** Why does truncated BPTT still let the model use information older than the window?

<details>
<summary>Show answer</summary>

The hidden state is carried forward across window boundaries (detached from the graph), so earlier information can influence predictions — it just cannot receive gradient from them. The model can *use* long context but only *learn* dependencies up to $k$ steps.

</details>

**Q5.** Show that for the scalar linear RNN $h_t=wh_{t-1}+x_t$, a perturbation $\\varepsilon$ at $t=0$ changes $h_T$ by $w^{T}\\varepsilon$. For what $w$ is the memory "perfect"?

<details>
<summary>Show answer</summary>

By induction $h_T=w^Th_0+\\sum_{k=1}^{T}w^{T-k}x_k$, so $\\partial h_T/\\partial h_0=w^T$. Perfect memory needs $|w|=1$ — a knife-edge; learning must keep the dynamics precisely there (or use gates to *switch* retention on and off).

</details>

**Q6 (code).** Verify your BPTT implementation with a finite-difference gradient check.

<details>
<summary>Show answer</summary>

\`rnn_bptt.py\` in the **Code** tab does exactly that: perturb each parameter by $\\pm10^{-5}$, compute $(L_+-L_-)/(2\\cdot10^{-5})$, and compare with the analytic gradient; relative error should be $\\lesssim10^{-6}$.

</details>

## In practice

- **Where plain RNNs still appear**: tiny on-device models, streaming keyword spotting, low-latency control; most others use GRU/LSTM, 1-D CNNs/TCNs, or Transformers.
- **Packed sequences** (\`pack_padded_sequence\`) avoid wasted compute on padding and prevent padded steps from contaminating the final state.
- **Stateful inference**: for streaming, keep $h_t$ between chunks; reset it at sequence boundaries.
- **Optimizer**: Adam with gradient clipping; learning-rate ≈ $10^{-3}$.
- **Modern relevance**: understanding the RNN is the prerequisite for state-space models (Mamba), RWKV and linear attention.

## Common pitfalls

- Forgetting to \`detach()\` the hidden state between truncated windows → memory grows without bound.
- Applying dropout to the recurrent connection with a fresh mask each step (destroys memory).
- Evaluating in \`train()\` mode.
- Feeding unsorted, unmasked padded batches.
- Using sigmoid/tanh with huge inputs → saturation → zero gradients.
`,b=`import numpy as np

rng = np.random.default_rng(0)
n, d, T = 5, 3, 7                      # hidden, input, sequence length

params = {
    "Wxh": rng.standard_normal((n, d)) * 0.5,
    "Whh": rng.standard_normal((n, n)) * 0.5,
    "Why": rng.standard_normal((1, n)) * 0.5,
    "b": np.zeros(n),
}
xs = rng.standard_normal((T, d))
target = 0.7


def forward(p):
    h = np.zeros(n)
    cache = [(None, h)]
    for t in range(T):
        h = np.tanh(p["Whh"] @ h + p["Wxh"] @ xs[t] + p["b"])
        cache.append((xs[t], h))
    y = (p["Why"] @ h)[0]
    loss = 0.5 * (y - target) ** 2
    return loss, y, cache


def backward(p, y, cache):
    g = {k: np.zeros_like(v) for k, v in p.items()}
    h_T = cache[-1][1]
    dy = y - target
    g["Why"] += dy * h_T[None, :]
    dh = dy * p["Why"][0]                       # dL/dh_T
    for t in range(T, 0, -1):                   # walk back through time
        x, h = cache[t]
        h_prev = cache[t - 1][1]
        da = dh * (1 - h ** 2)                  # through tanh
        g["Wxh"] += np.outer(da, x)             # shared weights: gradients SUM over time
        g["Whh"] += np.outer(da, h_prev)
        g["b"] += da
        dh = p["Whh"].T @ da                    # to h_{t-1}
    return g


loss, y, cache = forward(params)
grads = backward(params, y, cache)

# finite-difference check
worst = 0.0
for k, v in params.items():
    for idx in np.ndindex(*v.shape):
        old = v[idx]
        v[idx] = old + 1e-5; lp = forward(params)[0]
        v[idx] = old - 1e-5; lm = forward(params)[0]
        v[idx] = old
        num = (lp - lm) / 2e-5
        worst = max(worst, abs(num - grads[k][idx]) / (abs(num) + abs(grads[k][idx]) + 1e-12))
print(f"loss {loss:.5f}   worst relative gradient error: {worst:.2e}")

# vanishing gradient demo: norm of dh_t as we go back
for rho in (0.5, 0.9, 1.1):
    W = rho * np.linalg.qr(rng.standard_normal((n, n)))[0]
    v = np.ones(n)
    norms = []
    for _ in range(30):
        v = W.T @ v
        norms.append(np.linalg.norm(v))
    print(f"rho={rho}: gradient norm after 30 steps back = {norms[-1]:.3e}")
`,x=`import torch
import torch.nn as nn
import torch.nn.functional as F


class CharModel(nn.Module):
    def __init__(self, vocab, hidden=256, layers=2, kind="lstm", dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn(hidden, hidden, layers, batch_first=True, dropout=dropout)  # dropout between layers only
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x, state=None):
        out, state = self.rnn(self.embed(x), state)
        return self.head(out), state


def detach(state):
    """Cut the graph between truncated-BPTT windows but keep the values."""
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()


def train(text, kind="lstm", bptt=128, batch=64, epochs=5, lr=2e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    # arrange text into \`batch\` parallel streams so hidden state can carry across windows
    n = (len(data) - 1) // batch * batch
    X = data[:n].view(batch, -1)
    Y = data[1:n + 1].view(batch, -1)

    model = CharModel(len(chars), kind=kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        state = None
        for i in range(0, X.size(1) - bptt, bptt):
            x, y = X[:, i:i + bptt].to(device), Y[:, i:i + bptt].to(device)
            logits, state = model(x, state)
            state = detach(state)                       # truncated BPTT
            loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # essential for RNNs
            opt.step()
        print(f"epoch {epoch + 1}  loss {loss.item():.3f}")
    return model, chars, stoi


@torch.no_grad()
def sample(model, chars, stoi, prompt="The ", n=300, temperature=0.8, device="cpu"):
    model.eval()
    x = torch.tensor([[stoi[c] for c in prompt]], device=device)
    logits, state = model(x)
    out = prompt
    for _ in range(n):
        probs = F.softmax(logits[0, -1] / temperature, -1)
        idx = torch.multinomial(probs, 1)
        out += chars[idx.item()]
        logits, state = model(idx[None], state)
    return out


if __name__ == "__main__":
    text = open("input.txt").read()      # e.g. tiny-shakespeare
    model, chars, stoi = train(text, kind="gru")
    print(sample(model, chars, stoi))
`,S=n(),C=8;function w(e){let t=a(e),n=Array.from({length:C},()=>Array.from({length:C},()=>s(t))),r=[];for(let e of n){let t=[...e];for(let e of r){let n=e.reduce((e,n,r)=>e+n*t[r],0);t=t.map((t,r)=>t-n*e[r])}let n=Math.hypot(...t);r.push(t.map(e=>e/n))}return r}var T=w(7),E=(()=>{let e=a(3);return Array.from({length:C},()=>s(e)*.8)})();function D({rho:e,T:t,xs:n,perturb:r=0}){let i=T.map(t=>t.map(t=>t*e)),a=Array(C).fill(0),o=[a];for(let e=0;e<t;e++){let t=n[e]+(e===0?r:0),s=i.map((e,n)=>Math.tanh(e.reduce((e,t,n)=>e+t*a[n],0)+E[n]*t));o.push(s),a=s}let s=Array(t+1),c=Array(C).fill(1/Math.sqrt(C));s[t]=Math.hypot(...c);for(let e=t;e>=1;e--){let t=c.map((t,n)=>t*(1-o[e][n]**2));c=Array.from({length:C},(e,n)=>i.reduce((e,r,i)=>e+r[n]*t[i],0)),s[e-1]=Math.hypot(...c)}return{H:o,g:s}}function O(e,t){let n=a(11);return Array.from({length:e},()=>s(n)*t)}function k(){let[e,t]=(0,g.useState)(.8),[n,a]=(0,g.useState)(20),[s,f]=(0,g.useState)(!1),_=(0,g.useMemo)(()=>O(40,1),[]),{H:v,g:y}=(0,g.useMemo)(()=>D({rho:e,T:n,xs:_}),[e,n,_]),[b]=m(430,(t,a,u)=>{let d=(a-54-16)/n;l(t,`Forward: hidden state  h_t  (8 units, colour = activation)`,54,14,{size:12,color:c.mute,weight:600});for(let e=1;e<=n;e++){let n=54+(e-1)*d;for(let i=0;i<C;i++)t.fillStyle=r(v[e][i]),t.fillRect(n+1,28+i*13,d-2,12);(d>14||e%2==0)&&l(t,e,n+d/2,142,{size:10,align:`center`,color:c.dim,mono:!0}),t.strokeStyle=c.line,e>1&&(t.beginPath(),t.moveTo(n-3,80),t.lineTo(n+1,80),t.stroke())}l(t,`t →`,22,142,{size:10,color:c.dim,mono:!0});let f=u-184-30;l(t,`Backward: ‖∂h_T / ∂h_t‖  (log scale — bars above the line explode, below vanish)`,54,170,{size:12,color:c.mute,weight:600});let p=184+f/2;t.strokeStyle=c.grid,t.lineWidth=1;for(let e=-8;e<=8;e+=4){let n=p-e/8*(f/2);t.beginPath(),t.moveTo(54,n),t.lineTo(a-16,n),t.stroke(),l(t,`1e${e}`,48,n,{size:10,align:`right`,color:c.dim,mono:!0})}t.strokeStyle=`rgba(255,255,255,.35)`,t.beginPath(),t.moveTo(54,p),t.lineTo(a-16,p),t.stroke();for(let e=0;e<n;e++){let n=y[e];s&&(n=Math.min(n,5));let r=o(Math.log10(Math.max(n,1e-30)),-8,8),i=54+e*d,a=r/8*(f/2);t.fillStyle=Math.abs(r)<1?c.e:r>0?c.d:c.c,t.fillRect(i+2,a>0?p-a:p,d-4,Math.abs(a)||1)}let m=[];for(let t=0;t<=n;t++)m.push([54+(t-.5)*d,p-o((n-t)*Math.log10(e),-8,8)/8*(f/2)]);i(t,m,`rgba(255,255,255,.7)`,1.5,[4,3]),l(t,`dashed: ρ^(T−t)`,a-16,170,{size:11,align:`right`,color:`#fff`,mono:!0})}),x=y[0],w=x<.001?[`vanished`,`r`]:x>1e3?[`exploded`,`w`]:[`healthy`,`g`];return(0,S.jsxs)(S.Fragment,{children:[(0,S.jsx)(`canvas`,{...b}),(0,S.jsxs)(d,{children:[(0,S.jsx)(u,{label:`Spectral radius ρ of W`,min:.3,max:1.8,step:.02,value:e,onChange:t,fmt:e=>e.toFixed(2)}),(0,S.jsx)(u,{label:`Sequence length T`,min:5,max:40,value:n,onChange:a}),(0,S.jsx)(h,{label:`Clip gradient norm at 5`,value:s,onChange:f})]}),(0,S.jsxs)(p,{children:[`‖∂h_T/∂h_1‖ = `,(0,S.jsx)(`b`,{children:x.toExponential(2)}),` after `,n-1,` steps — `,(0,S.jsx)(`span`,{className:w[1],children:w[0]}),`. Naive estimate ρ^`,n-1,` = `,(e**(n-1)).toExponential(2),`; tanh′ ≤ 1 shrinks it further.`]})]})}function A(){let[e,t]=(0,g.useState)(.9),[n,r]=(0,g.useState)(30),[a,s]=(0,g.useState)(1),f=(0,g.useMemo)(()=>O(60,.6),[]),h=(0,g.useMemo)(()=>{let t=D({rho:e,T:n,xs:f}).H,r=D({rho:e,T:n,xs:f,perturb:a}).H;return t.map((e,t)=>Math.hypot(...e.map((e,n)=>e-r[t][n])))},[e,n,a,f]),[_]=m(320,(e,t,r)=>{let s=t-56-20,u=r-56,d=e=>56+e/n*s,f=e=>20+u-(o(Math.log10(Math.max(e,1e-12)),-8,1)- -8)/9*u;e.strokeStyle=c.grid;for(let t=-8;t<=1;t+=2)e.beginPath(),e.moveTo(56,f(10**t)),e.lineTo(56+s,f(10**t)),e.stroke(),l(e,`1e${t}`,48,f(10**t),{size:10,align:`right`,color:c.dim,mono:!0});for(let t=0;t<=n;t+=Math.max(1,Math.round(n/10)))l(e,t,d(t),20+u+14,{size:10.5,align:`center`,color:c.mute,mono:!0});l(e,`time step t`,56+s/2,r-8,{size:11,align:`center`,color:c.mute}),l(e,`‖h_t − h'_t‖ after perturbing x_0 by ${a}`,56,8,{size:12,color:c.mute,weight:600}),i(e,h.map((e,t)=>[d(t),f(e)]),c.b,2.5),h.forEach((t,n)=>{e.fillStyle=c.b,e.beginPath(),e.arc(d(n),f(t),3,0,7),e.fill()})});return(0,S.jsxs)(S.Fragment,{children:[(0,S.jsx)(`canvas`,{..._}),(0,S.jsxs)(d,{children:[(0,S.jsx)(u,{label:`Spectral radius ρ`,min:.3,max:1.8,step:.02,value:e,onChange:t,fmt:e=>e.toFixed(2)}),(0,S.jsx)(u,{label:`Length T`,min:10,max:60,value:n,onChange:r}),(0,S.jsx)(u,{label:`Perturbation of x₀`,min:.1,max:3,step:.1,value:a,onChange:s})]}),(0,S.jsxs)(p,{children:[`Forward-time view of the same story: with `,(0,S.jsx)(`b`,{children:`ρ<1`}),` a change at t=0 is forgotten (memory fades); with `,(0,S.jsx)(`b`,{children:`ρ>1`}),` it is amplified until tanh saturates, so the state becomes `,(0,S.jsx)(`span`,{className:`w`,children:`chaotic`}),` and sensitive to everything. Useful memory lives at the `,(0,S.jsx)(`b`,{children:`edge of stability`}),`.`]})]})}function j(){return(0,S.jsx)(f,{views:[{id:`grad`,label:`Backprop through time`,render:()=>(0,S.jsx)(k,{})},{id:`mem`,label:`Memory fade / chaos`,render:()=>(0,S.jsx)(A,{})}]})}var M={Lab:j,vizTitle:`Watch gradients vanish and explode through time`,tryIt:[`Slide **ρ below 1** and see the backward bars shrink exponentially toward the first time step.`,`Slide **ρ above ~1.3** — the bars explode; then switch on **clipping** to see the cap.`,`Increase **T** to 40: even ρ = 0.95 loses the early signal.`,`In the memory view find the ρ where a perturbation neither dies nor blows up.`],theory:_,math:v,practice:y,code:[{title:`BPTT from scratch with a numerical gradient check`,lang:`python`,note:`A tiny tanh RNN: forward, backward through time, and finite-difference verification.`,src:b},{title:`Character-level language model (PyTorch, RNN/LSTM/GRU switchable)`,lang:`python`,src:x}],quiz:[{q:`Backpropagating through $T$ steps multiplies roughly $T$ Jacobians of the form $\\mathrm{diag}(1-h^2)W_{hh}$. The gradient vanishes when:`,options:[`The product has norm $>1$`,`The product has norm $<1$ consistently`,`The loss is zero`,`The sequence is short`],answer:1,why:`Norms of products shrink (or grow) exponentially in $T$ — the root of vanishing/exploding gradients.`},{q:`Which technique fixes **exploding** (not vanishing) gradients?`,options:[`Gradient-norm clipping`,`More layers`,`Smaller batch`,`Sigmoid activations`],answer:0,why:`Clipping rescales the gradient when its norm is too large; it cannot revive vanished gradients.`},{q:`Weight sharing across time in an RNN means:`,options:[`The same $W_{hh}$ is applied at every step, so parameters do not grow with sequence length`,`Each step has its own weights`,`Only the output layer is shared`,`The hidden state is shared across batches`],answer:0,why:`This is the temporal analogue of convolutional weight sharing.`},{q:`Truncated BPTT with window $k$ trades:`,options:[`Accuracy for nothing`,`Compute/memory for the ability to learn dependencies longer than $k$`,`Batch size for epochs`,`Speed for determinism`],answer:1,why:`Gradients are cut every $k$ steps, bounding memory but preventing credit assignment beyond the window (the hidden state still carries forward).`},{q:`Why can't a vanilla RNN be parallelized over time during training?`,options:[`Because of softmax`,`Each $h_t$ depends on $h_{t-1}$, forcing sequential evaluation`,`GPUs lack tanh`,`Because of dropout`],answer:1,why:`The recurrence is inherently sequential — one motivation for Transformers (and for parallel-scan SSMs).`}]};export{M as default};