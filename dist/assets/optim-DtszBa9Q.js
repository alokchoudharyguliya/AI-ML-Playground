import{r as e}from"./rolldown-runtime-hePW80VL.js";import{a as t,b as n,d as r,m as i,o as a,p as o,s,u as c,v as l}from"./r3f-x1z21uF6.js";import{f as u,i as d,t as f,u as p,y as m}from"./viz-CPys2405.js";import{t as h}from"./Stage3D-CWOT2b2l.js";import{a as g,c as _,i as v,l as y,o as b,r as x,s as S,t as C}from"./hooks-Dw7oo1m3.js";var w=e(n(),1),T=`## Training is geometry

A network defines a loss surface $L(\\theta)$ over millions or billions of parameters. Training is a walk across it. The 3D lab above shows 2-parameter toys, but the intuition transfers: the shape of the surface — long narrow valleys, flat plateaus, saddles, multiple basins — decides which optimizer behaves well.

<div class="callout">

**Key idea.** In high dimensions, bad *local minima* are rare. The real obstacles are **saddle points**, **ill-conditioning** (steep in some directions, flat in others) and **noise**. Every optimizer below is a different answer to one of those.

</div>

## Stochastic gradient descent

SGD estimates the gradient on a mini-batch: $\\theta\\leftarrow\\theta-\\eta\\,\\hat g$. The estimate is noisy, which is both a cost (jitter) and a feature (it helps escape sharp basins and acts as implicit regularization). Batch size trades gradient noise for hardware efficiency; the **learning rate $\\eta$ is the single most important hyper-parameter** — too small crawls, too large diverges, and the usable range is bounded by the sharpest curvature direction ($\\eta<2/\\lambda_{max}$).

**The ravine problem.** On the ravine surface the gradient is huge across the valley and tiny along it. SGD must use a small $\\eta$ to stay stable across, so it crawls along — the classic ill-conditioning picture. The ratio of largest to smallest curvature is the **condition number** $\\kappa$.

## Momentum and Nesterov

Momentum keeps a velocity $v\\leftarrow\\beta v+g$ and steps along it. Oscillating components cancel; consistent components accumulate, giving up to $1/(1-\\beta)$ times the step along the valley. Convergence on quadratics improves from $O(\\kappa)$ to $O(\\sqrt\\kappa)$ iterations. **Nesterov** momentum evaluates the gradient at the look-ahead point, damping overshoot slightly.

## Adaptive methods

Rather than one global step size, give each parameter its own, scaled by recent gradient magnitude:

- **AdaGrad** divides by the root of the *sum* of squared gradients — great for sparse features, but the step eventually shrinks to zero.
- **RMSProp** uses an exponential *moving average* instead, fixing the decay.
- **Adam** = momentum on the first moment + RMSProp on the second moment + **bias correction** for the zero-initialised averages. Steps are roughly $\\eta\\cdot\\text{sign}$-like, so $\\eta\\approx10^{-3}$–$10^{-4}$ works across wildly different gradient scales.
- **AdamW** *decouples* weight decay from the adaptive step. It is the default for Transformers and most modern training.

Rules of thumb: **SGD + momentum** (often with large LR and long schedules) is still strong for CNNs and generalizes well; **AdamW** is the safe default for Transformers, embeddings and anything with heterogeneous gradient scales. Newer optimizers (Lion, Sophia, Shampoo, Muon) trade memory/compute for faster convergence and see use in large-scale training.

## Learning-rate schedules and warmup

A constant LR rarely works best. Typical recipes:

| Schedule | Used for |
|---|---|
| Step decay (×0.1 at milestones) | Classic ImageNet ResNet training |
| **Cosine decay** | Default for vision and LLM pre-training |
| **Linear warmup + cosine/linear decay** | Transformers, LLMs |
| Inverse-sqrt ("Noam") | Original Transformer |
| One-cycle / super-convergence | Fast CNN training with large peak LR |
| WSD (warmup–stable–decay) | LLM pre-training where the final length is unknown |

**Warmup** matters because Adam's variance estimates are unreliable at step 1 and Transformers are fragile early on; ramping LR from ~0 over the first 0.5–5% of steps avoids early divergence.

## Batch size, noise and scaling

Gradient noise scale $\\propto \\eta/B$. When you multiply the batch size by $k$, multiply the LR by $k$ (SGD, the *linear scaling rule*) or by $\\sqrt k$ (Adam-style) and add warmup. Beyond the **critical batch size** larger batches stop reducing the number of steps you need.

## Practical stabilizers

- **Gradient clipping** (global norm $\\le1$) is nearly universal in sequence models and LLMs.
- **Mixed precision** needs loss scaling (FP16) or uses BF16, which does not.
- **Gradient accumulation** simulates large batches under memory limits.
- **Weight decay** ($0.01$–$0.1$ with AdamW) and **EMA of weights** improve generalization.
- **Loss spikes** in huge runs are mitigated by lower $\\beta_2$ (0.95), QK-norm, z-loss and skipping bad batches.

## Flat vs sharp minima

Solutions found by small-batch SGD tend to be *flatter* (low curvature) and generalize better; sharpness-aware minimization (**SAM**) optimizes for this explicitly. The idea is why "train longer with noise" often beats "converge fast".
`,E=`## Gradient descent on a quadratic

For $L(\\theta)=\\tfrac12\\theta^\\top H\\theta$ with eigenvalues $\\lambda_i$ of $H$, gradient descent $\\theta\\leftarrow\\theta-\\eta H\\theta$ acts independently on each eigen-direction:

$$
\\theta_i^{(t+1)}=(1-\\eta\\lambda_i)\\,\\theta_i^{(t)} .
$$

Stable iff $|1-\\eta\\lambda_i|<1$ for all $i$, i.e.

$$
\\eta<\\frac{2}{\\lambda_{max}} .
$$

With the best step $\\eta^\\star=2/(\\lambda_{max}+\\lambda_{min})$ the contraction factor per step is

$$
\\rho=\\frac{\\kappa-1}{\\kappa+1},\\qquad\\kappa=\\frac{\\lambda_{max}}{\\lambda_{min}} ,
$$

so reaching accuracy $\\varepsilon$ needs $\\approx\\tfrac{\\kappa}{2}\\ln(1/\\varepsilon)$ iterations.

## Momentum (heavy ball)

$$
v_{t+1}=\\beta v_t+g_t,\\qquad \\theta_{t+1}=\\theta_t-\\eta\\,v_{t+1} .
$$

In the steady state with a constant gradient, $v=g/(1-\\beta)$: the *effective* learning rate is $\\eta/(1-\\beta)$ (10× for $\\beta=0.9$). With optimal $(\\eta,\\beta)$ the rate becomes

$$
\\rho_{mom}=\\frac{\\sqrt\\kappa-1}{\\sqrt\\kappa+1},
$$

needing $O(\\sqrt\\kappa)$ iterations — a quadratic improvement. **Nesterov**:

$$
v_{t+1}=\\beta v_t+\\nabla L(\\theta_t-\\eta\\beta v_t),\\qquad\\theta_{t+1}=\\theta_t-\\eta v_{t+1}.
$$

## RMSProp

$$
s_t=\\beta\\, s_{t-1}+(1-\\beta)\\,g_t^2,\\qquad
\\theta_{t+1}=\\theta_t-\\frac{\\eta}{\\sqrt{s_t}+\\epsilon}\\,g_t
$$

(all operations element-wise). Each coordinate's step is normalised to $\\approx\\eta$ regardless of gradient scale.

## Adam

$$
\\begin{aligned}
m_t&=\\beta_1 m_{t-1}+(1-\\beta_1)g_t, &\\qquad v_t&=\\beta_2 v_{t-1}+(1-\\beta_2)g_t^2,\\\\
\\hat m_t&=\\frac{m_t}{1-\\beta_1^t}, &\\qquad \\hat v_t&=\\frac{v_t}{1-\\beta_2^t},\\\\
\\theta_{t+1}&=\\theta_t-\\eta\\,\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}. &&
\\end{aligned}
$$

**Why bias correction?** With $m_0=0$, $\\mathbb E[m_t]=(1-\\beta_1^t)\\,\\mathbb E[g]$ when gradients are stationary, so dividing by $1-\\beta_1^t$ restores an unbiased estimate. At $t=1$: $\\hat m_1=g_1$, $\\hat v_1=g_1^2$, so the first step is $\\eta\\,g_1/|g_1|=\\eta\\,\\mathrm{sign}(g_1)$.

Defaults: $\\beta_1=0.9$, $\\beta_2=0.999$ (LLMs often $0.95$), $\\epsilon=10^{-8}$.

## L2 regularization vs decoupled weight decay

Adding $\\tfrac\\lambda2\\|\\theta\\|^2$ to the loss adds $\\lambda\\theta$ to $g_t$. In Adam that term is then divided by $\\sqrt{\\hat v_t}$, so parameters with large gradient history are decayed *less*. **AdamW** applies decay outside the adaptive rescaling:

$$
\\theta_{t+1}=\\theta_t-\\eta\\left(\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}+\\lambda\\,\\theta_t\\right).
$$

## SGD noise and the scaling rules

With batch size $B$, the gradient estimate has covariance $\\Sigma/B$. The SGD "temperature" scales like $\\eta/B$ (in the SDE view, noise scale $g\\approx\\eta N/B$). Keeping $\\eta/B$ constant — the **linear scaling rule** — preserves the dynamics for moderate $B$. For Adam-style methods a square-root rule, $\\eta\\propto\\sqrt B$, is often better.

## Schedules

With peak rate $\\eta_{max}$, total steps $T$ and warmup $T_w$:

$$
\\eta_t=\\begin{cases}
\\eta_{max}\\,\\dfrac{t}{T_w} & t<T_w\\\\[2mm]
\\eta_{min}+\\tfrac12(\\eta_{max}-\\eta_{min})\\left(1+\\cos\\dfrac{\\pi\\,(t-T_w)}{T-T_w}\\right) & t\\ge T_w
\\end{cases}
$$

**Inverse-sqrt (Noam):** $\\eta_t=\\eta_{max}\\min\\!\\left(\\dfrac{t}{T_w},\\sqrt{\\dfrac{T_w}{t}}\\right)$.

## Gradient clipping

Global-norm clipping rescales when the norm exceeds $c$:

$$
g\\leftarrow g\\cdot\\min\\!\\left(1,\\frac{c}{\\|g\\|_2}\\right).
$$

It preserves the gradient *direction* and bounds the step, which tames exploding gradients and loss spikes.
`,D=`## Exercises

**Q1.** For $f(x,y)=\\tfrac12(a x^2+b y^2)$ with $a=100$, $b=1$, what is the largest stable plain-GD learning rate, and how many iterations does GD need (order of magnitude) to shrink the error by $10^{-6}$ with the optimal $\\eta$?

<details>
<summary>Show answer</summary>

$\\lambda_{max}=100$ so $\\eta<2/100=0.02$. With $\\kappa=100$ and $\\eta^\\star=2/101$, $\\rho=99/101\\approx0.980$; iterations $\\approx\\ln(10^{6})/(-\\ln\\rho)\\approx13.8/0.0200\\approx690$. With momentum the rate is $\\tfrac{\\sqrt{100}-1}{\\sqrt{100}+1}=0.82$, i.e. $\\approx70$ iterations — a ~10× speed-up ($\\sqrt\\kappa$).

</details>

**Q2.** You use SGD with momentum $\\beta=0.9$ and $\\eta=0.01$. What is the effective step size on a constant gradient, and what happens if you raise $\\beta$ to $0.99$ without changing $\\eta$?

<details>
<summary>Show answer</summary>

Effective LR $=\\eta/(1-\\beta)=0.1$. With $\\beta=0.99$ it becomes $1.0$ — a **10× larger effective step**, which will likely diverge. If you tune $\\beta$, rescale $\\eta$ so $\\eta/(1-\\beta)$ stays comparable.

</details>

**Q3.** The first Adam update uses bias-corrected moments. What is its size per parameter, and why does this matter for the first few iterations?

<details>
<summary>Show answer</summary>

$\\hat m_1/\\sqrt{\\hat v_1}=g/|g|=\\pm1$, so every parameter moves by exactly $\\eta$ regardless of gradient scale. Early updates are therefore *large and sign-like*, which is a reason Transformers need **warmup** — a full-size first step on noisy, uncalibrated moments can derail training.

</details>

**Q4.** You scale pre-training from batch size 256 to 2048 on 8× more GPUs. How do you adjust the hyper-parameters?

<details>
<summary>Show answer</summary>

With SGD: multiply LR by 8 (linear rule) and lengthen warmup (5 epochs in the Goyal et al. ImageNet recipe). With AdamW: start from $\\sqrt8\\approx2.8\\times$ and sweep, keep weight decay, and check for loss spikes. Verify against the **critical batch size** — beyond it you burn compute without fewer steps.

</details>

**Q5.** Why is weight decay applied differently in AdamW, and which parameters should usually be excluded?

<details>
<summary>Show answer</summary>

Decay is *decoupled* from the adaptive scaling, so every weight shrinks at the same relative rate. Exclude **biases and normalization gains/shifts** (and often embeddings) — decaying them hurts and has little regularization value.

</details>

**Q6 (code).** Implement an *LR range test*: ramp the LR exponentially from $10^{-7}$ to $1$ over 200 steps, record the loss, and choose the LR.

<details>
<summary>Show answer</summary>

Train for 200 steps with \`lr = 1e-7 * (1e7) ** (step/200)\`; plot loss vs LR on a log axis. Pick an LR about **10× below the point where the loss is lowest** (before it blows up), or the steepest-descent region. This takes a minute and often saves hours of sweeps.

</details>

## In practice

**Defaults that rarely fail.** AdamW, peak LR $3\\times10^{-4}$ (small models) down to $10^{-5}$–$10^{-4}$ (large models), $\\beta=(0.9,0.95)$, weight decay 0.1, linear warmup for ~1% of steps, cosine decay to 10% of peak, gradient clip at 1.0, BF16.

**Fine-tuning.** Use 10–100× smaller LRs than pre-training; LoRA adapters tolerate larger ones ($10^{-4}$–$10^{-3}$).

**Debug order when loss is bad:** (1) overfit one batch, (2) LR range test, (3) check data/labels/normalization, (4) add clipping, (5) then change optimizers.

**Monitoring.** Log grad-norm, update-to-weight ratio ($\\|\\Delta\\theta\\|/\\|\\theta\\|\\approx10^{-3}$ is healthy), LR and loss per step; spikes tell you more than epoch averages.

**Memory.** Adam keeps two extra fp32 states per parameter (8 bytes) — for a 7B model that is ~56 GB on top of weights and gradients; this is why ZeRO/FSDP shard optimizer state and 8-bit optimizers exist.

## Common pitfalls

- Forgetting to zero gradients (or accumulating them unintentionally).
- Applying the schedule per-epoch when it was designed per-step.
- Decaying biases/LayerNorm parameters.
- Changing batch size without touching LR/warmup.
- Comparing optimizers at a single LR — each needs its own sweep.
- Using FP16 without loss scaling (silent underflow of gradients).
`,O=`import numpy as np


def ravine(p):
    x, y = p
    return 0.05 * x * x + y * y, np.array([0.1 * x, 2 * y])


class SGD:
    def __init__(self, lr): self.lr = lr
    def step(self, p, g): return p - self.lr * g


class Momentum:
    def __init__(self, lr, beta=0.9): self.lr, self.beta, self.v = lr, beta, 0
    def step(self, p, g):
        self.v = self.beta * self.v + g
        return p - self.lr * self.v


class RMSProp:
    def __init__(self, lr, beta=0.9, eps=1e-8): self.lr, self.beta, self.eps, self.s = lr, beta, eps, 0
    def step(self, p, g):
        self.s = self.beta * self.s + (1 - self.beta) * g * g
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, weight_decay
        self.m = self.v = 0
        self.t = 0

    def step(self, p, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)           # bias correction
        v_hat = self.v / (1 - self.b2 ** self.t)
        update = m_hat / (np.sqrt(v_hat) + self.eps)
        return p - self.lr * (update + self.wd * p)         # wd != 0  ->  AdamW (decoupled)


def run(opt, steps=300, start=(-7.0, 3.2)):
    p = np.array(start)
    for _ in range(steps):
        _, g = ravine(p)
        p = opt.step(p, g)
    return ravine(p)[0]


if __name__ == "__main__":
    for name, opt in [("SGD", SGD(0.4)), ("Momentum", Momentum(0.05)), ("RMSProp", RMSProp(0.1)), ("Adam", Adam(0.3))]:
        print(f"{name:9s} final loss after 300 steps: {run(opt):.3e}")
`,k=`import math
import torch
import torch.nn.functional as F


def build_optimizer(model, lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)):
    """AdamW with weight decay only on matrices (not biases / norm gains)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())


def warmup_cosine(step, total, warmup, min_ratio=0.1):
    """Multiplier in [min_ratio, 1]; use with LambdaLR."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def train(model, loader, total_steps, accum=4, lr=3e-4, warmup=200, clip=1.0, device="cuda"):
    opt = build_optimizer(model, lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total_steps, warmup))
    use_amp = device == "cuda"
    step, micro = 0, 0
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(model(x), y) / accum          # average over micro-batches
        loss.backward()
        micro += 1
        if micro % accum:
            continue

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # returns pre-clip norm
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            print(f"step {step:6d}  loss {loss.item() * accum:.4f}  grad-norm {gnorm:.2f}  lr {sched.get_last_lr()[0]:.2e}")
        if step >= total_steps:
            break
`,A=l(),j={ravine:{name:`Ravine (ill-conditioned bowl)`,dom:[-8,8,-4,4],start:[-7,3.2],lr:[.15,.3],f:(e,t)=>.05*e*e+t*t,g:(e,t)=>[.1*e,2*t]},himmelblau:{name:`Himmelblau (4 minima)`,dom:[-5,5,-5,5],start:[-.8,-4.2],lr:[.012,.12],f:(e,t)=>(e*e+t-11)**2+(e+t*t-7)**2,g:(e,t)=>[4*e*(e*e+t-11)+2*(e+t*t-7),2*(e*e+t-11)+4*t*(e+t*t-7)]},rosenbrock:{name:`Rosenbrock (curved valley)`,dom:[-2,2,-1,3],start:[-1.5,2.3],lr:[5e-4,.05],f:(e,t)=>(1-e)**2+100*(t-e*e)**2,g:(e,t)=>[-2*(1-e)-400*e*(t-e*e),200*(t-e*e)]}},M=[{id:`sgd`,name:`SGD`,color:`#fb7185`,group:0},{id:`mom`,name:`Momentum`,color:`#fbbf24`,group:0},{id:`rms`,name:`RMSProp`,color:`#4ade80`,group:1},{id:`adam`,name:`Adam`,color:`#22d3ee`,group:1}],N=8,P=3;function F(e){let t=0,[n,r,i,a]=e.dom;for(let o=0;o<=60;o++)for(let s=0;s<=60;s++)t=Math.max(t,e.f(n+(r-n)*o/60,i+(a-i)*s/60));return t}function I(e,t,n){let[r,i]=t.g(e.p[0],e.p[1]),a=[r,i],o=Math.hypot(r,i);o>1e3&&(a[0]*=1e3/o,a[1]*=1e3/o),e.t++;let s=1e-8;for(let t=0;t<2;t++)e.id===`sgd`?e.p[t]-=n*a[t]:e.id===`mom`?(e.m[t]=.9*e.m[t]+a[t],e.p[t]-=n*e.m[t]):e.id===`rms`?(e.v[t]=.9*e.v[t]+.1*a[t]*a[t],e.p[t]-=n*a[t]/(Math.sqrt(e.v[t])+s)):(e.m[t]=.9*e.m[t]+.1*a[t],e.v[t]=.999*e.v[t]+.001*a[t]*a[t],e.p[t]-=n*(e.m[t]/(1-.9**e.t))/(Math.sqrt(e.v[t]/(1-.999**e.t))+s));let[c,l,u,f]=t.dom;e.p[0]=d(e.p[0],c,l),e.p[1]=d(e.p[1],u,f),e.loss=t.f(e.p[0],e.p[1]),e.t%2==0&&(e.trail.push([e.p[0],e.p[1]]),e.trail.length>700&&e.trail.shift())}function L({fn:e,fm:t,onPick:n}){let r=(0,w.useMemo)(()=>{let[n,r,i,o]=e.dom,c=new Float32Array(24843),l=new Float32Array(24843),u=[];for(let a=0;a<=90;a++)for(let s=0;s<=90;s++){let u=n+(r-n)*s/90,d=i+(o-i)*a/90,f=Math.log1p(e.f(u,d))/Math.log1p(t),m=(a*91+s)*3;c[m]=(s/90-.5)*N,c[m+1]=f*P,c[m+2]=(a/90-.5)*N;let[h,g,_]=p(1-f**.6);l[m]=h/255,l[m+1]=g/255,l[m+2]=_/255}for(let e=0;e<90;e++)for(let t=0;t<90;t++){let n=e*91+t,r=n+1,i=n+90+1,a=i+1;u.push(n,i,r,r,i,a)}let d=new s;return d.setAttribute(`position`,new a(c,3)),d.setAttribute(`color`,new a(l,3)),d.setIndex(u),d.computeVertexNormals(),d},[e,t]);return(0,A.jsxs)(`group`,{children:[(0,A.jsx)(`mesh`,{geometry:r,onClick:t=>{let[r,i,a,o]=e.dom;n([r+(t.point.x/N+.5)*(i-r),a+(t.point.z/N+.5)*(o-a)])},children:(0,A.jsx)(`meshStandardMaterial`,{vertexColors:!0,side:2,roughness:.75,metalness:.05})}),(0,A.jsx)(`mesh`,{geometry:r,children:(0,A.jsx)(`meshBasicMaterial`,{wireframe:!0,color:`#ffffff`,transparent:!0,opacity:.05})})]})}function R({r:e,fn:n,fm:l}){let u=(0,w.useMemo)(()=>{let e=new s;return e.setAttribute(`position`,new a(new Float32Array(2103),3)),e.setDrawRange(0,0),e},[]),d=(0,w.useMemo)(()=>[new c(u,new r({color:e.color})),new o(u,new i({color:e.color,size:.07,sizeAttenuation:!0}))],[u,e.color]),f=(0,w.useRef)(),[p,m,h,g]=n.dom,_=(e,t)=>{let r=Math.log1p(n.f(e,t))/Math.log1p(l)*P+.06;return[(e-p)/(m-p)*N-N/2,r,(t-h)/(g-h)*N-N/2]};return t(()=>{let t=u.attributes.position.array,n=e.trail;for(let e=0;e<n.length;e++){let r=_(n[e][0],n[e][1]);t[e*3]=r[0],t[e*3+1]=r[1],t[e*3+2]=r[2]}u.attributes.position.needsUpdate=!0,u.setDrawRange(0,n.length);let r=_(e.p[0],e.p[1]);f.current&&f.current.position.set(r[0],r[1]+.07,r[2])}),(0,A.jsxs)(A.Fragment,{children:[d.map((e,t)=>(0,A.jsx)(`primitive`,{object:e},t)),(0,A.jsxs)(`mesh`,{ref:f,children:[(0,A.jsx)(`sphereGeometry`,{args:[.13,20,20]}),(0,A.jsx)(`meshStandardMaterial`,{color:e.color,emissive:e.color,emissiveIntensity:.9})]})]})}function z({fn:e,fm:n,sim:r,lrRef:i,speedRef:a,runKey:o}){let s=(0,w.useMemo)(()=>{let t=M.map(t=>({...t,p:[...r.start],m:[0,0],v:[0,0],t:0,trail:[[...r.start]],loss:e.f(...r.start)}));return r.runners=t,t},[e,r.start,o]),c=(0,w.useRef)(0);return t((t,n)=>{for(c.current+=Math.min(n,.05)*a.current;c.current>=1;)--c.current,s.forEach(t=>I(t,e,i.current[t.group]))}),s.map(t=>(0,A.jsx)(R,{r:t,fn:e,fm:n},t.id+o+e.name))}function B(){let[e,t]=(0,w.useState)(`ravine`),n=j[e],r=(0,w.useMemo)(()=>F(n),[n]),[i,a]=(0,w.useState)(n.start),[o,s]=(0,w.useState)(Math.log10(n.lr[0])),[c,l]=(0,w.useState)(Math.log10(n.lr[1])),[u,d]=(0,w.useState)(30),[p,m]=(0,w.useState)(0),[,y]=(0,w.useState)(0),C=(0,w.useRef)({start:n.start,runners:[]});C.current.start=i;let T=(0,w.useRef)([0,0]);T.current=[10**o,10**c];let E=(0,w.useRef)(30);E.current=u;let D=e=>{t(e);let n=j[e];a(n.start),s(Math.log10(n.lr[0])),l(Math.log10(n.lr[1])),m(e=>e+1)};(0,w.useEffect)(()=>{let e=setInterval(()=>y(e=>e+1),250);return()=>clearInterval(e)},[]);let O=C.current.runners;return(0,A.jsxs)(A.Fragment,{children:[(0,A.jsx)(h,{height:430,camera:[7.5,7,9],overlay:(0,A.jsxs)(A.Fragment,{children:[(0,A.jsx)(`b`,{children:`Click the surface`}),` to choose a start point.`,(0,A.jsx)(`br`,{}),`Colour = loss (bright = high). Height = log(1+loss).`]}),hint:`drag to orbit · click surface to move the start`,children:(0,A.jsxs)(`group`,{position:[0,-1,0],children:[(0,A.jsx)(L,{fn:n,fm:r,onPick:e=>{a(e),m(e=>e+1)}}),(0,A.jsx)(z,{fn:n,fm:r,sim:C.current,lrRef:T,speedRef:E,runKey:p})]})}),(0,A.jsxs)(v,{children:[(0,A.jsx)(S,{label:`Surface`,value:e,onChange:D,options:Object.entries(j).map(([e,t])=>[e,t.name])}),(0,A.jsx)(_,{label:`lr (SGD, Momentum)`,min:-4,max:0,step:.05,value:o,onChange:s,fmt:e=>(10**e).toPrecision(2)}),(0,A.jsx)(_,{label:`lr (RMSProp, Adam)`,min:-3,max:0,step:.05,value:c,onChange:l,fmt:e=>(10**e).toPrecision(2)}),(0,A.jsx)(_,{label:`Steps/s`,min:5,max:200,value:u,onChange:d}),(0,A.jsx)(x,{onClick:()=>m(e=>e+1),children:`Restart`})]}),(0,A.jsx)(g,{items:M.map(e=>[e.color,e.name])}),(0,A.jsx)(b,{children:O.map(e=>(0,A.jsxs)(`span`,{style:{marginRight:22},children:[e.name,`: `,(0,A.jsx)(`b`,{style:{color:e.color},children:Number.isFinite(e.loss)?e.loss.toPrecision(3):`—`}),` `,(0,A.jsxs)(`span`,{style:{color:f.dim},children:[`@ `,e.t]})]},e.id))})]})}var V=[{id:`const`,short:`Const`,name:`Constant`,color:f.mute,f:(e,t,n,r)=>r},{id:`step`,short:`Step`,name:`Step decay (×0.1 ×3)`,color:f.r,f:(e,t,n,r)=>r*.1**Math.floor(e/(t/3))},{id:`cos`,short:`Cos`,name:`Cosine`,color:f.b,f:(e,t,n,r)=>.5*r*(1+Math.cos(Math.PI*e/t))},{id:`wcos`,short:`W+cos`,name:`Warmup + cosine`,color:f.a,f:(e,t,n,r)=>e<n?r*e/n:.5*r*(1+Math.cos(Math.PI*(e-n)/(t-n)))},{id:`wlin`,short:`W+lin`,name:`Warmup + linear`,color:f.e,f:(e,t,n,r)=>e<n?r*e/n:r*(1-(e-n)/(t-n))},{id:`noam`,short:`Noam`,name:`Inverse-sqrt (Noam)`,color:f.d,f:(e,t,n,r)=>r*Math.min((e+1)/n,Math.sqrt(n/(e+1)))},{id:`one`,short:`1cycle`,name:`One-cycle`,color:f.c,f:(e,t,n,r)=>{let i=e/t;return i<.3?r/25+(r-r/25)*.5*(1-Math.cos(Math.PI*i/.3)):r+(r/1e4-r)*.5*(1-Math.cos(Math.PI*(i-.3)/.7))}}];function H(){let[e,t]=(0,w.useState)(1e4),[n,r]=(0,w.useState)(500),[i,a]=(0,w.useState)(3e-4),[o,s]=(0,w.useState)(2500),[c]=C(330,(t,r,a)=>{let s=r-60-16,c=a-54,l=t=>60+t/e*s,d=e=>16+c-e/(i*1.08)*c;t.strokeStyle=f.grid,t.lineWidth=1;for(let e=0;e<=4;e++){let n=i*e/4;t.beginPath(),t.moveTo(60,d(n)),t.lineTo(60+s,d(n)),t.stroke(),m(t,n.toExponential(1),52,d(n),{size:10.5,align:`right`,mono:!0,color:f.mute})}for(let n=0;n<=5;n++)m(t,Math.round(e*n/5),l(e*n/5),16+c+14,{size:10.5,align:`center`,mono:!0,color:f.mute});m(t,`step`,60+s/2,a-8,{size:11,align:`center`,color:f.mute}),V.forEach(r=>{let a=[];for(let t=0;t<=240;t++){let o=e*t/240;a.push([l(o),d(Math.max(0,r.f(Math.min(o,e-1),e,Math.min(n,e/2),i)))])}u(t,a,r.color,2.2)}),t.strokeStyle=`rgba(255,255,255,.3)`,t.setLineDash([3,3]),t.beginPath(),t.moveTo(l(o),16),t.lineTo(l(o),16+c),t.stroke(),t.setLineDash([])});return(0,A.jsxs)(A.Fragment,{children:[(0,A.jsx)(`canvas`,{...c}),(0,A.jsxs)(v,{children:[(0,A.jsx)(_,{label:`Total steps`,min:2e3,max:5e4,step:1e3,value:e,onChange:t}),(0,A.jsx)(_,{label:`Warmup steps`,min:0,max:2e3,step:50,value:n,onChange:r}),(0,A.jsx)(_,{label:`Peak lr`,min:-5,max:-2,step:.1,value:Math.log10(i),onChange:e=>a(10**e),fmt:e=>(10**e).toExponential(1)}),(0,A.jsx)(_,{label:`Inspect step`,min:0,max:e-1,step:50,value:Math.min(o,e-1),onChange:s})]}),(0,A.jsx)(g,{items:V.map(e=>[e.color,e.name])}),(0,A.jsx)(b,{children:V.map(t=>(0,A.jsxs)(`span`,{style:{marginRight:18},children:[t.short,`: `,(0,A.jsx)(`b`,{style:{color:t.color},children:Math.max(0,t.f(Math.min(o,e-1),e,Math.min(n,e/2)||1,i)).toExponential(2)})]},t.id))})]})}function U(){return(0,A.jsx)(y,{views:[{id:`race`,label:`3D optimizer race`,render:()=>(0,A.jsx)(B,{})},{id:`sched`,label:`Learning-rate schedules`,render:()=>(0,A.jsx)(H,{})}]})}var W={Lab:U,vizTitle:`Race SGD, Momentum, RMSProp and Adam across a loss landscape`,tryIt:[`On the **Ravine**, watch SGD zig-zag across the narrow axis while Momentum builds speed along the valley.`,`On **Himmelblau** click different start points — different optimizers can fall into *different* minima.`,`On **Rosenbrock**, crank the SGD learning rate up until it diverges (the marker gets clamped to the wall).`,`In the schedules view, add **warmup** and compare cosine, linear and inverse-sqrt decay shapes.`],theory:T,math:E,practice:D,code:[{title:`SGD, Momentum, RMSProp and Adam from scratch (NumPy)`,lang:`python`,note:`Exactly the update rules used by the 3D lab. Runs on the ravine and prints the final loss for each.`,src:O},{title:`Production training loop: AdamW + warmup-cosine + clipping + AMP + accumulation`,lang:`python`,src:k}],quiz:[{q:`For gradient descent on a quadratic with largest Hessian eigenvalue $\\lambda_{max}$, the largest stable learning rate is:`,options:[`$1/\\lambda_{max}$`,`$2/\\lambda_{max}$`,`$\\lambda_{max}$`,`Independent of curvature`],answer:1,why:`Each eigen-direction contracts by $|1-\\eta\\lambda|$, which is $<1$ iff $\\eta<2/\\lambda$.`},{q:`Why does momentum help in a ravine?`,options:[`It reduces memory use`,`It cancels oscillating components and accumulates the consistent one`,`It adds noise that escapes minima`,`It makes the loss convex`],answer:1,why:`Gradient components across the valley alternate sign and average out; the component along the valley accumulates.`},{q:`With bias correction, the very first Adam step has magnitude about:`,options:[`$\\eta\\,g$`,`$\\eta\\,g^2$`,`$\\eta\\,\\mathrm{sign}(g)$`,`$0$`],answer:2,why:`$\\hat m_1=g,\\ \\hat v_1=g^2$ so the step is $\\eta g/|g|=\\eta\\,\\mathrm{sign}(g)$.`},{q:`What is the difference between L2 regularization and decoupled weight decay (AdamW)?`,options:[`None`,`L2 is added to the gradient and so gets rescaled by Adam; AdamW shrinks weights directly`,`AdamW only decays biases`,`L2 is used only at inference`],answer:1,why:`Inside Adam the L2 term is divided by $\\sqrt{\\hat v}$, weakening the decay for large-gradient weights; AdamW applies decay outside the adaptive step.`},{q:`Why is learning-rate warmup common for Transformers?`,options:[`To save memory`,`Early second-moment estimates and attention logits are unreliable, so large steps destabilize training`,`To increase batch size`,`It is required by softmax`],answer:1,why:`Adam's variance estimate is poor in the first steps and post-LN Transformers are sensitive early; ramping the LR avoids divergence.`}]};export{W as default};