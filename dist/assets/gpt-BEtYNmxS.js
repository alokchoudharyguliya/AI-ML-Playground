import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{c as r,f as i,o as a,s as o,t as s,v as c,y as l}from"./viz-CPys2405.js";import{c as u,i as d,l as f,o as p,r as m,s as h,t as g,u as _}from"./hooks-Dw7oo1m3.js";var v=e(t(),1),y=`## One objective to rule them all

A **GPT-style** model is a Transformer *decoder* (causal mask, no encoder) trained on a single, brutally simple objective: **predict the next token**. Factorise the probability of a text with the chain rule,

$$p(x_1,\\dots,x_n)=\\prod_{t=1}^{n}p(x_t\\mid x_{<t}),$$

and minimise the cross-entropy at *every* position of *every* sequence. No labels, no task engineering — just trillions of tokens of text and code. Everything else (translation, summarisation, arithmetic, reasoning, tool use) must be learned *in service of* predicting the next token well.

| Generation | Year | Params | Notes |
|---|---|---|---|
| GPT-1 | 2018 | 117 M | pre-train + fine-tune |
| GPT-2 | 2019 | 1.5 B | zero-shot task transfer emerges |
| GPT-3 | 2020 | 175 B | **in-context learning**: examples in the prompt, no gradient updates |
| InstructGPT / ChatGPT | 2022 | — | supervised fine-tuning + RLHF → follows instructions |
| GPT-4, Llama 2/3, Mistral, Claude, Gemini… | 2023– | 7 B – 1 T+ | scaling, data quality, MoE, multimodality, long context |

## From base model to assistant

1. **Pre-training** (≈ 99% of compute): next-token prediction on web, books, code.
2. **Supervised fine-tuning (SFT)** on curated instruction–response pairs.
3. **Preference optimisation**: RLHF (reward model + PPO), or DPO / IPO / KTO directly on preference pairs; RL against verifiable rewards for reasoning.
4. **Safety and tool training**: refusal behaviour, function calling, retrieval.

## Decoding: turning probabilities into text

The model outputs logits over the vocabulary. *How you pick the next token* changes the text dramatically (first lab):

- **Greedy** ($T\\to0$): always the argmax — deterministic, repetitive, can loop.
- **Beam search**: tracks the best $k$ sequences; good for translation/ASR, dull for open-ended text.
- **Temperature** $T$: divide logits by $T$ before the softmax.
- **Top-k**: sample only from the $k$ most likely tokens.
- **Top-p (nucleus)**: sample from the smallest set whose cumulative probability $\\ge p$ — adapts to confidence.
- **Min-p**: drop tokens with probability $<\\text{min\\_p}\\times p_{max}$.
- **Penalties**: repetition / frequency / presence penalties discourage loops.
- **Constrained decoding**: mask tokens that would violate a JSON schema or grammar (guaranteed valid output).
- **Speculative decoding**: a small draft model proposes $\\gamma$ tokens, the big model verifies them in one parallel pass — 2–3× faster with *identical* output distribution.

## Inference has two very different phases

| Phase | What happens | Bound by |
|---|---|---|
| **Prefill** | process the whole prompt in parallel; build the KV cache | **compute** (big GEMMs, Tensor Cores) |
| **Decode** | one token at a time, reading all weights + KV cache | **memory bandwidth** |

**KV cache.** At step $t$ the new token's query must attend to the keys/values of all previous tokens. Those never change (causal mask), so we cache them: each step computes only one new $(k,v)$ per layer instead of re-encoding the entire prefix — $O(N)$ instead of $O(N^2)$ passes. The price is memory: $2\\cdot L\\cdot n_{kv}\\cdot d_{head}\\cdot\\text{bytes}$ per token. For Llama-3-8B at 128k context that is 16 GB *per sequence* — often more than the weights. Second lab tab.

How production systems cope:

- **GQA / MQA** — share K/V heads across query heads (4–8× smaller cache).
- **PagedAttention** (vLLM) — page-table the cache to eliminate fragmentation.
- **Continuous batching** — admit/evict requests every step so the GPU stays full.
- **KV quantisation** (fp8 / int4), **prefix caching** (reuse the cache of a shared system prompt), **sliding-window / sparse** attention, **FlashAttention / FlashDecoding** kernels.
- **Tensor / pipeline parallelism** for models larger than one GPU.

Metrics that matter: **TTFT** (time to first token, prefill-bound), **TPOT** (time per output token, decode-bound), throughput (tokens/s/GPU) and cost per million tokens.

## In-context learning and prompting

Large models perform **in-context learning**: put a few examples in the prompt and the model infers the task without weight updates. Mechanistically, *induction heads* (attend to the token after a previous occurrence of the current token) and larger circuits implement pattern completion. Prompt techniques — few-shot examples, system prompts, **chain-of-thought** ("think step by step"), self-consistency, tool use and retrieval-augmented generation — all work by shaping the context the model conditions on.

## Known failure modes

- **Hallucination**: fluent but false content; mitigated by retrieval, tool use, calibration training and verification.
- **Exposure bias & repetition** under greedy decoding.
- **Context-window limits** and "lost in the middle" effects.
- **Tokenisation artefacts**: counting letters, arithmetic on long numbers, non-English cost.
- **Prompt injection** and jailbreaks in tool-using agents.
- **Evaluation contamination**: benchmark leakage in training data.
`,b=`## Language-model loss, perplexity, bits

$$
\\mathcal L=-\\frac1N\\sum_{t=1}^{N}\\log p_\\theta(x_t\\mid x_{<t}),\\qquad
\\mathrm{PPL}=e^{\\mathcal L},\\qquad
\\text{bits/token}=\\mathcal L/\\ln2 .
$$

Perplexity is the effective branching factor: a uniform guess over $V$ tokens has PPL $=V$. Bits-per-*byte* normalises across tokenisers: $\\mathrm{BPB}=\\mathcal L\\cdot\\frac{\\#\\text{tokens}}{\\#\\text{bytes}\\cdot\\ln2}$.

## Temperature

$$
p_i(T)=\\frac{\\exp(z_i/T)}{\\sum_j\\exp(z_j/T)}.
$$

$T\\to0$: argmax; $T=1$: model distribution; $T\\to\\infty$: uniform. Entropy increases monotonically with $T$. The ratio of two probabilities changes as $p_i/p_j=\\exp\\big((z_i-z_j)/T\\big)$.

## Truncation samplers

Sort tokens by probability $p_{(1)}\\ge p_{(2)}\\ge\\cdots$.

**Top-k**: keep $\\{(1),\\dots,(k)\\}$, renormalise.

**Top-p** (nucleus): keep the smallest $m$ with $\\sum_{i\\le m}p_{(i)}\\ge p$.

**Min-p**: keep tokens with $p_i\\ge\\alpha\\,p_{(1)}$.

All then renormalise: $\\tilde p_i=p_i\\,\\mathbb 1[i\\in S]\\big/\\sum_{j\\in S}p_j$. The lab applies temperature → top-k → top-p in that order (as Hugging Face does).

## Penalties

Repetition penalty $\\theta>1$ on tokens already generated (CTRL-style): $z_i\\leftarrow z_i/\\theta$ if $z_i>0$ else $z_i\\theta$. Frequency / presence penalties (OpenAI): $z_i\\leftarrow z_i-\\alpha_f\\,c_i-\\alpha_p\\,\\mathbb 1[c_i>0]$, with $c_i$ the count so far.

## Beam search

Keep the $B$ highest-scoring partial sequences $y_{1:t}$ with score $\\sum\\log p(y_t\\mid y_{<t})$, optionally divided by $\\mathrm{len}^{\\alpha}$ to avoid short-sequence bias.

## KV-cache memory

$$
\\text{KV bytes}=2\\;\\underbrace{L}_{\\text{layers}}\\;\\underbrace{n_{kv}}_{\\text{KV heads}}\\;\\underbrace{d_{head}}_{\\text{head dim}}\\;\\underbrace{b}_{\\text{bytes}}\\;\\times\\;\\underbrace{n}_{\\text{tokens}}\\times\\underbrace{B}_{\\text{batch}} .
$$

Llama-3-8B ($L=32,n_{kv}=8,d_{head}=128$, fp16): $2\\cdot32\\cdot8\\cdot128\\cdot2=131{,}072$ B $=128$ KiB/token → $1$ GiB at 8k context, $16$ GiB at 128k, per sequence.

**Cost without cache.** Generating $N$ tokens by re-running the prefix each step costs $\\sum_{t=1}^N t=\\tfrac{N(N+1)}2$ token-forward passes; with a cache, $N$.

## Decode is memory-bound

For batch size $B$, one decode step performs about $2NB$ FLOPs on weights (with $N$ parameters) but must read $2N$ bytes of fp16 weights (plus the KV cache) from HBM:

$$
\\text{arithmetic intensity}\\approx\\frac{2NB}{2N+\\text{KV}}\\approx B\\ \\text{FLOP/byte}.
$$

A GPU with bandwidth $\\mathrm{BW}$ and peak $\\Pi$ has ridge point $\\Pi/\\mathrm{BW}$ (≈ 150–300 FLOP/B for H100 tensor cores). Below it the speed limit is

$$
\\text{tokens/s per stream}\\le\\frac{\\mathrm{BW}}{2N+\\text{KV bytes}},
$$

e.g. an 8B fp16 model on 3.35 TB/s: $3.35\\times10^{12}/16\\times10^{9}\\approx210$ tokens/s upper bound at batch 1. Batching raises throughput almost linearly until the ridge point.

## Speculative decoding

With draft length $\\gamma$ and per-token acceptance rate $\\alpha$ (i.i.d.), the expected tokens produced per target-model pass is

$$
\\mathbb E[\\text{tokens}]=\\frac{1-\\alpha^{\\gamma+1}}{1-\\alpha}.
$$

For $\\alpha=0.8,\\gamma=4$: $\\frac{1-0.8^5}{0.2}=3.36$ tokens per big-model step. Accept/reject uses $\\min\\big(1,p_{target}(x)/p_{draft}(x)\\big)$, preserving the target distribution exactly.

## Scaling of FLOPs

Training: $\\approx6ND$ FLOPs for $N$ parameters and $D$ tokens. Inference: $\\approx2N$ FLOPs/token plus $\\approx4Lnd$ for attention over $n$ cached tokens.
`,x=`## Exercises

**Q1.** Logits for three tokens are $(4,2,0)$. Compute the probabilities at $T=1$ and $T=0.5$.

<details>
<summary>Show answer</summary>

$T=1$: $e^{(4,2,0)}=(54.6,7.39,1)\\Rightarrow(0.867,0.117,0.016)$. $T=0.5$: logits $(8,4,0)$ → $(2981,54.6,1)\\Rightarrow(0.982,0.018,0.0003)$. Lower temperature concentrates mass on the top token.

</details>

**Q2.** With probabilities $(0.5,0.3,0.1,0.05,0.03,0.02)$, which tokens survive top-k=3 and top-p=0.85? What are the renormalised probabilities for top-p?

<details>
<summary>Show answer</summary>

Top-k=3 → tokens 1–3 (mass 0.9). Top-p=0.85: cumulative $0.5,0.8,0.9\\ge0.85$ → keep tokens 1–3 as well; renormalised by 0.9: $(0.556,0.333,0.111)$. With $p=0.75$ only tokens 1–2 would stay ($0.5,0.8$) → $(0.625,0.375)$.

</details>

**Q3.** Compute the KV-cache size for Mistral-7B ($L=32$, 8 KV heads, $d_{head}=128$) at 32k context, batch 16, fp16.

<details>
<summary>Show answer</summary>

Per token: $2\\cdot32\\cdot8\\cdot128\\cdot2=131{,}072$ B. Total: $131{,}072\\cdot32{,}768\\cdot16=6.87\\times10^{10}$ B $\\approx64$ GiB — more than four times the 14.5 GB of weights. Quantising KV to fp8 halves it; paging eliminates waste.

</details>

**Q4.** Estimate the max decode throughput of a 70B fp16 model at batch 1 on an H100 (3.35 TB/s). Why does it not fit on one GPU, and what changes with 8-way tensor parallelism?

<details>
<summary>Show answer</summary>

Weights: 140 GB > 80 GB, so it needs ≥2 GPUs (or 4-bit quantisation: 35 GB). Bandwidth bound: $3.35\\times10^{12}/1.4\\times10^{11}\\approx24$ tokens/s per GPU-equivalent. With TP=8 each GPU reads 1/8 of the weights concurrently: ideal $\\approx190$ tokens/s, reduced by all-reduce latency.

</details>

**Q5.** Speculative decoding with $\\alpha=0.7$, $\\gamma=5$, draft cost 5% of the target. Estimate the speed-up.

<details>
<summary>Show answer</summary>

Tokens per cycle: $\\frac{1-0.7^6}{0.3}=2.94$. Cost per cycle: 1 target pass + 5 draft passes $=1+5\\cdot0.05=1.25$ target-equivalents. Speed-up $\\approx2.94/1.25\\approx2.4\\times$.

</details>

**Q6 (code).** Prove numerically that incremental decoding with a KV cache gives the same logits as recomputing the whole prefix.

<details>
<summary>Show answer</summary>

\`kv_cache.py\` implements both paths for a single attention layer and asserts \`torch.allclose\` on the last-token output at every step; the cache path touches one token per step.

</details>

## In practice

- **Sampling presets**: factual/coding → $T\\in[0,0.3]$; creative writing → $T\\approx0.8$–$1.0$ with top-p 0.9–0.95; always cap \`max_tokens\`.
- **Serving stacks**: vLLM, TensorRT-LLM, SGLang, llama.cpp (CPU/Metal, GGUF quantisation), TGI. Choose by hardware, latency vs throughput goals.
- **Quantisation**: weights int8/int4 (GPTQ, AWQ), KV fp8 — 2–4× less memory traffic with small quality loss.
- **Evaluation**: perplexity is necessary but insufficient — use task evals, human preference and regression suites; watch for contamination.
- **Cost model**: price ≈ GPU-hour cost ÷ (tokens/s × 3600); batching and caching dominate economics.
- **RAG**: retrieve passages → put in context; use prefix caching for static instructions.

## Common pitfalls

- Using temperature 0 with sampling code that divides by zero (special-case greedy).
- Applying top-p before temperature (order matters).
- Forgetting to apply the *same* tokenizer chat template used in training.
- Position-id bugs when decoding with a cache (off-by-one) or with left-padded batches.
- Comparing perplexities across different tokenisers.
- Assuming longer context = better recall (needle-in-a-haystack ≠ reasoning over long documents).
`,S=`import numpy as np


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, min_p=None, rng=None):
    """Temperature -> top-k -> top-p -> min-p, then sample. Returns (token_id, final_probs)."""
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0:                                   # greedy
        probs = np.zeros_like(logits); probs[logits.argmax()] = 1.0
        return int(logits.argmax()), probs

    probs = softmax(logits / temperature)
    order = np.argsort(-probs)                             # indices by descending probability
    keep = np.zeros_like(probs, dtype=bool)

    k = len(probs) if top_k is None else top_k
    keep[order[:k]] = True                                 # top-k

    if top_p is not None:                                  # nucleus on the renormalised top-k set
        p = np.where(keep, probs, 0.0); p /= p.sum()
        cum = np.cumsum(p[order])
        cutoff = np.searchsorted(cum, top_p) + 1           # smallest prefix with cum >= top_p
        nucleus = np.zeros_like(keep); nucleus[order[:cutoff]] = True
        keep &= nucleus

    if min_p is not None:
        keep &= probs >= min_p * probs[keep].max()

    final = np.where(keep, probs, 0.0)
    final /= final.sum()
    return int(rng.choice(len(final), p=final)), final


if __name__ == "__main__":
    logits = np.log([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])

    _, p = sample_next(logits, top_k=3);    print("top-k=3      :", p.round(3))
    _, p = sample_next(logits, top_p=0.85); print("top-p=0.85   :", p.round(3))
    _, p = sample_next(logits, top_p=0.75); print("top-p=0.75   :", p.round(3))
    _, p = sample_next(logits, min_p=0.2);  print("min-p=0.2    :", p.round(3))
    _, p = sample_next(logits, temperature=0.5); print("T=0.5        :", p.round(3))

    rng = np.random.default_rng(0)
    draws = [sample_next(logits, temperature=1.0, top_p=0.9, rng=rng)[0] for _ in range(20000)]
    print("empirical    :", (np.bincount(draws, minlength=6) / 20000).round(3))
    assert sample_next(logits, temperature=0)[0] == 0
`,C=`import math
import torch
import torch.nn as nn


class CachedAttention(nn.Module):
    """Single causal attention layer that supports both full-prefix and cached decoding."""

    def __init__(self, d_model=64, n_heads=4, n_kv_heads=2):
        super().__init__()
        self.h, self.kvh, self.dh = n_heads, n_kv_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)   # GQA: fewer K/V heads
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)        # (B, H, T, dh)
        k = self.wk(x).view(B, T, self.kvh, self.dh).transpose(1, 2)      # (B, KVH, T, dh)
        v = self.wv(x).view(B, T, self.kvh, self.dh).transpose(1, 2)

        if cache is not None:                                             # append new K/V to the cache
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        rep = self.h // self.kvh                                          # share each KV head across \`rep\` query heads
        kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)

        S = kk.size(2)
        scores = q @ kk.transpose(-2, -1) / math.sqrt(self.dh)            # (B, H, T, S)
        # query i (absolute position S-T+i) may see keys <= its own position
        mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
        scores = scores.masked_fill(~mask, float("-inf"))
        out = (scores.softmax(-1) @ vv).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out), new_cache


if __name__ == "__main__":
    torch.manual_seed(0)
    attn = CachedAttention().eval()
    x = torch.randn(1, 12, 64)

    with torch.no_grad():
        full, _ = attn(x)                                   # recompute the whole prefix once

        cache, outs = None, []
        for t in range(x.size(1)):                          # one token per step, reusing the cache
            y, cache = attn(x[:, t:t + 1], cache)
            outs.append(y)
        cached = torch.cat(outs, 1)

    print("max |full - cached| =", (full - cached).abs().max().item())
    kv_bytes = sum(t.numel() * t.element_size() for t in cache)
    print("cache after 12 tokens:", tuple(cache[0].shape), f"-> {kv_bytes} bytes (GQA: 2 KV heads vs 4 query heads)")
`,w=n(),T=[[` Paris`,8.6],[` the`,5.4],[` a`,5],[` located`,4.2],[` Lyon`,4],[` known`,3.4],[` France`,3.1],[` not`,2.9],[` also`,2.7],[` Marseille`,2.4],[` one`,2.2],[` city`,2],[` Berlin`,1.2],[` banana`,-2]],E=T.map(e=>e[1]);function D(e,t,n){let r=c(E,e),i=r.map((e,t)=>t).sort((e,t)=>r[t]-r[e]),a=new Set(i.slice(0,t)),o=[...a].reduce((e,t)=>e+r[t],0),s=0,l=new Set;for(let e of i)if(a.has(e)&&(l.add(e),s+=r[e]/o,s>=n))break;a=l;let u=[...a].reduce((e,t)=>e+r[t],0);return{base:r,final:r.map((e,t)=>a.has(t)?e/u:0),keep:a}}function O(){let[e,t]=(0,v.useState)(1),[n,r]=(0,v.useState)(14),[i,o]=(0,v.useState)(1),[f,h]=(0,v.useState)(null),[_,y]=(0,v.useState)(null),{base:b,final:x,keep:S}=(0,v.useMemo)(()=>D(e,n,i),[e,n,i]),C=(0,v.useMemo)(()=>c(E,1),[]),O=e=>{let t=f&&e===1?[...f]:Array(T.length).fill(0),n=0;for(let r=0;r<e;r++){let e=Math.random(),r=0,i=T.length-1;for(let t=0;t<x.length;t++)if(r+=x[t],e<=r){i=t;break}t[i]++,n=i}h(t),y(n)},k=f?f.reduce((e,t)=>e+t,0):0,[A]=g(340,(e,t,n)=>{let r=t-60,i=r/T.length,a=n-62,o=n-120;l(e,`Prompt: “The capital of France is”  →  next-token distribution`,30,20,{size:13,color:`#fff`,weight:600}),e.strokeStyle=s.grid,e.lineWidth=1,[.25,.5,.75,1].forEach(t=>{e.beginPath(),e.moveTo(30,a-t*o),e.lineTo(30+r,a-t*o),e.stroke(),l(e,t,24,a-t*o,{size:9.5,align:`right`,color:s.dim,mono:!0})}),T.forEach(([t],n)=>{let r=30+n*i+i/2,c=i*.62;e.strokeStyle=`rgba(255,255,255,.28)`,e.setLineDash([3,3]),e.lineWidth=1.2,e.strokeRect(r-c/2,a-C[n]*o,c,C[n]*o),e.setLineDash([]);let u=S.has(n);e.fillStyle=u?n===_?s.d:s.a:`#222842`;let d=(u?x[n]:b[n])*o;if(e.globalAlpha=u?.95:.6,e.fillRect(r-c/2,a-d,c,Math.max(d,1)),e.globalAlpha=1,u||(e.strokeStyle=s.r,e.lineWidth=1.5,e.beginPath(),e.moveTo(r-c/2,a-d),e.lineTo(r+c/2,a),e.stroke()),l(e,u?x[n].toFixed(2):`✕`,r,a-d-10,{size:10,align:`center`,mono:!0,color:u?`#fff`:s.r}),e.save(),e.translate(r,a+10),e.rotate(.55),l(e,t,0,0,{size:11,color:u?`#fff`:s.dim,weight:600}),e.restore(),f&&k){let t=f[n]/k;e.fillStyle=s.e,e.fillRect(r-c/2-3,a-t*o,3,t*o)}}),l(e,`dashed = raw model (T=1)   solid = after temperature / top-k / top-p   green ticks = empirical samples`,30,n-10,{size:10.5,color:s.mute})}),j=a(x);return(0,w.jsxs)(w.Fragment,{children:[(0,w.jsx)(`canvas`,{...A}),(0,w.jsxs)(d,{children:[(0,w.jsx)(u,{label:`Temperature`,min:.05,max:3,step:.05,value:e,onChange:t,fmt:e=>e.toFixed(2)}),(0,w.jsx)(u,{label:`Top-k`,min:1,max:14,value:n,onChange:r,fmt:e=>e===14?`off`:e}),(0,w.jsx)(u,{label:`Top-p`,min:.05,max:1,step:.01,value:i,onChange:o,fmt:e=>e===1?`off`:e.toFixed(2)}),(0,w.jsx)(m,{primary:!0,onClick:()=>O(1),children:`Sample 1`}),(0,w.jsx)(m,{onClick:()=>O(1e3),children:`Sample 1000`})]}),(0,w.jsxs)(p,{children:[`Kept `,(0,w.jsx)(`b`,{children:S.size}),` of `,T.length,` tokens · entropy `,(0,w.jsx)(`b`,{children:j.toFixed(2)}),` bits · P(“Paris”) = `,(0,w.jsxs)(`b`,{children:[(x[0]*100).toFixed(1),`%`]}),` `,_!=null&&(0,w.jsxs)(w.Fragment,{children:[` · last draw: `,(0,w.jsx)(`b`,{className:`w`,children:T[_][0].trim()})]}),`. `,(0,w.jsx)(`b`,{children:`T→0`}),` is greedy; `,(0,w.jsx)(`b`,{children:`T>1`}),` flattens (creative, error-prone); `,(0,w.jsx)(`b`,{children:`top-k`}),` cuts a fixed count, `,(0,w.jsx)(`b`,{children:`top-p`}),` cuts by cumulative probability so it adapts when the model is confident vs uncertain.`]})]})}var k={l3_8:{name:`Llama-3-8B (GQA)`,L:32,heads:32,kv:8,hd:128,params:803e7},m7:{name:`Mistral-7B (GQA)`,L:32,heads:32,kv:8,hd:128,params:724e7},l2_7:{name:`Llama-2-7B (MHA)`,L:32,heads:32,kv:32,hd:128,params:674e7},l3_70:{name:`Llama-3-70B (GQA)`,L:80,heads:64,kv:8,hd:128,params:706e8},g3:{name:`GPT-3 175B (MHA)`,L:96,heads:96,kv:96,hd:128,params:175e9}},A={a100:[`A100 80GB`,8e10,2e12],h100:[`H100 80GB`,8e10,335e10],l4:[`L4 24GB`,24e9,3e11]};function j(){let[e,t]=(0,v.useState)(`l3_8`),[n,a]=(0,v.useState)(8192),[c,f]=(0,v.useState)(8),[m,y]=(0,v.useState)(2),[b,x]=(0,v.useState)(`h100`),[S,C]=(0,v.useState)(!1),T=k[e],E=S?T.heads:T.kv,D=2*T.L*E*T.hd*m,O=D*n*c,j=T.params*2,[,M,N]=A[b],P=N/(j+O)*c,[F]=g(300,(e,t,n)=>{let a=t*.46;l(e,`GPU memory — ${A[b][0]}`,24,22,{size:12,color:s.mute,weight:600});let c=Math.max(M,j+O)*1.02,u=e=>24+e/c*a;e.fillStyle=s.a,e.fillRect(24,52,u(j)-24,38),e.fillStyle=s.c,e.fillRect(u(j),52,u(j+O)-u(j),38),e.strokeStyle=`#fff`,e.lineWidth=2,e.setLineDash([5,3]),e.beginPath(),e.moveTo(u(M),42),e.lineTo(u(M),100),e.stroke(),e.setLineDash([]);let d=j+O>M;l(e,`weights ${o(j)}`,24,112,{size:12,color:s.a,mono:!0,weight:600}),l(e,`KV cache ${o(O)}`,214,112,{size:12,color:s.c,mono:!0,weight:600}),l(e,d?`OUT OF MEMORY`:`fits`,u(M)-6,36,{size:12,align:`right`,color:d?s.r:s.e,mono:!0,weight:700}),l(e,`${(D/1024).toFixed(0)} KB of KV per token`,24,144,{size:11.5,color:s.ink,mono:!0}),l(e,`= 2 × ${T.L} layers × ${E} KV-heads × ${T.hd} × ${m} B`,24,186,{size:10.5,color:s.mute,mono:!0}),l(e,`decode ≤ BW / (weights + KV) × batch`,24,202,{size:11.5,color:s.d,mono:!0}),l(e,`≈ ${r(Math.round(P))} tokens/s across all streams`,24,222,{size:11.5,color:s.d,mono:!0});let f=t*.56,p=t-f-20,h=n-90;l(e,`Work to generate N tokens (token-forward-passes)`,f,22,{size:12,color:s.mute,weight:600});let g=e=>e*(e+1)/2,_=e=>f+e/256*p,v=e=>44+h-e/g(256)*h;e.strokeStyle=s.grid,e.lineWidth=1,e.strokeRect(f,44,p,h);let y=[],x=[];for(let e=0;e<=256;e+=4)y.push([_(e),v(g(e))]),x.push([_(e),v(e)]);i(e,y,s.r,2.4),i(e,x,s.e,2.4),l(e,`no cache: O(N²)`,f+p-8,58,{size:11,align:`right`,color:s.r,mono:!0}),l(e,`KV cache: O(N)`,f+p-8,44+h-10,{size:11,align:`right`,color:s.e,mono:!0}),l(e,`at N=256: ${r(g(256))} vs 256 passes (${Math.round(g(256)/256)}× less)`,f,44+h+22,{size:11.5,color:s.ink,mono:!0})});return(0,w.jsxs)(w.Fragment,{children:[(0,w.jsx)(`canvas`,{...F}),(0,w.jsxs)(d,{children:[(0,w.jsx)(h,{label:`Model`,value:e,onChange:t,options:Object.entries(k).map(([e,t])=>[e,t.name])}),(0,w.jsx)(h,{label:`GPU`,value:b,onChange:x,options:Object.entries(A).map(([e,t])=>[e,t[0]])}),(0,w.jsx)(u,{label:`Context`,min:1024,max:131072,step:1024,value:n,onChange:a,fmt:e=>Math.round(e/1024)+`k`}),(0,w.jsx)(u,{label:`Batch`,min:1,max:128,value:c,onChange:f}),(0,w.jsx)(h,{label:`KV dtype`,value:m,onChange:e=>y(+e),options:[[2,`fp16 / bf16`],[1,`fp8 / int8`],[.5,`4-bit`]]}),(0,w.jsx)(_,{label:`Pretend MHA (no GQA)`,value:S,onChange:C})]}),(0,w.jsxs)(p,{children:[`Autoregressive decoding is `,(0,w.jsx)(`b`,{children:`memory-bandwidth-bound`}),`: each new token must stream `,(0,w.jsx)(`i`,{children:`all`}),` weights plus the whole KV cache from HBM, doing only ~1–2 FLOPs per byte. That is why `,(0,w.jsx)(`b`,{children:`GQA`}),`, `,(0,w.jsx)(`b`,{children:`KV quantisation`}),`, `,(0,w.jsx)(`b`,{children:`batching`}),` and `,(0,w.jsx)(`b`,{children:`PagedAttention`}),` matter more for serving cost than raw TFLOPs. Switch off GQA to see the cache balloon by `,Math.round(T.heads/T.kv),`×.`]})]})}function M(){return(0,w.jsx)(f,{views:[{id:`sample`,label:`Sampling & decoding`,render:()=>(0,w.jsx)(O,{})},{id:`kv`,label:`KV cache & inference cost`,render:()=>(0,w.jsx)(j,{})}]})}var N={Lab:M,vizTitle:`Control next-token sampling and see what serving a long context costs`,tryIt:[`Set **temperature → 0.05** (greedy) and press *Sample 1000*: always “Paris”. Now go to **T = 2.5** and watch “banana” appear.`,`Use **top-p = 0.9** at T=1 versus **top-k = 3** and compare how many tokens survive.`,`In the KV tab, select **Llama-3-70B**, context **128k**, batch **8** on an A100 — then toggle GQA off.`,`Notice how the decode-speed bound changes with batch size: batching amortises the weight reads.`],theory:y,math:b,practice:x,code:[{title:`Temperature, top-k, top-p, min-p from scratch (NumPy, with tests)`,lang:`python`,src:S},{title:`Incremental decoding with a KV cache (PyTorch)`,lang:`python`,note:`Proves that cached decoding is numerically identical to recomputing the full prefix.`,src:C}],quiz:[{q:`What does a decoder-only LM optimize at pre-training time?`,options:[`Next-token cross-entropy over all positions`,`Masked tokens only`,`Sentence order`,`Image–text similarity`],answer:0,why:`With a causal mask every position gives a training signal: predict $x_{t+1}$ from $x_{\\le t}$.`},{q:`Raising the sampling temperature:`,options:[`Sharpens the distribution`,`Flattens the distribution (more randomness)`,`Changes the model weights`,`Shortens the output`],answer:1,why:`Logits are divided by $T$; $T>1$ reduces the gaps between them.`},{q:`Why is nucleus (top-p) often preferred to top-k?`,options:[`It is faster`,`The number of kept tokens adapts to the model's confidence`,`It needs no softmax`,`It is deterministic`],answer:1,why:`When the distribution is peaked few tokens reach the mass $p$; when it is flat many do.`},{q:`The KV cache converts the cost of generating $N$ tokens from roughly:`,options:[`$O(N)$ to $O(N^2)$`,`$O(N^2)$ passes to $O(N)$ passes (each attending to a growing cache)`,`$O(N^3)$ to $O(N)$`,`No change`],answer:1,why:`Past keys/values are reused, so each step processes one new token instead of the whole prefix.`},{q:`Single-stream LLM decoding is usually limited by:`,options:[`Tensor Core FLOPs`,`Memory bandwidth (weights + KV reads)`,`Disk speed`,`Softmax`],answer:1,why:`Arithmetic intensity at batch 1 is ≈ 1–2 FLOP/byte, far below the GPU ridge point.`}]};export{N as default};