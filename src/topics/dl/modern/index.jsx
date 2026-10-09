import React, { useMemo, useState } from 'react'
import { Controls, Slider, Select, Readout, SubViews, Toggle } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, rng, randn, softmax, text, rr, polyline, clamp, fmtInt, fmtBytes, alpha, mix } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import blocks from './code/modern_blocks.py?raw'
import moe from './code/moe_layer.py?raw'
import lora from './code/lora.py?raw'
import dpo from './code/dpo_loss.py?raw'

/* ---------- RoPE ---------- */
const NP = 8
const qv = (() => { const r = rng(4); return Array.from({ length: NP }, () => [randn(r), randn(r)]) })()
const kv = (() => { const r = rng(9); return Array.from({ length: NP }, () => [randn(r), randn(r)]) })()
const rot = ([x, y], a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)]
function ropeScore(m, n, base) {
  let s = 0
  for (let i = 0; i < NP; i++) {
    const th = Math.pow(base, -i / NP)
    const a = rot(qv[i], m * th), b = rot(kv[i], n * th)
    s += a[0] * b[0] + a[1] * b[1]
  }
  return s
}
function RopeLab() {
  const [m, setM] = useState(20)
  const [n, setN] = useState(12)
  const [base, setBase] = useState(10000)
  const [freq, setFreq] = useState(1)
  const [shift, setShift] = useState(30)
  const th = Math.pow(base, -freq / NP)
  const s1 = ropeScore(m, n, base), s2 = ropeScore(m + shift, n + shift, base)

  const [cp] = useCanvas(340, (ctx, W, H) => {
    const R = Math.min(H / 2 - 30, W * 0.2), cx = W * 0.22, cy = H / 2 + 6
    text(ctx, `Frequency pair #${freq}:  θ = base^(−${freq}/${NP}) = ${th.toFixed(4)} rad/token`, 20, 18, { size: 12, color: C.mute, weight: 600 })
    ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke()
    const sc = R / 2.4
    const a = rot(qv[freq], m * th), b = rot(kv[freq], n * th)
    const draw = (v, col, lab) => {
      const x = cx + v[0] * sc, y = cy - v[1] * sc
      ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke()
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.fill(); text(ctx, lab, x + 10, y - 10, { size: 12, color: col, mono: true, weight: 700 })
    }
    draw(a, C.b, `q@${m}`); draw(b, C.c, `k@${n}`)
    const ang = Math.atan2(b[1], b[0]) - Math.atan2(a[1], a[0])
    text(ctx, `angle between = (n−m)·θ = ${(((n - m) * th * 180 / Math.PI) % 360).toFixed(0)}°`, cx, cy + R + 18, { size: 11.5, align: 'center', color: C.d, mono: true })

    // score vs relative offset
    const px = W * 0.45, pw = W - px - 20, py = 40, ph = H - 90
    text(ctx, `attention logit  q·k  vs relative offset (n − m)   — sum over ${NP} frequency pairs`, px, 18, { size: 12, color: C.mute, weight: 600 })
    const D = 80, X = d => px + ((d + D) / (2 * D)) * pw
    const vals = []; for (let d = -D; d <= D; d++) vals.push([d, ropeScore(0, d, base)])
    const ymax = Math.max(...vals.map(v => Math.abs(v[1])), 1), Y = v => py + ph / 2 - (v / ymax) * (ph / 2)
    ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(px, Y(0)); ctx.lineTo(px + pw, Y(0)); ctx.stroke()
    polyline(ctx, vals.map(([d, v]) => [X(d), Y(v)]), C.a, 2.4)
    const cur = n - m
    if (Math.abs(cur) <= D) { ctx.fillStyle = C.d; ctx.beginPath(); ctx.arc(X(cur), Y(ropeScore(0, cur, base)), 6, 0, 7); ctx.fill() }
    text(ctx, '← keys before query', px, py + ph + 18, { size: 11, color: C.mute }); text(ctx, 'keys after query →', px + pw, py + ph + 18, { size: 11, align: 'right', color: C.mute })
    ;[-D, -D / 2, 0, D / 2, D].forEach(d => text(ctx, d, X(d), py + ph + 4, { size: 9.5, align: 'center', color: C.dim, mono: true }))
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Query position m" min={0} max={80} value={m} onChange={setM} />
        <Slider label="Key position n" min={0} max={80} value={n} onChange={setN} />
        <Slider label="Frequency pair" min={0} max={NP - 1} value={freq} onChange={setFreq} />
        <Slider label="RoPE base" min={100} max={1000000} step={100} value={base} onChange={setBase} fmt={v => (v >= 1e6 ? '1M' : v >= 1000 ? Math.round(v / 1000) + 'k' : v)} />
        <Slider label="Shift both by" min={0} max={200} value={shift} onChange={setShift} />
      </Controls>
      <Readout>score(m={m}, n={n}) = <b>{s1.toFixed(4)}</b> · score(m+{shift}, n+{shift}) = <b>{s2.toFixed(4)}</b> · difference <b className="g">{Math.abs(s1 - s2).toExponential(1)}</b> → the logit depends <b>only on n−m</b>, even though absolute positions changed. RoPE encodes <i>relative</i> position inside the dot product, with high-frequency pairs resolving nearby tokens and low-frequency pairs long range.</Readout>
    </>
  )
}

/* ---------- MoE ---------- */
const NT = 28
const TOPIC = Array.from({ length: NT }, (_, i) => (i * 7 + (i >> 2)) % 4)
const TCOL = [C.c, C.b, C.d, C.e]
function MoeLab() {
  const [E, setE] = useState(8)
  const [k, setK] = useState(2)
  const [cap, setCap] = useState(1.25)
  const [skew, setSkew] = useState(1.5)
  const R = useMemo(() => {
    const r = rng(31)
    const bias = Array.from({ length: 16 }, () => randn(r))
    const logits = TOPIC.map(tp => Array.from({ length: E }, (_, e) => (e % 4 === tp ? 2.2 : 0) + skew * bias[e] * 0.8 + (r() - 0.5) * 1.2))
    const probs = logits.map(l => softmax(l))
    const picks = logits.map(l => l.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(x => x[1]))
    const C_ = Math.ceil(cap * NT * k / E)
    const load = Array(E).fill(0), bins = Array.from({ length: E }, () => []), dropped = []
    picks.forEach((ps, t) => ps.forEach(e => { if (load[e] < C_) { load[e]++; bins[e].push(t) } else dropped.push([t, e]) }))
    const f = Array(E).fill(0); picks.forEach(ps => { f[ps[0]] += 1 / NT })
    const P = Array(E).fill(0); probs.forEach(p => p.forEach((v, e) => { P[e] += v / NT }))
    const aux = E * f.reduce((s, v, e) => s + v * P[e], 0)
    return { picks, C_, bins, dropped, aux, load }
  }, [E, k, cap, skew])

  const [cp] = useCanvas(380, (ctx, W, H) => {
    const ty = 40, tw = (W - 40) / NT
    text(ctx, 'tokens (colour = what the token is about)  →  router picks top-k experts', 20, 18, { size: 12, color: C.mute, weight: 600 })
    const ex = e => 20 + (e + 0.5) * ((W - 40) / E), ey = H - 150
    R.picks.forEach((ps, t) => ps.forEach((e, rank) => {
      const x = 20 + (t + 0.5) * tw
      const drop = R.dropped.some(([tt, ee]) => tt === t && ee === e)
      ctx.strokeStyle = drop ? alpha(C.r, 0.6) : alpha(TCOL[TOPIC[t]], rank ? 0.28 : 0.55); ctx.lineWidth = rank ? 1 : 1.6
      if (drop) ctx.setLineDash([3, 3])
      ctx.beginPath(); ctx.moveTo(x, ty + 12); ctx.bezierCurveTo(x, ty + 90, ex(e), ey - 90, ex(e), ey); ctx.stroke(); ctx.setLineDash([])
    }))
    TOPIC.forEach((tp, t) => { ctx.fillStyle = TCOL[tp]; ctx.beginPath(); ctx.arc(20 + (t + 0.5) * tw, ty, Math.min(8, tw / 2 - 1), 0, 7); ctx.fill() })
    const bw = Math.min(70, (W - 40) / E - 8), cellH = 12
    for (let e = 0; e < E; e++) {
      const x = ex(e)
      const over = R.load[e] >= R.C_
      ctx.fillStyle = '#10131f'; rr(ctx, x - bw / 2, ey, bw, 118, 8); ctx.fill(); ctx.strokeStyle = over ? C.d : C.line; ctx.lineWidth = over ? 2 : 1; ctx.stroke()
      R.bins[e].forEach((t, i) => { ctx.fillStyle = TCOL[TOPIC[t]]; ctx.fillRect(x - bw / 2 + 5 + (i % 5) * ((bw - 10) / 5), ey + 8 + Math.floor(i / 5) * (cellH + 1), (bw - 10) / 5 - 2, cellH) })
      text(ctx, `E${e}`, x, ey + 132, { size: 11, align: 'center', color: C.ink, weight: 700 })
      text(ctx, `${R.load[e]}/${R.C_}`, x, ey + 146, { size: 10, align: 'center', color: over ? C.d : C.mute, mono: true })
    }
  })
  const total = R.picks.length * k
  const balanced = R.aux < 1.15
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Experts E" min={4} max={16} step={2} value={E} onChange={setE} />
        <Slider label="Top-k" min={1} max={4} value={k} onChange={setK} />
        <Slider label="Capacity factor" min={0.8} max={2.5} step={0.05} value={cap} onChange={setCap} fmt={v => v.toFixed(2)} />
        <Slider label="Router imbalance" min={0} max={3} step={0.1} value={skew} onChange={setSkew} fmt={v => v.toFixed(1)} />
      </Controls>
      <Readout>Capacity per expert = ⌈{cap.toFixed(2)} × {NT}·{k} / {E}⌉ = <b>{R.C_}</b> slots · dropped assignments: <b className={R.dropped.length ? 'r' : 'g'}>{R.dropped.length}</b> of {total} · load-balance loss E·Σfᵢ·Pᵢ = <b className={balanced ? 'g' : 'w'}>{R.aux.toFixed(2)}</b> (1.0 = perfectly balanced). Only <b>{k}/{E}</b> of the expert parameters run per token — <b>{Math.round(100 * k / E)}% active compute</b>, 100% of the memory.</Readout>
    </>
  )
}

/* ---------- LoRA ---------- */
function LoraLab() {
  const [d, setD] = useState(4096)
  const [r, setR] = useState(16)
  const [Ln, setLn] = useState(32)
  const [tgt, setTgt] = useState('qv')
  const dff = Math.round(3.5 * d)
  const per = tgt === 'qv' ? 2 * r * (d + d) : tgt === 'attn' ? 4 * r * (d + d) : 4 * r * (d + d) + 3 * r * (d + dff)
  const trainable = per * Ln
  const total = Ln * (4 * d * d + 3 * d * dff)
  const mem = {
    full: total * 16,
    lora: total * 2 + trainable * 16,
    qlora: total * 0.5 + trainable * 16
  }
  const [cp] = useCanvas(330, (ctx, W, H) => {
    // left: matrix picture
    const S = Math.min(H - 80, W * 0.22), x0 = 40, y0 = 50
    text(ctx, 'ΔW = (α/r) · B · A', x0, 22, { size: 13, color: '#fff', weight: 700 })
    ctx.fillStyle = alpha(C.a, 0.35); ctx.fillRect(x0, y0, S, S); ctx.strokeStyle = C.a; ctx.lineWidth = 1.5; ctx.strokeRect(x0, y0, S, S)
    text(ctx, `W  (${d}×${d}) frozen`, x0 + S / 2, y0 + S / 2, { size: 12, align: 'center', color: '#fff', weight: 600 })
    text(ctx, `${fmtInt(d * d)} params`, x0 + S / 2, y0 + S / 2 + 18, { size: 10.5, align: 'center', color: C.mute, mono: true })
    const rb = Math.max(3, (r / d) * S * 6)
    const bx = x0 + S + 40
    ctx.fillStyle = alpha(C.c, 0.8); ctx.fillRect(bx, y0, rb, S); text(ctx, `B (${d}×${r})`, bx + rb / 2, y0 + S + 14, { size: 11, align: 'center', color: C.c, mono: true })
    const ax = bx + rb + 34
    ctx.fillStyle = alpha(C.e, 0.8); ctx.fillRect(ax, y0, S, rb); text(ctx, `A (${r}×${d})`, ax + S / 2, y0 + rb + 14, { size: 11, align: 'center', color: C.e, mono: true })
    text(ctx, '×', bx + rb + 17, y0 + S / 2, { size: 22, align: 'center', color: C.mute, weight: 700 })
    text(ctx, `trainable per matrix: 2·d·r = ${fmtInt(2 * d * r)}  (${(100 * 2 * d * r / (d * d)).toFixed(2)}% of W)`, bx, y0 + S + 44, { size: 11.5, color: C.ink, mono: true })
    // right: memory bars
    const px = W * 0.62, pw = W - px - 30
    text(ctx, `Training memory — ${(total / 1e9).toFixed(1)}B-param model`, px, 22, { size: 12, color: C.mute, weight: 600 })
    const rows = [['Full fine-tune (Adam, mixed prec.)', mem.full, C.r], ['LoRA (fp16 base)', mem.lora, C.a], ['QLoRA (4-bit base)', mem.qlora, C.e]]
    rows.forEach(([nm, v, col], i) => {
      const y = 56 + i * 70
      text(ctx, nm, px, y, { size: 12, color: '#fff', weight: 600 })
      ctx.fillStyle = col; ctx.fillRect(px, y + 12, (v / mem.full) * pw * 0.72, 24)
      text(ctx, fmtBytes(v), px + (v / mem.full) * pw * 0.72 + 8, y + 24, { size: 12, mono: true, color: C.ink })
    })
    text(ctx, 'activations & KV excluded', px, 56 + 3 * 70 - 6, { size: 10, color: C.dim })
    text(ctx, 'full FT: 2 wt + 2 grad + 4 master + 8 Adam = 16 B/param', px, 56 + 3 * 70 + 8, { size: 10, color: C.dim })
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Hidden size d" min={1024} max={8192} step={512} value={d} onChange={setD} />
        <Slider label="Layers" min={12} max={96} value={Ln} onChange={setLn} />
        <Slider label="Rank r" min={1} max={256} value={r} onChange={setR} />
        <Select label="Adapt" value={tgt} onChange={setTgt} options={[['qv', 'q, v only'], ['attn', 'q, k, v, o'], ['all', 'all linear layers']]} />
      </Controls>
      <Readout>Trainable parameters: <b>{fmtInt(trainable)}</b> = <b className="g">{(100 * trainable / total).toFixed(3)}%</b> of the model · adapter file ≈ <b>{fmtBytes(trainable * 2)}</b> (fp16) — swap per task without storing a full copy. The frozen base weights can be <b>quantised to 4-bit</b> (QLoRA) because gradients flow only through the small A, B.</Readout>
    </>
  )
}

/* ---------- scaling laws ---------- */
const CH = { E: 1.69, A: 406.4, B: 410.7, a: 0.34, b: 0.28 }
const loss = (N, D) => CH.E + CH.A / Math.pow(N, CH.a) + CH.B / Math.pow(D, CH.b)
function ScalingLab() {
  const [lc, setLc] = useState(23)       // log10 compute (FLOPs)
  const [ln, setLn] = useState(10.2)     // log10 params chosen by the user
  const Cc = Math.pow(10, lc), N = Math.pow(10, ln), D = Cc / (6 * N)
  const G = Math.pow(CH.a * CH.A / (CH.b * CH.B), 1 / (CH.a + CH.b))
  const Nopt = G * Math.pow(Cc / 6, CH.b / (CH.a + CH.b)), Dopt = Cc / (6 * Nopt)
  const [cp] = useCanvas(340, (ctx, W, H) => {
    const px = 70, pw = W - px - 24, py = 36, ph = H - 90
    const X = v => px + ((v - 7) / 6.5) * pw
    const pts = []; let lo = 1e9, hi = 0
    for (let l = 7; l <= 13.5; l += 0.05) { const L = loss(Math.pow(10, l), Cc / (6 * Math.pow(10, l))); pts.push([l, L]); lo = Math.min(lo, L); hi = Math.max(hi, L) }
    hi = Math.min(hi, lo + 2.5)
    const Y = v => py + ph - ((clamp(v, lo - 0.05, hi) - lo + 0.05) / (hi - lo + 0.05)) * ph
    text(ctx, `Loss at fixed compute C = 10^${lc.toFixed(1)} FLOPs  (D = C / 6N)`, px, 18, { size: 12, color: C.mute, weight: 600 })
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1
    for (let l = 7; l <= 13; l++) { ctx.beginPath(); ctx.moveTo(X(l), py); ctx.lineTo(X(l), py + ph); ctx.stroke(); text(ctx, `1e${l}`, X(l), py + ph + 14, { size: 10, align: 'center', color: C.dim, mono: true }) }
    text(ctx, 'model parameters N →', px + pw / 2, H - 10, { size: 11, align: 'center', color: C.mute })
    polyline(ctx, pts.map(([l, L]) => [X(l), Y(L)]), C.a, 2.6)
    const lo_ = Math.log10(Nopt); ctx.fillStyle = C.e; ctx.beginPath(); ctx.arc(X(lo_), Y(loss(Nopt, Dopt)), 7, 0, 7); ctx.fill(); text(ctx, 'compute-optimal', X(lo_), Y(loss(Nopt, Dopt)) - 16, { size: 11.5, align: 'center', color: C.e, weight: 700 })
    ctx.fillStyle = C.d; ctx.beginPath(); ctx.arc(X(ln), Y(loss(N, D)), 6, 0, 7); ctx.fill(); text(ctx, 'your model', X(ln), Y(loss(N, D)) + 18, { size: 11.5, align: 'center', color: C.d, weight: 700 })
    text(ctx, 'too small / too many tokens', px + 4, py + 12, { size: 10.5, color: C.dim }); text(ctx, 'too big / too few tokens', px + pw - 4, py + 12, { size: 10.5, align: 'right', color: C.dim })
  })
  const fmtN = v => (v >= 1e12 ? (v / 1e12).toFixed(1) + 'T' : v >= 1e9 ? (v / 1e9).toFixed(1) + 'B' : (v / 1e6).toFixed(0) + 'M')
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Compute log₁₀ FLOPs" min={19} max={26} step={0.1} value={lc} onChange={setLc} fmt={v => '10^' + v.toFixed(1)} />
        <Slider label="Your model log₁₀ N" min={7} max={13} step={0.05} value={ln} onChange={setLn} fmt={v => fmtN(Math.pow(10, v))} />
      </Controls>
      <Readout>Compute-optimal: <b>N* = {fmtN(Nopt)}</b> params on <b>D* = {fmtN(Dopt)}</b> tokens (<b>{(Dopt / Nopt).toFixed(0)}</b> tokens/param), predicted loss <b>{loss(Nopt, Dopt).toFixed(3)}</b>. Your choice: N = {fmtN(N)}, D = {fmtN(D)} → loss <b className={loss(N, D) - loss(Nopt, Dopt) > 0.05 ? 'w' : 'g'}>{loss(N, D).toFixed(3)}</b> (+{(loss(N, D) - loss(Nopt, Dopt)).toFixed(3)}). Parametric fit L = E + A/N^α + B/D^β with the constants of Hoffmann et al. (2022); alternative fits give tokens/param ratios nearer 20 — the shape of the trade-off is the lesson. <i>Inference-aware</i> labs deliberately <b>over-train</b> smaller models (Llama-3-8B saw ~15T tokens) to cut serving cost.</Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'rope', label: 'RoPE (rotary positions)', render: () => <RopeLab /> },
    { id: 'moe', label: 'Mixture of Experts', render: () => <MoeLab /> },
    { id: 'lora', label: 'LoRA / QLoRA', render: () => <LoraLab /> },
    { id: 'scale', label: 'Scaling laws', render: () => <ScalingLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'RoPE, expert routing, low-rank adapters and compute-optimal scaling',
  tryIt: [
    'In **RoPE**, change both positions by the same shift — the logit does not change. Change the *difference* and it does.',
    'In **MoE**, crank *router imbalance* up: some experts overflow (yellow border), tokens get dropped, and the load-balancing loss rises above 1.',
    'In **LoRA**, compare full fine-tuning memory for a 7B-class model with QLoRA at rank 16.',
    'In **Scaling laws**, move your model size away from the optimum at fixed compute and watch the loss penalty.'
  ],
  theory, math, practice,
  code: [
    { title: 'RoPE, RMSNorm, SwiGLU, GQA — with a relative-position test (NumPy)', lang: 'python', src: blocks },
    { title: 'Top-k Mixture-of-Experts layer with load-balancing loss (PyTorch)', lang: 'python', src: moe },
    { title: 'LoRA linear layer with merge (PyTorch)', lang: 'python', src: lora },
    { title: 'Direct Preference Optimization (DPO) loss', lang: 'python', src: dpo }
  ],
  quiz: [
    { q: 'What property makes RoPE attractive?', options: ['It adds no compute', 'The attention logit depends only on relative position $n-m$', 'It removes the softmax', 'It makes keys orthogonal'], answer: 1, why: 'Rotating $q$ and $k$ by $m\\theta$ and $n\\theta$ makes $\\langle R_m q,R_nk\\rangle$ a function of $n-m$ only.' },
    { q: 'In a Mixture-of-Experts layer with $E$ experts and top-$k$ routing, per-token compute scales with:', options: ['$E$', '$k$', '$E\\cdot k$', '$1/E$'], answer: 1, why: 'Only $k$ experts execute per token; memory still holds all $E$.' },
    { q: 'LoRA fine-tunes $W$ by learning:', options: ['A sparse mask', 'A low-rank update $\\Delta W=BA$ while $W$ stays frozen', 'A smaller vocabulary', 'Larger heads'], answer: 1, why: 'Trainable parameters drop from $d^2$ to $2dr$ per matrix.' },
    { q: 'Chinchilla\'s main message for a fixed training-compute budget:', options: ['Always make the model larger', 'Scale parameters and tokens roughly in proportion', 'More data is useless', 'Use bigger batches'], answer: 1, why: 'Earlier models were under-trained: they had too many parameters for their token count.' },
    { q: 'Grouped-query attention reduces:', options: ['Training FLOPs by 50%', 'KV-cache size by sharing K/V heads across query heads', 'Vocabulary size', 'Layer count'], answer: 1, why: '$n_{kv}<n_{heads}$ shrinks the cache by $n_{heads}/n_{kv}$ with small quality loss.' }
  ]
}
