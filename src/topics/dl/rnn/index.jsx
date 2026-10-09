import React, { useMemo, useState } from 'react'
import { Controls, Slider, Toggle, Btn, Readout, SubViews, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, rng, randn, div, text, rr, polyline, clamp } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import scratch from './code/rnn_bptt.py?raw'
import charrnn from './code/char_rnn.py?raw'

const n = 8

/* Orthogonal matrix (Gram-Schmidt on a seeded Gaussian) so that spectral radius = rho exactly. */
function orthogonal(seed) {
  const r = rng(seed)
  const A = Array.from({ length: n }, () => Array.from({ length: n }, () => randn(r)))
  const Q = []
  for (const a of A) {
    let v = [...a]
    for (const q of Q) { const d = q.reduce((s, x, i) => s + x * v[i], 0); v = v.map((x, i) => x - d * q[i]) }
    const nm = Math.hypot(...v); Q.push(v.map(x => x / nm))
  }
  return Q
}
const Q = orthogonal(7)
const U = (() => { const r = rng(3); return Array.from({ length: n }, () => randn(r) * 0.8) })()

function simulate({ rho, T, xs, perturb = 0 }) {
  const W = Q.map(row => row.map(v => v * rho))
  let h = Array(n).fill(0)
  const H = [h]
  for (let t = 0; t < T; t++) {
    const x = xs[t] + (t === 0 ? perturb : 0)
    const nh = W.map((row, i) => Math.tanh(row.reduce((s, w, j) => s + w * h[j], 0) + U[i] * x))
    H.push(nh); h = nh
  }
  // backward: g_T = e (unit), g_{t-1} = W^T (g_t * (1 - h_t^2))
  const g = Array(T + 1)
  let gv = Array(n).fill(1 / Math.sqrt(n))
  g[T] = Math.hypot(...gv)
  for (let t = T; t >= 1; t--) {
    const d = gv.map((v, i) => v * (1 - H[t][i] ** 2))
    gv = Array.from({ length: n }, (_, j) => W.reduce((s, row, i) => s + row[j] * d[i], 0))
    g[t - 1] = Math.hypot(...gv)
  }
  return { H, g }
}

function inputs(T, scale) {
  const r = rng(11)
  return Array.from({ length: T }, () => randn(r) * scale)
}

function GradientLab() {
  const [rho, setRho] = useState(0.8)
  const [T, setT] = useState(20)
  const [clip, setClip] = useState(false)
  const xs = useMemo(() => inputs(40, 1), [])
  const { H, g } = useMemo(() => simulate({ rho, T, xs }), [rho, T, xs])

  const [cp] = useCanvas(430, (ctx, W, Hh) => {
    const left = 54, right = 16
    const cw = (W - left - right) / T
    text(ctx, 'Forward: hidden state  h_t  (8 units, colour = activation)', left, 14, { size: 12, color: C.mute, weight: 600 })
    const cellH = 13
    for (let t = 1; t <= T; t++) {
      const x = left + (t - 1) * cw
      for (let i = 0; i < n; i++) { ctx.fillStyle = div(H[t][i]); ctx.fillRect(x + 1, 28 + i * cellH, cw - 2, cellH - 1) }
      if (cw > 14 || t % 2 === 0) text(ctx, t, x + cw / 2, 28 + n * cellH + 10, { size: 10, align: 'center', color: C.dim, mono: true })
      ctx.strokeStyle = C.line
      if (t > 1) { ctx.beginPath(); ctx.moveTo(x - 3, 28 + n * cellH / 2); ctx.lineTo(x + 1, 28 + n * cellH / 2); ctx.stroke() }
    }
    text(ctx, 't →', 22, 28 + n * cellH + 10, { size: 10, color: C.dim, mono: true })

    // backward
    const y0 = 28 + n * cellH + 52, ph = Hh - y0 - 30
    text(ctx, 'Backward: ‖∂h_T / ∂h_t‖  (log scale — bars above the line explode, below vanish)', left, y0 - 14, { size: 12, color: C.mute, weight: 600 })
    const mid = y0 + ph / 2, span = 8
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1
    for (let k = -span; k <= span; k += 4) { const y = mid - (k / span) * (ph / 2); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(W - right, y); ctx.stroke(); text(ctx, `1e${k}`, left - 6, y, { size: 10, align: 'right', color: C.dim, mono: true }) }
    ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.beginPath(); ctx.moveTo(left, mid); ctx.lineTo(W - right, mid); ctx.stroke()
    for (let t = 0; t < T; t++) {
      let v = g[t]
      if (clip) v = Math.min(v, 5)
      const lg = clamp(Math.log10(Math.max(v, 1e-30)), -span, span)
      const x = left + t * cw, h = (lg / span) * (ph / 2)
      ctx.fillStyle = Math.abs(lg) < 1 ? C.e : lg > 0 ? C.d : C.c
      ctx.fillRect(x + 2, h > 0 ? mid - h : mid, cw - 4, Math.abs(h) || 1)
    }
    // theoretical envelope rho^(T-t)
    const pts = []
    for (let t = 0; t <= T; t++) pts.push([left + (t - 0.5) * cw, mid - clamp((T - t) * Math.log10(rho), -span, span) / span * (ph / 2)])
    polyline(ctx, pts, 'rgba(255,255,255,.7)', 1.5, [4, 3])
    text(ctx, 'dashed: ρ^(T−t)', W - right, y0 - 14, { size: 11, align: 'right', color: '#fff', mono: true })
  })
  const first = g[0]
  const verdict = first < 1e-3 ? ['vanished', 'r'] : first > 1e3 ? ['exploded', 'w'] : ['healthy', 'g']
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Spectral radius ρ of W" min={0.3} max={1.8} step={0.02} value={rho} onChange={setRho} fmt={v => v.toFixed(2)} />
        <Slider label="Sequence length T" min={5} max={40} value={T} onChange={setT} />
        <Toggle label="Clip gradient norm at 5" value={clip} onChange={setClip} />
      </Controls>
      <Readout>‖∂h_T/∂h_1‖ = <b>{first.toExponential(2)}</b> after {T - 1} steps — <span className={verdict[1]}>{verdict[0]}</span>. Naive estimate ρ^{T - 1} = {Math.pow(rho, T - 1).toExponential(2)}; tanh′ ≤ 1 shrinks it further.</Readout>
    </>
  )
}

function MemoryLab() {
  const [rho, setRho] = useState(0.9)
  const [T, setT] = useState(30)
  const [eps, setEps] = useState(1)
  const xs = useMemo(() => inputs(60, 0.6), [])
  const data = useMemo(() => {
    const a = simulate({ rho, T, xs }).H, b = simulate({ rho, T, xs, perturb: eps }).H
    return a.map((h, t) => Math.hypot(...h.map((v, i) => v - b[t][i])))
  }, [rho, T, eps, xs])
  const [cp] = useCanvas(320, (ctx, W, Hh) => {
    const px = 56, py = 20, pw = W - px - 20, ph = Hh - 56
    const lo = -8, hi = 1
    const X = t => px + (t / T) * pw, Y = v => py + ph - ((clamp(Math.log10(Math.max(v, 1e-12)), lo, hi) - lo) / (hi - lo)) * ph
    ctx.strokeStyle = C.grid
    for (let k = lo; k <= hi; k += 2) { ctx.beginPath(); ctx.moveTo(px, Y(10 ** k)); ctx.lineTo(px + pw, Y(10 ** k)); ctx.stroke(); text(ctx, `1e${k}`, px - 8, Y(10 ** k), { size: 10, align: 'right', color: C.dim, mono: true }) }
    for (let t = 0; t <= T; t += Math.max(1, Math.round(T / 10))) text(ctx, t, X(t), py + ph + 14, { size: 10.5, align: 'center', color: C.mute, mono: true })
    text(ctx, 'time step t', px + pw / 2, Hh - 8, { size: 11, align: 'center', color: C.mute })
    text(ctx, `‖h_t − h'_t‖ after perturbing x_0 by ${eps}`, px, 8, { size: 12, color: C.mute, weight: 600 })
    polyline(ctx, data.map((v, t) => [X(t), Y(v)]), C.b, 2.5)
    data.forEach((v, t) => { ctx.fillStyle = C.b; ctx.beginPath(); ctx.arc(X(t), Y(v), 3, 0, 7); ctx.fill() })
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Spectral radius ρ" min={0.3} max={1.8} step={0.02} value={rho} onChange={setRho} fmt={v => v.toFixed(2)} />
        <Slider label="Length T" min={10} max={60} value={T} onChange={setT} />
        <Slider label="Perturbation of x₀" min={0.1} max={3} step={0.1} value={eps} onChange={setEps} />
      </Controls>
      <Readout>Forward-time view of the same story: with <b>ρ&lt;1</b> a change at t=0 is forgotten (memory fades); with <b>ρ&gt;1</b> it is amplified until tanh saturates, so the state becomes <span className="w">chaotic</span> and sensitive to everything. Useful memory lives at the <b>edge of stability</b>.</Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'grad', label: 'Backprop through time', render: () => <GradientLab /> },
    { id: 'mem', label: 'Memory fade / chaos', render: () => <MemoryLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Watch gradients vanish and explode through time',
  tryIt: [
    'Slide **ρ below 1** and see the backward bars shrink exponentially toward the first time step.',
    'Slide **ρ above ~1.3** — the bars explode; then switch on **clipping** to see the cap.',
    'Increase **T** to 40: even ρ = 0.95 loses the early signal.',
    'In the memory view find the ρ where a perturbation neither dies nor blows up.'
  ],
  theory, math, practice,
  code: [
    { title: 'BPTT from scratch with a numerical gradient check', lang: 'python', note: 'A tiny tanh RNN: forward, backward through time, and finite-difference verification.', src: scratch },
    { title: 'Character-level language model (PyTorch, RNN/LSTM/GRU switchable)', lang: 'python', src: charrnn }
  ],
  quiz: [
    { q: 'Backpropagating through $T$ steps multiplies roughly $T$ Jacobians of the form $\\mathrm{diag}(1-h^2)W_{hh}$. The gradient vanishes when:', options: ['The product has norm $>1$', 'The product has norm $<1$ consistently', 'The loss is zero', 'The sequence is short'], answer: 1, why: 'Norms of products shrink (or grow) exponentially in $T$ — the root of vanishing/exploding gradients.' },
    { q: 'Which technique fixes **exploding** (not vanishing) gradients?', options: ['Gradient-norm clipping', 'More layers', 'Smaller batch', 'Sigmoid activations'], answer: 0, why: 'Clipping rescales the gradient when its norm is too large; it cannot revive vanished gradients.' },
    { q: 'Weight sharing across time in an RNN means:', options: ['The same $W_{hh}$ is applied at every step, so parameters do not grow with sequence length', 'Each step has its own weights', 'Only the output layer is shared', 'The hidden state is shared across batches'], answer: 0, why: 'This is the temporal analogue of convolutional weight sharing.' },
    { q: 'Truncated BPTT with window $k$ trades:', options: ['Accuracy for nothing', 'Compute/memory for the ability to learn dependencies longer than $k$', 'Batch size for epochs', 'Speed for determinism'], answer: 1, why: 'Gradients are cut every $k$ steps, bounding memory but preventing credit assignment beyond the window (the hidden state still carries forward).' },
    { q: 'Why can\'t a vanilla RNN be parallelized over time during training?', options: ['Because of softmax', 'Each $h_t$ depends on $h_{t-1}$, forcing sequential evaluation', 'GPUs lack tanh', 'Because of dropout'], answer: 1, why: 'The recurrence is inherently sequential — one motivation for Transformers (and for parallel-scan SSMs).' }
  ]
}
