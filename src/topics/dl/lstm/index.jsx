import React, { useMemo, useState } from 'react'
import { Controls, Slider, Toggle, Readout, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, rng, randn, sigmoid, text, rr, polyline, arrow, clamp, alpha } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import scratch from './code/lstm_cell.py?raw'
import model from './code/lstm_classifier.py?raw'

const T = 32

function makeInput(noise, pulses) {
  const r = rng(5)
  return Array.from({ length: T }, (_, t) => (pulses.includes(t) ? 1 : 0) + (noise ? 0.25 * randn(r) : 0))
}

/* scalar LSTM with fixed illustrative weights; biases are user controlled */
function runLstm(xs, bf, bi, bo) {
  let h = 0, c = 0, hr = 0
  const out = []
  for (let t = 0; t < T; t++) {
    const x = xs[t]
    const f = sigmoid(0.5 * x + 0.5 * h + bf)
    const i = sigmoid(3 * x + 0.5 * h + bi)
    const o = sigmoid(1.0 * x + 0.5 * h + bo)
    const g = Math.tanh(2 * x + 0.5 * h)
    const cPrev = c, hPrev = h
    c = f * c + i * g
    h = o * Math.tanh(c)
    hr = Math.tanh(2 * x + 0.9 * hr)          // vanilla RNN for comparison
    out.push({ x, f, i, o, g, c, h, hr, cPrev, hPrev })
  }
  return out
}

function Lab() {
  const [bf, setBf] = useState(3)
  const [bi, setBi] = useState(-2)
  const [bo, setBo] = useState(0)
  const [noise, setNoise] = useState(false)
  const [second, setSecond] = useState(false)
  const [cur, setCur] = useState(8)
  const xs = useMemo(() => makeInput(noise, second ? [3, 16] : [3]), [noise, second])
  const D = useMemo(() => runLstm(xs, bf, bi, bo), [xs, bf, bi, bo])
  const s = D[cur]

  const [cpA] = useCanvas(300, (ctx, W, H) => {
    const cL = 150, cR = W - 150, cy = 56, gy = 190, by = H - 24
    const fX = cL + (cR - cL) * 0.18, iX = cL + (cR - cL) * 0.42, gX = cL + (cR - cL) * 0.58, oX = cL + (cR - cL) * 0.86
    // cell body
    ctx.strokeStyle = C.line; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); rr(ctx, cL - 30, 20, cR - cL + 60, H - 40, 16); ctx.stroke(); ctx.setLineDash([])
    // cell-state highway
    const wc = 2 + Math.min(6, Math.abs(s.cPrev) * 3)
    arrow(ctx, 20, cy, cR + 90, cy, C.b, 3, 9)
    text(ctx, `c(t−1) = ${s.cPrev.toFixed(2)}`, 26, cy - 16, { size: 12, mono: true, color: C.b })
    text(ctx, `c(t) = ${s.c.toFixed(2)}`, W - 20, cy - 16, { size: 12, mono: true, color: C.b, align: 'right' })
    // gate boxes
    const gate = (x, label, v, col, fn = 'σ') => {
      ctx.fillStyle = alpha(col, 0.12 + 0.45 * Math.abs(v)); rr(ctx, x - 34, gy - 22, 68, 44, 10); ctx.fill()
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke()
      text(ctx, `${fn} ${label}`, x, gy - 7, { size: 11, align: 'center', color: C.mute })
      text(ctx, v.toFixed(2), x, gy + 8, { size: 14, align: 'center', mono: true, color: '#fff', weight: 700 })
    }
    gate(fX, 'forget', s.f, C.d); gate(iX, 'input', s.i, C.c); gate(gX, 'cand.', s.g, C.a, 'tanh'); gate(oX, 'output', s.o, C.e)
    // multiplication / addition nodes
    const node = (x, y, sym, col) => { ctx.fillStyle = C.panel2; ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 14, 0, 7); ctx.fill(); ctx.stroke(); text(ctx, sym, x, y + 1, { size: 17, align: 'center', color: '#fff', weight: 700 }) }
    // forget path: gate -> × on highway
    arrow(ctx, fX, gy - 22, fX, cy + 15, C.d, 1 + 4 * s.f, 7)
    node(fX, cy, '×', C.d)
    // i*g -> + on highway
    const mx = (iX + gX) / 2, my = 112
    arrow(ctx, iX, gy - 22, iX, my + 12, C.c, 1 + 4 * s.i, 6)
    arrow(ctx, gX, gy - 22, gX, my + 12, C.a, 1 + 4 * Math.abs(s.g), 6)
    ctx.strokeStyle = C.mute; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(iX, my + 12); ctx.lineTo(iX, my); ctx.lineTo(gX, my); ctx.lineTo(gX, my + 12); ctx.stroke()
    arrow(ctx, mx, my, mx, cy + 15, C.mute, 2, 7)
    node(mx, my, '×', C.c)
    // + node
    node((gX + oX) / 2 - 20, cy, '+', C.b)
    // output: tanh(c) × o -> h
    const hx = oX
    arrow(ctx, hx, gy - 22, hx, 108, C.e, 1 + 4 * s.o, 7)
    node(hx, 94, '×', C.e)
    ctx.strokeStyle = C.b; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(hx - 52, cy); ctx.lineTo(hx - 52, 94); ctx.lineTo(hx - 14, 94); ctx.stroke()
    text(ctx, 'tanh', hx - 52, 78, { size: 10.5, align: 'center', color: C.mute })
    arrow(ctx, hx + 14, 94, W - 24, 94, C.a, 2.4, 8)
    text(ctx, `h(t) = ${s.h.toFixed(2)}`, W - 20, 78, { size: 12, mono: true, color: C.a, align: 'right' })
    // inputs bus
    ctx.strokeStyle = C.dim; ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(40, by); ctx.lineTo(oX + 40, by); ctx.stroke()
    ;[fX, iX, gX, oX].forEach(x => { ctx.beginPath(); ctx.moveTo(x, by); ctx.lineTo(x, gy + 22); ctx.stroke() })
    text(ctx, `h(t−1) = ${s.hPrev.toFixed(2)}   x(t) = ${s.x.toFixed(2)}`, 40, by - 12, { size: 12, mono: true, color: C.ink })
    text(ctx, `t = ${cur}`, W - 20, H - 26, { size: 12, mono: true, color: C.mute, align: 'right' })
  })

  const [cpB] = useCanvas(280, (ctx, W, H) => {
    const px = 40, py = 16, pw = W - px - 16, ph = H - 48
    const X = t => px + (t / (T - 1)) * pw, Y = v => py + ph / 2 - clamp(v, -1.5, 1.5) / 1.5 * (ph / 2)
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1
    ;[-1, 0, 1].forEach(v => { ctx.beginPath(); ctx.moveTo(px, Y(v)); ctx.lineTo(px + pw, Y(v)); ctx.stroke(); text(ctx, v, px - 8, Y(v), { size: 10.5, align: 'right', color: C.dim, mono: true }) })
    for (let t = 0; t < T; t += 4) text(ctx, t, X(t), py + ph + 14, { size: 10.5, align: 'center', color: C.dim, mono: true })
    D.forEach((d, t) => { ctx.fillStyle = 'rgba(255,255,255,.18)'; const h = clamp(d.x, -1.5, 1.5) / 1.5 * (ph / 2); ctx.fillRect(X(t) - 3, h > 0 ? Y(0) - h : Y(0), 6, Math.abs(h)) })
    const line = (k, col, w, dash) => polyline(ctx, D.map((d, t) => [X(t), Y(d[k])]), col, w, dash)
    line('hr', 'rgba(255,255,255,.75)', 1.6, [4, 4])
    line('f', C.d, 1.4); line('i', C.c, 1.4); line('o', C.e, 1.4)
    line('h', C.a, 2.2); line('c', C.b, 3)
    ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(cur), py); ctx.lineTo(X(cur), py + ph); ctx.stroke(); ctx.setLineDash([])
  })

  const peak = D[3].c
  const later = D[Math.min(T - 1, 28)].c
  return (
    <>
      <canvas {...cpA} />
      <canvas {...cpB} />
      <Legend items={[[C.b, 'cell state c'], [C.a, 'hidden h'], [C.d, 'forget f'], [C.c, 'input i'], [C.e, 'output o'], ['rgba(255,255,255,.7)', 'vanilla RNN h'], ['rgba(255,255,255,.3)', 'input x']]} />
      <Controls>
        <Slider label="Forget bias b_f" min={-4} max={6} step={0.25} value={bf} onChange={setBf} fmt={v => v.toFixed(2)} />
        <Slider label="Input bias b_i" min={-5} max={4} step={0.25} value={bi} onChange={setBi} fmt={v => v.toFixed(2)} />
        <Slider label="Output bias b_o" min={-4} max={4} step={0.25} value={bo} onChange={setBo} fmt={v => v.toFixed(2)} />
        <Slider label="Inspect step t" min={0} max={T - 1} value={cur} onChange={setCur} />
        <Toggle label="Noisy input" value={noise} onChange={setNoise} />
        <Toggle label="Second pulse at t=16" value={second} onChange={setSecond} />
      </Controls>
      <Readout>Cell state after the pulse (t=3): <b>{peak.toFixed(2)}</b> → at t=28: <b className={later > 0.3 ? 'g' : 'r'}>{later.toFixed(2)}</b>. The vanilla RNN (dashed) forgets within a few steps; the LSTM keeps what it wrote as long as <span className="w">f ≈ 1</span> and <span className="p">i ≈ 0</span> in between.</Readout>
    </>
  )
}

export default {
  Lab,
  vizTitle: 'Inside the cell: gates, the cell-state highway and memory over time',
  tryIt: [
    'Set **forget bias = −4**: the cell forgets almost instantly, like a vanilla RNN. Set it to **+5**: it holds the pulse for the whole sequence.',
    'Raise the **input bias** and enable the **noisy input**: now junk gets written into memory — selective writing needs a *closed* input gate by default.',
    'Turn on the **second pulse** to watch a new value overwrite the old one when the forget gate is low.',
    'Slide the inspect step and read the four gate values driving the diagram.'
  ],
  theory, math, practice,
  code: [
    { title: 'An LSTM cell from scratch (and checked against nn.LSTM)', lang: 'python', src: scratch },
    { title: 'Sequence classifier with LSTM: packing, bidirectionality, pooling', lang: 'python', src: model }
  ],
  quiz: [
    { q: 'Why is the LSTM cell-state update $c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde g_t$ good for gradients?', options: ['It uses ReLU', 'The path through $c$ is additive, so $\\partial c_t/\\partial c_{t-1}=\\mathrm{diag}(f_t)$ — no repeated weight-matrix multiplication', 'It removes the need for training', 'It is linear in the input'], answer: 1, why: 'Gradients flow along the cell state multiplied only by forget-gate values, which the network can keep near 1.' },
    { q: 'Common LSTM initialization trick:', options: ['Zero forget bias', 'Forget-gate bias ≈ 1', 'All biases −10', 'Orthogonal biases'], answer: 1, why: 'A positive $b_f$ starts the network remembering by default (Jozefowicz et al., 2015).' },
    { q: 'How many weight matrices pairs ($W$, $U$) does an LSTM layer have relative to a vanilla RNN?', options: ['1×', '2×', '3×', '4×'], answer: 3, why: 'Four transformations: forget, input, output gates and the candidate.' },
    { q: 'Which gate controls what is exposed to the next layer?', options: ['Forget', 'Input', 'Output', 'Candidate'], answer: 2, why: '$h_t=o_t\\odot\\tanh(c_t)$ — the output gate filters the cell state.' },
    { q: 'Peephole connections let gates see:', options: ['The next input', 'The cell state', 'The loss', 'The batch'], answer: 1, why: 'Gers & Schmidhuber added $c_{t-1}$ (or $c_t$ for the output gate) as an input to the gates.' }
  ]
}
