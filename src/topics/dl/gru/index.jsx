import React, { useMemo, useState } from 'react'
import { Controls, Slider, Toggle, Readout, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, rng, randn, sigmoid, text, rr, polyline, clamp, fmtInt, alpha } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import scratch from './code/gru_cell.py?raw'
import compare from './code/compare_cells.py?raw'

const T = 32

function run(xs, bz, br) {
  let h = 0, hr = 0
  return xs.map(x => {
    const z = sigmoid(2 * x + 0.5 * h + bz)
    const r = sigmoid(1 * x + 0.5 * h + br)
    const ht = Math.tanh(2.2 * x + 1.5 * r * h)
    const hp = h
    h = (1 - z) * h + z * ht
    hr = Math.tanh(2 * x + 0.9 * hr)
    return { x, z, r, ht, h, hp, hr }
  })
}

function Lab() {
  const [bz, setBz] = useState(-2)
  const [br, setBr] = useState(0)
  const [noise, setNoise] = useState(false)
  const [cur, setCur] = useState(3)
  const [n, setN] = useState(512)
  const xs = useMemo(() => {
    const r = rng(9)
    return Array.from({ length: T }, (_, t) => (t === 3 ? 1 : t === 18 ? -0.8 : 0) + (noise ? 0.25 * randn(r) : 0))
  }, [noise])
  const D = useMemo(() => run(xs, bz, br), [xs, bz, br])
  const s = D[cur]

  const [cpA] = useCanvas(290, (ctx, W, H) => {
    const px = 40, py = 14, pw = W - px - 16, ph = H - 46
    const X = t => px + (t / (T - 1)) * pw, Y = v => py + ph / 2 - clamp(v, -1.1, 1.1) / 1.1 * (ph / 2)
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1
    ;[-1, 0, 1].forEach(v => { ctx.beginPath(); ctx.moveTo(px, Y(v)); ctx.lineTo(px + pw, Y(v)); ctx.stroke(); text(ctx, v, px - 8, Y(v), { size: 10.5, align: 'right', color: C.dim, mono: true }) })
    for (let t = 0; t < T; t += 4) text(ctx, t, X(t), py + ph + 14, { size: 10.5, align: 'center', color: C.dim, mono: true })
    D.forEach((d, t) => { ctx.fillStyle = 'rgba(255,255,255,.18)'; const h = clamp(d.x, -1.1, 1.1) / 1.1 * (ph / 2); ctx.fillRect(X(t) - 3, h > 0 ? Y(0) - h : Y(0), 6, Math.abs(h)) })
    const line = (k, col, w, dash) => polyline(ctx, D.map((d, t) => [X(t), Y(d[k])]), col, w, dash)
    line('hr', 'rgba(255,255,255,.75)', 1.6, [4, 4]); line('z', C.d, 1.4); line('r', C.c, 1.4); line('ht', 'rgba(139,123,255,.55)', 1.4); line('h', C.b, 3)
    ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(cur), py); ctx.lineTo(X(cur), py + ph); ctx.stroke(); ctx.setLineDash([])
  })

  // interpolation + parameter-count panel
  const [cpB] = useCanvas(190, (ctx, W, H) => {
    const half = W / 2
    // left: h_t = (1-z) h_prev + z h~
    text(ctx, `Update gate at t=${cur}: h(t) = (1−z)·h(t−1) + z·h̃`, 18, 18, { size: 12, color: C.mute, weight: 600 })
    const x0 = 40, x1 = half - 40, y = 92
    const P = v => x0 + ((clamp(v, -1.1, 1.1) + 1.1) / 2.2) * (x1 - x0)
    ctx.strokeStyle = C.line; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(P(s.hp), y); ctx.lineTo(P(s.ht), y); ctx.stroke()
    ctx.strokeStyle = C.d; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(P(s.hp), y); ctx.lineTo(P(s.h), y); ctx.stroke()
    const dot = (v, col, lab, dy) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(P(v), y, 7, 0, 7); ctx.fill(); text(ctx, `${lab} ${v.toFixed(2)}`, P(v), y + dy, { size: 11, align: 'center', mono: true, color: col }) }
    dot(s.hp, C.mute, 'h(t−1)', 26); dot(s.ht, C.a, 'h̃', 26); dot(s.h, C.b, 'h(t)', -22)
    text(ctx, `z = ${s.z.toFixed(2)}  (${s.z < 0.2 ? 'copy old state' : s.z > 0.8 ? 'overwrite with candidate' : 'blend'})   r = ${s.r.toFixed(2)}`, 18, H - 22, { size: 12, mono: true, color: C.ink })
    // right: parameters
    const bx = half + 30, by = 38, bw = W - bx - 30, bh = 26
    text(ctx, `Parameters, hidden = input = ${n}`, bx, 18, { size: 12, color: C.mute, weight: 600 })
    const base = n * (n + n) + n
    const rows = [['RNN', base, C.mute], ['GRU', 3 * base, C.b], ['LSTM', 4 * base, C.a]]
    rows.forEach(([nm, v, col], i) => {
      const w = (v / (4 * base)) * (bw - 120)
      ctx.fillStyle = col; rr(ctx, bx + 46, by + i * (bh + 12), w, bh, 6); ctx.fill()
      text(ctx, nm, bx, by + i * (bh + 12) + bh / 2, { size: 12, color: C.ink, weight: 600 })
      text(ctx, fmtInt(v), bx + 46 + w + 8, by + i * (bh + 12) + bh / 2, { size: 11, mono: true, color: C.mute })
    })
    text(ctx, 'GRU ≈ 75% of LSTM', bx, by + 3 * (bh + 12) + 6, { size: 12, color: C.b, mono: true })
  })

  return (
    <>
      <canvas {...cpA} />
      <canvas {...cpB} />
      <Legend items={[[C.b, 'hidden h'], [C.d, 'update z'], [C.c, 'reset r'], ['rgba(139,123,255,.8)', 'candidate h̃'], ['rgba(255,255,255,.7)', 'vanilla RNN'], ['rgba(255,255,255,.3)', 'input x']]} />
      <Controls>
        <Slider label="Update bias b_z" min={-5} max={4} step={0.25} value={bz} onChange={setBz} fmt={v => v.toFixed(2)} />
        <Slider label="Reset bias b_r" min={-4} max={4} step={0.25} value={br} onChange={setBr} fmt={v => v.toFixed(2)} />
        <Slider label="Inspect step t" min={0} max={T - 1} value={cur} onChange={setCur} />
        <Slider label="Hidden size n" min={64} max={2048} step={64} value={n} onChange={setN} />
        <Toggle label="Noisy input" value={noise} onChange={setNoise} />
      </Controls>
      <Readout>The GRU has <b>one</b> state and <b>two</b> gates. A low update gate <b>z≈0</b> copies <i>h(t−1)</i> forward (memory); a high one overwrites it with the candidate. The reset gate <b>r</b> decides how much of the past the candidate may look at.</Readout>
    </>
  )
}

export default {
  Lab,
  vizTitle: 'Update gate as a dial between "keep" and "overwrite"',
  tryIt: [
    'Set the **update bias very negative (−4)**: z≈0, the state is copied and the pulse is remembered. Set it to **+3**: the GRU behaves like a fast vanilla RNN.',
    'Inspect step t=3 and t=18: the blue dot lands between the old state and the candidate at a position set by z.',
    'Slide the **hidden size** and compare parameter counts of RNN, GRU and LSTM.'
  ],
  theory, math, practice,
  code: [
    { title: 'GRU cell from scratch, verified against nn.GRU', lang: 'python', src: scratch },
    { title: 'Benchmark RNN / GRU / LSTM: parameters, speed, accuracy on a toy memory task', lang: 'python', src: compare }
  ],
  quiz: [
    { q: 'In the GRU, $h_t=(1-z_t)\\odot h_{t-1}+z_t\\odot\\tilde h_t$. If $z_t\\to0$:', options: ['The state is overwritten', 'The previous state is copied unchanged', 'The reset gate closes', 'The gradient is zero'], answer: 1, why: 'A small $z$ means "keep the old state", giving an identity path for gradients like the LSTM\'s constant error carousel.' },
    { q: 'What does the reset gate $r_t$ do?', options: ['Resets the weights', 'Controls how much of $h_{t-1}$ is used to compute the candidate', 'Sets the output to zero', 'Clips the gradient'], answer: 1, why: '$\\tilde h_t=\\tanh(Wx_t+U(r_t\\odot h_{t-1}))$ — it lets the candidate ignore irrelevant history.' },
    { q: 'Relative to an LSTM of the same width, a GRU has about:', options: ['Half the parameters', '3/4 of the parameters', 'The same', '2× the parameters'], answer: 1, why: 'Three weight blocks instead of four: $3/4$.' },
    { q: 'Which statement about GRU vs LSTM is most accurate?', options: ['GRU always wins', 'LSTM always wins', 'They perform similarly on most tasks; GRU is cheaper, LSTM has an extra memory/output decoupling', 'GRUs cannot be stacked'], answer: 2, why: 'Empirical comparisons (Chung et al. 2014, Greff et al. 2017) show task-dependent, small differences.' },
    { q: 'The GRU merges which two LSTM components?', options: ['Input and output gates', 'Cell state and hidden state', 'Forget and candidate', 'Weights and biases'], answer: 1, why: 'There is no separate cell state; $h$ is both memory and output.' }
  ]
}
