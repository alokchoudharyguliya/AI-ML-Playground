import React, { useMemo, useRef, useState } from 'react'
import { Controls, Slider, Toggle, Select, Readout, SubViews } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, rng, softmax, entropy, heat, text, rr, arrow, alpha, pt, clamp } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import seq2seq from './code/seq2seq_attn.py?raw'
import numpyAttn from './code/attention_numpy.py?raw'

/* ---------- alignment lab ---------- */
const SRC = ['The', 'black', 'cat', 'sat', 'on', 'the', 'mat', '.']
const TGT = ['Le', 'chat', 'noir', 's’est', 'assis', 'sur', 'le', 'tapis', '.']
const ALIGN = [0, 2, 1, 3, 3, 4, 5, 6, 7]
const LOGITS = (() => {
  const r = rng(21)
  return TGT.map((_, j) => SRC.map((__, i) => (i === ALIGN[j] ? 6 : Math.abs(i - ALIGN[j]) === 1 ? 1.8 : 0) + (r() - 0.5)))
})()
LOGITS[1][1] += 2.2 // "chat" also glances at "black"

function AlignLab() {
  const [sel, setSel] = useState(2)
  const [temp, setTemp] = useState(1)
  const [bottleneck, setBottleneck] = useState(false)
  const W_ = useMemo(() => LOGITS.map(row => softmax(row, temp)), [temp])
  const hit = useRef([])

  const [cp] = useCanvas(380, (ctx, W, H) => {
    const dW = W * 0.6, bw = Math.min(56, dW / SRC.length - 6), bh = 32
    const sx = i => 12 + (i + 0.5) * ((dW - 12) / SRC.length)
    const tx = j => 12 + (j + 0.5) * ((dW - 12) / TGT.length)
    const sy = 70, ty = H - 96
    text(ctx, 'ENCODER — source (English)', 12, 18, { size: 11, color: C.mute, weight: 600 })
    text(ctx, 'DECODER — target (French), click a word', 12, H - 18, { size: 11, color: C.mute, weight: 600 })
    const w = W_[sel]
    hit.current = []

    if (bottleneck) {
      // everything squeezed through one vector
      const cx = dW / 2, cy = (sy + ty) / 2
      SRC.forEach((_, i) => { ctx.strokeStyle = 'rgba(251,191,36,.35)'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(sx(i), sy + bh / 2); ctx.lineTo(cx, cy - 16); ctx.stroke() })
      TGT.forEach((_, j) => { ctx.strokeStyle = 'rgba(251,191,36,.35)'; ctx.beginPath(); ctx.moveTo(cx, cy + 16); ctx.lineTo(tx(j), ty - bh / 2); ctx.stroke() })
      ctx.fillStyle = alpha(C.d, 0.2); rr(ctx, cx - 70, cy - 16, 140, 32, 8); ctx.fill(); ctx.strokeStyle = C.d; ctx.lineWidth = 1.5; ctx.stroke()
      text(ctx, 'one fixed vector', cx, cy, { size: 12, align: 'center', color: C.d, weight: 600 })
    } else {
      SRC.forEach((_, i) => {
        ctx.strokeStyle = alpha(C.c, 0.18 + 0.82 * w[i]); ctx.lineWidth = 0.8 + 9 * w[i]
        ctx.beginPath(); ctx.moveTo(tx(sel), ty - bh / 2); ctx.bezierCurveTo(tx(sel), (ty + sy) / 2, sx(i), (ty + sy) / 2, sx(i), sy + bh / 2); ctx.stroke()
      })
    }
    SRC.forEach((s, i) => {
      const a = bottleneck ? 0.25 : 0.12 + 0.88 * w[i]
      ctx.fillStyle = alpha(C.c, a); rr(ctx, sx(i) - bw / 2, sy - bh / 2, bw, bh, 8); ctx.fill()
      ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.stroke()
      text(ctx, s, sx(i), sy, { size: 13, align: 'center', color: '#fff', weight: 600 })
      if (!bottleneck) text(ctx, w[i].toFixed(2), sx(i), sy - 28, { size: 11, align: 'center', mono: true, color: w[i] > 0.15 ? C.c : C.dim })
    })
    TGT.forEach((s, j) => {
      const on = j === sel
      ctx.fillStyle = on ? alpha(C.b, 0.35) : C.panel2; rr(ctx, tx(j) - bw / 2, ty - bh / 2, bw, bh, 8); ctx.fill()
      ctx.strokeStyle = on ? C.b : C.line; ctx.lineWidth = on ? 2 : 1; ctx.stroke()
      text(ctx, s, tx(j), ty, { size: 13, align: 'center', color: '#fff', weight: 600 })
      hit.current.push([tx(j) - bw / 2, ty - bh / 2, bw, bh, j])
    })

    // heat-map
    const hx = dW + 36, hy = 60, hw = W - hx - 14, cs = Math.min(hw / SRC.length, (H - hy - 50) / TGT.length)
    text(ctx, 'Attention matrix α[target, source]', hx, 24, { size: 11, color: C.mute, weight: 600 })
    for (let j = 0; j < TGT.length; j++) for (let i = 0; i < SRC.length; i++) {
      ctx.fillStyle = bottleneck ? '#141830' : heat(W_[j][i]); ctx.fillRect(hx + i * cs, hy + j * cs, cs - 1, cs - 1)
    }
    ctx.strokeStyle = C.b; ctx.lineWidth = 2; ctx.strokeRect(hx - 1, hy + sel * cs - 1, SRC.length * cs + 1, cs)
    SRC.forEach((s, i) => text(ctx, s, hx + i * cs + cs / 2, hy - 10, { size: 9.5, align: 'center', color: C.mute }))
    TGT.forEach((s, j) => text(ctx, s, hx - 6, hy + j * cs + cs / 2, { size: 9.5, align: 'right', color: j === sel ? C.b : C.mute }))
  })

  const w = W_[sel]
  const top = w.indexOf(Math.max(...w))
  return (
    <>
      <canvas {...cp} onPointerDown={e => { const p = pt(e); const h = hit.current.find(([x, y, ww, hh]) => p.x >= x && p.x <= x + ww && p.y >= y && p.y <= y + hh); if (h) setSel(h[4]) }} style={{ ...cp.style, cursor: 'pointer' }} />
      <Controls>
        <Slider label="Softmax temperature" min={0.3} max={4} step={0.1} value={temp} onChange={setTemp} fmt={v => v.toFixed(1)} />
        <Toggle label="No attention (fixed-vector bottleneck)" value={bottleneck} onChange={setBottleneck} />
      </Controls>
      <Readout>
        {bottleneck
          ? <>Without attention the decoder must reconstruct <b>every</b> word from a single fixed-size vector — information about early words is crushed as sentences get longer.</>
          : <>Generating “<b>{TGT[sel]}</b>”: context c = Σ αᵢ·hᵢ is dominated by “<b>{SRC[top]}</b>” (α = {w[top].toFixed(2)}). Entropy <b>{entropy(w).toFixed(2)}</b> bits — a low temperature focuses, a high one blurs toward a uniform average.</>}
      </Readout>
    </>
  )
}

/* ---------- geometry of attention ---------- */
const KEYS = [['the', -0.75, 0.45], ['black', 0.1, 0.85], ['cat', 0.75, 0.45], ['sat', 0.65, -0.45], ['on', -0.1, -0.8], ['mat', -0.7, -0.45], ['.', 0.0, 0.0]]

function GeometryLab() {
  const [q, setQ] = useState([0.55, 0.3])
  const [mode, setMode] = useState('dot')
  const [sharp, setSharp] = useState(4)
  const drag = useRef(false)
  const scores = KEYS.map(([, x, y]) => mode === 'dot' ? sharp * (q[0] * x + q[1] * y) : -sharp * ((q[0] - x) ** 2 + (q[1] - y) ** 2))
  const w = softmax(scores)
  const ctxv = [w.reduce((s, a, i) => s + a * KEYS[i][1], 0), w.reduce((s, a, i) => s + a * KEYS[i][2], 0)]

  const [cp] = useCanvas(380, (c, W, H) => {
    const R = Math.min(W * 0.34, H / 2 - 20), cx = W * 0.3, cy = H / 2
    const P = (x, y) => [cx + x * R, cy - y * R]
    c.strokeStyle = C.grid; c.lineWidth = 1
    c.beginPath(); c.moveTo(cx - R * 1.1, cy); c.lineTo(cx + R * 1.1, cy); c.moveTo(cx, cy - R * 1.1); c.lineTo(cx, cy + R * 1.1); c.stroke()
    c.beginPath(); c.arc(cx, cy, R, 0, 7); c.stroke()
    const [qx, qy] = P(q[0], q[1])
    KEYS.forEach(([n, x, y], i) => {
      const [px, py] = P(x, y)
      c.strokeStyle = alpha(C.c, 0.1 + 0.9 * w[i]); c.lineWidth = 0.6 + 8 * w[i]; c.beginPath(); c.moveTo(qx, qy); c.lineTo(px, py); c.stroke()
    })
    KEYS.forEach(([n, x, y], i) => {
      const [px, py] = P(x, y)
      c.fillStyle = C.a; c.beginPath(); c.arc(px, py, 6 + 8 * w[i], 0, 7); c.fill()
      text(c, n, px, py - 18 - 6 * w[i], { size: 12, align: 'center', color: '#fff', weight: 600 })
    })
    // context vector
    const [vx, vy] = P(ctxv[0], ctxv[1])
    c.fillStyle = C.d; c.beginPath()
    for (let k = 0; k < 10; k++) { const r = k % 2 ? 5 : 11, a = (k * Math.PI) / 5 - Math.PI / 2; c.lineTo(vx + r * Math.cos(a), vy + r * Math.sin(a)) }
    c.closePath(); c.fill()
    text(c, 'context c', vx + 14, vy + 4, { size: 11, color: C.d, mono: true })
    // query
    c.fillStyle = C.b; c.beginPath(); c.moveTo(qx, qy - 10); c.lineTo(qx + 10, qy); c.lineTo(qx, qy + 10); c.lineTo(qx - 10, qy); c.closePath(); c.fill()
    text(c, 'query q (drag me)', qx + 14, qy - 14, { size: 11, color: C.b, mono: true })
    // bars
    const bx = W * 0.62, bw = W - bx - 60, by = 50, bh = 26
    text(c, `α = softmax(${mode === 'dot' ? 'q·k' : '−‖q−k‖²'} × ${sharp})`, bx, 24, { size: 12, color: C.mute, weight: 600 })
    KEYS.forEach(([n], i) => {
      const y = by + i * (bh + 10)
      text(c, n, bx, y + bh / 2, { size: 12, color: '#fff', weight: 600 })
      c.fillStyle = C.panel2; rr(c, bx + 48, y, bw - 40, bh, 6); c.fill()
      c.fillStyle = C.c; rr(c, bx + 48, y, Math.max(2, (bw - 40) * w[i]), bh, 6); c.fill()
      text(c, w[i].toFixed(2), bx + 48 + bw - 34, y + bh / 2, { size: 11, mono: true, color: C.ink })
    })
  })
  const move = e => {
    if (!drag.current) return
    const rect = cp.ref.current.getBoundingClientRect()
    const R = Math.min(rect.width * 0.34, rect.height / 2 - 20)
    const p = pt(e)
    setQ([clamp((p.x - rect.width * 0.3) / R, -1.1, 1.1), clamp(-(p.y - rect.height / 2) / R, -1.1, 1.1)])
  }
  return (
    <>
      <canvas {...cp} style={{ ...cp.style, cursor: 'grab', touchAction: 'none' }}
        onPointerDown={e => { drag.current = true; e.currentTarget.setPointerCapture(e.pointerId); move(e) }} onPointerMove={move} onPointerUp={() => { drag.current = false }} />
      <Controls>
        <Select label="Score function" value={mode} onChange={setMode} options={[['dot', 'Dot product (Luong / Transformer)'], ['dist', 'Negative squared distance (RBF-like)']]} />
        <Slider label="Sharpness" min={1} max={14} step={0.5} value={sharp} onChange={setSharp} />
      </Controls>
      <Readout>Attention is a <b>soft nearest-neighbour lookup</b>: move the query and the weights re-distribute; the output (★) is the weighted average of the values. Higher sharpness → closer to a hard <i>argmax</i> lookup.</Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'align', label: 'Alignment (translation)', render: () => <AlignLab /> },
    { id: 'geo', label: 'Attention as soft lookup', render: () => <GeometryLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'See which source words the decoder looks at',
  tryIt: [
    'Click **“noir”** (black) — attention jumps backwards to “black”, showing word-order reordering between languages.',
    'Click **“s’est assis”** words: both point to “sat” — one source word aligned to several target words.',
    'Lower the temperature to sharpen the alignment; raise it to watch attention blur into an average.',
    'Toggle the **bottleneck** to see the encoder–decoder design attention was invented to fix.',
    'In the second view, drag the query across the plane to see the weighted average (★) follow it.'
  ],
  theory, math, practice,
  code: [
    { title: 'Seq2Seq with Bahdanau attention (PyTorch)', lang: 'python', note: 'GRU encoder, additive attention, GRU decoder with teacher forcing and padding masks.', src: seq2seq },
    { title: 'Attention in 25 lines of NumPy (with masking)', lang: 'python', src: numpyAttn }
  ],
  quiz: [
    { q: 'What problem did attention solve in encoder–decoder RNNs?', options: ['Slow training', 'Compressing the entire source into a single fixed-size vector', 'Overfitting', 'Vocabulary size'], answer: 1, why: 'Attention lets the decoder access *all* encoder states at every step instead of one summary vector.' },
    { q: 'In $c_t=\\sum_i\\alpha_{ti}h_i$ the weights $\\alpha_{ti}$:', options: ['Are learned parameters fixed after training', 'Are computed on the fly from the decoder state and encoder states and sum to 1', 'Are one-hot', 'Are independent of the input'], answer: 1, why: '$\\alpha_t=\\mathrm{softmax}(e_t)$ with scores that depend on the current decoder state and each $h_i$.' },
    { q: 'Bahdanau (additive) vs Luong (multiplicative) attention differ mainly in:', options: ['The score function', 'The loss', 'The optimizer', 'The tokenizer'], answer: 0, why: 'Additive: $v^\\top\\tanh(Ws+Uh)$; multiplicative: $s^\\top Wh$ or $s^\\top h$.' },
    { q: 'What do padding masks do in attention?', options: ['Speed up softmax', 'Set scores of padded positions to $-\\infty$ so they receive zero weight', 'Normalize embeddings', 'Add noise'], answer: 1, why: 'Otherwise the model would attend to meaningless padding tokens.' },
    { q: 'Why is attention called "interpretable-ish"?', options: ['It proves causality', 'The weights give a visible alignment between target and source, but are not guaranteed explanations', 'It has no parameters', 'It uses rules'], answer: 1, why: 'Alignments are informative but attention weights alone are not faithful explanations (Jain & Wallace, 2019).' }
  ]
}
