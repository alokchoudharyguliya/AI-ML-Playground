import React, { useMemo, useState } from 'react'
import { Controls, Slider, Toggle, Select, Readout, SubViews, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, rng, randn, softmax, entropy, heat, div, text, rr, polyline, clamp, fmtInt, pt } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import scratch from './code/transformer_scratch.py?raw'
import sdpa from './code/sdpa_check.py?raw'

const TOK = ['The', 'animal', "didn't", 'cross', 'the', 'street', 'because', 'it', 'was', 'tired']
const n = TOK.length

function drawMatrix(ctx, M, x, y, cell, colorFn, labels, selRow, mask) {
  M.forEach((row, i) => row.forEach((v, j) => {
    ctx.fillStyle = mask && mask[i][j] ? '#0b0d16' : colorFn(v)
    ctx.fillRect(x + j * cell, y + i * cell, cell - 1, cell - 1)
  }))
  if (labels) {
    labels.forEach((l, j) => text(ctx, l, x + j * cell + cell / 2, y - 10, { size: 9.5, align: 'center', color: C.mute }))
    labels.forEach((l, i) => text(ctx, l, x - 6, y + i * cell + cell / 2, { size: 9.5, align: 'right', color: i === selRow ? C.b : C.mute }))
  }
  if (selRow != null) { ctx.strokeStyle = C.b; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y + selRow * cell - 1, M[0].length * cell + 1, cell) }
}

/* ---------- scaled dot-product attention ---------- */
const MAXD = 256
const QK = (() => {
  const r = rng(42)
  return {
    q: TOK.map(() => Array.from({ length: MAXD }, () => randn(r))),
    k: TOK.map(() => Array.from({ length: MAXD }, () => randn(r)))
  }
})()

function AttnLab() {
  const [dk, setDk] = useState(64)
  const [scaled, setScaled] = useState(true)
  const [causal, setCausal] = useState(false)
  const [sel, setSel] = useState(7)
  const { raw, P, mask } = useMemo(() => {
    const raw = TOK.map((_, i) => TOK.map((__, j) => {
      let s = 0; for (let d = 0; d < dk; d++) s += QK.q[i][d] * QK.k[j][d]
      return scaled ? s / Math.sqrt(dk) : s
    }))
    const mask = TOK.map((_, i) => TOK.map((__, j) => causal && j > i))
    const P = raw.map((row, i) => {
      const e = row.map((v, j) => (mask[i][j] ? -1e9 : v))
      return softmax(e)
    })
    return { raw, P, mask }
  }, [dk, scaled, causal])

  const rowStd = useMemo(() => {
    const all = raw.flat(); const m = all.reduce((a, b) => a + b, 0) / all.length
    return Math.sqrt(all.reduce((a, b) => a + (b - m) ** 2, 0) / all.length)
  }, [raw])
  const p = P[sel]
  const jac = useMemo(() => { // Frobenius norm of softmax Jacobian diag(p) - p p^T
    let s = 0; for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { const v = (a === b ? p[a] : 0) - p[a] * p[b]; s += v * v }
    return Math.sqrt(s)
  }, [p])
  const meanMax = P.reduce((s, r) => s + Math.max(...r), 0) / n

  const [cp] = useCanvas(420, (ctx, W, H) => {
    const cell = Math.min(34, (W * 0.5 - 90) / n, (H - 66 - 12) / n), mx = 90, my = 66
    text(ctx, 'Attention weights  softmax(QKᵀ' + (scaled ? '/√d_k' : '') + ')', mx, 22, { size: 12, color: C.mute, weight: 600 })
    drawMatrix(ctx, P, mx, my, cell, v => heat(v), TOK, sel, mask)
    // right: bars for selected query
    const bx = mx + n * cell + 50, bw = W - bx - 24, rowH = Math.min(24, (H - my - 10) / n)
    text(ctx, `Query “${TOK[sel]}” — raw score  q·k${scaled ? '/√d_k' : ''}   |   weight`, bx, 22, { size: 12, color: C.mute, weight: 600 })
    const mxv = Math.max(1, ...raw[sel].map(Math.abs))
    TOK.forEach((t, j) => {
      const y = my + j * rowH
      text(ctx, t, bx, y + rowH / 2 - 1, { size: 11, color: mask[sel][j] ? C.dim : '#fff', weight: 600 })
      const half = (bw * 0.5 - 70) / 2, cx = bx + 66 + half
      ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(cx, y); ctx.lineTo(cx, y + rowH - 2); ctx.stroke()
      const v = raw[sel][j] / mxv
      ctx.fillStyle = mask[sel][j] ? '#1a1e30' : v >= 0 ? C.c : C.b
      ctx.fillRect(Math.min(cx, cx + v * half), y + 3, Math.abs(v * half) || 1, rowH - 8)
      const wx = bx + bw * 0.5 + 40, ww = bw * 0.5 - 90
      ctx.fillStyle = C.panel2; ctx.fillRect(wx, y + 3, ww, rowH - 8)
      ctx.fillStyle = C.e; ctx.fillRect(wx, y + 3, Math.max(1, ww * p[j]), rowH - 8)
      text(ctx, p[j].toFixed(2), wx + ww + 6, y + rowH / 2 - 1, { size: 10.5, mono: true, color: C.mute })
    })
  })
  return (
    <>
      <canvas {...cp} style={{ ...cp.style, cursor: 'pointer' }} onPointerDown={e => {
        const q = pt(e); const rect = e.currentTarget.getBoundingClientRect()
        const cell = Math.min(34, (rect.width * 0.5 - 90) / n, (rect.height - 66 - 12) / n)
        const i = Math.floor((q.y - 66) / cell)
        if (q.x > 90 && q.x < 90 + n * cell + 60 && i >= 0 && i < n) setSel(i)
      }} />
      <Controls>
        <Slider label="Key dimension d_k" min={4} max={256} step={4} value={dk} onChange={setDk} />
        <Toggle label="Scale by 1/√d_k" value={scaled} onChange={setScaled} />
        <Toggle label="Causal mask (decoder)" value={causal} onChange={setCausal} />
        <Select label="Query token" value={sel} onChange={v => setSel(+v)} options={TOK.map((t, i) => [i, `${i}: ${t}`])} />
      </Controls>
      <Readout>
        std of scores = <b className={rowStd > 3 ? 'r' : 'g'}>{rowStd.toFixed(2)}</b> {scaled ? '(≈1 by design)' : <>(≈√d_k = {Math.sqrt(dk).toFixed(1)})</>} · mean max weight = <b>{meanMax.toFixed(2)}</b> · entropy of this row = <b>{entropy(p).toFixed(2)}</b> bits · softmax Jacobian norm = <b className={jac < 0.05 ? 'r' : 'g'}>{jac.toFixed(3)}</b> {jac < 0.05 && '← saturated: gradients vanish'}
      </Readout>
    </>
  )
}

/* ---------- positional encoding ---------- */
function PeLab() {
  const [d, setD] = useState(64)
  const [base, setBase] = useState(4)  // log10
  const L = 64
  const pe = useMemo(() => Array.from({ length: L }, (_, pos) => Array.from({ length: d }, (_, i) => {
    const k = Math.floor(i / 2), w = pos / 10 ** (base * (2 * k) / d)
    return i % 2 ? Math.cos(w) : Math.sin(w)
  })), [d, base])
  const sim = useMemo(() => Array.from({ length: L }, (_, q) => pe[32].reduce((s, v, i) => s + v * pe[q][i], 0) / (d / 2)), [pe, d])
  const [cp] = useCanvas(340, (ctx, W, H) => {
    const hx = 56, hy = 40, hw = W * 0.4, hh = H - 70
    text(ctx, 'PE[pos, dim]  (rows = position 0…63, columns = dimension)', hx, 18, { size: 12, color: C.mute, weight: 600 })
    const cw = hw / d, ch = hh / L
    pe.forEach((row, pos) => row.forEach((v, i) => { ctx.fillStyle = div(v); ctx.fillRect(hx + i * cw, hy + pos * ch, cw + 0.5, ch + 0.5) }))
    text(ctx, 'pos →', hx - 8, hy + 6, { size: 10, align: 'right', color: C.dim, mono: true })
    text(ctx, 'low freq ←  dimension  → high freq', hx + hw / 2, hy + hh + 14, { size: 10.5, align: 'center', color: C.dim })
    // similarity plot
    const px = hx + hw + 70, pw = W - px - 20, py = hy, ph = hh * 0.5
    text(ctx, 'similarity  PE(32)·PE(q) / (d/2)', px, 18, { size: 12, color: C.mute, weight: 600 })
    const X = q => px + (q / (L - 1)) * pw, Y = v => py + ph / 2 - clamp(v, -1.1, 1.1) / 1.1 * (ph / 2)
    ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(px, Y(0)); ctx.lineTo(px + pw, Y(0)); ctx.stroke()
    polyline(ctx, sim.map((v, q) => [X(q), Y(v)]), C.b, 2.4)
    ctx.fillStyle = C.c; ctx.beginPath(); ctx.arc(X(32), Y(sim[32]), 4, 0, 7); ctx.fill()
    text(ctx, 'position q →', px + pw / 2, py + ph + 14, { size: 10.5, align: 'center', color: C.dim })
    // sinusoids for 3 dims
    const sy = py + ph + 50, sh = hh - ph - 50
    text(ctx, 'single dimensions over position', px, sy - 14, { size: 12, color: C.mute, weight: 600 })
    ;[[0, C.a], [Math.floor(d / 4) * 2, C.e], [Math.floor(d / 2) * 2 - 2 < d ? Math.min(d - 2, Math.floor(d * 0.75 / 2) * 2) : 0, C.d]].forEach(([dim, col], k) => {
      polyline(ctx, pe.map((row, pos) => [X(pos), sy + sh / 2 - row[dim] * (sh / 2) * 0.9]), col, 1.8)
      text(ctx, `dim ${dim}`, px + pw - 2, sy + 8 + k * 13, { size: 10.5, align: 'right', color: col, mono: true })
    })
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Model dimension d" min={16} max={128} step={16} value={d} onChange={setD} />
        <Slider label="Wavelength base (10^x)" min={2} max={5} step={0.25} value={base} onChange={setBase} fmt={v => '10^' + v} />
      </Controls>
      <Readout>Each pair of dimensions is a sine/cosine at a geometrically spaced wavelength (from 2π up to ~2π·10^4). Nearby positions have <b>high dot-product similarity</b> that decays with distance, and the encoding of position <i>p+k</i> is a <b>fixed rotation</b> of position <i>p</i> — so relative offsets are easy to learn. The Transformer adds this to the token embeddings.</Readout>
    </>
  )
}

/* ---------- multi-head ---------- */
const D_MODEL = 32
const EMB = (() => { const r = rng(5); return TOK.map(() => Array.from({ length: D_MODEL }, () => randn(r))) })()
const HEADS = (() => {
  const r = rng(77)
  return Array.from({ length: 8 }, () => ({
    wq: Array.from({ length: D_MODEL }, () => Array.from({ length: D_MODEL }, () => randn(r) / Math.sqrt(D_MODEL))),
    wk: Array.from({ length: D_MODEL }, () => Array.from({ length: D_MODEL }, () => randn(r) / Math.sqrt(D_MODEL)))
  }))
})()
const ARCH = ['content', 'local window', 'previous token', 'first-token sink', 'content', 'wide local', 'previous token', 'content']
function bias(kind, i, j) {
  if (kind === 'local window') return -Math.abs(i - j) * 1.2
  if (kind === 'previous token') return j === i - 1 ? 4 : 0
  if (kind === 'first-token sink') return j === 0 ? 3.5 : 0
  if (kind === 'wide local') return -Math.abs(i - j) * 0.45
  return 0
}
function MultiHeadLab() {
  const [h, setH] = useState(4)
  const [strength, setStrength] = useState(1)
  const [causal, setCausal] = useState(true)
  const dh = D_MODEL / h
  const maps = useMemo(() => Array.from({ length: h }, (_, k) => {
    const { wq, wk } = HEADS[k]
    const q = EMB.map(e => Array.from({ length: dh }, (_, c) => e.reduce((s, v, a) => s + v * wq[a][c], 0)))
    const kk = EMB.map(e => Array.from({ length: dh }, (_, c) => e.reduce((s, v, a) => s + v * wk[a][c], 0)))
    return TOK.map((_, i) => softmax(TOK.map((__, j) => {
      if (causal && j > i) return -1e9
      return q[i].reduce((s, v, c) => s + v * kk[j][c], 0) / Math.sqrt(dh) * 2 + strength * bias(ARCH[k], i, j)
    })))
  }), [h, dh, strength, causal])
  const [cp] = useCanvas(h > 4 ? 380 : 225, (ctx, W, H) => {
    const cols = Math.min(h, 4), cw = W / cols, cell = Math.min((cw - 54) / n, (h > 4 ? 130 : 150) / n)
    maps.forEach((M, k) => {
      const col = k % cols, row = Math.floor(k / cols)
      const x = col * cw + (cw - n * cell) / 2, y = 42 + row * 150
      text(ctx, `head ${k + 1} · ${ARCH[k]}`, x + n * cell / 2, y - 24, { size: 11, align: 'center', color: C.mute, weight: 600 })
      drawMatrix(ctx, M, x, y, cell, v => heat(clamp(v * 1.4, 0, 1)), null, null, null)
      text(ctx, `H=${(M.reduce((s, r) => s + entropy(r), 0) / n).toFixed(1)}b`, x + n * cell / 2, y + n * cell + 12, { size: 10, align: 'center', color: C.dim, mono: true })
    })
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Heads h" min={1} max={8} value={h} onChange={setH} />
        <Slider label="Pattern strength" min={0} max={2} step={0.1} value={strength} onChange={setStrength} fmt={v => v.toFixed(1)} />
        <Toggle label="Causal mask" value={causal} onChange={setCausal} />
      </Controls>
      <Readout>d_model = {D_MODEL} → each head works in <b>d_head = {dh}</b> dims. Parameters of the attention block: 4·d_model² = <b>{fmtInt(4 * D_MODEL * D_MODEL)}</b> — <i>independent of h</i>. Heads are random projections plus the structural patterns trained models are known to learn (local, previous-token, first-token “sink”); the point is that <b>different heads can specialise</b> in different relations.</Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'attn', label: 'Scaled dot-product attention', render: () => <AttnLab /> },
    { id: 'heads', label: 'Multi-head attention', render: () => <MultiHeadLab /> },
    { id: 'pe', label: 'Positional encoding', render: () => <PeLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Dissect self-attention, heads and positional encodings',
  tryIt: [
    'Turn **off** the $1/\\sqrt{d_k}$ scaling and drag $d_k$ to 256: the softmax saturates (Jacobian norm → 0) — the reason the scale factor exists.',
    'Enable the **causal mask**: the upper triangle goes dark — this is what makes a decoder autoregressive.',
    'Click different rows of the heat-map to inspect any query token’s raw scores and weights.',
    'In *multi-head*, raise heads from 1 to 8: parameter count stays fixed while patterns diversify.',
    'In *positional encoding*, shrink the base to 10² and watch the long wavelengths disappear.'
  ],
  theory, math, practice,
  code: [
    { title: 'A GPT-style Transformer from scratch (PyTorch)', lang: 'python', note: 'Causal multi-head attention, pre-LN blocks, learned positions, weight-tied LM head.', src: scratch },
    { title: 'Check against F.scaled_dot_product_attention (FlashAttention backend)', lang: 'python', src: sdpa }
  ],
  quiz: [
    { q: 'Why divide attention scores by $\\sqrt{d_k}$?', options: ['To save memory', 'For unit-variance scores, preventing softmax saturation and vanishing gradients', 'To make the matrix symmetric', 'To enforce causality'], answer: 1, why: 'If $q,k$ have unit-variance entries, $q\\cdot k$ has variance $d_k$; scaling restores variance 1.' },
    { q: 'Self-attention is permutation-equivariant. What gives the Transformer a notion of order?', options: ['The softmax', 'Positional encodings added to (or applied inside) attention', 'Layer norm', 'Residual connections'], answer: 1, why: 'Without position information, shuffling the tokens would merely shuffle the outputs.' },
    { q: 'Parameters in the attention sublayer of a block with width $d$ (ignoring biases):', options: ['$d^2$', '$2d^2$', '$4d^2$', '$12d^2$'], answer: 2, why: '$W_Q,W_K,W_V,W_O$, each $d\\times d$ — independent of the number of heads. (The FFN adds $8d^2$.)' },
    { q: 'The compute/memory of vanilla self-attention scales with sequence length $n$ as:', options: ['$O(n)$', '$O(n\\log n)$', '$O(n^2)$', '$O(n^3)$'], answer: 2, why: 'Every token attends to every other: an $n\\times n$ score matrix.' },
    { q: 'What does the causal mask do in a decoder?', options: ['Hides padding', 'Prevents a position from attending to later positions', 'Drops 15% of tokens', 'Normalizes weights'], answer: 1, why: 'It sets scores for $j>i$ to $-\\infty$, so training with teacher forcing cannot "peek" at the future.' }
  ]
}
