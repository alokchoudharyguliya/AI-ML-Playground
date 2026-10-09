import React, { useMemo, useRef, useState } from 'react'
import { Controls, Slider, Select, Btn, Readout, SubViews, Toggle } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, softmax, entropy, text, rr, polyline, clamp, fmtInt, fmtBytes, alpha } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import sampling from './code/sampling.py?raw'
import kvcache from './code/kv_cache.py?raw'

/* ---------- sampling lab ---------- */
const CAND = [[' Paris', 8.6], [' the', 5.4], [' a', 5.0], [' located', 4.2], [' Lyon', 4.0], [' known', 3.4], [' France', 3.1], [' not', 2.9], [' also', 2.7], [' Marseille', 2.4], [' one', 2.2], [' city', 2.0], [' Berlin', 1.2], [' banana', -2]]
const LOGITS = CAND.map(c => c[1])

function process(T, k, pp) {
  const p = softmax(LOGITS, T)
  const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a])
  let keep = new Set(order.slice(0, k))
  // renormalise after top-k, then nucleus
  const mass = [...keep].reduce((s, i) => s + p[i], 0)
  let cum = 0; const nuc = new Set()
  for (const i of order) {
    if (!keep.has(i)) continue
    nuc.add(i); cum += p[i] / mass
    if (cum >= pp) break
  }
  keep = nuc
  const z = [...keep].reduce((s, i) => s + p[i], 0)
  return { base: p, final: p.map((v, i) => (keep.has(i) ? v / z : 0)), keep }
}

function SamplingLab() {
  const [T, setT] = useState(1)
  const [k, setK] = useState(14)
  const [pp, setPp] = useState(1)
  const [counts, setCounts] = useState(null)
  const [last, setLast] = useState(null)
  const { base, final, keep } = useMemo(() => process(T, k, pp), [T, k, pp])
  const orig = useMemo(() => softmax(LOGITS, 1), [])
  const draw = n => {
    const c = counts && n === 1 ? [...counts] : Array(CAND.length).fill(0)
    let lastI = 0
    for (let s = 0; s < n; s++) { let u = Math.random(), a = 0, idx = CAND.length - 1; for (let i = 0; i < final.length; i++) { a += final[i]; if (u <= a) { idx = i; break } } c[idx]++; lastI = idx }
    setCounts(c); setLast(lastI)
  }
  const tot = counts ? counts.reduce((a, b) => a + b, 0) : 0
  const [cp] = useCanvas(340, (ctx, W, H) => {
    const x0 = 30, w = W - 60, slot = w / CAND.length, by = H - 62, bh = H - 120
    text(ctx, 'Prompt: “The capital of France is”  →  next-token distribution', x0, 20, { size: 13, color: '#fff', weight: 600 })
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1
    ;[0.25, 0.5, 0.75, 1].forEach(v => { ctx.beginPath(); ctx.moveTo(x0, by - v * bh); ctx.lineTo(x0 + w, by - v * bh); ctx.stroke(); text(ctx, v, x0 - 6, by - v * bh, { size: 9.5, align: 'right', color: C.dim, mono: true }) })
    CAND.forEach(([tk], i) => {
      const cx = x0 + i * slot + slot / 2, bw = slot * 0.62
      // ghost: raw model distribution (T=1, no truncation)
      ctx.strokeStyle = 'rgba(255,255,255,.28)'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.2
      ctx.strokeRect(cx - bw / 2, by - orig[i] * bh, bw, orig[i] * bh); ctx.setLineDash([])
      // processed
      const kept = keep.has(i)
      ctx.fillStyle = kept ? (i === last ? C.d : C.a) : '#222842'
      const h = (kept ? final[i] : base[i]) * bh
      ctx.globalAlpha = kept ? 0.95 : 0.6
      ctx.fillRect(cx - bw / 2, by - h, bw, Math.max(h, 1)); ctx.globalAlpha = 1
      if (!kept) { ctx.strokeStyle = C.r; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(cx - bw / 2, by - h); ctx.lineTo(cx + bw / 2, by); ctx.stroke() }
      text(ctx, kept ? final[i].toFixed(2) : '✕', cx, by - h - 10, { size: 10, align: 'center', mono: true, color: kept ? '#fff' : C.r })
      ctx.save(); ctx.translate(cx, by + 10); ctx.rotate(0.55); text(ctx, tk, 0, 0, { size: 11, color: kept ? '#fff' : C.dim, weight: 600 }); ctx.restore()
      if (counts && tot) { const f = counts[i] / tot; ctx.fillStyle = C.e; ctx.fillRect(cx - bw / 2 - 3, by - f * bh, 3, f * bh) }
    })
    text(ctx, 'dashed = raw model (T=1)   solid = after temperature / top-k / top-p   green ticks = empirical samples', x0, H - 10, { size: 10.5, color: C.mute })
  })
  const H = entropy(final)
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Temperature" min={0.05} max={3} step={0.05} value={T} onChange={setT} fmt={v => v.toFixed(2)} />
        <Slider label="Top-k" min={1} max={14} value={k} onChange={setK} fmt={v => (v === 14 ? 'off' : v)} />
        <Slider label="Top-p" min={0.05} max={1} step={0.01} value={pp} onChange={setPp} fmt={v => (v === 1 ? 'off' : v.toFixed(2))} />
        <Btn primary onClick={() => draw(1)}>Sample 1</Btn>
        <Btn onClick={() => draw(1000)}>Sample 1000</Btn>
      </Controls>
      <Readout>Kept <b>{keep.size}</b> of {CAND.length} tokens · entropy <b>{H.toFixed(2)}</b> bits · P(“Paris”) = <b>{(final[0] * 100).toFixed(1)}%</b> {last != null && <> · last draw: <b className="w">{CAND[last][0].trim()}</b></>}. <b>T→0</b> is greedy; <b>T&gt;1</b> flattens (creative, error-prone); <b>top-k</b> cuts a fixed count, <b>top-p</b> cuts by cumulative probability so it adapts when the model is confident vs uncertain.</Readout>
    </>
  )
}

/* ---------- KV cache lab ---------- */
const MODELS = {
  l3_8: { name: 'Llama-3-8B (GQA)', L: 32, heads: 32, kv: 8, hd: 128, params: 8.03e9 },
  m7: { name: 'Mistral-7B (GQA)', L: 32, heads: 32, kv: 8, hd: 128, params: 7.24e9 },
  l2_7: { name: 'Llama-2-7B (MHA)', L: 32, heads: 32, kv: 32, hd: 128, params: 6.74e9 },
  l3_70: { name: 'Llama-3-70B (GQA)', L: 80, heads: 64, kv: 8, hd: 128, params: 70.6e9 },
  g3: { name: 'GPT-3 175B (MHA)', L: 96, heads: 96, kv: 96, hd: 128, params: 175e9 }
}
const GPUS = { a100: ['A100 80GB', 80e9, 2.0e12], h100: ['H100 80GB', 80e9, 3.35e12], l4: ['L4 24GB', 24e9, 0.3e12] }

function KvLab() {
  const [mk, setMk] = useState('l3_8')
  const [ctxLen, setCtxLen] = useState(8192)
  const [batch, setBatch] = useState(8)
  const [bytes, setBytes] = useState(2)
  const [gk, setGk] = useState('h100')
  const [mha, setMha] = useState(false)
  const m = MODELS[mk]
  const kvh = mha ? m.heads : m.kv
  const perTok = 2 * m.L * kvh * m.hd * bytes
  const kvTotal = perTok * ctxLen * batch
  const wBytes = m.params * 2
  const [, gmem, bw] = GPUS[gk]
  const tps = bw / (wBytes + kvTotal) * batch   // tokens/s across the batch (memory-bound decode)
  const [cp] = useCanvas(300, (ctx, W, H) => {
    // left: memory bar
    const x0 = 24, bw_ = W * 0.46, y0 = 52, bh = 38
    text(ctx, `GPU memory — ${GPUS[gk][0]}`, x0, 22, { size: 12, color: C.mute, weight: 600 })
    const scale = Math.max(gmem, wBytes + kvTotal) * 1.02
    const sx = v => x0 + (v / scale) * bw_
    ctx.fillStyle = C.a; ctx.fillRect(x0, y0, sx(wBytes) - x0, bh)
    ctx.fillStyle = C.c; ctx.fillRect(sx(wBytes), y0, sx(wBytes + kvTotal) - sx(wBytes), bh)
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.setLineDash([5, 3]); ctx.beginPath(); ctx.moveTo(sx(gmem), y0 - 10); ctx.lineTo(sx(gmem), y0 + bh + 10); ctx.stroke(); ctx.setLineDash([])
    const over = wBytes + kvTotal > gmem
    text(ctx, `weights ${fmtBytes(wBytes)}`, x0, y0 + bh + 22, { size: 12, color: C.a, mono: true, weight: 600 })
    text(ctx, `KV cache ${fmtBytes(kvTotal)}`, x0 + 190, y0 + bh + 22, { size: 12, color: C.c, mono: true, weight: 600 })
    text(ctx, over ? 'OUT OF MEMORY' : 'fits', sx(gmem) - 6, y0 - 16, { size: 12, align: 'right', color: over ? C.r : C.e, mono: true, weight: 700 })
    text(ctx, `${(perTok / 1024).toFixed(0)} KB of KV per token`, x0, y0 + bh + 54, { size: 11.5, color: C.ink, mono: true })
    text(ctx, `= 2 × ${m.L} layers × ${kvh} KV-heads × ${m.hd} × ${bytes} B`, x0, y0 + bh + 72 + 24, { size: 10.5, color: C.mute, mono: true })
    text(ctx, `decode ≤ BW / (weights + KV) × batch`, x0, y0 + bh + 112, { size: 11.5, color: C.d, mono: true })
    text(ctx, `≈ ${fmtInt(Math.round(tps))} tokens/s across all streams`, x0, y0 + bh + 132, { size: 11.5, color: C.d, mono: true })
    // right: compute with and without cache
    const px = W * 0.56, pw = W - px - 20, py = 44, ph = H - 90
    text(ctx, 'Work to generate N tokens (token-forward-passes)', px, 22, { size: 12, color: C.mute, weight: 600 })
    const N = 256, nc = i => i * (i + 1) / 2, X = i => px + (i / N) * pw, Y = v => py + ph - (v / nc(N)) * ph
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.strokeRect(px, py, pw, ph)
    const a = [], b = []
    for (let i = 0; i <= N; i += 4) { a.push([X(i), Y(nc(i))]); b.push([X(i), Y(i)]) }
    polyline(ctx, a, C.r, 2.4); polyline(ctx, b, C.e, 2.4)
    text(ctx, 'no cache: O(N²)', px + pw - 8, py + 14, { size: 11, align: 'right', color: C.r, mono: true })
    text(ctx, 'KV cache: O(N)', px + pw - 8, py + ph - 10, { size: 11, align: 'right', color: C.e, mono: true })
    text(ctx, `at N=${N}: ${fmtInt(nc(N))} vs ${N} passes (${Math.round(nc(N) / N)}× less)`, px, py + ph + 22, { size: 11.5, color: C.ink, mono: true })
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Select label="Model" value={mk} onChange={setMk} options={Object.entries(MODELS).map(([k, v]) => [k, v.name])} />
        <Select label="GPU" value={gk} onChange={setGk} options={Object.entries(GPUS).map(([k, v]) => [k, v[0]])} />
        <Slider label="Context" min={1024} max={131072} step={1024} value={ctxLen} onChange={setCtxLen} fmt={v => Math.round(v / 1024) + 'k'} />
        <Slider label="Batch" min={1} max={128} value={batch} onChange={setBatch} />
        <Select label="KV dtype" value={bytes} onChange={v => setBytes(+v)} options={[[2, 'fp16 / bf16'], [1, 'fp8 / int8'], [0.5, '4-bit']]} />
        <Toggle label="Pretend MHA (no GQA)" value={mha} onChange={setMha} />
      </Controls>
      <Readout>Autoregressive decoding is <b>memory-bandwidth-bound</b>: each new token must stream <i>all</i> weights plus the whole KV cache from HBM, doing only ~1–2 FLOPs per byte. That is why <b>GQA</b>, <b>KV quantisation</b>, <b>batching</b> and <b>PagedAttention</b> matter more for serving cost than raw TFLOPs. Switch off GQA to see the cache balloon by {Math.round(m.heads / m.kv)}×.</Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'sample', label: 'Sampling & decoding', render: () => <SamplingLab /> },
    { id: 'kv', label: 'KV cache & inference cost', render: () => <KvLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Control next-token sampling and see what serving a long context costs',
  tryIt: [
    'Set **temperature → 0.05** (greedy) and press *Sample 1000*: always “Paris”. Now go to **T = 2.5** and watch “banana” appear.',
    'Use **top-p = 0.9** at T=1 versus **top-k = 3** and compare how many tokens survive.',
    'In the KV tab, select **Llama-3-70B**, context **128k**, batch **8** on an A100 — then toggle GQA off.',
    'Notice how the decode-speed bound changes with batch size: batching amortises the weight reads.'
  ],
  theory, math, practice,
  code: [
    { title: 'Temperature, top-k, top-p, min-p from scratch (NumPy, with tests)', lang: 'python', src: sampling },
    { title: 'Incremental decoding with a KV cache (PyTorch)', lang: 'python', note: 'Proves that cached decoding is numerically identical to recomputing the full prefix.', src: kvcache }
  ],
  quiz: [
    { q: 'What does a decoder-only LM optimize at pre-training time?', options: ['Next-token cross-entropy over all positions', 'Masked tokens only', 'Sentence order', 'Image–text similarity'], answer: 0, why: 'With a causal mask every position gives a training signal: predict $x_{t+1}$ from $x_{\\le t}$.' },
    { q: 'Raising the sampling temperature:', options: ['Sharpens the distribution', 'Flattens the distribution (more randomness)', 'Changes the model weights', 'Shortens the output'], answer: 1, why: 'Logits are divided by $T$; $T>1$ reduces the gaps between them.' },
    { q: 'Why is nucleus (top-p) often preferred to top-k?', options: ['It is faster', 'The number of kept tokens adapts to the model\'s confidence', 'It needs no softmax', 'It is deterministic'], answer: 1, why: 'When the distribution is peaked few tokens reach the mass $p$; when it is flat many do.' },
    { q: 'The KV cache converts the cost of generating $N$ tokens from roughly:', options: ['$O(N)$ to $O(N^2)$', '$O(N^2)$ passes to $O(N)$ passes (each attending to a growing cache)', '$O(N^3)$ to $O(N)$', 'No change'], answer: 1, why: 'Past keys/values are reused, so each step processes one new token instead of the whole prefix.' },
    { q: 'Single-stream LLM decoding is usually limited by:', options: ['Tensor Core FLOPs', 'Memory bandwidth (weights + KV reads)', 'Disk speed', 'Softmax'], answer: 1, why: 'Arithmetic intensity at batch 1 is ≈ 1–2 FLOP/byte, far below the GPU ridge point.' }
  ]
}
