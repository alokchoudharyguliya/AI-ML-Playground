import React, { useMemo, useRef, useState } from 'react'
import { Controls, Slider, Select, Btn, Readout, SubViews, Toggle } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, rng, randn, div, text, rr, clamp, fmtInt, fmtBytes, alpha } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import masking from './code/mlm_masking.py?raw'
import finetune from './code/bert_finetune.py?raw'

/* ---------- MLM masking sampler ---------- */
const SENT = 'the quick brown fox jumps over the lazy dog near the river bank'.split(' ')
const VOCAB = ['apple', 'seven', 'cloud', 'blue', 'runs', 'piano', 'window', 'green', 'under', 'table', 'cold', 'paper', 'while', 'engine']

function sampleMask(rate, rand) {
  const n = SENT.length
  const k = Math.max(1, Math.round(rate * n))
  const idx = [...Array(n).keys()].sort(() => rand() - 0.5).slice(0, k)
  const kind = {}, rnd = {}
  idx.forEach(i => {
    const u = rand()
    kind[i] = u < 0.8 ? 'mask' : u < 0.9 ? 'rand' : 'keep'
    if (kind[i] === 'rand') rnd[i] = VOCAB[Math.floor(rand() * VOCAB.length)]
  })
  return { kind, rnd }
}

function MaskLab() {
  const [rate, setRate] = useState(0.15)
  const rand = useRef(rng(Date.now() & 0xffff))
  const [s, setS] = useState(() => sampleMask(0.15, rand.current))
  const [tally, setTally] = useState({ mask: 0, rand: 0, keep: 0 })
  const draw = () => {
    const ns = sampleMask(rate, rand.current)
    setS(ns)
    setTally(t => { const c = { ...t }; Object.values(ns.kind).forEach(k => { c[k]++ }); return c })
  }
  const bulk = () => {
    let c = { ...tally }
    for (let r = 0; r < 500; r++) Object.values(sampleMask(rate, rand.current).kind).forEach(k => { c[k]++ })
    setTally(c)
  }
  const tot = tally.mask + tally.rand + tally.keep
  const pct = k => (tot ? (100 * tally[k] / tot).toFixed(1) : { mask: '80', rand: '10', keep: '10' }[k])
  const inputTok = SENT.map((w, i) => (s.kind[i] === 'mask' ? '[MASK]' : s.kind[i] === 'rand' ? s.rnd[i] : w))
  return (
    <>
      <div className="lab-card">
        <div className="tokrow"><span className="rl">Original</span>{SENT.map((w, i) => <span key={i} className={'tk' + (s.kind[i] ? ' sel' : '')}>{w}</span>)}</div>
        <div className="tokrow"><span className="rl">Model input</span>{inputTok.map((w, i) => <span key={i} className={'tk ' + (s.kind[i] || '')}>{w}</span>)}</div>
        <div className="tokrow"><span className="rl">Predict (loss)</span>{SENT.map((w, i) => <span key={i} className={'tk ' + (s.kind[i] ? 'keep' : 'dim')}>{s.kind[i] ? w : '·'}</span>)}</div>
        <div className="tokrow" style={{ fontSize: 12, color: C.mute }}><span className="rl" />
          <span><span className="tk mask">[MASK]</span> 80% replaced by the mask token</span>
          <span><span className="tk rand">random</span> 10% replaced by a random token</span>
          <span><span className="tk keep">same</span> 10% left unchanged</span></div>
        <div>
          <div className="statbar">
            <i style={{ width: (tot ? 100 * tally.mask / tot : 80) + '%', background: C.c }}>{pct('mask')}%</i>
            <i style={{ width: (tot ? 100 * tally.rand / tot : 10) + '%', background: C.d }}>{pct('rand')}%</i>
            <i style={{ width: (tot ? 100 * tally.keep / tot : 10) + '%', background: C.b }}>{pct('keep')}%</i>
          </div>
        </div>
      </div>
      <Controls>
        <Slider label="Mask rate" min={0.05} max={0.5} step={0.01} value={rate} onChange={setRate} fmt={v => Math.round(v * 100) + '%'} />
        <Btn primary onClick={draw}>Resample</Btn>
        <Btn onClick={bulk}>+500 samples</Btn>
        <Btn onClick={() => setTally({ mask: 0, rand: 0, keep: 0 })}>Reset tally</Btn>
      </Controls>
      <Readout>Only the highlighted <b>{Object.keys(s.kind).length}</b> of {SENT.length} positions ({Math.round(100 * Object.keys(s.kind).length / SENT.length)}%) contribute to the loss — MLM is <b>sample-inefficient</b> (ELECTRA fixes this). The 10% random / 10% unchanged cases stop the model learning “only predict when I see [MASK]”, since [MASK] never appears at fine-tuning time. Tally over {fmtInt(tot)} selected tokens converges to 80 / 10 / 10.</Readout>
    </>
  )
}

/* ---------- attention masks ---------- */
const NM = 12
function allowed(kind, i, j, pre) {
  if (kind === 'bi') return true
  if (kind === 'causal') return j <= i
  return j < pre || j <= i       // prefix-LM: bidirectional over the prefix, causal after
}
function MaskPatternLab() {
  const [kind, setKind] = useState('bi')
  const [pre, setPre] = useState(5)
  const [row, setRow] = useState(8)
  const info = {
    bi: ['Bidirectional (encoder)', 'BERT, RoBERTa, DeBERTa, ViT encoders: every token sees every token → rich representations, but cannot generate left-to-right.'],
    causal: ['Causal (decoder)', 'GPT, Llama: token i sees only ≤ i → can be trained on every position as next-token prediction and generate autoregressively.'],
    prefix: ['Prefix-LM', 'T5-style decoders / UL2 / PaLM-prefix: bidirectional over the prompt (prefix), causal over the continuation — the best of both for conditional generation.']
  }[kind]
  const [cp] = useCanvas(360, (ctx, W, H) => {
    const cell = Math.min(26, (H - 80) / NM, (W * 0.5 - 70) / NM), x0 = 70, y0 = 56
    text(ctx, 'who can attend to whom  (row = query, column = key)', x0, 20, { size: 12, color: C.mute, weight: 600 })
    for (let i = 0; i < NM; i++) for (let j = 0; j < NM; j++) {
      const ok = allowed(kind, i, j, pre)
      ctx.fillStyle = ok ? (i === row ? C.c : alpha(C.a, 0.8)) : '#10131f'
      ctx.fillRect(x0 + j * cell, y0 + i * cell, cell - 1.5, cell - 1.5)
    }
    for (let k = 0; k < NM; k++) {
      text(ctx, k, x0 + k * cell + cell / 2, y0 - 10, { size: 10, align: 'center', color: C.dim, mono: true })
      text(ctx, 'tok ' + k, x0 - 8, y0 + k * cell + cell / 2, { size: 10, align: 'right', color: k === row ? C.c : C.dim, mono: true })
    }
    if (kind === 'prefix') { ctx.strokeStyle = C.d; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.strokeRect(x0 - 1, y0 - 1, pre * cell, NM * cell + 1); ctx.setLineDash([]) }
    // right: for selected row show the visible context as a token strip
    const rx = x0 + NM * cell + 60, rw = W - rx - 20
    text(ctx, `Token ${row} can see:`, rx, 20, { size: 12, color: C.mute, weight: 600 })
    const tw = Math.min(40, rw / NM)
    for (let j = 0; j < NM; j++) {
      const ok = allowed(kind, row, j, pre)
      ctx.fillStyle = ok ? alpha(C.c, 0.85) : '#141830'; rr(ctx, rx + j * tw, 40, tw - 4, 34, 6); ctx.fill()
      text(ctx, j, rx + j * tw + tw / 2 - 2, 57, { size: 11, align: 'center', color: ok ? '#fff' : C.dim, mono: true })
    }
    const vis = Array.from({ length: NM }, (_, j) => allowed(kind, row, j, pre)).filter(Boolean).length
    text(ctx, `${vis} of ${NM} positions visible`, rx, 100, { size: 12, color: C.ink, mono: true })
    text(ctx, info[0], rx, 140, { size: 14, color: '#fff', weight: 700 })
    // wrapped description
    const words = info[1].split(' '); let line = '', yy = 168
    words.forEach(w => { if ((line + w).length * 6.6 > rw) { text(ctx, line, rx, yy, { size: 12, color: C.mute }); line = ''; yy += 18 } line += w + ' ' })
    text(ctx, line, rx, yy, { size: 12, color: C.mute })
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Select label="Mask" value={kind} onChange={setKind} options={[['bi', 'Bidirectional (BERT)'], ['causal', 'Causal (GPT)'], ['prefix', 'Prefix-LM (T5/UL2)']]} />
        <Slider label="Prefix length" min={1} max={NM - 1} value={pre} onChange={setPre} />
        <Slider label="Inspect token" min={0} max={NM - 1} value={row} onChange={setRow} />
      </Controls>
    </>
  )
}

/* ---------- input representation ---------- */
const TOKS = ['[CLS]', 'my', 'dog', 'is', 'cute', '[SEP]', 'he', 'likes', 'play', '##ing', '[SEP]']
const SEG = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
const EMB_D = 10
const mk = seed => { const r = rng(seed); return Array.from({ length: 40 }, () => Array.from({ length: EMB_D }, () => randn(r) * 0.6)) }
const TOKE = mk(1), SEGE = mk(2).slice(0, 2), POSE = mk(3)

function InputLab() {
  const [showLN, setShowLN] = useState(false)
  const [cp] = useCanvas(480, (ctx, W, H) => {
    const cw = Math.min(70, (W - 150) / TOKS.length), x0 = 130, ch = 8
    const rows = [['Token emb', i => TOKE[i % 40], C.c], ['Segment emb', i => SEGE[SEG[i]], C.d], ['Position emb', i => POSE[i], C.e]]
    let y = 54
    TOKS.forEach((t, i) => { text(ctx, t, x0 + i * cw + cw / 2 - 2, 26, { size: 11, align: 'center', color: '#fff', weight: 600 }) })
    const sums = TOKS.map((_, i) => Array.from({ length: EMB_D }, (_, d) => rows.reduce((s, [, f]) => s + f(i)[d], 0)))
    rows.forEach(([name, f, col], k) => {
      text(ctx, name, x0 - 12, y + (EMB_D * ch) / 2, { size: 12, align: 'right', color: col, weight: 600 })
      TOKS.forEach((_, i) => f(i).forEach((v, d) => { ctx.fillStyle = div(v / 1.5); ctx.fillRect(x0 + i * cw + 4, y + d * ch, cw - 8, ch - 1) }))
      y += EMB_D * ch + 16
      text(ctx, k < 2 ? '+' : '=', x0 - 60, y - 14, { size: 18, align: 'center', color: C.mute, weight: 700 })
    })
    text(ctx, showLN ? 'LayerNorm(sum)' : 'Sum → Transformer', x0 - 12, y + (EMB_D * ch) / 2, { size: 12, align: 'right', color: '#fff', weight: 700 })
    sums.forEach((v, i) => {
      const m = v.reduce((a, b) => a + b, 0) / v.length, sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
      v.forEach((val, d) => { ctx.fillStyle = div((showLN ? (val - m) / sd : val) / (showLN ? 2.2 : 2.5)); ctx.fillRect(x0 + i * cw + 4, y + d * ch, cw - 8, ch - 1) })
    })
    // segment brace
    const sx = x0 + 4, mid = x0 + 6 * cw
    ctx.strokeStyle = C.d; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(sx, H - 14); ctx.lineTo(mid - 4, H - 14); ctx.stroke()
    ctx.strokeStyle = C.b; ctx.beginPath(); ctx.moveTo(mid + 4, H - 14); ctx.lineTo(x0 + TOKS.length * cw - 4, H - 14); ctx.stroke()
    text(ctx, 'sentence A (segment 0)', (sx + mid) / 2, H - 28, { size: 11, align: 'center', color: C.d })
    text(ctx, 'sentence B (segment 1)', (mid + x0 + TOKS.length * cw) / 2, H - 28, { size: 11, align: 'center', color: C.b })
  })
  return (
    <>
      <canvas {...cp} />
      <Controls><Toggle label="Apply LayerNorm to the sum" value={showLN} onChange={setShowLN} /></Controls>
      <Readout>BERT's input to layer 1 is the <b>element-wise sum</b> of three learned lookups — token (WordPiece vocabulary of 30,522), segment (A/B) and absolute position (≤512). Each column is one token's vector (10 of 768 dims shown). Sub-word pieces like <b>##ing</b> let a fixed vocabulary cover any word.</Readout>
    </>
  )
}

/* ---------- parameter calculator ---------- */
const PRESETS = {
  base: { name: 'BERT-base', L: 12, H: 768, A: 12, V: 30522, P: 512, F: 4 },
  large: { name: 'BERT-large', L: 24, H: 1024, A: 16, V: 30522, P: 512, F: 4 },
  distil: { name: 'DistilBERT', L: 6, H: 768, A: 12, V: 30522, P: 512, F: 4 },
  tiny: { name: 'TinyBERT-4', L: 4, H: 312, A: 12, V: 30522, P: 512, F: 4 }
}
function params({ L, H, V, P, F }) {
  const emb = V * H + P * H + 2 * H + 2 * H
  const attn = L * 4 * (H * H + H)
  const ffn = L * (2 * H * F * H + F * H + H)
  const ln = L * 4 * H
  const pool = H * H + H
  return { emb, attn, ffn, ln, pool, total: emb + attn + ffn + ln + pool }
}
function SizeLab() {
  const [pk, setPk] = useState('base')
  const [cfg, setCfg] = useState(PRESETS.base)
  const [n, setN] = useState(512)
  const set = (k, v) => setCfg(c => ({ ...c, [k]: v, name: 'custom' }))
  const p = params(cfg)
  const flopsTok = 2 * (p.total - p.emb) + 4 * cfg.L * n * cfg.H
  const [cp] = useCanvas(210, (ctx, W, H) => {
    const parts = [['Embeddings', p.emb, C.c], ['Attention (QKVO)', p.attn, C.a], ['FFN', p.ffn, C.b], ['LayerNorm', p.ln, C.d], ['Pooler', p.pool, C.e]]
    const x0 = 24, w = W - 48
    text(ctx, `${cfg.name} — ${fmtInt(p.total)} parameters`, x0, 22, { size: 14, color: '#fff', weight: 700 })
    let x = x0
    parts.forEach(([nm, v, col]) => { const ww = (v / p.total) * w; ctx.fillStyle = col; ctx.fillRect(x, 44, Math.max(1, ww - 1), 40); x += ww })
    parts.forEach(([nm, v, col], i) => {
      const lx = x0 + (i % 3) * (w / 3), ly = 112 + Math.floor(i / 3) * 34
      ctx.fillStyle = col; ctx.fillRect(lx, ly - 6, 12, 12)
      text(ctx, `${nm}`, lx + 20, ly, { size: 12, color: C.ink, weight: 600 })
      text(ctx, `${(v / 1e6).toFixed(2)}M · ${(100 * v / p.total).toFixed(1)}%`, lx + 20, ly + 15, { size: 11, mono: true, color: C.mute })
    })
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Select label="Preset" value={pk} onChange={k => { setPk(k); setCfg(PRESETS[k]) }} options={Object.entries(PRESETS).map(([k, v]) => [k, v.name])} />
        <Slider label="Layers L" min={2} max={48} value={cfg.L} onChange={v => set('L', v)} />
        <Slider label="Hidden H" min={128} max={2048} step={64} value={cfg.H} onChange={v => set('H', v)} />
        <Slider label="FFN ratio" min={2} max={8} step={1} value={cfg.F} onChange={v => set('F', v)} />
        <Slider label="Seq length n" min={64} max={2048} step={64} value={n} onChange={setN} />
      </Controls>
      <Readout>fp32 weights: <b>{fmtBytes(p.total * 4)}</b> · fp16: <b>{fmtBytes(p.total * 2)}</b> · ≈ <b>{(flopsTok / 1e9).toFixed(2)} GFLOPs</b>/token forward at n={n}. BERT-base check: the formula gives <b>{fmtInt(params(PRESETS.base).total)}</b> (published: 109,482,240). Note how <b>FFN ≈ 2× attention</b> and the embedding table is a fixed cost that dominates tiny models.</Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'mask', label: 'Masked language modeling', render: () => <MaskLab /> },
    { id: 'attn', label: 'Attention masks', render: () => <MaskPatternLab /> },
    { id: 'input', label: 'Input representation', render: () => <InputLab /> },
    { id: 'size', label: 'Model size calculator', render: () => <SizeLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'How BERT sees text: masking, attention patterns, inputs and size',
  tryIt: [
    'Hit **+500 samples** and watch the tally converge to the 80 / 10 / 10 split.',
    'Raise the mask rate to 40% — the model must predict many tokens with little context; 15% was a tuned compromise.',
    'Compare **bidirectional vs causal vs prefix** masks and count how many positions token 8 can see.',
    'In the size calculator pick **BERT-large**, then shrink the hidden size and notice FFN dominating compute.'
  ],
  theory, math, practice,
  code: [
    { title: 'BERT-style 80/10/10 masking (vectorised, with tests)', lang: 'python', src: masking },
    { title: 'Fine-tune BERT for text classification (Hugging Face)', lang: 'python', note: 'Plain PyTorch loop: tokenizer, AdamW with layer-wise LR decay, linear warmup, mixed precision.', src: finetune }
  ],
  quiz: [
    { q: 'Why does BERT sometimes replace selected tokens with a random token or leave them unchanged, rather than always using [MASK]?', options: ['To speed up training', 'Because [MASK] never appears at fine-tuning time; the mismatch is reduced', 'To regularize embeddings', 'To increase vocabulary'], answer: 1, why: 'The 80/10/10 recipe forces the model to maintain a good representation of every token, not just masked ones.' },
    { q: 'What fraction of tokens contribute to the MLM loss?', options: ['100%', '≈50%', '≈15%', '≈1%'], answer: 2, why: 'Only the selected 15% are predicted — one reason MLM is less sample-efficient than next-token prediction.' },
    { q: 'Which statement distinguishes BERT from GPT?', options: ['BERT uses recurrence', 'BERT is bidirectional and cannot generate text autoregressively without adaptation', 'GPT uses an encoder', 'BERT has no attention'], answer: 1, why: 'BERT\'s unmasked attention sees both sides; GPT\'s causal mask enables left-to-right generation.' },
    { q: 'ELECTRA improves on MLM by:', options: ['Using more layers', 'Training a discriminator to detect replaced tokens at every position', 'Adding recurrence', 'Removing attention'], answer: 1, why: 'Replaced-token detection gives a learning signal at all positions, ≈ 4× more sample-efficient.' },
    { q: 'In fine-tuning for sentence classification, the standard BERT head sits on:', options: ['The first token\'s ([CLS]) final hidden state', 'The embedding layer', 'The attention weights', 'The positional table'], answer: 0, why: '[CLS] aggregates sequence information through attention; a linear layer maps it to class logits.' }
  ]
}
