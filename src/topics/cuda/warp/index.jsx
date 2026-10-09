import React, { useMemo, useState } from 'react'
import Stage3D from '../../../components/Stage3D.jsx'
import { Controls, Slider, Select, Readout, SubViews, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, text, rr, alpha } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import divSrc from './code/divergence.cu?raw'
import redSrc from './code/warp_reduce.cu?raw'
import ballotSrc from './code/ballot.cu?raw'

const LANES = 32
const COLORS = [C.e, C.d, C.c, C.b]

function pathsFor(kind, cut) {
  const maskOf = pred => Array.from({ length: LANES }, (_, i) => pred(i))
  if (kind === 'all') return [{ name: 'A', color: C.e, mask: maskOf(() => true) }]
  if (kind === 'cut') {
    const out = []
    const a = maskOf(i => i < cut)
    const b = a.map(x => !x)
    if (a.some(Boolean)) out.push({ name: 'then', color: C.e, mask: a })
    if (b.some(Boolean)) out.push({ name: 'else', color: C.d, mask: b })
    return out
  }
  if (kind === 'even') {
    return [
      { name: 'even', color: C.e, mask: maskOf(i => i % 2 === 0) },
      { name: 'odd', color: C.d, mask: maskOf(i => i % 2 === 1) }
    ]
  }
  return [0, 1, 2, 3].map(r => ({
    name: '%' + r,
    color: COLORS[r],
    mask: maskOf(i => i % 4 === r)
  }))
}

/* =====================================================================
   TAB A — divergence
   ===================================================================== */
function DivergenceLab() {
  const [kind, setKind] = useState('cut')
  const [cut, setCut] = useState(16)
  const [body, setBody] = useState(8)
  const paths = useMemo(() => pathsFor(kind, cut), [kind, cut])
  const issues = paths.length * body

  const [cv] = useCanvas(360, (ctx, W) => {
    const ml = 36, mr = 12
    const barW = W - ml - mr
    const lw = Math.min(18, (barW - 31 * 3) / 32)
    const gap = (barW - 32 * lw) / 31
    const xOf = i => ml + i * (lw + gap)

    text(ctx, 'which arm each lane takes', ml, 16, { size: 12, color: C.mute, weight: 600 })
    text(ctx, `${paths.length} pass${paths.length > 1 ? 'es' : ''} · ${issues} issues`, W - mr, 16, {
      size: 13, align: 'right', weight: 700, color: paths.length === 1 ? C.e : paths.length === 2 ? C.d : C.r
    })
    paths.forEach(p => p.mask.forEach((on, i) => {
      if (!on) return
      ctx.fillStyle = p.color
      rr(ctx, xOf(i), 32, lw, 26, 3); ctx.fill()
    }))
    text(ctx, '0', xOf(0), 70, { size: 10, color: C.mute, mono: true })
    text(ctx, '31', xOf(31) + lw, 70, { size: 10, color: C.mute, mono: true, align: 'right' })

    text(ctx, 'issued one arm at a time  ·  dim lanes are masked off', ml, 96, { size: 12, color: C.mute, weight: 600 })
    const rowH = 32
    const top = 112
    paths.forEach((p, r) => {
      const y = top + r * (rowH + 10)
      text(ctx, p.name, 4, y + rowH / 2, { size: 10, color: C.mute, align: 'left' })
      for (let i = 0; i < LANES; i++) {
        ctx.fillStyle = p.mask[i] ? p.color : '#1a1e32'
        rr(ctx, xOf(i), y, lw, rowH, 3); ctx.fill()
      }
    })
    const foot = top + paths.length * (rowH + 10) + 8
    text(ctx, `each row is ${body} instruction${body > 1 ? 's' : ''} issued for the whole warp`, ml, Math.min(344, foot), { size: 11, color: C.mute })
  })

  return (
    <>
      <canvas {...cv} />
      <Controls>
        <Select label="Predicate" value={kind} onChange={setKind} options={[
          ['all', 'every lane takes A'],
          ['cut', 'lane < K'],
          ['even', 'even / odd'],
          ['mod4', 'lane % 4']
        ]} />
        {kind === 'cut' && <Slider label="K" min={0} max={32} value={cut} onChange={setCut} />}
        <Slider label="Arm length" min={1} max={20} value={body} onChange={setBody} fmt={v => v + ' inst'} />
      </Controls>
      <Legend items={paths.map(p => [p.color, p.name])} />
      <Readout>
        The warp issues <b className={paths.length === 1 ? 'g' : 'r'}>{issues}</b> instructions for this region
        {paths.length === 1
          ? <>, one arm, every lane live.</>
          : <>. A uniform warp would issue <b>{body}</b>. Lane utilization is <b>{Math.round(100 / paths.length)}%</b>.</>}
        {kind === 'cut' && cut > 0 && cut < 32 && <> {cut} lanes take <b>then</b>, {32 - cut} take <b>else</b>. Both arms are paid in full.</>}
        {kind === 'cut' && (cut === 0 || cut === 32) && <> The predicate is uniform, so the empty arm is never issued.</>}
        {kind === 'mod4' && <> Eight lanes are live on each pass. A four-way split costs four issue passes.</>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB B — 3D mask
   ===================================================================== */
function MaskScene({ paths, step }) {
  const issued = paths[Math.min(step, paths.length - 1)]
  return (
    <group>
      {Array.from({ length: LANES }, (_, i) => {
        const on = issued.mask[i]
        const col = i % 16
        const row = i < 16 ? 0 : 1
        const x = (col - 7.5) * 0.46
        const z = row === 0 ? 0.9 : -0.9
        return (
          <mesh key={i} position={[x, on ? 0.28 : 0, z]}>
            <boxGeometry args={[0.34, on ? 0.7 : 0.22, 0.34]} />
            <meshStandardMaterial
              color={on ? issued.color : '#3d4660'}
              emissive={on ? issued.color : '#000'}
              emissiveIntensity={on ? 0.55 : 0}
              roughness={0.45}
            />
          </mesh>
        )
      })}
    </group>
  )
}

function Divergence3D() {
  const [kind, setKind] = useState('cut')
  const [cut, setCut] = useState(16)
  const [step, setStep] = useState(0)
  const paths = useMemo(() => pathsFor(kind, cut), [kind, cut])
  const safe = Math.min(step, paths.length - 1)
  const issued = paths[safe]
  const live = issued.mask.filter(Boolean).length
  const overlay = (
    <>
      pass <b style={{ color: issued.color }}>{safe + 1}</b> of {paths.length}
      {' · '}arm <b style={{ color: issued.color }}>{issued.name}</b>
      <br />{live} lanes live · {LANES - live} masked off
    </>
  )
  return (
    <>
      <Stage3D height={420} camera={[0, 6.2, 8.2]} target={[0, 0.15, 0]} fov={42} overlay={overlay} hint="drag to orbit · tall boxes are the lanes issued on this pass">
        <MaskScene paths={paths} step={safe} />
      </Stage3D>
      <Controls>
        <Select label="Predicate" value={kind} onChange={k => { setKind(k); setStep(0) }} options={[
          ['all', 'every lane takes A'],
          ['cut', 'lane < K'],
          ['even', 'even / odd'],
          ['mod4', 'lane % 4']
        ]} />
        {kind === 'cut' && <Slider label="K" min={0} max={32} value={cut} onChange={v => { setCut(v); setStep(0) }} />}
        <Slider label="Pass" min={0} max={Math.max(0, paths.length - 1)} value={safe} onChange={setStep} fmt={v => (v + 1) + ' / ' + paths.length} />
      </Controls>
      <Readout>
        {paths.length === 1
          ? <>One pass. All 32 lanes stay tall — there is nothing to mask off.</>
          : <>Step the pass slider. Each pass stands up a different subset, and the warp waits until every pass has been issued.</>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB C — shuffle reduction
   ===================================================================== */
function initialVals(fill) {
  if (fill === 'ones') return Array.from({ length: LANES }, () => 1)
  if (fill === 'lane') return Array.from({ length: LANES }, (_, i) => i)
  return Array.from({ length: LANES }, (_, i) => (i * 3) % 7)
}

function shuffleSteps(mode, init) {
  const steps = [{ vals: init.slice(), delta: null }]
  let v = init.slice()
  for (let delta = 16; delta >= 1; delta >>= 1) {
    const n = v.slice()
    for (let lane = 0; lane < LANES; lane++) {
      const src = mode === 'xor' ? (lane ^ delta) : lane + delta
      if (src < LANES) n[lane] = v[lane] + v[src]
    }
    v = n
    steps.push({ vals: v.slice(), delta })
  }
  return steps
}

const stepLabel = v => (v === 0 ? 'start' : 'Δ ' + (32 >> v))

function ShuffleLab() {
  const [mode, setMode] = useState('down')
  const [fill, setFill] = useState('ones')
  const [step, setStep] = useState(0)
  const [watch, setWatch] = useState(0)
  const init = useMemo(() => initialVals(fill), [fill])
  const steps = useMemo(() => shuffleSteps(mode, init), [mode, init])
  const cur = steps[step]
  const total = init.reduce((a, b) => a + b, 0)
  const partner = cur.delta == null ? null : (mode === 'xor' ? (watch ^ cur.delta) : watch + cur.delta)
  const partnerLive = partner != null && partner < LANES

  const [cv] = useCanvas(390, (ctx, W) => {
    const ml = 16
    const gap = 4
    const cw = Math.min(34, (W - ml - 8 - 15 * gap) / 16)
    const ch = 48
    const cell = i => {
      const col = i % 16
      const row = i < 16 ? 0 : 1
      return { x: ml + col * (cw + gap), y: 86 + row * 92, cw, ch }
    }
    text(ctx, mode === 'xor' ? '__shfl_xor_sync' : '__shfl_down_sync', ml, 16, { size: 12, color: C.mute, weight: 600 })
    const label = cur.delta == null ? 'initial registers' : `after Δ = ${cur.delta}`
    text(ctx, label, W - 12, 16, { size: 13, color: C.b, align: 'right', weight: 700 })
    text(ctx, `warp sum ${total}`, ml, 36, { size: 12, color: C.ink, mono: true })

    const maxV = Math.max(...cur.vals, 1)
    for (let i = 0; i < LANES; i++) {
      const { x, y } = cell(i)
      const hot = i === watch || (partnerLive && i === partner)
      ctx.fillStyle = hot ? alpha(C.a, 0.95) : alpha(C.b, 0.18 + 0.55 * (cur.vals[i] / maxV))
      rr(ctx, x, y, cw, ch, 4); ctx.fill()
      text(ctx, String(i), x + cw / 2, y - 8, { size: 9, color: C.dim, align: 'center', mono: true })
      text(ctx, String(cur.vals[i]), x + cw / 2, y + ch / 2, {
        size: cw < 26 ? 9 : 12, color: hot ? '#0a0c14' : C.ink, align: 'center', weight: 700, mono: true
      })
    }

    if (partnerLive) {
      const a = cell(partner), b = cell(watch)
      const x1 = a.x + cw / 2, y1 = a.y + ch + 2
      const x2 = b.x + cw / 2, y2 = b.y - 2
      ctx.strokeStyle = C.c
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.bezierCurveTo(x1, y1 + 16, x2, y2 - 16, x2, y2)
      ctx.stroke()
    }
    const noteY = 86 + 92 + ch + 28
    const note = cur.delta == null
      ? 'step forward to add a partner into each lane'
      : partnerLive
        ? `lane ${watch} adds lane ${partner}`
        : `lane ${watch} is past the warp — it keeps its own value`
    text(ctx, note, ml, Math.min(372, noteY), { size: 12, color: partnerLive ? C.c : C.mute })
  })

  const done = step === steps.length - 1
  const lane0 = cur.vals[0]
  const allFull = cur.vals.every(v => v === total)

  return (
    <>
      <canvas {...cv} />
      <Controls>
        <Select label="Primitive" value={mode} onChange={m => { setMode(m); setStep(0) }} options={[['down', 'shfl_down → lane 0'], ['xor', 'shfl_xor → every lane']]} />
        <Select label="Input" value={fill} onChange={f => { setFill(f); setStep(0) }} options={[['ones', 'all ones'], ['lane', 'lane index'], ['mix', 'small mix']]} />
        <Slider label="Step" min={0} max={5} value={step} onChange={setStep} fmt={stepLabel} />
        <Slider label="Watch lane" min={0} max={31} value={watch} onChange={setWatch} />
      </Controls>
      <Legend items={[[C.a, 'watched lane and its partner'], [C.b, 'register value (brighter = larger)']]} />
      <Readout>
        {cur.delta == null && <>Five steps, offsets 16, 8, 4, 2, 1. No shared memory and no barrier.</>}
        {cur.delta != null && mode === 'down' && <>Lane {watch} {partnerLive ? <>reads lane <b>{partner}</b> and adds it.</> : <>has no source lane, so the shuffle returns its own value.</>}</>}
        {cur.delta != null && mode === 'xor' && <>Lane {watch} reads lane <b>{watch ^ cur.delta}</b> (<b>{watch} XOR {cur.delta}</b>). Every partner lands inside the warp.</>}
        {done && mode === 'down' && <> Lane 0 holds <b className={lane0 === total ? 'g' : 'r'}>{lane0}</b>{lane0 === total ? ', the full sum.' : '.'} The other lanes hold suffixes, not the total.</>}
        {done && mode === 'xor' && allFull && <> Every lane holds <b className="g">{total}</b>. The butterfly broadcasts the sum for free.</>}
      </Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'd', label: 'Divergence', render: () => <DivergenceLab /> },
    { id: '3', label: '3D: masked passes', render: () => <Divergence3D /> },
    { id: 's', label: 'Shuffle reduction', render: () => <ShuffleLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Watch a warp issue each arm, then reduce 32 registers in five shuffles',
  tryIt: [
    'Predicate **lane < K**, K = 16. Two rows light up, 16 lanes each. The issue count is twice the arm length.',
    'Drag **K** to 0, then to 32. One row disappears — a uniform predicate never issues the empty arm.',
    'Switch to **lane % 4**. Four passes, eight lanes live on each, utilization 25%.',
    'Open **3D** and step **Pass**. Each pass stands up a different subset of the warp.',
    'Open **Shuffle reduction**, input **all ones**. Step Δ 16 through Δ 1. Lane 0 grows 2, 4, 8, 16, 32.',
    'Switch the primitive to **shfl_xor**. The same five steps leave 32 in every lane, and the watched lane always has a partner.'
  ],
  theory, math, practice,
  code: [
    { title: 'Uniform predicate vs even/odd split', lang: 'cuda', note: 'Same arithmetic, different predicate. The split warp issues the heavy loop twice. A __syncthreads inside one arm is undefined.', src: divSrc },
    { title: 'Block sum with five shuffles and one barrier', lang: 'cuda', note: 'warp_sum_down leaves the result in lane 0. warp_sum_all leaves it in every lane. Padding lanes must still execute the shuffle.', src: redSrc },
    { title: 'Compact a warp with ballot and popc', lang: 'cuda', note: 'The dense index of a lane is the number of set bits below it. The warp writes popc(ballot) outputs and no shared-memory scan.', src: ballotSrc }
  ],
  quiz: [
    { q: 'A warp takes both arms of an if/else, and each arm is 10 instructions. How many instructions does the warp issue?', options: ['10', '20', '320', '32'], answer: 1, why: 'Taken arms are issued one after another. $T = 10 + 10$. Inactive lanes are masked, not skipped.' },
    { q: 'Which predicate does **not** diverge a full warp?', options: ['threadIdx.x & 1', 'threadIdx.x < 16', 'blockIdx.x == 0', 'threadIdx.x % 4'], answer: 2, why: 'blockIdx.x is the same for every lane of the warp. The other three take more than one arm inside a single warp.' },
    { q: 'After a full __shfl_down_sync reduction, where is the sum of the warp?', options: ['In every lane', 'In lane 0 only', 'In shared memory', 'In lane 31'], answer: 1, why: 'Each lane adds the lane Δ above it. Only lane 0 has absorbed all 31 neighbours. Lane 31 never gains a source.' },
    { q: 'What does the xor (butterfly) reduction produce that the down-sweep does not?', options: ['A sum in every lane', 'Fewer than five steps', 'A shared-memory write', 'A block barrier'], answer: 0, why: 'lane ^ Δ stays inside the warp, so every lane accumulates the full sum. Still five steps, still no shared memory.' },
    { q: 'A 256-thread block reduces with warp shuffles. How many __syncthreads() calls does the reduction need?', options: ['0', '1', '5', '8'], answer: 1, why: 'Warps reduce independently. One barrier publishes the per-warp partials, then warp 0 reduces those 8 values.' },
    { q: 'Why can if (tid < n) __shfl_down_sync(0xffffffff, v, 16) hang?', options: ['Shuffles cannot add', 'The mask waits for lanes that never reach the call', 'n must be a power of two', 'Down-sweeps require shared memory'], answer: 1, why: '0xffffffff means all 32 lanes participate. Lanes that skip the call never arrive. They must execute the shuffle too, contributing 0.' },
    { q: '__ballot_sync returns 0b1010 (bits 1 and 3). The compaction index of lane 3 is:', options: ['0', '1', '3', '4'], answer: 1, why: 'Index = popcount of bits below lane 3. Only bit 1 is set below bit 3, so the index is 1.' }
  ]
}
