import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Edges } from '@react-three/drei'
import Stage3D from '../../../components/Stage3D.jsx'
import Md from '../../../components/Md.jsx'
import CodeBlock from '../../../components/CodeBlock.jsx'
import { Controls, Slider, Select, Toggle, Readout, SubViews, Btn, Legend } from '../../../components/ui.jsx'
import { useCanvas, useTicker } from '../../../lib/hooks.js'
import { C, text, rr, alpha, clamp, fmtInt, fmtBytes, polyline, pt } from '../../../lib/viz.js'
import { GPUS, GPU_OPTIONS, levelsFor, fmtBW } from './specs.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import stream from './code/bandwidth_stream.cu?raw'
import spaces from './code/memory_spaces.cu?raw'
import chase from './code/pointer_chase.cu?raw'
import probe from './code/roofline_probe.cu?raw'
import plot from './code/plot_roofline.py?raw'

/* =====================================================================
   TAB A — 3D memory hierarchy with load packets
   ===================================================================== */
const SLAB_H = 0.46
const Y_LEVEL = [3.2, 2.2, 1.2, 0, -1.3]          // reg, smem, l1, l2, hbm (top → bottom)
const Y_THREAD = 4.4
const IDX = { smem: 1, l1: 2, l2: 3, hbm: 4 }
const SERVED_COLOR = { smem: '#22d3ee', l1: '#76d12a', l2: '#fbbf24', hbm: '#fb7185' }

const metricOf = (lv, by) => (by === 'cap' ? lv.capTotal : by === 'lat' ? lv.lat : lv.bw)
function slabWidths(levels, by) {
  const vs = levels.map(l => Math.log10(metricOf(l, by)))
  const lo = Math.min(...vs), hi = Math.max(...vs)
  return vs.map(v => 1.7 + 5.0 * (v - lo) / (hi - lo || 1))
}

function buildTimeline(ys, dw, idxs) {
  const k = [{ t: 0, y: ys[0], lvl: -1 }]
  let t = 0
  for (let i = 1; i < ys.length; i++) {
    t += 0.28; k.push({ t, y: ys[i], lvl: idxs[i] })
    t += dw[i]; k.push({ t, y: ys[i], lvl: idxs[i] })
  }
  for (let i = ys.length - 2; i >= 0; i--) { t += 0.2; k.push({ t, y: ys[i], lvl: -1 }) }
  return { k, total: t }
}
function sampleTimeline(tl, t) {
  const k = tl.k
  for (let i = 1; i < k.length; i++) {
    if (t <= k[i].t) {
      const a = k[i - 1], b = k[i], f = (t - a.t) / ((b.t - a.t) || 1)
      return { y: a.y + (b.y - a.y) * f, lvl: a.y === b.y ? b.lvl : -1 }
    }
  }
  return { y: k[k.length - 1].y, lvl: -1 }
}

/** Billboard text drawn to a canvas texture (no DOM, no network font). align: 'right' puts the text to the left of `position`. */
function Label3D({ text, color = '#e7e9f4', position, align = 'center', height = 0.4 }) {
  const tex = useMemo(() => {
    const cv = document.createElement('canvas'), ctx = cv.getContext('2d')
    const font = '600 44px "JetBrains Mono Variable", ui-monospace, monospace'
    ctx.font = font
    cv.width = Math.ceil(ctx.measureText(text).width) + 16; cv.height = 64
    ctx.font = font; ctx.fillStyle = color; ctx.textBaseline = 'middle'; ctx.fillText(text, 8, 34)
    const t = new THREE.CanvasTexture(cv)
    t.colorSpace = THREE.SRGBColorSpace
    t.userData.aspect = cv.width / cv.height
    return t
  }, [text, color])
  useEffect(() => () => tex.dispose(), [tex])
  const wW = height * tex.userData.aspect
  const dx = align === 'right' ? -wW / 2 : align === 'left' ? wW / 2 : 0
  return (
    <sprite position={[position[0] + dx, position[1], position[2]]} scale={[wW, height, 1]} renderOrder={10}>
      <spriteMaterial map={tex} transparent depthTest={false} />
    </sprite>
  )
}

function Slab({ lv, idx, w, sel, onSel, live }) {
  const mat = useRef()
  useFrame((_, dt) => {
    const m = mat.current
    if (!m) return
    const target = live.current.lvl === idx ? 1.2 : sel ? 0.5 : 0.12
    m.emissiveIntensity += (target - m.emissiveIntensity) * Math.min(1, dt * 10)
  })
  return (
    <group position={[0, Y_LEVEL[idx], 0]}>
      <mesh onClick={e => { e.stopPropagation(); onSel(lv.id) }}
        onPointerOver={() => (document.body.style.cursor = 'pointer')} onPointerOut={() => (document.body.style.cursor = '')}>
        <boxGeometry args={[w, SLAB_H, 3.4]} />
        <meshStandardMaterial ref={mat} color={lv.color} emissive={lv.color} emissiveIntensity={0.12} roughness={0.5} transparent opacity={0.9} />
        <Edges color={sel ? '#ffffff' : '#0a0c14'} />
      </mesh>
      <Label3D text={lv.name} color={lv.color} position={[-w / 2 - 0.35, 0, 1.7]} align="right" />
    </group>
  )
}

function Journey({ job, live }) {
  const ref = useRef()
  const st = useRef({ t: 0, tl: null })
  useEffect(() => { if (job) st.current = { t: 0, tl: buildTimeline(job.ys, job.dw, job.idxs) } }, [job])
  useFrame((_, dt) => {
    const s = st.current, m = ref.current
    if (!m) return
    if (!s.tl || s.t > s.tl.total) { m.visible = false; live.current = { lvl: -1 }; return }
    s.t += Math.min(dt, 0.05)
    const r = sampleTimeline(s.tl, s.t)
    m.visible = true; m.position.y = r.y; live.current = { lvl: r.lvl }
  })
  const col = job ? job.color : '#ffffff'
  return (
    <mesh ref={ref} visible={false}>
      <sphereGeometry args={[0.2, 20, 20]} />
      <meshStandardMaterial color={col} emissive={col} emissiveIntensity={1} />
    </mesh>
  )
}

function HierarchyLab() {
  const [gk, setGk] = useState('a100')
  const [by, setBy] = useState('cap')
  const [sel, setSel] = useState('l2')
  const [h1, setH1] = useState(60)
  const [h2, setH2] = useState(70)
  const [path, setPath] = useState('global')
  const [auto, setAuto] = useState(true)
  const [job, setJob] = useState(null)
  const [stats, setStats] = useState({ n: 0, cyc: 0, c: { smem: 0, l1: 0, l2: 0, hbm: 0 } })
  const live = useRef({ lvl: -1 })

  const g = GPUS[gk]
  const levels = useMemo(() => levelsFor(g), [g])
  const lv = useMemo(() => Object.fromEntries(levels.map(l => [l.id, l])), [levels])
  const widths = useMemo(() => slabWidths(levels, by), [levels, by])

  const draw = () => {
    if (path === 'shared') return 'smem'
    return Math.random() < h1 / 100 ? 'l1' : Math.random() < h2 / 100 ? 'l2' : 'hbm'
  }
  const launch = served => {
    const chain = served === 'smem' ? ['smem'] : served === 'l1' ? ['l1'] : served === 'l2' ? ['l1', 'l2'] : ['l1', 'l2', 'hbm']
    const ys = [Y_THREAD, ...chain.map(s => Y_LEVEL[IDX[s]])]
    const idxs = [-1, ...chain.map(s => IDX[s])]
    const dw = [0, ...chain.map((s, i) => (i === chain.length - 1 ? 0.12 + 0.5 * Math.log10(lv[s].lat) / 3 : 0.08))]
    setJob({ ys, idxs, dw, color: SERVED_COLOR[served], id: Math.random() })
  }
  const fire = () => {
    const served = draw()
    setStats(s => ({ n: s.n + 1, cyc: s.cyc + lv[served].lat, c: { ...s.c, [served]: s.c[served] + 1 } }))
    launch(served)
  }
  const bulk = () => {
    const c = { smem: 0, l1: 0, l2: 0, hbm: 0 }; let cyc = 0
    for (let i = 0; i < 1000; i++) { const s = draw(); c[s]++; cyc += lv[s].lat }
    setStats(s => ({ n: s.n + 1000, cyc: s.cyc + cyc, c: { smem: s.c.smem + c.smem, l1: s.c.l1 + c.l1, l2: s.c.l2 + c.l2, hbm: s.c.hbm + c.hbm } }))
  }
  const reset = () => setStats({ n: 0, cyc: 0, c: { smem: 0, l1: 0, l2: 0, hbm: 0 } })
  useTicker(auto, 1700, fire)
  useEffect(reset, [gk, h1, h2, path])

  const p1 = h1 / 100, p2 = (1 - p1) * h2 / 100, p3 = 1 - p1 - p2
  const amat = p1 * lv.l1.lat + p2 * lv.l2.lat + p3 * lv.hbm.lat
  const obs = stats.n ? stats.cyc / stats.n : null
  const L = lv[sel]

  const overlay = (
    <>
      <b style={{ color: L.color }}>{L.name}</b> · {L.where}<br />
      capacity {L.capText}<br />
      latency ≈ <b>{L.lat}</b> cycles ({(L.lat / g.clk).toFixed(0)} ns) · bandwidth ≈ <b>{fmtBW(L.bw)}</b><br />
      scope: {L.scope} · lifetime: {L.life}<br />
      <span style={{ color: '#fbbf24' }}>{L.kw}</span>
    </>
  )

  return (
    <>
      <Stage3D height={540} camera={[9, 5.2, 12.5]} target={[0, 1.6, 0]} fov={42} overlay={overlay} hint="drag to orbit · click a slab for its specs · packets show a load's journey">
        <mesh position={[0, (Y_THREAD + Y_LEVEL[4]) / 2, 0]}>
          <boxGeometry args={[0.04, Y_THREAD - Y_LEVEL[4], 0.04]} />
          <meshStandardMaterial color="#3a4170" />
        </mesh>
        {levels.map((l, i) => <Slab key={l.id} lv={l} idx={i} w={widths[i]} sel={sel === l.id} onSel={setSel} live={live} />)}
        <mesh position={[0, Y_THREAD, 0]}>
          <boxGeometry args={[0.55, 0.55, 0.55]} />
          <meshStandardMaterial color="#ffffff" emissive="#8b7bff" emissiveIntensity={0.4} />
        </mesh>
        <Label3D text="thread" position={[0.55, Y_THREAD, 0]} align="left" />
        <Journey job={job} live={live} />
      </Stage3D>
      <Controls>
        <Select label="GPU" value={gk} onChange={setGk} options={GPU_OPTIONS} />
        <Select label="Slab width = (log)" value={by} onChange={setBy} options={[['cap', 'capacity'], ['lat', 'latency'], ['bw', 'bandwidth']]} />
        <Select label="Load type" value={path} onChange={setPath} options={[['global', 'global load (via L1 → L2 → HBM)'], ['shared', 'shared-memory load']]} />
        {path === 'global' && <>
          <Slider label="L1 hit rate" min={0} max={100} value={h1} onChange={setH1} fmt={v => v + '%'} />
          <Slider label="L2 hit rate (of L1 misses)" min={0} max={100} value={h2} onChange={setH2} fmt={v => v + '%'} />
        </>}
        <Toggle label="Auto-fire loads" value={auto} onChange={setAuto} />
        <Btn primary onClick={fire}>Fire a load</Btn>
        <Btn onClick={bulk}>Sample 1000</Btn>
        <Btn onClick={reset}>Reset stats</Btn>
      </Controls>
      <Legend items={[[SERVED_COLOR.l1, 'served by L1'], [SERVED_COLOR.l2, 'served by L2'], [SERVED_COLOR.hbm, 'served by HBM'], [SERVED_COLOR.smem, 'shared memory']]} />
      <Readout>
        {path === 'shared' ? <>
          Shared memory is addressed explicitly (<b>__shared__</b>) so there is no hit-or-miss: every load costs ≈ <b>{lv.smem.lat}</b> cycles — about <b>{(lv.hbm.lat / lv.smem.lat).toFixed(0)}×</b> faster than a trip to {lv.hbm.name}.
        </> : <>
          AMAT = {h1}%·{lv.l1.lat} + {(100 * p2).toFixed(0)}%·{lv.l2.lat} + {(100 * p3).toFixed(0)}%·{lv.hbm.lat} = <b>{amat.toFixed(0)}</b> cycles
          {obs != null && <> · observed over {fmtInt(stats.n)} loads: <b>{obs.toFixed(0)}</b> cycles (L1 {stats.c.l1} · L2 {stats.c.l2} · HBM {stats.c.hbm})</>}
          {' '}· {(lv.hbm.lat / amat).toFixed(1)}× better than every load going to HBM. Drag the hit rates: a kernel is only as fast as its <b className="w">miss path</b>.
        </>}
        {' '}<span style={{ opacity: 0.7 }}>(packet dwell times are log-scaled so L1 stays visible; latencies are approximate.)</span>
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB B — latency / bandwidth ladder + Little's law
   ===================================================================== */
function LatencyLab() {
  const [gk, setGk] = useState('a100')
  const [lvlId, setLvlId] = useState('hbm')
  const [warps, setWarps] = useState(16)
  const [bytes, setBytes] = useState(4)
  const [ilp, setIlp] = useState(1)
  const g = GPUS[gk]
  const levels = useMemo(() => levelsFor(g), [g])
  const L = levels.find(l => l.id === lvlId)

  const perCycle = L.bw / g.sms / g.clk              // bytes per cycle per SM (GB/s ÷ GHz)
  const need = perCycle * L.lat                      // bytes that must be in flight per SM
  const have = warps * 32 * bytes * ilp
  const frac = Math.min(1, have / need)
  const warpsNeeded = need / (32 * bytes * ilp)

  const [cl] = useCanvas(250, (ctx, W) => {
    const x0 = 128, half = (W - x0 - 24) / 2 - 8, x1 = x0 + half + 32
    text(ctx, 'latency · cycles (log)', x0, 14, { size: 11, color: C.mute, weight: 600 })
    text(ctx, 'aggregate bandwidth · GB/s (log)', x1, 14, { size: 11, color: C.mute, weight: 600 })
    levels.forEach((l, i) => {
      const y = 30 + i * 40
      text(ctx, l.name, x0 - 12, y + 14, { align: 'right', size: 12, color: l.color, weight: 600 })
      const bar = (x, f, label) => {
        const w = Math.max(5, f * half)
        ctx.fillStyle = alpha(l.color, 0.85); rr(ctx, x, y, w, 28, 5); ctx.fill()
        if (w > 150) text(ctx, label, x + w - 8, y + 14, { align: 'right', size: 11, color: '#0a0c14', mono: true, weight: 700 })
        else text(ctx, label, x + w + 7, y + 14, { size: 11, color: C.ink, mono: true })
      }
      bar(x0, clamp(Math.log10(l.lat) / 3, 0, 1), `${l.lat} cyc · ${(l.lat / g.clk).toFixed(0)} ns`)
      bar(x1, clamp(Math.log10(l.bw) / 6, 0, 1), fmtBW(l.bw))
    })
    const reg = levels[0], hbm = levels[4]
    text(ctx, `HBM vs register: ${(hbm.lat / reg.lat).toFixed(0)}× slower to reach, ${(reg.bw / hbm.bw).toFixed(0)}× less bandwidth`, x0, 238, { size: 11, color: C.d, mono: true })
  })

  const [cb] = useCanvas(250, (ctx, W, H) => {
    const ml = 58, mr = 24, mt = 22, mb = 40, pw = W - ml - mr, ph = H - mt - mb
    const X = w => ml + (w - 1) / 63 * pw, Y = f => mt + ph - f * ph
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.fillStyle = C.mute
    for (let f = 0; f <= 1.001; f += 0.25) {
      ctx.beginPath(); ctx.moveTo(ml, Y(f)); ctx.lineTo(ml + pw, Y(f)); ctx.stroke()
      text(ctx, Math.round(f * 100) + '%', ml - 8, Y(f), { align: 'right', size: 10.5, color: C.mute, mono: true })
    }
    ;[1, 8, 16, 32, 48, 64].forEach(w => text(ctx, w, X(w), mt + ph + 14, { align: 'center', size: 10.5, color: C.mute, mono: true }))
    text(ctx, 'resident warps per SM →', ml + pw / 2, H - 8, { align: 'center', size: 11, color: C.mute })
    text(ctx, `achieved ${L.name} bandwidth`, ml, 9, { size: 11, color: C.mute, weight: 600 })
    const curve = k => Array.from({ length: 64 }, (_, i) => [X(i + 1), Y(Math.min(1, (i + 1) * 32 * bytes * k / need))])
    if (ilp > 1) polyline(ctx, curve(1), C.dim, 1.5, [4, 3])
    polyline(ctx, curve(ilp), L.color, 2.8)
    if (warpsNeeded <= 64) {
      ctx.strokeStyle = alpha('#ffffff', 0.35); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(warpsNeeded), mt); ctx.lineTo(X(warpsNeeded), mt + ph); ctx.stroke(); ctx.setLineDash([])
      text(ctx, `saturates at ${Math.ceil(warpsNeeded)} warps`, X(warpsNeeded) - 6, mt + 10, { align: 'right', size: 10.5, color: C.ink, mono: true })
    } else text(ctx, `needs ${warpsNeeded.toFixed(0)} warps — more than the SM can hold!`, ml + pw - 6, mt + 10, { align: 'right', size: 10.5, color: C.r, mono: true })
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X(warps), Y(frac), 5.5, 0, 7); ctx.fill()
    if (ilp > 1) text(ctx, '1 load / thread', X(40), Y(Math.min(1, 40 * 32 * bytes / need)) + 14, { size: 10.5, color: C.dim, mono: true })
  })

  return (
    <>
      <canvas {...cl} />
      <canvas {...cb} />
      <Controls>
        <Select label="GPU" value={gk} onChange={setGk} options={GPU_OPTIONS} />
        <Select label="Level to saturate" value={lvlId} onChange={setLvlId} options={[['smem', 'Shared memory / L1'], ['l2', 'L2'], ['hbm', 'HBM / global']]} />
        <Slider label="Resident warps / SM" min={1} max={64} value={warps} onChange={setWarps} />
        <Select label="Bytes per load" value={bytes} onChange={v => setBytes(+v)} options={[[4, '4 B (float)'], [8, '8 B (float2)'], [16, '16 B (float4)']]} />
        <Slider label="Independent loads in flight / thread" min={1} max={8} value={ilp} onChange={setIlp} />
      </Controls>
      <Readout>
        Little’s law: bytes in flight = bandwidth × latency = <b>{perCycle.toFixed(1)}</b> B/cycle/SM × <b>{L.lat}</b> cycles = <b>{fmtBytes(need)}</b> per SM.
        You supply {warps} warps × 32 lanes × {bytes} B × {ilp} = <b>{fmtBytes(have)}</b> → <b className={frac >= 0.99 ? 'g' : 'w'}>{(100 * frac).toFixed(0)}%</b> of peak {L.name} bandwidth.
        {warpsNeeded > 64 && <> Even a full SM cannot hide this latency with {ilp} load/thread — <b className="r">raise ILP or widen the loads</b>.</>}
        {warpsNeeded <= 64 && frac < 0.99 && <> You need ≈ <b>{Math.ceil(warpsNeeded)}</b> warps (or more loads per thread) to saturate it.</>}
        {frac >= 0.99 && <> Saturated: extra warps no longer help — bandwidth is now the limit, not latency.</>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB C — Roofline explorer
   ===================================================================== */
const PREC = { fp64: 8, fp32: 4, fp16: 2 }
const KERNELS = [
  { id: 'vecadd', name: 'vector add', eff: 0.9, ai: c => 1 / (3 * c.s), how: '1 add per 3 elements moved (2 loads + 1 store)' },
  { id: 'saxpy', name: 'SAXPY', eff: 0.9, ai: c => 2 / (3 * c.s), how: '1 FMA = 2 FLOP per 3 elements moved' },
  { id: 'reduce', name: 'sum reduction', eff: 0.8, ai: c => 1 / c.s, how: '1 add per element read' },
  { id: 'stencil', name: '3-pt stencil', eff: 0.75, ai: c => 5 / (2 * c.s), how: '5 FLOP per element, neighbours reused on-chip' },
  { id: 'spmv', name: 'SpMV (CSR)', eff: 0.45, ai: c => 2 / (2 * c.s + 4), how: '2 FLOP per non-zero; value + column index + gathered x' },
  { id: 'lnorm', name: 'LayerNorm', eff: 0.8, ai: c => 8 / (2 * c.s), how: '≈8 FLOP per element, one read + one write' },
  { id: 'naive', name: 'naive GEMM', eff: 0.35, ai: c => 1 / c.s, how: 'each output re-reads a full row and column from global memory: 2N FLOP per 2N elements' },
  { id: 'tiled', name: 'tiled GEMM', eff: 0.6, tc: true, ai: c => c.T / c.s, how: 'T×T tiles cut global traffic by T: AI = T / bytes' },
  { id: 'gemm', name: 'library GEMM', eff: 0.9, tc: true, ai: c => 2 * c.N / (3 * c.s), how: 'ideal traffic: 2N³ FLOP over 3N² elements' },
  { id: 'decode', name: 'LLM decode', eff: 0.85, tc: true, ai: c => 2 * c.B / c.s, how: '2 FLOP per weight per sequence; each weight is read once per step' }
]
const CEIL_NAME = { fp64: 'FP64', fp32: 'FP32', tf32: 'TF32 TC', fp16: 'FP16 TC' }
const CEIL_LONG = { fp64: 'FP64', fp32: 'FP32 (CUDA cores)', tf32: 'TF32 Tensor Cores', fp16: 'FP16 Tensor Cores' }
const ceilKey = (tc, prec, useTC) => (tc && useTC ? (prec === 'fp16' ? 'fp16' : prec === 'fp32' ? 'tf32' : 'fp64') : prec === 'fp64' ? 'fp64' : 'fp32')

const RH = 430, ML = 64, MR = 118, MT = 18, MB = 46, LX0 = -5, LX1 = 12, LY0 = -1.5, LY1 = 3.3
const tickAI = v => (v < 1 ? `1/${Math.round(1 / v)}` : v >= 1024 ? `${v / 1024}k` : String(v))

function RooflineLab() {
  const [gk, setGk] = useState('a100')
  const [prec, setPrec] = useState('fp32')
  const [useTC, setUseTC] = useState(false)
  const [showL2, setShowL2] = useState(false)
  const [nE, setNE] = useState(10)
  const [tE, setTE] = useState(5)
  const [bE, setBE] = useState(0)
  const [focus, setFocus] = useState('saxpy')
  const [my, setMy] = useState({ lx: 1, lp: 0.2 })
  const drag = useRef(false)

  const g = GPUS[gk]
  const s = PREC[prec]
  const cx = { s, N: 2 ** nE, T: 2 ** tE, B: 2 ** bE }
  const bwT = g.bw / 1000                              // TB/s == TFLOP/s per (FLOP/B)
  const activeKey = ceilKey(true, prec, useTC)
  const topPeak = Math.max(...Object.values(g.peak))

  const rows = KERNELS.map((k, i) => {
    const ai = k.ai(cx), key = ceilKey(k.tc, prec, useTC), peak = g.peak[key]
    const att = Math.min(peak, bwT * ai), rg = peak / bwT
    return { ...k, n: i + 1, ai, key, peak, att, ach: att * k.eff, rg, mem: ai < rg }
  })
  const myAi = 2 ** my.lx
  const myPeak = g.peak[activeKey], myAtt = Math.min(myPeak, bwT * myAi), myPerf = Math.min(10 ** my.lp, myAtt)
  const mine = { id: 'my', name: 'your kernel', n: '★', ai: myAi, key: activeKey, peak: myPeak, att: myAtt, ach: myPerf, rg: myPeak / bwT, mem: myAi < myPeak / bwT, eff: myPerf / myAtt, how: 'drag the ★ on the plot' }
  const cur = focus === 'my' ? mine : rows.find(r => r.id === focus)

  const [cv] = useCanvas(RH, (ctx, W, H) => {
    const pw = W - ML - MR, ph = H - MT - MB
    const X = ai => ML + (Math.log2(ai) - LX0) / (LX1 - LX0) * pw
    const Y = p => MT + ph - (Math.log10(p) - LY0) / (LY1 - LY0) * ph
    // grid + labels
    for (let lx = -4; lx <= 12; lx += 2) {
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(2 ** lx), MT); ctx.lineTo(X(2 ** lx), MT + ph); ctx.stroke()
      text(ctx, tickAI(2 ** lx), X(2 ** lx), MT + ph + 14, { align: 'center', size: 10.5, color: C.mute, mono: true })
    }
    for (let ly = -1; ly <= 3; ly++) {
      ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(ML, Y(10 ** ly)); ctx.lineTo(ML + pw, Y(10 ** ly)); ctx.stroke()
      text(ctx, String(10 ** ly), ML - 8, Y(10 ** ly), { align: 'right', size: 10.5, color: C.mute, mono: true })
    }
    text(ctx, 'arithmetic intensity (FLOP / byte) →', ML + pw / 2, H - 8, { align: 'center', size: 11, color: C.mute })
    ctx.save(); ctx.translate(14, MT + ph / 2); ctx.rotate(-Math.PI / 2); text(ctx, 'attainable TFLOP/s', 0, 0, { align: 'center', size: 11, color: C.mute }); ctx.restore()

    ctx.save(); ctx.beginPath(); ctx.rect(ML, MT, pw, ph); ctx.clip()
    // region tint + ridge of the active ceiling
    const rgA = g.peak[activeKey] / bwT
    ctx.fillStyle = alpha(C.b, 0.04); ctx.fillRect(ML, MT, X(rgA) - ML, ph)
    ctx.fillStyle = alpha(C.g, 0.04); ctx.fillRect(X(rgA), MT, ML + pw - X(rgA), ph)
    // L2 roof
    if (showL2) polyline(ctx, [[X(2 ** LX0), Y(g.l2bw / 1000 * 2 ** LX0)], [X(topPeak / (g.l2bw / 1000)), Y(topPeak)]], alpha(C.b, 0.8), 1.8, [6, 4])
    // HBM roof
    polyline(ctx, [[X(2 ** LX0), Y(bwT * 2 ** LX0)], [X(topPeak / bwT), Y(topPeak)]], C.a, 3.2)
    // ceilings
    Object.entries(g.peak).forEach(([k, p]) => {
      const on = k === activeKey
      polyline(ctx, [[X(p / bwT), Y(p)], [ML + pw, Y(p)]], on ? C.g : alpha('#ffffff', 0.28), on ? 3 : 1.5, on ? null : [5, 4])
    })
    ctx.strokeStyle = alpha(C.g, 0.6); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(rgA), Y(g.peak[activeKey])); ctx.lineTo(X(rgA), MT + ph); ctx.stroke(); ctx.setLineDash([])
    text(ctx, 'memory-bound', ML + 10, MT + 14, { size: 11, color: alpha(C.b, 0.9), weight: 600 })
    text(ctx, 'compute-bound', ML + pw - 10, MT + 14, { size: 11, color: alpha(C.g, 0.9), weight: 600, align: 'right' })
    text(ctx, `ridge ${rgA.toFixed(rgA < 10 ? 1 : 0)}`, X(rgA) + 6, MT + ph - 10, { size: 10.5, color: C.g, mono: true })
    // kernel dots
    const dot = (r, color, isMy) => {
      const x = X(clamp(r.ai, 2 ** LX0, 2 ** (LX1 - 0.15))), y = Y(r.ach), foc = cur && cur.id === r.id
      ctx.strokeStyle = alpha('#ffffff', 0.35); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, Y(r.att)); ctx.stroke(); ctx.setLineDash([])
      ctx.fillStyle = color
      ctx.beginPath()
      if (isMy) { ctx.moveTo(x, y - 12); ctx.lineTo(x + 12, y); ctx.lineTo(x, y + 12); ctx.lineTo(x - 12, y); ctx.closePath() } else ctx.arc(x, y, 10, 0, 7)
      ctx.fill()
      if (foc) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2.2; ctx.stroke() }
      text(ctx, r.n, x, y + 0.5, { align: 'center', size: 10.5, color: '#0a0c14', mono: true, weight: 800 })
    }
    rows.forEach(r => dot(r, r.mem ? C.d : C.e, false))
    dot(mine, C.c, true)
    ctx.restore()
    // right-hand ceiling labels (merge equal values)
    const lines = []
    Object.entries(g.peak).sort((a, b) => b[1] - a[1]).forEach(([k, p]) => {
      const last = lines[lines.length - 1]
      if (last && Math.abs(last.p - p) / p < 0.01) last.names.push(CEIL_NAME[k]); else lines.push({ p, names: [CEIL_NAME[k]], on: k === activeKey })
      if (k === activeKey) lines[lines.length - 1].on = true
    })
    lines.forEach(l => {
      text(ctx, `${l.p >= 100 ? Math.round(l.p) : l.p} TF`, ML + pw + 8, Y(l.p) - 7, { size: 10.5, color: l.on ? C.g : C.mute, mono: true, weight: 700 })
      text(ctx, l.names.join(' / '), ML + pw + 8, Y(l.p) + 6, { size: 9.5, color: l.on ? C.g : C.dim, mono: true })
    })
    const lx = Math.min(Math.log2(topPeak / bwT) - 1.6, LX1 - 2)     // label sits on the upper part of the slope, clear of the kernel dots
    text(ctx, `HBM ${fmtBW(g.bw)}`, X(2 ** lx) + 10, Y(bwT * 2 ** lx) + 14, { size: 10.5, color: C.a, mono: true, weight: 700 })
  })

  const setFromEvent = e => {
    const { x, y } = pt(e)
    const r = e.currentTarget.getBoundingClientRect()
    const pw = r.width - ML - MR, ph = RH - MT - MB
    setMy({ lx: clamp(LX0 + (x - ML) / pw * (LX1 - LX0), LX0, LX1 - 0.2), lp: clamp(LY0 + (MT + ph - y) / ph * (LY1 - LY0), LY0, LY1) })
  }

  const pct = 100 * cur.att / cur.peak
  const advice = cur.mem
    ? (cur.ai < cur.rg / 8
      ? <>Far left of the ridge ({(cur.rg / cur.ai).toFixed(0)}× below it): extra FLOPs are free, <b>bytes are the cost</b>. Fuse kernels, vectorise loads (<b>float4</b>), store data in fewer bits, never re-read what you can keep in registers/shared memory.</>
      : <>Memory-bound but near the ridge: raising reuse (tiling, register blocking, bigger tiles) pushes you right; after that you will hit the compute roof.</>)
    : (cur.tc && !useTC && prec !== 'fp64'
      ? <>Compute-bound on CUDA cores — but this kernel could use <b>Tensor Cores</b>. Flip the toggle and watch the roof jump.</>
      : <>Compute-bound: memory is no longer the problem. Gains now come from instruction efficiency, ILP, and (if available) Tensor Cores / lower precision.</>)

  return (
    <>
      <canvas {...cv}
        style={{ ...cv.style, cursor: 'crosshair' }}
        onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); drag.current = true; setFocus('my'); setFromEvent(e) }}
        onPointerMove={e => { if (drag.current) setFromEvent(e) }}
        onPointerUp={() => { drag.current = false }} />
      <Legend items={[...rows.map(r => [r.mem ? C.d : C.e, `${r.n} ${r.name}`]), [C.c, '★ your kernel (drag on the plot)']]} />
      <Controls>
        <Select label="GPU" value={gk} onChange={setGk} options={GPU_OPTIONS} />
        <Select label="Data type" value={prec} onChange={setPrec} options={[['fp64', 'FP64 (8 B)'], ['fp32', 'FP32 (4 B)'], ['fp16', 'FP16 (2 B)']]} />
        <Toggle label="Tensor Cores for GEMM-class kernels" value={useTC} onChange={setUseTC} />
        <Toggle label="Show L2 roof" value={showL2} onChange={setShowL2} />
        <Select label="Focus" value={focus} onChange={setFocus} options={[...rows.map(r => [r.id, `${r.n} ${r.name}`]), ['my', '★ your kernel']]} />
      </Controls>
      <Controls>
        <Slider label="GEMM size N" min={6} max={14} value={nE} onChange={setNE} fmt={v => 2 ** v} />
        <Slider label="Tile size T" min={2} max={7} value={tE} onChange={setTE} fmt={v => 2 ** v} />
        <Slider label="Decode batch" min={0} max={9} value={bE} onChange={setBE} fmt={v => 2 ** v} />
      </Controls>
      <Readout>
        <b>{cur.n} {cur.name}</b>: AI = <b>{cur.ai.toFixed(cur.ai < 10 ? 3 : 1)}</b> FLOP/B <span style={{ opacity: 0.7 }}>({cur.how})</span><br />
        Ceiling <b>{CEIL_LONG[cur.key]}</b> {cur.peak} TF, ridge <b>{cur.rg.toFixed(cur.rg < 10 ? 1 : 0)}</b> FLOP/B → attainable min({cur.peak}, {bwT.toFixed(2)}×{cur.ai.toFixed(2)}) = <b>{cur.att < 1 ? (cur.att * 1000).toFixed(0) + ' GFLOP/s' : cur.att.toFixed(1) + ' TFLOP/s'}</b> ({pct.toFixed(pct < 10 ? 1 : 0)}% of peak) →
        {' '}<b className={cur.mem ? 'w' : 'g'}>{cur.mem ? 'memory-bound' : 'compute-bound'}</b>. {advice}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB D — Where does it live?
   ===================================================================== */
const SPACES = [['reg', 'Register'], ['local', 'Local memory'], ['shared', 'Shared memory'], ['const', 'Constant memory'], ['global', 'Global memory']]
const CHALLENGES = [
  { ask: 'Where does `x` live?', ans: 'reg', code: '__global__ void k(const float* a, float* o) {\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  float x = a[i] * 2.0f;      // <- x\n  o[i] = x;\n}', why: 'A scalar automatic variable is assigned a **register**: fastest storage on the chip, private to the thread.' },
  { ask: 'Where does `tile` live?', ans: 'shared', code: '__global__ void k(...) {\n  __shared__ float tile[32][33];   // <- tile\n  ...\n}', why: '`__shared__` puts one copy **per block** in on-chip SRAM (32 banks). The `33` pads rows to dodge bank conflicts (next chapters).' },
  { ask: 'Where does `coeff` live?', ans: 'const', code: '__constant__ float coeff[16];       // <- coeff\n// host: cudaMemcpyToSymbol(coeff, h_coeff, sizeof(h_coeff));\n__global__ void k(...) { y = coeff[3] * x; }', why: '**Constant memory** (64 KB total) is backed by device memory but read through a dedicated cache that **broadcasts** one address to a whole warp at register-like speed.' },
  { ask: 'Where does `buf` live?', ans: 'local', code: '__global__ void k(const int* idx, float* o) {\n  float buf[64];\n  for (int j = 0; j < 64; ++j) buf[j] = j * 0.5f;\n  o[threadIdx.x] = buf[idx[threadIdx.x] % 64];   // runtime index\n}', why: 'Registers are not addressable, so a per-thread array indexed with a **runtime value** goes to **local memory**: private to the thread but physically in device memory (cached by L1/L2). Name says local, speed says global.' },
  { ask: 'Where does the *data* behind `d_x` live?', ans: 'global', code: 'float* d_x;\ncudaMalloc(&d_x, n * sizeof(float));   // <- data behind d_x', why: '`cudaMalloc` returns **global memory** (HBM / GDDR). Visible to every thread and kernel until `cudaFree`.' },
  { ask: 'Where does `acc` live?', ans: 'reg', code: 'float acc[4] = {0, 0, 0, 0};\n#pragma unroll\nfor (int j = 0; j < 4; ++j) acc[j] += a[i + j] * w;', why: 'After the loop is **unrolled** every index is a compile-time constant, so the compiler keeps `acc[0..3]` in four **registers**. Without the unroll (runtime index) it could fall back to local memory.' },
  { ask: 'Where does `s` live?', ans: 'shared', code: 'extern __shared__ float s[];          // <- s\n...\nkernel<<<grid, block, 4096>>>(...);   // 4096 B of dynamic shared memory', why: 'The third launch parameter sizes **dynamic shared memory**: still one block-private copy in on-chip SRAM.' },
  { ask: 'Where does `counter` live?', ans: 'global', code: '__device__ unsigned int counter;      // <- counter\n__global__ void k() { atomicAdd(&counter, 1u); }', why: 'A `__device__` variable is statically allocated **global memory**, visible to all threads. The atomic is performed in the **L2** cache, which is why contended atomics are slow.' },
  { ask: 'A kernel needs ~300 live floats per thread but a thread may use at most 255 registers. Where does the excess go?', ans: 'local', code: '// nvcc -Xptxas -v prints:\n//   ptxas info : 168 bytes stack frame,\n//                120 bytes spill stores, 132 bytes spill loads', why: 'The compiler **spills** to **local memory** (in device memory, cached in L1/L2). It also lowers occupancy. Check the spill counters in `-Xptxas -v` output.' },
  { ask: 'Where does `tid` live?', ans: 'reg', code: 'int tid = threadIdx.x;   // <- tid', why: '`threadIdx` itself is a read-only **special register**; copying it into a local gives an ordinary **register**.' }
]

function SpacesLab() {
  const [i, setI] = useState(0)
  const [pick, setPick] = useState(null)
  const [score, setScore] = useState({ ok: 0, n: 0 })
  const c = CHALLENGES[i]
  const choose = a => {
    if (pick) return
    setPick(a)
    setScore(s => ({ ok: s.ok + (a === c.ans ? 1 : 0), n: s.n + 1 }))
  }
  const next = () => { setPick(null); setI(v => (v + 1) % CHALLENGES.length) }
  return (
    <>
      <div style={{ display: 'grid', gap: 12 }}>
        <div style={{ font: '600 11px var(--f-mono)', letterSpacing: '.12em', textTransform: 'uppercase', color: C.mute }}>
          Challenge {i + 1} / {CHALLENGES.length} · score {score.ok}/{score.n}
        </div>
        <CodeBlock lang="cuda" title="Which memory space?" src={c.code} />
        <div style={{ fontSize: 15 }}><Md inline>{c.ask}</Md></div>
      </div>
      <Controls>
        {SPACES.map(([k, l]) => (
          <Btn key={k} primary={pick === k} onClick={() => choose(k)}
            className={pick ? (k === c.ans ? 'ok' : k === pick ? 'bad' : '') : ''}>
            {pick && k === c.ans ? '✓ ' : pick === k ? '✗ ' : ''}{l}
          </Btn>
        ))}
        {pick && <Btn onClick={next}>Next →</Btn>}
      </Controls>
      <Readout>
        {pick
          ? <><b className={pick === c.ans ? 'g' : 'r'}>{pick === c.ans ? 'Correct. ' : 'Not quite. '}</b><Md inline>{c.why}</Md></>
          : <>Pick the memory space. Remember: <b>scope</b> (who can see it) and <b>physical location</b> (where the bytes are) are different questions.</>}
      </Readout>
    </>
  )
}

/* ===================================================================== */
function Lab() {
  return <SubViews views={[
    { id: 'h', label: '3D: the memory hierarchy', render: () => <HierarchyLab /> },
    { id: 'l', label: 'Latency, bandwidth & Little’s law', render: () => <LatencyLab /> },
    { id: 'r', label: 'Roofline explorer', render: () => <RooflineLab /> },
    { id: 's', label: 'Where does it live?', render: () => <SpacesLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Descend the memory hierarchy, then find where your kernel sits on the roofline',
  tryIt: [
    'In the 3D tab set **L1 hit = 0%** and **L2 hit = 0%**: every load pays the full HBM trip. Now raise the hit rates and watch **AMAT** collapse.',
    'Click each slab and switch **slab width** between capacity, latency and bandwidth: capacity grows downward while bandwidth and speed shrink.',
    'In the Little’s-law tab pick **HBM** with 4-byte loads: you need ~50 warps/SM. Raise **loads in flight** to 4 and the same bandwidth needs only a few.',
    'In the Roofline tab compare **vector add** with **library GEMM** while sweeping **N**; then enable Tensor Cores with **FP16** and see the GEMM ridge move far to the right.',
    'Select **LLM decode**, set FP16 and slide the **batch** from 1 to 512: it only becomes compute-bound near the ridge.',
    'Drag the **★** into the compute-bound region with Tensor Cores off — see the advice change.'
  ],
  theory, math, practice,
  code: [
    { title: 'STREAM-style bandwidth benchmark: copy / scale / add / triad (+ float4)', lang: 'cuda', note: 'Measures effective HBM bandwidth and compares it with the theoretical number from the device attributes.', src: stream },
    { title: 'The memory spaces in one kernel: register, local, shared, constant, global', lang: 'cuda', note: 'Compile with -Xptxas -v to see register counts and spills.', src: spaces },
    { title: 'Pointer chasing: measure the latency ladder (L1 → L2 → HBM)', lang: 'cuda', note: 'Latency jumps as the working set outgrows each cache level. This is the real version of the lab ladder.', src: chase },
    { title: 'Roofline probe: sweep arithmetic intensity and measure your own roofline', lang: 'cuda', note: 'Prints CSV: FMAs per element, FLOP/B, GFLOP/s, GB/s.', src: probe },
    { title: 'Plot the measured roofline (matplotlib)', lang: 'python', src: plot }
  ],
  quiz: [
    { q: 'Order these from lowest to highest access latency on a modern NVIDIA GPU:', options: ['HBM, L2, shared memory, registers', 'Registers, shared memory / L1, L2, HBM', 'Registers, L2, shared memory, HBM', 'Shared memory, registers, HBM, L2'], answer: 1, why: 'Registers (a few cycles) → shared memory / L1 (tens of cycles) → L2 (a couple of hundred) → HBM (several hundred cycles).' },
    { q: 'A per-thread array `float buf[64]` indexed by a runtime value lives in:', options: ['Registers', 'Shared memory', 'Local memory (backed by device memory)', 'Constant memory'], answer: 2, why: 'Registers cannot be indexed dynamically. The array goes to **local memory**: private to the thread but physically in device memory, cached by L1/L2.' },
    { q: 'SAXPY (`y = a*x + y`, FP32) on an A100 (≈19.5 TFLOP/s, ≈2 TB/s) is:', options: ['Compute-bound; it can reach ~19.5 TFLOP/s', 'Memory-bound at ≈0.17 FLOP/B; roughly 0.34 TFLOP/s attainable', 'Latency-bound at exactly 1 TFLOP/s', 'Limited by shared memory'], answer: 1, why: '2 FLOP per 12 bytes = $0.167$ FLOP/B, far below the ridge ($\\approx9.6$). Attainable $\\approx 0.167\\times2.04\\text{ TB/s}\\approx0.34$ TFLOP/s ($\\approx1.7\\%$ of peak).' },
    { q: 'The ridge point of the roofline is:', options: ['The point where latency equals bandwidth', '$\\pi/\\beta$: peak FLOP/s divided by peak bytes/s', 'The L2 cache size', 'The number of SMs'], answer: 1, why: 'Kernels with arithmetic intensity below $I^*=\\pi/\\beta$ are bandwidth-limited; above it they are limited by compute.' },
    { q: 'Tiling a GEMM with $T\\times T$ shared-memory tiles changes its global-memory arithmetic intensity by about:', options: ['×1 (no change)', '×T', '×T²', '÷T'], answer: 1, why: 'Each element loaded into a tile is reused $T$ times, so global traffic drops by $T$ and AI rises by $T$.' },
    { q: 'Why is batch-1 LLM decoding usually memory-bound?', options: ['Softmax is slow', 'Each weight is read once but used for only ~2 FLOP, so AI ≈ 1 FLOP/B', 'Tensor Cores cannot run it', 'Kernel launch overhead dominates'], answer: 1, why: 'Per token every weight streams from HBM and performs one multiply-add. Batching reuses each weight across sequences and raises AI.' },
    { q: 'Little’s law says to saturate memory bandwidth you need roughly ___ bytes in flight per SM.', options: ['Bandwidth ÷ latency', 'Bandwidth × latency', 'Latency ÷ clock', 'The L2 size'], answer: 1, why: '$\\text{in-flight}=\\text{bandwidth}\\times\\text{latency}$. That is why many resident warps, or many independent (and wide) loads per thread, are needed.' }
  ]
}
