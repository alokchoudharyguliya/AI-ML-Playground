import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import Stage3D from '../../../components/Stage3D.jsx'
import { Controls, Slider, Select, Btn, Readout, SubViews, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, clamp, heatRGB, text, polyline } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import optimizers from './code/optimizers.py?raw'
import trainLoop from './code/train_loop.py?raw'

/* ---------- test functions ---------- */
const FUNCS = {
  ravine: {
    name: 'Ravine (ill-conditioned bowl)', dom: [-8, 8, -4, 4], start: [-7, 3.2], lr: [0.15, 0.3],
    f: (x, y) => 0.05 * x * x + y * y, g: (x, y) => [0.1 * x, 2 * y]
  },
  himmelblau: {
    name: 'Himmelblau (4 minima)', dom: [-5, 5, -5, 5], start: [-0.8, -4.2], lr: [0.012, 0.12],
    f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    g: (x, y) => [4 * x * (x * x + y - 11) + 2 * (x + y * y - 7), 2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)]
  },
  rosenbrock: {
    name: 'Rosenbrock (curved valley)', dom: [-2, 2, -1, 3], start: [-1.5, 2.3], lr: [0.0005, 0.05],
    f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    g: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)]
  }
}
const OPTS = [
  { id: 'sgd', name: 'SGD', color: '#fb7185', group: 0 },
  { id: 'mom', name: 'Momentum', color: '#fbbf24', group: 0 },
  { id: 'rms', name: 'RMSProp', color: '#4ade80', group: 1 },
  { id: 'adam', name: 'Adam', color: '#22d3ee', group: 1 }
]
const SIZE = 8, HEIGHT = 3

function fmax(fn) {
  let m = 0
  const [x0, x1, y0, y1] = fn.dom
  for (let i = 0; i <= 60; i++) for (let j = 0; j <= 60; j++) m = Math.max(m, fn.f(x0 + (x1 - x0) * i / 60, y0 + (y1 - y0) * j / 60))
  return m
}

function stepRunner(r, fn, lr) {
  const [gx, gy] = fn.g(r.p[0], r.p[1])
  const g = [gx, gy]
  const gn = Math.hypot(gx, gy)
  if (gn > 1e3) { g[0] *= 1e3 / gn; g[1] *= 1e3 / gn }  // safety clip for display only
  r.t++
  const eps = 1e-8
  for (let i = 0; i < 2; i++) {
    if (r.id === 'sgd') r.p[i] -= lr * g[i]
    else if (r.id === 'mom') { r.m[i] = 0.9 * r.m[i] + g[i]; r.p[i] -= lr * r.m[i] }
    else if (r.id === 'rms') { r.v[i] = 0.9 * r.v[i] + 0.1 * g[i] * g[i]; r.p[i] -= lr * g[i] / (Math.sqrt(r.v[i]) + eps) }
    else {
      r.m[i] = 0.9 * r.m[i] + 0.1 * g[i]
      r.v[i] = 0.999 * r.v[i] + 0.001 * g[i] * g[i]
      r.p[i] -= lr * (r.m[i] / (1 - 0.9 ** r.t)) / (Math.sqrt(r.v[i] / (1 - 0.999 ** r.t)) + eps)
    }
  }
  const [x0, x1, y0, y1] = fn.dom
  r.p[0] = clamp(r.p[0], x0, x1); r.p[1] = clamp(r.p[1], y0, y1)
  r.loss = fn.f(r.p[0], r.p[1])
  if (r.t % 2 === 0) { r.trail.push([r.p[0], r.p[1]]); if (r.trail.length > 700) r.trail.shift() }
}

/* ---------- 3D pieces ---------- */
function Surface({ fn, fm, onPick }) {
  const geo = useMemo(() => {
    const n = 90, [x0, x1, y0, y1] = fn.dom
    const pos = new Float32Array((n + 1) * (n + 1) * 3), col = new Float32Array((n + 1) * (n + 1) * 3)
    const idx = []
    for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) {
      const x = x0 + (x1 - x0) * i / n, y = y0 + (y1 - y0) * j / n
      const h = Math.log1p(fn.f(x, y)) / Math.log1p(fm)
      const k = (j * (n + 1) + i) * 3
      pos[k] = (i / n - 0.5) * SIZE; pos[k + 1] = h * HEIGHT; pos[k + 2] = (j / n - 0.5) * SIZE
      const [r, g, b] = heatRGB(1 - Math.pow(h, 0.6)); col[k] = r / 255; col[k + 1] = g / 255; col[k + 2] = b / 255
    }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1
      idx.push(a, c, b, b, c, d)
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('color', new THREE.BufferAttribute(col, 3))
    g.setIndex(idx); g.computeVertexNormals()
    return g
  }, [fn, fm])
  return (
    <group>
      <mesh geometry={geo} onClick={e => { const [x0, x1, y0, y1] = fn.dom; onPick([x0 + (e.point.x / SIZE + 0.5) * (x1 - x0), y0 + (e.point.z / SIZE + 0.5) * (y1 - y0)]) }}>
        <meshStandardMaterial vertexColors side={THREE.DoubleSide} roughness={0.75} metalness={0.05} />
      </mesh>
      <mesh geometry={geo}><meshBasicMaterial wireframe color="#ffffff" transparent opacity={0.05} /></mesh>
    </group>
  )
}

function Trail({ r, fn, fm }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(701 * 3), 3))
    g.setDrawRange(0, 0)
    return g
  }, [])
  const objs = useMemo(() => [
    new THREE.Line(geo, new THREE.LineBasicMaterial({ color: r.color })),
    new THREE.Points(geo, new THREE.PointsMaterial({ color: r.color, size: 0.07, sizeAttenuation: true }))
  ], [geo, r.color])
  const marker = useRef()
  const [x0, x1, y0, y1] = fn.dom
  const W = (x, y) => {
    const h = Math.log1p(fn.f(x, y)) / Math.log1p(fm) * HEIGHT + 0.06
    return [(x - x0) / (x1 - x0) * SIZE - SIZE / 2, h, (y - y0) / (y1 - y0) * SIZE - SIZE / 2]
  }
  useFrame(() => {
    const a = geo.attributes.position.array
    const tr = r.trail
    for (let i = 0; i < tr.length; i++) { const w = W(tr[i][0], tr[i][1]); a[i * 3] = w[0]; a[i * 3 + 1] = w[1]; a[i * 3 + 2] = w[2] }
    geo.attributes.position.needsUpdate = true
    geo.setDrawRange(0, tr.length)
    const w = W(r.p[0], r.p[1])
    if (marker.current) marker.current.position.set(w[0], w[1] + 0.07, w[2])
  })
  return (
    <>
      {objs.map((o, i) => <primitive key={i} object={o} />)}
      <mesh ref={marker}>
        <sphereGeometry args={[0.13, 20, 20]} />
        <meshStandardMaterial color={r.color} emissive={r.color} emissiveIntensity={0.9} />
      </mesh>
    </>
  )
}

function Sim({ fn, fm, sim, lrRef, speedRef, runKey }) {
  const runners = useMemo(() => {
    const rs = OPTS.map(o => ({ ...o, p: [...sim.start], m: [0, 0], v: [0, 0], t: 0, trail: [[...sim.start]], loss: fn.f(...sim.start) }))
    sim.runners = rs
    return rs
  }, [fn, sim.start, runKey]) // eslint-disable-line
  const acc = useRef(0)
  useFrame((_, dt) => {
    acc.current += Math.min(dt, 0.05) * speedRef.current
    while (acc.current >= 1) {
      acc.current -= 1
      runners.forEach(r => stepRunner(r, fn, lrRef.current[r.group]))
    }
  })
  return runners.map(r => <Trail key={r.id + runKey + fn.name} r={r} fn={fn} fm={fm} />)
}

/* ---------- race lab ---------- */
function RaceLab() {
  const [fk, setFk] = useState('ravine')
  const fn = FUNCS[fk]
  const fm = useMemo(() => fmax(fn), [fn])
  const [start, setStart] = useState(fn.start)
  const [lrS, setLrS] = useState(Math.log10(fn.lr[0]))
  const [lrA, setLrA] = useState(Math.log10(fn.lr[1]))
  const [speed, setSpeed] = useState(30)
  const [runKey, setRunKey] = useState(0)
  const [, force] = useState(0)
  const sim = useRef({ start: fn.start, runners: [] })
  sim.current.start = start
  const lrRef = useRef([0, 0]); lrRef.current = [10 ** lrS, 10 ** lrA]
  const speedRef = useRef(30); speedRef.current = speed

  const changeFn = k => {
    setFk(k); const f = FUNCS[k]
    setStart(f.start); setLrS(Math.log10(f.lr[0])); setLrA(Math.log10(f.lr[1])); setRunKey(x => x + 1)
  }
  useEffect(() => { const id = setInterval(() => force(x => x + 1), 250); return () => clearInterval(id) }, [])
  const rs = sim.current.runners

  return (
    <>
      <Stage3D height={430} camera={[7.5, 7, 9]} overlay={<><b>Click the surface</b> to choose a start point.<br />Colour = loss (bright = high). Height = log(1+loss).</>} hint="drag to orbit · click surface to move the start">
        <group position={[0, -1, 0]}>
          <Surface fn={fn} fm={fm} onPick={p => { setStart(p); setRunKey(x => x + 1) }} />
          <Sim fn={fn} fm={fm} sim={sim.current} lrRef={lrRef} speedRef={speedRef} runKey={runKey} />
        </group>
      </Stage3D>
      <Controls>
        <Select label="Surface" value={fk} onChange={changeFn} options={Object.entries(FUNCS).map(([k, v]) => [k, v.name])} />
        <Slider label="lr (SGD, Momentum)" min={-4} max={0} step={0.05} value={lrS} onChange={setLrS} fmt={v => (10 ** v).toPrecision(2)} />
        <Slider label="lr (RMSProp, Adam)" min={-3} max={0} step={0.05} value={lrA} onChange={setLrA} fmt={v => (10 ** v).toPrecision(2)} />
        <Slider label="Steps/s" min={5} max={200} value={speed} onChange={setSpeed} />
        <Btn onClick={() => setRunKey(x => x + 1)}>Restart</Btn>
      </Controls>
      <Legend items={OPTS.map(o => [o.color, o.name])} />
      <Readout>
        {rs.map(r => <span key={r.id} style={{ marginRight: 22 }}>{r.name}: <b style={{ color: r.color }}>{Number.isFinite(r.loss) ? r.loss.toPrecision(3) : '—'}</b> <span style={{ color: C.dim }}>@ {r.t}</span></span>)}
      </Readout>
    </>
  )
}

/* ---------- schedules lab ---------- */
const SCHEDS = [
  { id: 'const', short: 'Const', name: 'Constant', color: C.mute, f: (t, T, w, lr) => lr },
  { id: 'step', short: 'Step', name: 'Step decay (×0.1 ×3)', color: C.r, f: (t, T, w, lr) => lr * 0.1 ** Math.floor(t / (T / 3)) },
  { id: 'cos', short: 'Cos', name: 'Cosine', color: C.b, f: (t, T, w, lr) => 0.5 * lr * (1 + Math.cos(Math.PI * t / T)) },
  { id: 'wcos', short: 'W+cos', name: 'Warmup + cosine', color: C.a, f: (t, T, w, lr) => (t < w ? lr * t / w : 0.5 * lr * (1 + Math.cos(Math.PI * (t - w) / (T - w)))) },
  { id: 'wlin', short: 'W+lin', name: 'Warmup + linear', color: C.e, f: (t, T, w, lr) => (t < w ? lr * t / w : lr * (1 - (t - w) / (T - w))) },
  { id: 'noam', short: 'Noam', name: 'Inverse-sqrt (Noam)', color: C.d, f: (t, T, w, lr) => lr * Math.min((t + 1) / w, Math.sqrt(w / (t + 1))) },
  { id: 'one', short: '1cycle', name: 'One-cycle', color: C.c, f: (t, T, w, lr) => {
    const p = t / T
    return p < 0.3 ? lr / 25 + (lr - lr / 25) * 0.5 * (1 - Math.cos(Math.PI * p / 0.3)) : lr + (lr / 1e4 - lr) * 0.5 * (1 - Math.cos(Math.PI * (p - 0.3) / 0.7))
  } }
]
function SchedLab() {
  const [T, setT] = useState(10000)
  const [w, setW] = useState(500)
  const [lr, setLr] = useState(3e-4)
  const [cur, setCur] = useState(2500)
  const [cp] = useCanvas(330, (ctx, W, H) => {
    const px = 60, py = 16, pw = W - px - 16, ph = H - 54
    const X = t => px + (t / T) * pw, Y = v => py + ph - (v / (lr * 1.08)) * ph
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) { const v = lr * i / 4; ctx.beginPath(); ctx.moveTo(px, Y(v)); ctx.lineTo(px + pw, Y(v)); ctx.stroke(); text(ctx, v.toExponential(1), px - 8, Y(v), { size: 10.5, align: 'right', mono: true, color: C.mute }) }
    for (let i = 0; i <= 5; i++) text(ctx, Math.round(T * i / 5), X(T * i / 5), py + ph + 14, { size: 10.5, align: 'center', mono: true, color: C.mute })
    text(ctx, 'step', px + pw / 2, H - 8, { size: 11, align: 'center', color: C.mute })
    SCHEDS.forEach(s => {
      const pts = []
      for (let i = 0; i <= 240; i++) { const t = T * i / 240; pts.push([X(t), Y(Math.max(0, s.f(Math.min(t, T - 1), T, Math.min(w, T / 2), lr)))]) }
      polyline(ctx, pts, s.color, 2.2)
    })
    ctx.strokeStyle = 'rgba(255,255,255,.3)'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(cur), py); ctx.lineTo(X(cur), py + ph); ctx.stroke(); ctx.setLineDash([])
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Total steps" min={2000} max={50000} step={1000} value={T} onChange={setT} />
        <Slider label="Warmup steps" min={0} max={2000} step={50} value={w} onChange={setW} />
        <Slider label="Peak lr" min={-5} max={-2} step={0.1} value={Math.log10(lr)} onChange={v => setLr(10 ** v)} fmt={v => (10 ** v).toExponential(1)} />
        <Slider label="Inspect step" min={0} max={T - 1} step={50} value={Math.min(cur, T - 1)} onChange={setCur} />
      </Controls>
      <Legend items={SCHEDS.map(s => [s.color, s.name])} />
      <Readout>
        {SCHEDS.map(s => <span key={s.id} style={{ marginRight: 18 }}>{s.short}: <b style={{ color: s.color }}>{Math.max(0, s.f(Math.min(cur, T - 1), T, Math.min(w, T / 2) || 1, lr)).toExponential(2)}</b></span>)}
      </Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'race', label: '3D optimizer race', render: () => <RaceLab /> },
    { id: 'sched', label: 'Learning-rate schedules', render: () => <SchedLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Race SGD, Momentum, RMSProp and Adam across a loss landscape',
  tryIt: [
    'On the **Ravine**, watch SGD zig-zag across the narrow axis while Momentum builds speed along the valley.',
    'On **Himmelblau** click different start points — different optimizers can fall into *different* minima.',
    'On **Rosenbrock**, crank the SGD learning rate up until it diverges (the marker gets clamped to the wall).',
    'In the schedules view, add **warmup** and compare cosine, linear and inverse-sqrt decay shapes.'
  ],
  theory, math, practice,
  code: [
    { title: 'SGD, Momentum, RMSProp and Adam from scratch (NumPy)', lang: 'python', note: 'Exactly the update rules used by the 3D lab. Runs on the ravine and prints the final loss for each.', src: optimizers },
    { title: 'Production training loop: AdamW + warmup-cosine + clipping + AMP + accumulation', lang: 'python', src: trainLoop }
  ],
  quiz: [
    { q: 'For gradient descent on a quadratic with largest Hessian eigenvalue $\\lambda_{max}$, the largest stable learning rate is:', options: ['$1/\\lambda_{max}$', '$2/\\lambda_{max}$', '$\\lambda_{max}$', 'Independent of curvature'], answer: 1, why: 'Each eigen-direction contracts by $|1-\\eta\\lambda|$, which is $<1$ iff $\\eta<2/\\lambda$.' },
    { q: 'Why does momentum help in a ravine?', options: ['It reduces memory use', 'It cancels oscillating components and accumulates the consistent one', 'It adds noise that escapes minima', 'It makes the loss convex'], answer: 1, why: 'Gradient components across the valley alternate sign and average out; the component along the valley accumulates.' },
    { q: 'With bias correction, the very first Adam step has magnitude about:', options: ['$\\eta\\,g$', '$\\eta\\,g^2$', '$\\eta\\,\\mathrm{sign}(g)$', '$0$'], answer: 2, why: '$\\hat m_1=g,\\ \\hat v_1=g^2$ so the step is $\\eta g/|g|=\\eta\\,\\mathrm{sign}(g)$.' },
    { q: 'What is the difference between L2 regularization and decoupled weight decay (AdamW)?', options: ['None', 'L2 is added to the gradient and so gets rescaled by Adam; AdamW shrinks weights directly', 'AdamW only decays biases', 'L2 is used only at inference'], answer: 1, why: 'Inside Adam the L2 term is divided by $\\sqrt{\\hat v}$, weakening the decay for large-gradient weights; AdamW applies decay outside the adaptive step.' },
    { q: 'Why is learning-rate warmup common for Transformers?', options: ['To save memory', 'Early second-moment estimates and attention logits are unreliable, so large steps destabilize training', 'To increase batch size', 'It is required by softmax'], answer: 1, why: 'Adam\'s variance estimate is poor in the first steps and post-LN Transformers are sensitive early; ramping the LR avoids divergence.' }
  ]
}
