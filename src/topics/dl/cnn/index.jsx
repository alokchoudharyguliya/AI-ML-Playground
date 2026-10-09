import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Edges, Html } from '@react-three/drei'
import Stage3D from '../../../components/Stage3D.jsx'
import { Controls, Slider, Select, Toggle, Btn, Readout, SubViews, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, clamp, div, text, rr, mix, polyline, fmtInt } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import resnet from './code/resnet.py?raw'
import scratch from './code/conv_scratch.py?raw'

/* ---------------- data ---------------- */
const N = 12
const SEVEN = [
  '............', '.##########.', '.##########.', '.........##.', '........##..', '.......##...',
  '......##....', '.....##.....', '....##......', '....##......', '....##......', '............'
]
function makeScene(name) {
  const a = Array.from({ length: N }, () => Array(N).fill(0))
  if (name === 'seven') SEVEN.forEach((row, y) => [...row].forEach((ch, x) => { a[y][x] = ch === '#' ? 1 : 0 }))
  else if (name === 'shapes') {
    for (let y = 1; y < 5; y++) for (let x = 1; x < 5; x++) a[y][x] = 1
    for (let y = 1; y < 11; y++) a[y][8] = 1
    for (let i = 0; i < 6; i++) a[5 + i][1 + i] = 1
    for (let x = 4; x < 11; x++) a[9][x] = 1
  } else {
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) a[y][x] = x / (N - 1) * 0.6
    for (let y = 3; y < 9; y++) for (let x = 3; x < 9; x++) a[y][x] = 1
  }
  return a
}
const KERNELS = {
  identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  sobelx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  sobely: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
  blur: Array.from({ length: 3 }, () => Array(3).fill(1 / 9)),
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  laplace: [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
  emboss: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
}
const KNAMES = [['sobelx', 'Sobel-X (vertical edges)'], ['sobely', 'Sobel-Y (horizontal edges)'], ['laplace', 'Laplacian'], ['blur', 'Box blur'], ['sharpen', 'Sharpen'], ['emboss', 'Emboss'], ['identity', 'Identity']]

function conv2d(img, k, s, p) {
  const n = img.length, K = k.length
  const O = Math.floor((n + 2 * p - K) / s) + 1
  const at = (y, x) => (y >= 0 && x >= 0 && y < n && x < n ? img[y][x] : 0)
  return Array.from({ length: O }, (_, oy) => Array.from({ length: O }, (_, ox) => {
    let acc = 0
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) acc += k[i][j] * at(oy * s - p + i, ox * s - p + j)
    return acc
  }))
}
function maxpool2(m) {
  const O = Math.floor(m.length / 2)
  return Array.from({ length: O }, (_, y) => Array.from({ length: O }, (_, x) => Math.max(m[2 * y][2 * x], m[2 * y][2 * x + 1], m[2 * y + 1][2 * x], m[2 * y + 1][2 * x + 1])))
}

/* ---------------- 2D convolution lab ---------------- */
function ConvLab() {
  const [scene, setScene] = useState('seven')
  const [kn, setKn] = useState('sobelx')
  const [stride, setStride] = useState(1)
  const [pad, setPad] = useState(0)
  const [pool, setPool] = useState(false)
  const [relu, setRelu] = useState(true)
  const [speed, setSpeed] = useState(10)
  const [playing, setPlaying] = useState(true)
  const st = useRef({ i: 0, acc: 0 })

  const img = useMemo(() => makeScene(scene), [scene])
  const k = KERNELS[kn]
  const out = useMemo(() => conv2d(img, k, stride, pad), [img, k, stride, pad])
  const On = out.length
  const act = useMemo(() => (relu ? out.map(r => r.map(v => Math.max(0, v))) : out), [out, relu])
  const pooled = useMemo(() => (pool && On >= 2 ? maxpool2(act) : null), [pool, act, On])
  const maxAbs = useMemo(() => Math.max(1e-6, ...act.flat().map(Math.abs)), [act])
  useEffect(() => { st.current = { i: 0, acc: 0 } }, [scene, kn, stride, pad])

  const [cp] = useCanvas(390, (ctx, W, H, t, dt) => {
    const s = st.current
    const total = On * On
    if (playing) {
      s.acc += dt * speed
      while (s.acc >= 1) { s.acc -= 1; s.i = (s.i + 1) % (total + 8) }
    }
    const idx = Math.min(s.i, total - 1)
    const revealed = Math.min(s.i + 1, total)
    const oy = Math.floor(idx / On), ox = idx % On
    const Np = N + 2 * pad, K = 3
    const kcs = W < 560 ? 20 : 30, gap = W < 560 ? 18 : 34
    const pcells = pooled ? pooled.length : 0
    const cs = clamp((W - 28 - gap * (pooled ? 3 : 2) - K * kcs) / (Np + On + pcells), 5, 22)
    const y0 = 56
    const x0 = 14
    const x1 = x0 + Np * cs + gap
    const x2 = x1 + K * kcs + gap
    const x3 = x2 + On * cs + gap

    // titles
    text(ctx, `Input ${N}×${N}${pad ? ` + pad ${pad}` : ''}`, x0, 24, { size: 12, color: C.mute, weight: 600 })
    text(ctx, 'Kernel 3×3', x1, 24, { size: 12, color: C.mute, weight: 600 })
    text(ctx, `${relu ? 'ReLU(' : ''}Feature map${relu ? ')' : ''} ${On}×${On}`, x2, 24, { size: 12, color: C.mute, weight: 600 })
    if (pooled) text(ctx, `Max-pool 2×2 → ${pcells}×${pcells}`, x3, 24, { size: 12, color: C.mute, weight: 600 })

    // input
    for (let r = 0; r < Np; r++) for (let c = 0; c < Np; c++) {
      const ir = r - pad, ic = c - pad
      const inside = ir >= 0 && ic >= 0 && ir < N && ic < N
      ctx.fillStyle = inside ? mix('#10131f', '#e7e9f4', img[ir][ic]) : '#0d0f1a'
      ctx.fillRect(x0 + c * cs, y0 + r * cs, cs - 1, cs - 1)
      if (!inside) { ctx.strokeStyle = '#2a2f4a'; ctx.setLineDash([2, 2]); ctx.strokeRect(x0 + c * cs + 0.5, y0 + r * cs + 0.5, cs - 2, cs - 2); ctx.setLineDash([]) }
    }
    // window
    const wx = x0 + ox * stride * cs, wy = y0 + oy * stride * cs
    ctx.fillStyle = 'rgba(139,123,255,.18)'; ctx.fillRect(wx, wy, K * cs, K * cs)
    ctx.strokeStyle = C.a; ctx.lineWidth = 2.5; ctx.strokeRect(wx, wy, K * cs, K * cs)

    // kernel
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
      ctx.fillStyle = div(k[i][j] / 2)
      rr(ctx, x1 + j * kcs, y0 + i * kcs, kcs - 2, kcs - 2, 4); ctx.fill()
      text(ctx, Math.abs(k[i][j]) < 0.2 && k[i][j] !== 0 ? '⅑' : String(+k[i][j].toFixed(1)), x1 + j * kcs + kcs / 2 - 1, y0 + i * kcs + kcs / 2, { size: 11, align: 'center', mono: true, color: '#fff' })
    }
    ctx.strokeStyle = C.a; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3])
    ctx.beginPath(); ctx.moveTo(wx + K * cs, wy); ctx.lineTo(x1, y0); ctx.moveTo(wx + K * cs, wy + K * cs); ctx.lineTo(x1, y0 + K * kcs); ctx.stroke(); ctx.setLineDash([])

    // output
    for (let r = 0; r < On; r++) for (let c = 0; c < On; c++) {
      const n = r * On + c
      ctx.fillStyle = n < revealed ? div(act[r][c] / maxAbs) : '#0d0f1a'
      ctx.fillRect(x2 + c * cs, y0 + r * cs, cs - 1, cs - 1)
    }
    ctx.strokeStyle = C.b; ctx.lineWidth = 2.5; ctx.strokeRect(x2 + ox * cs, y0 + oy * cs, cs - 1, cs - 1)

    // pooled
    if (pooled) {
      for (let r = 0; r < pcells; r++) for (let c = 0; c < pcells; c++) {
        const done = (2 * r + 1) * On + 2 * c + 1 < revealed
        ctx.fillStyle = done ? div(pooled[r][c] / maxAbs) : '#0d0f1a'
        ctx.fillRect(x3 + c * cs * 2, y0 + r * cs * 2, cs * 2 - 1, cs * 2 - 1)
      }
    }

    // current dot-product
    let sum = 0, parts = []
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
      const ir = oy * stride - pad + i, ic = ox * stride - pad + j
      const v = ir >= 0 && ic >= 0 && ir < N && ic < N ? img[ir][ic] : 0
      sum += v * k[i][j]
      if (v * k[i][j] !== 0) parts.push(`${(v * k[i][j]).toFixed(2).replace(/\.?0+$/, '')}`)
    }
    const yb = y0 + Math.max(Np * cs, K * kcs) + 34
    text(ctx, `y[${oy},${ox}] = Σ x·k = ${parts.length ? parts.slice(0, 7).join(' + ').replace(/\+ -/g, '− ') + (parts.length > 7 ? ' …' : '') : '0'} = ${sum.toFixed(2).replace(/\.?0+$/, '') || '0'}`, x0, yb, { size: 12.5, mono: true, color: C.ink })
    text(ctx, `O = ⌊(N + 2P − K)/S⌋ + 1 = ⌊(${N} + ${2 * pad} − 3)/${stride}⌋ + 1 = ${On}      MACs = ${On}²×9 = ${fmtInt(On * On * 9)}`, x0, yb + 24, { size: 12.5, mono: true, color: C.b })
  }, { animate: true })

  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Select label="Image" value={scene} onChange={setScene} options={[['seven', 'Digit 7'], ['shapes', 'Shapes'], ['grad', 'Gradient + box']]} />
        <Select label="Kernel" value={kn} onChange={setKn} options={KNAMES} />
        <Slider label="Stride" min={1} max={3} value={stride} onChange={setStride} />
        <Slider label="Padding" min={0} max={2} value={pad} onChange={setPad} />
        <Toggle label="ReLU" value={relu} onChange={setRelu} />
        <Toggle label="Max-pool" value={pool} onChange={setPool} />
        <Slider label="Speed" min={2} max={40} value={speed} onChange={setSpeed} fmt={v => v + '/s'} />
        <Btn onClick={() => setPlaying(p => !p)}>{playing ? 'Pause' : 'Play'}</Btn>
        <Btn onClick={() => { st.current = { i: 0, acc: 0 } }}>Restart</Btn>
      </Controls>
    </>
  )
}

/* ---------------- 3D feature-volume lab ---------------- */
function buildLayers(c) {
  const L = [{ name: 'Input image', ch: 3, hw: 64, params: 0, rf: 1, op: 'RGB 64×64' }]
  let ch = 3, hw = 64, rf = 1, j = 1
  const conv = (name, cout, s) => {
    const params = ch * cout * 9 + cout
    hw = Math.floor((hw + 2 - 3) / s) + 1
    rf += 2 * j; j *= s
    L.push({ name, ch: cout, hw, params, rf, op: `3×3 conv, stride ${s}, + BN + ReLU` })
    ch = cout
  }
  conv('conv1', c, 1); conv('conv2', c * 2, 2); conv('conv3', c * 4, 2); conv('conv4', c * 8, 2)
  L.push({ name: 'global avg-pool', ch, hw: 1, params: 0, rf: Math.min(64, rf + 2 * j * 4), op: 'average over H×W' })
  L.push({ name: 'fc (10 classes)', ch: 10, hw: 1, params: ch * 10 + 10, rf: 64, op: 'linear layer' })
  return L
}

function Slab({ layer, x, sel, onSelect, color }) {
  const th = 0.22 + Math.log2(Math.max(2, layer.ch)) * 0.2
  const sz = layer.hw === 1 ? 0.28 : layer.hw / 16
  return (
    <group position={[x, 0, 0]}>
      <mesh onClick={e => { e.stopPropagation(); onSelect() }} onPointerOver={() => (document.body.style.cursor = 'pointer')} onPointerOut={() => (document.body.style.cursor = '')}>
        <boxGeometry args={[th, sz, sz]} />
        <meshStandardMaterial color={color} transparent opacity={sel ? 0.85 : 0.5} emissive={color} emissiveIntensity={sel ? 0.5 : 0.12} />
        <Edges color={sel ? '#ffffff' : color} />
      </mesh>
      <Html position={[0, -2.6, 0]} center distanceFactor={9} style={{ pointerEvents: 'none' }}>
        <div style={{ font: '600 11px Inter Variable,sans-serif', color: sel ? '#fff' : '#9aa1c0', whiteSpace: 'nowrap', textAlign: 'center' }}>
          {layer.name}<br /><span style={{ font: '500 10px JetBrains Mono Variable,monospace', color: '#22d3ee' }}>{layer.ch}@{layer.hw}×{layer.hw}</span>
        </div>
      </Html>
    </group>
  )
}
function Pulse({ x0, x1 }) {
  const ref = useRef()
  useFrame(({ clock }) => { if (ref.current) ref.current.position.x = x0 + ((clock.elapsedTime * 1.4) % 1) * (x1 - x0) })
  return (
    <mesh ref={ref}>
      <boxGeometry args={[0.04, 4.4, 4.4]} />
      <meshBasicMaterial color="#22d3ee" transparent opacity={0.1} />
    </mesh>
  )
}

function VolumeLab() {
  const [base, setBase] = useState(16)
  const [selIdx, setSel] = useState(1)
  const layers = useMemo(() => buildLayers(base), [base])
  const xs = useMemo(() => {
    let x = 0
    return layers.map((l, i) => {
      const th = 0.22 + Math.log2(Math.max(2, l.ch)) * 0.2
      if (i) x += (0.22 + Math.log2(Math.max(2, layers[i - 1].ch)) * 0.2) / 2 + th / 2 + 0.75
      return x
    })
  }, [layers])
  const mid = xs[xs.length - 1] / 2
  const total = layers.reduce((a, l) => a + l.params, 0)
  const L = layers[Math.min(selIdx, layers.length - 1)]
  const colors = ['#e7e9f4', '#8b7bff', '#7c8cff', '#5aa8ff', '#22d3ee', '#4ade80', '#f472b6']
  return (
    <>
      <Stage3D height={400} camera={[mid, 3.2, 12]} target={[mid, 0, 0]} overlay={
        <><b>{L.name}</b> — {L.op}<br />output {L.ch}×{L.hw}×{L.hw} · params {fmtInt(L.params)}<br />receptive field ≈ {L.rf}×{L.rf} px</>
      }>
        {layers.map((l, i) => <Slab key={i} layer={l} x={xs[i]} sel={i === selIdx} onSelect={() => setSel(i)} color={colors[i % colors.length]} />)}
        <Pulse x0={xs[0]} x1={xs[xs.length - 1]} />
      </Stage3D>
      <Controls>
        <Slider label="Base channels" min={8} max={64} step={8} value={base} onChange={setBase} />
        <Readout>Total parameters: <b>{fmtInt(total)}</b> · click a slab to inspect it · note how <b>depth (channels) grows</b> while <b>spatial size shrinks</b></Readout>
      </Controls>
    </>
  )
}

/* ---------------- receptive field lab ---------------- */
const RF_CONFIGS = [
  { id: 'k3', name: '3×3 stride 1', color: C.a, layer: () => ({ k: 3, s: 1, d: 1 }) },
  { id: 'k3s2', name: '3×3, stride 2 every layer', color: C.b, layer: () => ({ k: 3, s: 2, d: 1 }) },
  { id: 'dil', name: '3×3 dilated (1,2,4,8…)', color: C.e, layer: i => ({ k: 3, s: 1, d: 2 ** i }) },
  { id: 'k3pool', name: '3×3 conv + 2×2 pool', color: C.d, layer: i => (i % 2 ? { k: 2, s: 2, d: 1 } : { k: 3, s: 1, d: 1 }) }
]
function rfCurve(cfg, L) {
  let rf = 1, j = 1; const out = [1]
  for (let i = 0; i < L; i++) { const { k, s, d } = cfg.layer(i); rf += (k - 1) * d * j; j *= s; out.push(rf) }
  return out
}
function RfLab() {
  const [L, setL] = useState(6)
  const [sel, setSel] = useState('k3')
  const curves = RF_CONFIGS.map(c => ({ c, v: rfCurve(c, 10) }))
  const cur = curves.find(x => x.c.id === sel)
  const [cp] = useCanvas(320, (ctx, W, H) => {
    const px = 50, py = 20, pw = W * 0.55 - px, ph = H - 60
    const ymax = Math.max(...curves.map(x => x.v[10]), 20)
    const Y = v => py + ph - (Math.log(v) / Math.log(ymax)) * ph
    const X = i => px + (i / 10) * pw
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1
    for (let i = 0; i <= 10; i += 2) { ctx.beginPath(); ctx.moveTo(X(i), py); ctx.lineTo(X(i), py + ph); ctx.stroke(); text(ctx, i, X(i), py + ph + 14, { size: 11, align: 'center', color: C.mute, mono: true }) }
    ;[1, 3, 10, 30, 100, 300, 1000].filter(v => v <= ymax).forEach(v => { ctx.beginPath(); ctx.moveTo(px, Y(v)); ctx.lineTo(px + pw, Y(v)); ctx.stroke(); text(ctx, v, px - 8, Y(v), { size: 11, align: 'right', color: C.mute, mono: true }) })
    text(ctx, 'layers →', px + pw / 2, H - 8, { size: 11, align: 'center', color: C.mute })
    text(ctx, 'receptive field (log scale)', px, 8, { size: 11, color: C.mute })
    curves.forEach(({ c, v }) => {
      polyline(ctx, v.map((r, i) => [X(i), Y(r)]), c.color, c.id === sel ? 3 : 1.6)
      ctx.fillStyle = c.color; ctx.beginPath(); ctx.arc(X(L), Y(v[L]), 4.5, 0, 7); ctx.fill()
    })
    ctx.strokeStyle = 'rgba(255,255,255,.25)'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(L), py); ctx.lineTo(X(L), py + ph); ctx.stroke(); ctx.setLineDash([])

    // picture of the receptive field on a 41×41 input
    const g = 41, cs = Math.min((W * 0.4) / g, (H - 60) / g), gx = W * 0.58, gy = 30
    text(ctx, `Selected: ${cur.c.name} @ ${L} layers`, gx, 12, { size: 11.5, color: C.mute })
    const rf = cur.v[L]
    for (let y = 0; y < g; y++) for (let x = 0; x < g; x++) {
      const inRf = Math.abs(x - 20) <= (rf - 1) / 2 && Math.abs(y - 20) <= (rf - 1) / 2
      ctx.fillStyle = inRf ? cur.c.color : '#141830'
      ctx.globalAlpha = inRf ? 0.85 : 1
      ctx.fillRect(gx + x * cs, gy + y * cs, cs - 0.5, cs - 0.5)
    }
    ctx.globalAlpha = 1
    ctx.fillStyle = '#fff'; ctx.fillRect(gx + 20 * cs, gy + 20 * cs, cs, cs)
    text(ctx, `RF = ${rf}×${rf}${rf > g ? ' (exceeds this 41px view)' : ''}`, gx, gy + g * cs + 16, { size: 12, mono: true, color: C.ink })
  })
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Layers" min={1} max={10} value={L} onChange={setL} />
        <Select label="Highlight" value={sel} onChange={setSel} options={RF_CONFIGS.map(c => [c.id, c.name])} />
        <Legend items={RF_CONFIGS.map(c => [c.color, c.name])} />
      </Controls>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'conv', label: '2D convolution', render: () => <ConvLab /> },
    { id: 'vol', label: '3D feature volumes', render: () => <VolumeLab /> },
    { id: 'rf', label: 'Receptive field growth', render: () => <RfLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Slide a kernel, build a CNN in 3D, grow a receptive field',
  tryIt: [
    'Pick **Sobel-X** on the digit 7 and watch only the vertical strokes light up; switch to **Sobel-Y** for the horizontal bar.',
    'Set **stride = 2** and **padding = 1** and read the output-size formula update live.',
    'In the 3D view, drag the *Base channels* slider and watch the parameter count scale ~quadratically.',
    'In the receptive-field view compare plain 3×3 stacks with **dilated** convolutions — the gap is exponential.'
  ],
  theory, math, practice,
  code: [
    { title: 'A small ResNet for CIFAR-10 (PyTorch)', lang: 'python', note: 'Residual blocks, BatchNorm, global average pooling, AMP and channels_last — the practical defaults.', src: resnet },
    { title: 'Convolution from scratch: loops vs im2col', lang: 'python', note: 'Shows that a convolution is just one big matrix multiply — exactly what the GPU does (see the CUDA track).', src: scratch }
  ],
  quiz: [
    { q: 'A $7\\times7$ conv with stride 2, padding 3 on a $224\\times224$ input gives an output of size:', options: ['$224$', '$112$', '$111$', '$109$'], answer: 1, why: '$\\lfloor(224+6-7)/2\\rfloor+1 = 112$. This is the ResNet stem.' },
    { q: 'Why do two stacked $3\\times3$ convs usually beat one $5\\times5$ conv?', options: ['They have more parameters', 'Same receptive field, fewer parameters, extra non-linearity', 'They are faster only on CPUs', 'They avoid padding'], answer: 1, why: '$2\\cdot9C^2=18C^2 < 25C^2$ with the same $5\\times5$ field and an additional ReLU.' },
    { q: 'The gradient of the loss with respect to the *input* of a conv layer is computed by:', options: ['A max-pool', 'A transposed convolution with the same kernel', 'A matrix inverse', 'Batch normalization'], answer: 1, why: 'Backprop through a convolution is a convolution of the upstream gradient with the flipped kernel (a transposed conv).' },
    { q: 'A depthwise-separable $3\\times3$ conv with $C_{in}=C_{out}=256$ costs about what fraction of a standard conv?', options: ['1/2', '1/4', '≈1/9', '≈1/100'], answer: 2, why: 'Ratio $=1/C_{out}+1/k^2 \\approx 0.004+0.111\\approx 0.115$.' },
    { q: 'What does the skip connection in a residual block mainly fix?', options: ['Overfitting', 'Optimization of very deep nets (gradient flow)', 'Memory use', 'Translation invariance'], answer: 1, why: '$\\partial(x+F(x))/\\partial x = I + \\partial F/\\partial x$ guarantees a direct gradient path.' }
  ]
}
