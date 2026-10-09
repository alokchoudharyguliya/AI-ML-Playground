import React, { useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import Stage3D from '../../../components/Stage3D.jsx'
import { Controls, Slider, Toggle, Readout, SubViews, Btn } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, rng, text, rr, alpha, clamp, fmtInt } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import vecadd from './code/vecadd.cu?raw'
import gray from './code/grayscale2d.cu?raw'
import query from './code/device_query.cu?raw'

/* ---------- 3D thread hierarchy ---------- */
const warpColor = w => new THREE.Color().setHSL(((w * 0.137) % 1), 0.72, 0.58)
const tmp = new THREE.Object3D()

function Threads({ bx, by, hover, setHover, mode }) {
  const ref = useRef()
  const n = bx * by
  const c = Math.min(0.5, 5.2 / bx, 3.6 / by), sp = c * 1.18
  useLayoutEffect(() => {
    const m = ref.current
    m.count = n
    for (let i = 0; i < n; i++) {
      const tx = i % bx, ty = Math.floor(i / bx)
      tmp.position.set((tx - (bx - 1) / 2) * sp, -(ty - (by - 1) / 2) * sp, 0)
      const s = i === hover ? 1.35 : 1
      tmp.scale.setScalar(s); tmp.updateMatrix()
      m.setMatrixAt(i, tmp.matrix)
      const warp = Math.floor(i / 32)
      const col = mode === 'warp' ? warpColor(warp) : new THREE.Color().setHSL(0.6, 0.5, 0.35 + 0.3 * ((i % bx) / bx))
      if (i === hover) col.set('#ffffff')
      m.setColorAt(i, col)
    }
    m.instanceMatrix.needsUpdate = true
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  }, [bx, by, hover, mode, n, sp])
  return (
    <instancedMesh ref={ref} args={[null, null, 1024]}
      onPointerMove={e => { e.stopPropagation(); setHover(e.instanceId) }} onPointerOut={() => setHover(null)}>
      <boxGeometry args={[c, c, c]} />
      <meshStandardMaterial roughness={0.45} metalness={0.1} />
    </instancedMesh>
  )
}

function Blocks({ gx, gy, sel, setSel }) {
  const tile = Math.min(1.25, 6.4 / gx, 4.6 / gy), sp = tile * 1.14
  return (
    <group position={[-6.6, -0.6, 0]} rotation={[-0.35, 0.28, 0]}>
      {Array.from({ length: gx * gy }, (_, i) => {
        const x = i % gx, y = Math.floor(i / gx)
        const on = sel[0] === x && sel[1] === y
        return (
          <mesh key={i} position={[(x - (gx - 1) / 2) * sp * 1.0 + 0.5, -(y - (gy - 1) / 2) * sp * 0.9, 0]} onClick={e => { e.stopPropagation(); setSel([x, y]) }}
            onPointerOver={() => (document.body.style.cursor = 'pointer')} onPointerOut={() => (document.body.style.cursor = '')}>
            <boxGeometry args={[tile, tile * 0.82, on ? 0.5 : 0.25]} />
            <meshStandardMaterial color={on ? '#76d12a' : '#2d3560'} emissive={on ? '#3d6b12' : '#10142a'} roughness={0.5} />
          </mesh>
        )
      })}
    </group>
  )
}

function HierarchyLab() {
  const [gx, setGx] = useState(4)
  const [gy, setGy] = useState(3)
  const [bx, setBx] = useState(16)
  const [by, setBy] = useState(4)
  const [sel, setSel] = useState([1, 1])
  const [hover, setHover] = useState(null)
  const [mode, setMode] = useState('warp')
  const bxx = Math.min(bx, Math.floor(1024 / by)), threads = bxx * by
  const warps = Math.ceil(threads / 32), wasted = warps * 32 - threads
  const h = hover ?? 0
  const tx = h % bxx, ty = Math.floor(h / bxx)
  const gX = sel[0] * bxx + tx, gY = sel[1] * by + ty
  const tid = ty * bxx + tx
  const overlay = (
    <>
      <b>Block</b> ({sel[0]},{sel[1]}) of grid {gx}×{gy} · <b>{threads}</b> threads = <b>{warps}</b> warps<br />
      {hover != null ? <>
        <b>thread</b> ({tx},{ty}) → linear tid <b>{tid}</b> · <span style={{ color: '#fbbf24' }}>warp {Math.floor(tid / 32)}, lane {tid % 32}</span><br />
        global x = {sel[0]}·{bxx}+{tx} = <b>{gX}</b> · global y = {sel[1]}·{by}+{ty} = <b>{gY}</b>
      </> : <span style={{ opacity: 0.7 }}>hover a thread cube · click a block tile on the left</span>}
    </>
  )
  return (
    <>
      <Stage3D height={440} camera={[1.6, 2.2, 11.5]} target={[-0.5, -0.2, 0]} overlay={overlay} hint="drag to orbit · hover threads · click blocks">
        <Blocks gx={gx} gy={gy} sel={sel} setSel={setSel} />
        <group position={[2.2, 0, 0]} rotation={[0, -0.18, 0]}>
          <Threads bx={bxx} by={by} hover={hover} setHover={setHover} mode={mode} />
        </group>
      </Stage3D>
      <Controls>
        <Slider label="gridDim.x" min={1} max={8} value={gx} onChange={v => { setGx(v); setSel(s => [Math.min(s[0], v - 1), s[1]]) }} />
        <Slider label="gridDim.y" min={1} max={6} value={gy} onChange={v => { setGy(v); setSel(s => [s[0], Math.min(s[1], v - 1)]) }} />
        <Slider label="blockDim.x" min={1} max={64} value={bx} onChange={setBx} />
        <Slider label="blockDim.y" min={1} max={16} value={by} onChange={setBy} />
        <Toggle label="Colour by warp" value={mode === 'warp'} onChange={v => setMode(v ? 'warp' : 'x')} />
      </Controls>
      <Readout>
        &lt;&lt;&lt;dim3({gx},{gy}), dim3({bxx},{by})&gt;&gt;&gt; launches <b>{fmtInt(gx * gy)}</b> blocks × <b>{threads}</b> threads = <b>{fmtInt(gx * gy * threads)}</b> threads
        {bx !== bxx && <span className="w"> — blockDim.x clamped to {bxx}: a block holds at most 1024 threads</span>}.
        {wasted > 0 ? <> The last warp has <b className="r">{wasted}</b> idle lanes ({(100 * wasted / (warps * 32)).toFixed(1)}% of issue slots wasted) — pick a multiple of 32.</> : <> Block size is a multiple of 32 → <b className="g">no partially-filled warps</b>.</>}
      </Readout>
    </>
  )
}

/* ---------- block scheduler ---------- */
function SchedulerLab() {
  const [nb, setNb] = useState(40)
  const [sms, setSms] = useState(6)
  const [cap, setCap] = useState(2)
  const [vary, setVary] = useState(0.4)
  const [seed, setSeed] = useState(1)
  const clock = useRef(0)
  const sched = useMemo(() => {
    const r = rng(seed * 17)
    const dur = Array.from({ length: nb }, () => 1 + (r() * 2 - 1) * vary)
    const slots = Array.from({ length: sms * cap }, () => 0)
    const items = dur.map((d, b) => {
      let s = 0; for (let i = 1; i < slots.length; i++) if (slots[i] < slots[s]) s = i
      const start = slots[s]; slots[s] += d
      return { b, sm: Math.floor(s / cap), slot: s % cap, start, end: slots[s], d }
    })
    const total = Math.max(...slots)
    const busy = dur.reduce((a, b) => a + b, 0)
    return { items, total, busy, ideal: busy / (sms * cap) }
  }, [nb, sms, cap, vary, seed])

  const [cp] = useCanvas(340, (ctx, W, H, t, dt) => {
    clock.current = (clock.current + dt * sched.total * 0.18) % (sched.total * 1.25)
    const x0 = 64, y0 = 22, w = W - x0 - 20, laneH = Math.min(46, (H - 70) / sms), rowH = (laneH - 6) / cap
    const X = v => x0 + (v / (sched.total * 1.02)) * w
    for (let s = 0; s < sms; s++) {
      const y = y0 + s * laneH
      ctx.fillStyle = '#0f1220'; rr(ctx, x0, y, w, laneH - 4, 6); ctx.fill()
      text(ctx, `SM ${s}`, x0 - 8, y + laneH / 2 - 2, { size: 11, align: 'right', color: C.mute, mono: true })
    }
    const waveSize = sms * cap
    sched.items.forEach(it => {
      const y = y0 + it.sm * laneH + 1 + it.slot * rowH
      const wave = Math.floor(it.b / waveSize)
      const done = clock.current >= it.end, run = clock.current >= it.start && !done
      ctx.fillStyle = run ? C.g : done ? alpha(['#8b7bff', '#22d3ee', '#f472b6', '#fbbf24', '#4ade80'][wave % 5], 0.55) : alpha('#8b7bff', 0.12)
      ctx.fillRect(X(it.start) + 1, y + 1, Math.max(1, X(it.end) - X(it.start) - 2), rowH - 3)
      if (X(it.end) - X(it.start) > 22) text(ctx, it.b, (X(it.start) + X(it.end)) / 2, y + rowH / 2, { size: 9.5, align: 'center', color: '#fff', mono: true })
    })
    const cx = X(Math.min(clock.current, sched.total))
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(cx, y0 - 6); ctx.lineTo(cx, y0 + sms * laneH); ctx.stroke()
    // ideal finish
    ctx.strokeStyle = C.d; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(X(sched.ideal), y0 - 6); ctx.lineTo(X(sched.ideal), y0 + sms * laneH); ctx.stroke(); ctx.setLineDash([])
    text(ctx, 'ideal', X(sched.ideal) + 4, H - 28, { size: 10.5, color: C.d, mono: true })
    text(ctx, 'actual', X(sched.total) + 4, H - 12, { size: 10.5, color: '#fff', mono: true })
    text(ctx, 'time →', x0 + w, y0 + sms * laneH + 14, { size: 10.5, align: 'right', color: C.dim })
  }, { animate: true })
  const eff = sched.busy / (sched.total * sms * cap)
  const waves = Math.ceil(nb / (sms * cap))
  return (
    <>
      <canvas {...cp} />
      <Controls>
        <Slider label="Thread blocks" min={4} max={160} value={nb} onChange={setNb} />
        <Slider label="SMs" min={2} max={16} value={sms} onChange={setSms} />
        <Slider label="Resident blocks / SM" min={1} max={4} value={cap} onChange={setCap} />
        <Slider label="Duration variance" min={0} max={0.9} step={0.05} value={vary} onChange={setVary} fmt={v => v.toFixed(2)} />
        <Btn onClick={() => setSeed(s => s + 1)}>Re-roll durations</Btn>
      </Controls>
      <Readout>{nb} blocks on {sms} SMs × {cap} resident = {sms * cap} slots → <b>{waves}</b> wave(s) · makespan <b>{sched.total.toFixed(2)}</b> vs ideal {sched.ideal.toFixed(2)} · SM utilisation <b className={eff > 0.9 ? 'g' : 'w'}>{(100 * eff).toFixed(0)}%</b>. When {nb} is not a multiple of {sms * cap}, the <b>last wave leaves SMs idle (the “tail effect”)</b>; blocks run in no guaranteed order and never wait for each other.</Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'h', label: 'Grid → block → warp → thread (3D)', render: () => <HierarchyLab /> },
    { id: 's', label: 'Blocks scheduled onto SMs', render: () => <SchedulerLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Fly through a CUDA launch: grid, blocks, warps, threads — and the SMs that run them',
  tryIt: [
    'Hover any cube: read its **threadIdx**, **global index** and which **warp / lane** it belongs to (colour = warp).',
    'Set **blockDim = (20, 1)**: the single warp has 12 idle lanes. Set (32,1) or (16,2): none.',
    'Click different **block tiles** on the left; the global index formula changes with `blockIdx`.',
    'In the scheduler tab choose 41 blocks on 4 SMs×2: see the tail wave leave most SMs idle.'
  ],
  theory, math, practice,
  code: [
    { title: 'Vector add: 1-D indexing, bounds check, grid-stride loop, event timing', lang: 'cuda', src: vecadd },
    { title: '2-D launch: RGB → grayscale on an image', lang: 'cuda', src: gray },
    { title: 'Query your GPU: SMs, warp size, limits', lang: 'cuda', src: query }
  ],
  quiz: [
    { q: 'What is the maximum number of threads in one thread block on current NVIDIA GPUs?', options: ['256', '512', '1024', '2048'], answer: 2, why: 'The hardware limit is 1024 threads per block (e.g. 32×32 or 1024×1).' },
    { q: 'Threads of a warp execute:', options: ['Independently on different SMs', 'In lockstep (SIMT), 32 at a time', 'Sequentially', 'In random order of blocks'], answer: 1, why: 'A warp is 32 consecutive threads issued together; divergence is handled by masking.' },
    { q: 'Which statement about thread blocks is true?', options: ['Blocks of a grid can synchronise with `__syncthreads()`', 'Blocks execute in a guaranteed order', 'Blocks are independent and may run in any order on any SM', 'A block spans several SMs'], answer: 2, why: '`__syncthreads()` only syncs within a block; independence is what lets the same code scale across GPUs of any size.' },
    { q: 'For $N=1{,}000{,}000$ elements and 256 threads/block, the grid size should be:', options: ['3906', '3907', '1000000', '256'], answer: 1, why: '$\\lceil10^6/256\\rceil=3907$; the last block needs a bounds check `if (i < N)`.' },
    { q: 'Why choose block sizes that are multiples of 32?', options: ['The compiler requires it', 'To avoid partially filled warps that waste issue slots', 'To use more shared memory', 'To enable Tensor Cores'], answer: 1, why: 'A block of 100 threads still occupies 4 warps (128 lanes); 28 lanes do nothing.' }
  ]
}
