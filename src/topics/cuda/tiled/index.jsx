import React, { useLayoutEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { Edges } from '@react-three/drei'
import Stage3D from '../../../components/Stage3D.jsx'
import { Controls, Slider, Select, Toggle, Readout, SubViews, Btn, Legend } from '../../../components/ui.jsx'
import { useCanvas, useTicker } from '../../../lib/hooks.js'
import { C, text, rr, alpha, clamp, fmtInt, fmtBytes, polyline } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import naive from './code/naive_gemm.cu?raw'
import tiled from './code/tiled_gemm.cu?raw'
import stencil from './code/stencil1d.cu?raw'
import raceSrc from './code/syncthreads_race.cu?raw'

const tmp = new THREE.Object3D()
const PEAK = {
  a100: { name: 'A100', pi: 19.5, bw: 2.039 },
  h100: { name: 'H100', pi: 67, bw: 3.35 },
  rtx4090: { name: 'RTX 4090', pi: 82.6, bw: 1.008 }
}

/* ---------- decode a linear step into (block, k-tile, phase) ---------- */
function decode(step, N, T) {
  const nb = N / T, nt = nb
  const perTile = 1 + T, perBlock = nt * perTile + 1
  const max = nb * nb * perBlock
  const s = ((step % max) + max) % max
  const b = Math.floor(s / perBlock), rem = s % perBlock
  const bx = b % nb, by = Math.floor(b / nb)
  if (rem === perBlock - 1) return { bx, by, t: nt - 1, phase: 'write', k: T - 1, max, s }
  const t = Math.floor(rem / perTile), r = rem % perTile
  if (r === 0) return { bx, by, t, phase: 'load', k: -1, max, s }
  return { bx, by, t, phase: 'mac', k: r - 1, max, s }
}

/* =====================================================================
   TAB A — 3D tiled GEMM
   ===================================================================== */
function MatGrid({ N, s, origin, colorFor, hover, onHover, z = 0 }) {
  const ref = useRef()
  useLayoutEffect(() => {
    const m = ref.current
    if (!m) return
    m.count = N * N
    for (let i = 0; i < N * N; i++) {
      const c = i % N, r = Math.floor(i / N)
      tmp.position.set(origin[0] + (c - (N - 1) / 2) * s, origin[1] - (r - (N - 1) / 2) * s, origin[2] + z)
      const hov = hover && hover[0] === r && hover[1] === c
      tmp.scale.setScalar(hov ? 1.22 : 1)
      tmp.updateMatrix()
      m.setMatrixAt(i, tmp.matrix)
      m.setColorAt(i, colorFor(r, c, hov))
    }
    m.instanceMatrix.needsUpdate = true
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  })
  return (
    <instancedMesh ref={ref} args={[null, null, N * N]}
      onPointerMove={e => { e.stopPropagation(); const id = e.instanceId; onHover && onHover([Math.floor(id / N), id % N]) }}
      onPointerOut={() => onHover && onHover(null)}>
      <boxGeometry args={[s * 0.86, s * 0.86, s * 0.34]} />
      <meshStandardMaterial roughness={0.45} metalness={0.08} />
    </instancedMesh>
  )
}

function SmemGrid({ T, s, origin, fill, glow }) {
  const ref = useRef()
  useLayoutEffect(() => {
    const m = ref.current
    if (!m) return
    m.count = T * T
    const col = new THREE.Color(glow ? '#fbbf24' : '#3d4568')
    for (let i = 0; i < T * T; i++) {
      const c = i % T, r = Math.floor(i / T)
      tmp.position.set(origin[0] + (c - (T - 1) / 2) * s, origin[1] - (r - (T - 1) / 2) * s, origin[2])
      tmp.scale.setScalar(fill[r][c] ? 1 : 0.55)
      tmp.updateMatrix()
      m.setMatrixAt(i, tmp.matrix)
      const cc = fill[r][c] ? (glow ? col : new THREE.Color('#fbbf24')) : new THREE.Color('#1a1e32')
      m.setColorAt(i, cc)
    }
    m.instanceMatrix.needsUpdate = true
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  })
  return (
    <instancedMesh ref={ref} args={[null, null, 64]}>
      <boxGeometry args={[s * 0.86, s * 0.86, s * 0.4]} />
      <meshStandardMaterial roughness={0.4} emissive={glow ? '#fbbf24' : '#000'} emissiveIntensity={glow ? 0.35 : 0} />
    </instancedMesh>
  )
}

function Frame({ w, h, d, color }) {
  return (
    <mesh>
      <boxGeometry args={[w, h, d]} />
      <meshStandardMaterial color={color} transparent opacity={0.07} depthWrite={false} />
      <Edges color={color} />
    </mesh>
  )
}

function GemmLab() {
  const N = 8
  const [T, setT] = useState(4)
  const [step, setStep] = useState(0)
  const [play, setPlay] = useState(true)
  const [hover, setHover] = useState(null)          // [r,c] in C
  const st = decode(step, N, T)
  const { bx, by, t, phase, k } = st
  const r0 = by * T, c0 = bx * T, k0 = t * T
  useTicker(play, 420, () => setStep(s => s + 1))

  const accOf = (r, c) => {
    const br = Math.floor(r / T), bc = Math.floor(c / T)
    if (br < by || (br === by && bc < bx)) return N   // earlier blocks (row-major in blockIdx)
    if (br !== by || bc !== bx) return 0
    if (phase === 'write') return N
    return t * T + (phase === 'mac' ? k + 1 : 0)     // all-ones A,B → each MAC adds 1
  }

  const inTile = (r, c, r0_, c0_) => r >= r0_ && r < r0_ + T && c >= c0_ && c < c0_ + T
  const colA = (r, c) => {
    const on = inTile(r, c, r0, k0)
    const mac = phase === 'mac' && hover && r === hover[0] && c === k0 + k
    return new THREE.Color(mac ? '#ffffff' : on ? (phase === 'load' ? '#c4b5fd' : '#8b7bff') : '#2a3152')
  }
  const colB = (r, c) => {
    const on = inTile(r, c, k0, c0)
    const mac = phase === 'mac' && hover && c === hover[1] && r === k0 + k
    return new THREE.Color(mac ? '#ffffff' : on ? (phase === 'load' ? '#67e8f9' : '#22d3ee') : '#2a3152')
  }
  const colC = (r, c, hov) => {
    const acc = accOf(r, c)
    if (hov) return new THREE.Color('#ffffff')
    if (inTile(r, c, r0, c0) && phase !== 'write') return new THREE.Color().setHSL(0.33, 0.7, 0.25 + 0.4 * acc / N)
    if (acc >= N) return new THREE.Color('#3f6b28')
    return new THREE.Color('#1c2238')
  }

  const smemFill = Array.from({ length: T }, () => Array.from({ length: T }, () => true))
  const smemOn = phase === 'load' || phase === 'mac'

  const s = 0.32, sm = 0.3
  const hovered = hover && inTile(hover[0], hover[1], r0, c0) ? hover : [r0, c0]
  const [hr, hc] = hovered
  const tx = hc - c0, ty = hr - r0
  const overlay = (
    <>
      Block <b>({bx},{by})</b> owns C[{r0}:{r0 + T}, {c0}:{c0 + T}] · k-tile <b>{t}</b>/{N / T}<br />
      {phase === 'load' && <>Cooperative load → <span style={{ color: '#c4b5fd' }}>As</span> = A[{r0}:{r0 + T}, {k0}:{k0 + T}] · <span style={{ color: '#67e8f9' }}>Bs</span> = B[{k0}:{k0 + T}, {c0}:{c0 + T}]</>}
      {phase === 'mac' && <>MAC k=<b>{k}</b>: C[{hr},{hc}] += As[{ty}][{k}] × Bs[{k}][{tx}]  (acc = <b>{accOf(hr, hc)}</b>/{N})</>}
      {phase === 'write' && <>Write the C-tile (all-ones A,B → each entry is N = <b>{N}</b>)</>}
    </>
  )

  return (
    <>
      <Stage3D height={520} camera={[0, 3.4, 15.5]} target={[0, 0.2, 0]} fov={42} overlay={overlay} hint="drag to orbit · hover a C cell · play / step the k-tiles">
        <group position={[-5.4, 1.15, 0]}>
          <Frame w={N * s + 0.18} h={N * s + 0.18} d={0.16} color="#8b7bff" />
          <MatGrid N={N} s={s} origin={[0, 0, 0]} colorFor={colA} />
        </group>
        <group position={[5.4, 1.15, 0]}>
          <Frame w={N * s + 0.18} h={N * s + 0.18} d={0.16} color="#22d3ee" />
          <MatGrid N={N} s={s} origin={[0, 0, 0]} colorFor={colB} />
        </group>
        <group position={[0, -2.15, 0]}>
          <Frame w={N * s + 0.18} h={N * s + 0.18} d={0.18} color="#76d12a" />
          <MatGrid N={N} s={s} origin={[0, 0, 0]} colorFor={colC} hover={hover} onHover={setHover} z={0.05} />
        </group>
        {smemOn && (
          <>
            <group position={[-1.15, 3.15, 0.35]}>
              <SmemGrid T={T} s={sm} origin={[0, 0, 0]} fill={smemFill} glow={phase === 'load'} />
            </group>
            <group position={[1.15, 3.15, 0.35]}>
              <SmemGrid T={T} s={sm} origin={[0, 0, 0]} fill={smemFill} glow={phase === 'load'} />
            </group>
          </>
        )}
      </Stage3D>
      <Legend items={[['#8b7bff', 'A (left)'], ['#22d3ee', 'B (right)'], ['#76d12a', 'C (centre)'], ['#fbbf24', 'shared As / Bs']]} />
      <Controls>
        <Select label="TILE" value={String(T)} onChange={v => { setT(+v); setStep(0) }} options={[['2', '2 × 2'], ['4', '4 × 4']]} />
        <Toggle label="Auto-step" value={play} onChange={setPlay} />
        <Btn onClick={() => setStep(s => s - 1)}>←</Btn>
        <Btn primary onClick={() => setStep(s => s + 1)}>Step →</Btn>
        <Btn onClick={() => setStep(0)}>Reset</Btn>
      </Controls>
      <Readout>
        {fmtInt(N)}×{fmtInt(N)} all-ones GEMM, TILE=<b>{T}</b> → {N / T}×{N / T} blocks, {N / T} k-tiles each.
        Thread (<b>{tx}</b>,<b>{ty}</b>) writes C[<b>{hr}</b>,<b>{hc}</b>]; over the whole K loop it does <b>{N}</b> FMAs and only <b>{2 * (N / T)}</b> global loads (one A, one B per k-tile).
        Naive would issue <b>{2 * N}</b> global loads for the same output — <b className="g">{T}×</b> more.
        {phase === 'load' && <> Right now every thread of the block writes <b>one</b> element of As and of Bs, then the (unseen) <b>__syncthreads</b>.</>}
        {phase === 'mac' && <> Inner product step {k + 1}/{T} uses only shared memory. After {T} steps, a second barrier, then the next k-tile.</>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB B — traffic / reuse / roofline
   ===================================================================== */
function TrafficLab() {
  const [nE, setNE] = useState(10)             // N = 2^nE
  const [tE, setTE] = useState(4)              // T = 2^tE
  const [s, setS] = useState(4)
  const [gk, setGk] = useState('a100')
  const N = 2 ** nE, T = 2 ** tE
  const g = PEAK[gk]
  const W = 2 * N * N * N
  const naiveQ = (2 * N * N * N + N * N) * s
  const tiledQ = (2 * N * N * N / T + N * N) * s
  const idealQ = 3 * N * N * s
  const I = W / tiledQ, In = W / naiveQ, Ii = W / idealQ
  const ridge = (g.pi * 1e12) / (g.bw * 1e12)
  const tNaive = naiveQ / (g.bw * 1e12)
  const tTiled = Math.max(W / (g.pi * 1e12), tiledQ / (g.bw * 1e12))
  const tIdeal = Math.max(W / (g.pi * 1e12), idealQ / (g.bw * 1e12))
  const bound = (W / (g.pi * 1e12) > tiledQ / (g.bw * 1e12)) ? 'compute' : 'memory'

  const [cv] = useCanvas(280, (ctx, Wd, H) => {
    const ml = 58, mr = 24, mt = 28, mb = 36, pw = Wd - ml - mr, ph = H - mt - mb
    const lo = Math.log10(idealQ * 0.45), hi = Math.log10(naiveQ)
    const X = q => ml + (Math.log10(q) - lo) / (hi - lo) * pw
    const rows = [
      ['naive', naiveQ, C.r],
      [`tiled T=${T}`, tiledQ, C.d],
      ['ideal (A,B once)', idealQ, C.e]
    ]
    text(ctx, `global traffic for N=${fmtInt(N)}  FP${8 * s}`, ml, 12, { size: 12, color: C.mute, weight: 600 })
    rows.forEach(([lab, q, col], i) => {
      const y = mt + 6 + i * (ph / 3)
      const bh = ph / 3 - 16
      const w = Math.max(4, Math.min(pw - 4, X(q) - ml))
      ctx.fillStyle = alpha(col, 0.9); rr(ctx, ml, y, w, bh, 6); ctx.fill()
      text(ctx, lab, ml, y - 7, { size: 11, color: col, weight: 600 })
      const label = fmtBytes(q)
      const inside = w > 90 && ml + w + 8 + label.length * 7.2 > Wd - 12
      text(ctx, label, inside ? ml + w - 10 : ml + w + 8, y + bh / 2, { size: 12, color: inside ? '#0a0c14' : C.ink, mono: true, weight: 700, align: inside ? 'right' : 'left' })
    })
    text(ctx, `log scale · reuse vs naive: ${T}×`, ml, H - 12, { size: 12, color: C.d, mono: true })
  })

  const [cr] = useCanvas(220, (ctx, Wd, H) => {
    const ml = 48, mr = 16, mt = 18, mb = 32, pw = Wd - ml - mr, ph = H - mt - mb
    const lx0 = -3, lx1 = 8, ly0 = -1, ly1 = 2.2
    const X = ai => ml + (Math.log2(ai) - lx0) / (lx1 - lx0) * pw
    const Y = p => mt + ph - (Math.log10(p) - ly0) / (ly1 - ly0) * ph
    const bwT = g.bw, top = Math.max(g.pi, 1)
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1
    ctx.strokeRect(ml, mt, pw, ph)
    ctx.save(); ctx.beginPath(); ctx.rect(ml, mt, pw, ph); ctx.clip()
    polyline(ctx, [[X(2 ** lx0), Y(bwT * 2 ** lx0)], [X(top / bwT), Y(top)]], C.a, 2.4)
    polyline(ctx, [[X(g.pi / bwT), Y(g.pi)], [ml + pw, Y(g.pi)]], C.g, 2.4)
    const dots = [[In, Math.min(g.pi, bwT * In), C.r, 'naive'], [I, Math.min(g.pi, bwT * I), C.d, 'tiled'], [Ii, Math.min(g.pi, bwT * Ii), C.e, 'ideal']]
    dots.forEach(([ai, p, col]) => {
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(clamp(ai, 2 ** lx0, 2 ** lx1)), Y(p), 7, 0, 7); ctx.fill()
    })
    ctx.restore()
    text(ctx, `ridge ${ridge.toFixed(1)} FLOP/B`, X(clamp(ridge, 2 ** lx0, 2 ** lx1)) + 6, mt + ph - 8, { size: 10.5, color: C.g, mono: true })
    text(ctx, 'I (FLOP/B) →', ml + pw / 2, H - 8, { align: 'center', size: 11, color: C.mute })
  })

  return (
    <>
      <canvas {...cv} />
      <canvas {...cr} />
      <Legend items={[[C.r, 'naive'], [C.d, 'tiled'], [C.e, 'ideal']]} />
      <Controls>
        <Select label="GPU" value={gk} onChange={setGk} options={Object.entries(PEAK).map(([k, v]) => [k, v.name])} />
        <Select label="dtype" value={String(s)} onChange={v => setS(+v)} options={[['8', 'FP64'], ['4', 'FP32'], ['2', 'FP16']]} />
        <Slider label="N" min={6} max={14} value={nE} onChange={setNE} fmt={v => fmtInt(2 ** v)} />
        <Slider label="TILE" min={2} max={6} value={tE} onChange={setTE} fmt={v => 2 ** v} />
      </Controls>
      <Readout>
        W = 2N³ = <b>{(W / 1e9).toFixed(2)} GFLOP</b>. Tiled I ≈ T/s = <b>{I.toFixed(2)}</b> FLOP/B
        (naive {In.toFixed(2)}, ideal {Ii.toFixed(1)}) vs {g.name} ridge <b>{ridge.toFixed(1)}</b> →
        tiled is <b className={bound === 'memory' ? 'w' : 'g'}>{bound}-bound</b>.
        Lower bound: naive {(tNaive * 1e3).toFixed(2)} ms · tiled {(tTiled * 1e3).toFixed(2)} ms · ideal {(tIdeal * 1e3).toFixed(2)} ms.
        Shared footprint 2·T²·{s} = <b>{fmtBytes(2 * T * T * s)}</b>.
        {T * T > 1024 && <span className="r"> TILE² &gt; 1024: not a legal block.</span>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB C — barrier race + stencil halo
   ===================================================================== */
function RaceCanvas({ sync, play, nonce }) {
  const clock = useRef(0)
  useLayoutEffect(() => { clock.current = 0 }, [sync, nonce])
  useTicker(play, 40, () => { clock.current += 0.04 })
  const [cr] = useCanvas(300, (ctx, W, H) => {
    const t = clock.current % 6
    const n = 8, warp = 4
    const x0 = 70, y0 = 36, cw = Math.min(52, (W - x0 - 24) / n), ch = 36
    text(ctx, 'shared s[0..7]  —  each thread writes s[i]=i, then reads s[(i+1) mod 8]', x0, 16, { size: 11.5, color: C.mute })
    for (let i = 0; i < n; i++) {
      const x = x0 + i * cw, warp1 = i >= warp
      const written = sync ? t > 1.6 : (warp1 ? t > 2.4 : t > 0.8)
      const reading = t > (sync ? 2.2 : (warp1 ? 2.8 : 1.2)) && t < 4.8
      ctx.fillStyle = written ? (warp1 ? C.c : C.a) : '#1a1e32'
      rr(ctx, x + 4, y0, cw - 8, ch, 6); ctx.fill()
      text(ctx, written ? String(i) : '?', x + cw / 2, y0 + ch / 2, { align: 'center', size: 14, color: written ? '#0a0c14' : C.dim, mono: true, weight: 700 })
      text(ctx, 't' + i, x + cw / 2, y0 + ch + 14, { align: 'center', size: 10.5, color: warp1 ? C.c : C.a, mono: true })
      if (reading) {
        const src = (i + 1) % n
        const srcWritten = sync ? t > 1.6 : (src >= warp ? t > 2.4 : t > 0.8)
        ctx.strokeStyle = srcWritten ? C.e : C.r; ctx.lineWidth = 1.6
        ctx.beginPath(); ctx.moveTo(x0 + src * cw + cw / 2, y0 + ch); ctx.lineTo(x + cw / 2, y0 + ch + 28); ctx.stroke()
      }
    }
    text(ctx, 'warp 0', x0, y0 + ch + 40, { size: 11, color: C.a, weight: 600 })
    text(ctx, 'warp 1  (arrives later)', x0 + warp * cw, y0 + ch + 40, { size: 11, color: C.c, weight: 600 })
    // results
    const yR = y0 + 118
    text(ctx, 'out[i] = s[i+1]', x0 - 8, yR + 16, { size: 11, color: C.mute, align: 'right' })
    for (let i = 0; i < n; i++) {
      const src = (i + 1) % n
      const srcWritten = sync ? true : (src < warp)          // without sync, warp0 reads before warp1 wrote: s[4] may be unset
      const ok = sync || src < warp
      const shown = t > 3.2
      ctx.fillStyle = !shown ? '#1a1e32' : ok ? alpha(C.e, 0.85) : alpha(C.r, 0.85)
      rr(ctx, x0 + i * cw + 4, yR, cw - 8, 32, 6); ctx.fill()
      if (shown) text(ctx, ok ? String(src) : '??', x0 + i * cw + cw / 2, yR + 16, { align: 'center', size: 13, color: '#0a0c14', mono: true, weight: 700 })
    }
    const msg = sync
      ? 'Barrier: every write is visible. out = 1,2,3,4,5,6,7,0.'
      : 'No barrier: warp 0 reads s[4] before warp 1 writes it. out[3] is garbage — a data race, not a “sometimes works”.'
    text(ctx, msg, x0, H - 18, { size: 12, color: sync ? C.e : C.r, weight: 600 })
  }, { animate: true })
  return <canvas {...cr} />
}

function HaloCanvas({ play, nonce }) {
  const clock = useRef(0)
  useLayoutEffect(() => { clock.current = 0 }, [nonce])
  useTicker(play, 40, () => { clock.current += 0.04 })
  const [cs] = useCanvas(300, (ctx, W, H) => {
    const n = 24, B = 8, halo = 1
    const blk = Math.min(2, Math.floor((clock.current / 3) % 3))
    const phase = (clock.current % 3) < 1.35 ? 'load' : 'compute'
    const x0 = 36, y0 = 78, cw = (W - x0 - 20) / n, ch = 34
    text(ctx, `3-point stencil · block ${blk} of ${Math.ceil(n / B)} · ${phase === 'load' ? 'cooperative load + halo' : 'compute from shared memory'}`, x0, 18, { size: 12, color: C.mute, weight: 600 })
    for (let i = 0; i < n; i++) {
      const owner = Math.floor(i / B)
      const isHalo = (i === blk * B - 1 || i === blk * B + B) && i >= 0 && i < n
      const interior = owner === blk
      ctx.fillStyle = interior ? C.d : isHalo && phase === 'load' ? C.c : '#1a1e32'
      rr(ctx, x0 + i * cw + 1, y0, cw - 2, ch, 4); ctx.fill()
      text(ctx, String(i), x0 + i * cw + cw / 2, y0 + ch / 2, { align: 'center', size: 10, color: interior || isHalo ? '#0a0c14' : C.dim, mono: true })
    }
    text(ctx, 'global in[]', x0, y0 - 12, { size: 11, color: C.mute })
    // smem row
    const s0 = x0 + blk * B * cw, cells = B + 2 * halo
    const y1 = y0 + 70
    text(ctx, 'shared s[0..9]  (8 interior + 2 halo)', s0, y1 - 14, { size: 11, color: C.mute })
    for (let i = 0; i < cells; i++) {
      const g = blk * B - 1 + i
      ctx.fillStyle = i === 0 || i === cells - 1 ? C.c : C.d
      rr(ctx, s0 + (i - 0.5) * cw + 1, y1, cw - 2, ch, 4); ctx.fill()
      text(ctx, g < 0 || g >= n ? '0' : String(g), s0 + (i - 0.5) * cw + cw / 2, y1 + ch / 2, { align: 'center', size: 10, color: '#0a0c14', mono: true })
    }
    if (phase === 'compute') {
      const mid = s0 + (1 + 3) * cw
      ctx.strokeStyle = C.e; ctx.lineWidth = 1.5
      ctx.beginPath(); ctx.moveTo(s0 + 3.5 * cw, y1 + ch); ctx.lineTo(s0 + 3.5 * cw, y1 + ch + 36); ctx.stroke()
      text(ctx, 'y[i] = (s[i-1] + 2 s[i] + s[i+1]) / 4', s0, y1 + ch + 48, { size: 12, color: C.e, mono: true })
    } else {
      text(ctx, 'thread 0 also loads in[start-1] · thread 7 loads in[end]  — 2 extra global reads for the whole block', s0, y1 + ch + 22, { size: 11, color: C.c })
    }
    text(ctx, `traffic: naive 3 loads/point · tiled ${(B + 2) / B} loads/point  (${((B + 2) / (3 * B) * 100).toFixed(0)}% of naive)`, x0, H - 16, { size: 12, color: C.d, mono: true })
  }, { animate: true })
  return <canvas {...cs} />
}

function BarrierLab() {
  const [mode, setMode] = useState('race')
  const [sync, setSync] = useState(false)
  const [play, setPlay] = useState(true)
  const [nonce, setNonce] = useState(0)

  return (
    <>
      {mode === 'race' ? <RaceCanvas sync={sync} play={play} nonce={nonce} /> : <HaloCanvas play={play} nonce={nonce} />}
      <Controls>
        <Select label="Demo" value={mode} onChange={setMode} options={[['race', '__syncthreads race'], ['halo', 'Stencil halo']]} />
        {mode === 'race' && <Toggle label="Insert __syncthreads()" value={sync} onChange={setSync} />}
        <Toggle label="Animate" value={play} onChange={setPlay} />
        <Btn onClick={() => setNonce(n => n + 1)}>Replay</Btn>
      </Controls>
      <Readout>
        {mode === 'race'
          ? <>Two warps share <b>s[8]</b>. Warp 1 is scheduled later. Without a barrier, warp 0's read of <b>s[4]</b> is a data race — it may be empty, stale, or (on a lucky run) already written.
            A single-warp test hides this; <b>compute-sanitizer --tool racecheck</b> does not. The same bug in tiled GEMM is dropping the <b>second</b> barrier in the k-loop.</>
          : <>A radius-1 stencil needs neighbours. The block loads its <b>{8}</b> interior points plus a 1-cell <b>halo</b> (pink) = 10 shared slots, then every output is 3 shared reads.
            Halo overhead is 2/{8} = 25% here (tiny blocks for the picture); at 256 threads it is 0.8%. Same pattern as the GEMM tile, just 1-D.</>}
      </Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'g', label: '3D: tiled GEMM, step by step', render: () => <GemmLab /> },
    { id: 't', label: 'Traffic, reuse & the roofline', render: () => <TrafficLab /> },
    { id: 'b', label: '__syncthreads & stencil halo', render: () => <BarrierLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Stage a tile in shared memory, reuse it, slide along K — watch global traffic fall by T',
  tryIt: [
    'In the 3D tab **hover a C cell** in the active tile during a MAC: the matching A-element and B-element light up white. That pair lives in **As / Bs**, not HBM.',
    'Set **TILE = 2** and step: four k-tiles instead of two, twice the global loads. TILE is the reuse factor.',
    'Let it run through a **write** phase: the C-tile goes solid green (value N) and the next block starts.',
    'In Traffic, set **N = 1024, TILE = 16, FP32, A100**: tiled sits at 4 FLOP/B, still memory-bound but 16× less traffic than naive.',
    'Raise TILE to 32, then switch the GPU to **H100 FP16** in your head (ridge ~300): you now understand why FlashAttention tiles.',
    'In the race demo, leave **__syncthreads off** and watch `out[3]` go red; turn it on and the vector is a clean rotation.'
  ],
  theory, math, practice,
  code: [
    { title: 'Naive GEMM: one output per thread, 2N global loads each', lang: 'cuda', note: 'B\'s column is uncoalesced. This is the kernel the roofline in the previous chapter plotted as "naive GEMM".', src: naive },
    { title: 'Tiled GEMM: cooperative load, two __syncthreads, T× less traffic', lang: 'cuda', note: 'Build with -DTILE=32 to try a bigger tile. Spot-checks a few C[i,j] against a CPU inner product.', src: tiled },
    { title: '1-D stencil with a halo in dynamic shared memory', lang: 'cuda', src: stencil },
    { title: 'The missing-barrier data race (and why 32 threads hide it)', lang: 'cuda', src: raceSrc }
  ],
  quiz: [
    { q: 'Tiling a GEMM with TILE $=T$ changes global-memory traffic by about:', options: ['×T (more traffic)', '÷T', '÷T²', 'no change; only latency changes'], answer: 1, why: 'Each loaded element is reused $T$ times inside the block, so $Q\\approx 2N^3 s/T$.' },
    { q: 'Why does the k-loop of tiled GEMM contain **two** `__syncthreads()`?', options: ['The compiler requires pairs of barriers', 'One after the load so the tile is visible; one after the MAC so the next load cannot overwrite it', 'One for A and one for B', 'To synchronise with other blocks'], answer: 1, why: 'The second barrier is a write-after-read fence on `As`/`Bs`. Dropping it is a shared-memory race, often silent at small TILE.' },
    { q: 'Which of these is legal?', options: ['`if (row < N) { load; __syncthreads(); }`', '`As[ty][tx] = row<N && col<N ? A[...] : 0; __syncthreads();`', 'Skipping the barrier on the last k-tile', 'Synchronising two blocks through shared memory'], answer: 1, why: 'Every thread of the block must reach the same barrier. Mask the **memory op**, never the barrier. Shared memory does not exist across blocks.' },
    { q: 'Thread `(tx,ty)` in block `(bx,by)` with TILE $T$ owns which output?', options: ['$C[tx,ty]$', '$C[by\\cdot T+ty,\\ bx\\cdot T+tx]$', '$C[bx\\cdot T+tx,\\ by\\cdot T+ty]$', '$C[ty, tx]$'], answer: 1, why: '`row = blockIdx.y * T + threadIdx.y`, `col = blockIdx.x * T + threadIdx.x`. $x$ is the contiguous dimension so the B-load is coalesced.' },
    { q: 'A 3-point stencil on 256-thread blocks with a 1-cell halo loads how many global elements per output, versus 3 naive?', options: ['3', '≈1.008', '256', '2'], answer: 1, why: '$(256+2)/256=1.008$ loads/output. The halo is 2 extra loads for the whole block.' },
    { q: 'At TILE $=16$, FP32, the tiled-GEMM arithmetic intensity is about:', options: ['0.25 FLOP/B', '4 FLOP/B', '16 FLOP/B', '256 FLOP/B'], answer: 1, why: '$I\\approx T/s=16/4=4$ FLOP/B. Still below the A100 FP32 ridge (~9.6), which is why bigger tiles or register blocking still pay off.' },
    { q: 'A missing `__syncthreads` after a cooperative load is often invisible when tested with 32 threads because:', options: ['Barriers are optional inside a block', 'A warp already executes in lockstep, so the 32 writes happen together; a second warp exposes the race', 'Shared memory is coherent globally', 'The compiler inserts a barrier'], answer: 1, why: 'Correctness requires a block-wide fence. `racecheck` reports the hazard even at 32 threads.' }
  ]
}
