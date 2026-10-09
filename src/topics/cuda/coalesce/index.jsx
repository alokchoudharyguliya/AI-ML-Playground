import React, { useMemo, useState } from 'react'
import Stage3D from '../../../components/Stage3D.jsx'
import { Controls, Slider, Select, Readout, SubViews, Btn, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, text, rr, alpha, fmtBytes } from '../../../lib/viz.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import strideSrc from './code/stride_copy.cu?raw'
import banksSrc from './code/smem_banks.cu?raw'
import transposeSrc from './code/transpose.cu?raw'

const LANES = 32
const SECTOR = 32
const pct = eta => {
  const p = Math.round(1000 * eta) / 10
  return p >= 99.95 ? '100' : String(p)
}

/** One warp of loads. stride and offset are in elements; word is the byte width of each load. */
function warpSectors(stride, offset, word) {
  const hits = []
  const bySector = new Map()
  for (let lane = 0; lane < LANES; lane++) {
    const start = (offset + lane * stride) * word
    const secs = []
    for (let b = start; b < start + word; b++) {
      const s = Math.floor(b / SECTOR)
      if (secs[secs.length - 1] !== s) secs.push(s)
    }
    hits.push({ lane, start, secs })
    secs.forEach(s => {
      if (!bySector.has(s)) bySector.set(s, [])
      bySector.get(s).push(lane)
    })
  }
  const sectors = [...bySector.keys()].sort((a, b) => a - b)
  const useful = LANES * word
  const moved = sectors.length * SECTOR
  return { hits, sectors, bySector, useful, moved, eta: useful / moved }
}

function bankMap(addrs) {
  const byBank = Array.from({ length: 32 }, () => new Set())
  addrs.forEach(a => byBank[((a % 32) + 32) % 32].add(a))
  const degree = Math.max(...byBank.map(s => s.size))
  const broadcast = addrs.every(a => a === addrs[0])
  return { byBank, degree, broadcast }
}

/* =====================================================================
   TAB A — sector counting
   ===================================================================== */
const PRESETS = [
  ['s1', 'stride 1', 1, 0, 4],
  ['mis', 'offset +1', 1, 1, 4],
  ['s2', 'stride 2', 2, 0, 4],
  ['aos', 'AoS .x (stride 4)', 4, 0, 4],
  ['f4', 'float4', 1, 0, 16],
  ['col', 'column, stride 32', 32, 0, 4]
]

function CoalesceLab() {
  const [stride, setStride] = useState(1)
  const [offset, setOffset] = useState(0)
  const [word, setWord] = useState(4)
  const w = warpSectors(stride, offset, word)

  const [cv] = useCanvas(400, (ctx, W, H) => {
    const ml = 16, mr = 16
    text(ctx, 'one warp · 32 lanes', ml, 16, { size: 12, color: C.mute, weight: 600 })
    text(ctx, `${pct(w.eta)}% of bytes useful`, W - mr, 16, { size: 13, color: w.eta > 0.95 ? C.e : w.eta > 0.45 ? C.d : C.r, align: 'right', weight: 700 })

    const barY = 32, barH = 16, barW = W - ml - mr
    ctx.fillStyle = '#1a1e32'; rr(ctx, ml, barY, barW, barH, 5); ctx.fill()
    ctx.fillStyle = w.eta > 0.95 ? C.e : w.eta > 0.45 ? C.d : C.r
    rr(ctx, ml, barY, Math.max(4, barW * w.eta), barH, 5); ctx.fill()
    text(ctx, `${fmtBytes(w.useful)} useful  /  ${fmtBytes(w.moved)} moved  ·  ${w.sectors.length} sectors`, ml, barY + barH + 14, { size: 12, color: C.ink, mono: true })

    const laneY = 92, laneH = 28
    const lw = Math.min(18, (barW - 31 * 3) / 32)
    const laneGap = (barW - 32 * lw) / 31
    const laneX = i => ml + i * (lw + laneGap)
    text(ctx, 'lanes', ml, laneY - 12, { size: 11, color: C.mute })
    w.hits.forEach(h => {
      const span = h.secs.length > 1
      ctx.fillStyle = span ? C.c : C.a
      rr(ctx, laneX(h.lane), laneY, lw, laneH, 3); ctx.fill()
    })

    const secs = w.sectors
    const perRow = Math.min(16, Math.max(4, secs.length))
    const rows = Math.ceil(secs.length / perRow)
    const cellW = Math.min(72, (barW - (perRow - 1) * 6) / perRow)
    const cellH = 36
    const gridTop = 168
    text(ctx, '32-byte sectors this warp touches  ·  number = lanes in that sector', ml, gridTop - 12, { size: 11, color: C.mute })
    secs.forEach((s, i) => {
      const c = i % perRow, r = Math.floor(i / perRow)
      const x = ml + c * (cellW + 6), y = gridTop + r * (cellH + 8)
      const n = w.bySector.get(s).length
      const full = n * word >= SECTOR
      ctx.fillStyle = full ? alpha(C.e, 0.9) : n === 1 ? alpha(C.r, 0.85) : alpha(C.d, 0.9)
      rr(ctx, x, y, cellW, cellH, 6); ctx.fill()
      text(ctx, String(n), x + cellW / 2, y + 14, { align: 'center', size: 14, color: '#0a0c14', weight: 800, mono: true })
      text(ctx, 's' + s, x + cellW / 2, y + 28, { align: 'center', size: 9, color: alpha('#0a0c14', 0.7), mono: true })
    })
    const foot = gridTop + rows * (cellH + 8) + 6
    if (foot < H - 4) {
      const split = w.hits.some(h => h.secs.length > 1)
      text(ctx, split ? 'pink lanes: that load straddles two sectors (misaligned or wider than the gap)' : 'every load sits inside one sector', ml, Math.min(H - 12, foot), { size: 11, color: split ? C.c : C.mute })
    }
  })

  const apply = (s, o, wd) => { setStride(s); setOffset(o); setWord(wd) }

  return (
    <>
      <canvas {...cv} />
      <Controls>
        {PRESETS.map(([id, label, s, o, wd]) => (
          <Btn key={id} primary={stride === s && offset === o && word === wd} onClick={() => apply(s, o, wd)}>{label}</Btn>
        ))}
      </Controls>
      <Controls>
        <Slider label="Stride (elements)" min={1} max={32} value={stride} onChange={setStride} />
        <Slider label="Start element" min={0} max={32} value={offset} onChange={setOffset} />
        <Select label="Load width" value={String(word)} onChange={v => setWord(+v)} options={[['4', 'float · 4 B'], ['8', 'float2 · 8 B'], ['16', 'float4 · 16 B']]} />
      </Controls>
      <Legend items={[[C.e, 'sector fully used'], [C.d, 'sector partly used'], [C.r, 'one lane, one sector'], [C.c, 'lane straddles two sectors']]} />
      <Readout>
        {w.sectors.length} sector{w.sectors.length === 1 ? '' : 's'} × 32 B = <b>{fmtBytes(w.moved)}</b> moved for <b>{fmtBytes(w.useful)}</b> wanted
        → efficiency <b className={w.eta > 0.95 ? 'g' : 'w'}>{pct(w.eta)}%</b>.
        {stride === 1 && offset === 0 && word === 4 && <> This is the access a warp gets from <b>threadIdx.x</b> on a contiguous <b>float</b> array: one cache line.</>}
        {stride === 1 && offset === 1 && word === 4 && <> One float of misalignment turns 4 sectors into <b>5</b>. <b>cudaMalloc</b> is aligned; a column that starts at <b>x = 1</b> is not.</>}
        {stride > 1 && word === 4 && offset === 0 && <> Aligned stride {stride}: about <b>{stride <= 8 ? stride : 8}×</b> the traffic of stride 1. Past stride 8 a 4-byte load is already one sector per lane, so it cannot get worse.</>}
        {word === 16 && stride === 1 && <> <b>float4</b> touches more sectors and still scores 100% when 16-byte aligned — every extra byte is useful, and one instruction fills more of the memory pipe.</>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB B — 3D sectors
   ===================================================================== */
function SectorScene({ stride, offset, word }) {
  const w = useMemo(() => warpSectors(stride, offset, word), [stride, offset, word])
  const layout = useMemo(() => {
    let cursor = 0
    const sectors = w.sectors.map(s => {
      const n = w.bySector.get(s).length
      const width = Math.max(0.34, n * 0.2)
      const x = cursor + width / 2
      cursor += width + 0.1
      return { s, n, width, x }
    })
    const shift = cursor / 2
    sectors.forEach(sec => { sec.x -= shift })
    return sectors
  }, [w])
  const at = Object.fromEntries(layout.map(sec => [sec.s, sec]))
  return (
    <group>
      {layout.map(sec => {
        const full = sec.n * word >= SECTOR
        const color = full ? '#4ade80' : sec.n === 1 ? '#fb7185' : '#fbbf24'
        return (
          <mesh key={sec.s} position={[sec.x, 0, 0]}>
            <boxGeometry args={[sec.width * 0.92, 0.4, 0.85]} />
            <meshStandardMaterial color={color} roughness={0.45} emissive={color} emissiveIntensity={0.25} />
          </mesh>
        )
      })}
      {w.hits.map(h => {
        const sec = at[h.secs[0]]
        const group = w.bySector.get(h.secs[0])
        const k = group.indexOf(h.lane)
        const x = sec.x + (k - (group.length - 1) / 2) * (sec.width * 0.86 / group.length)
        return (
          <mesh key={h.lane} position={[x, 1.05, 0]}>
            <sphereGeometry args={[0.075, 12, 12]} />
            <meshStandardMaterial color={h.secs.length > 1 ? '#f472b6' : '#8b7bff'} emissive="#8b7bff" emissiveIntensity={0.4} />
          </mesh>
        )
      })}
    </group>
  )
}

function Coalesce3D() {
  const [stride, setStride] = useState(1)
  const [offset, setOffset] = useState(0)
  const [word, setWord] = useState(4)
  const w = warpSectors(stride, offset, word)
  const overlay = (
    <>
      stride <b>{stride}</b> · start element <b>{offset}</b> · {word} B loads<br />
      <b>{w.sectors.length}</b> sectors · efficiency <b>{pct(w.eta)}%</b><br />
      <span style={{ color: '#4ade80' }}>green</span> full · <span style={{ color: '#fbbf24' }}>amber</span> partial · <span style={{ color: '#fb7185' }}>red</span> one lane
    </>
  )
  return (
    <>
      <Stage3D height={420} camera={[0, 2.8, 9.5]} target={[0, 0.35, 0]} fov={42} overlay={overlay} hint="drag to orbit · spheres are the 32 lanes, boxes are the sectors they hit">
        <SectorScene stride={stride} offset={offset} word={word} />
      </Stage3D>
      <Controls>
        <Slider label="Stride (elements)" min={1} max={32} value={stride} onChange={setStride} />
        <Slider label="Start element" min={0} max={16} value={offset} onChange={setOffset} />
        <Select label="Load width" value={String(word)} onChange={v => setWord(+v)} options={[['4', 'float · 4 B'], ['8', 'float2 · 8 B'], ['16', 'float4 · 16 B']]} />
      </Controls>
      <Readout>
        Each sphere is a lane; each box is a 32-byte sector that warp actually requests.
        Stride 1 packs all 32 lanes into 4 green boxes. Stride 32 leaves a red box per lane — the same 128 useful bytes, eight times the traffic.
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB C — banks
   ===================================================================== */
function BankLab() {
  const [mode, setMode] = useState('col')
  const [width, setWidth] = useState(32)
  const [pad, setPad] = useState(0)
  const [col, setCol] = useState(0)
  const ld = width + pad
  const addrs = useMemo(() => {
    if (mode === 'stride') return Array.from({ length: 32 }, (_, lane) => lane * ld)
    if (mode === 'row') return Array.from({ length: 32 }, (_, lane) => col * ld + lane)
    return Array.from({ length: 32 }, (_, lane) => lane * ld + col)
  }, [mode, width, pad, col, ld])
  const map = useMemo(() => bankMap(addrs), [addrs])

  const [cv] = useCanvas(340, (ctx, W) => {
    const ml = 36, mt = 36, mr = 12
    const colW = (W - ml - mr) / 32
    text(ctx, 'shared-memory banks 0–31', ml, 16, { size: 12, color: C.mute, weight: 600 })
    text(ctx, map.broadcast ? 'broadcast' : map.degree === 1 ? 'no conflict' : `${map.degree}-way conflict`, W - mr, 16, { align: 'right', size: 13, weight: 700, color: map.degree === 1 ? C.e : C.r })
    for (let b = 0; b < 32; b++) {
      const x = ml + b * colW
      const words = [...map.byBank[b]]
      ctx.fillStyle = words.length > 1 ? alpha(C.r, 0.18) : '#14182a'
      ctx.fillRect(x + 1, mt, colW - 2, 250)
      words.forEach((addr, k) => {
        const y = mt + 8 + k * 7
        ctx.fillStyle = words.length > 1 ? C.r : C.e
        ctx.fillRect(x + 2, y, colW - 4, 5)
      })
      if (b % 4 === 0) text(ctx, b, x + colW / 2, mt + 264, { align: 'center', size: 10, color: C.mute, mono: true })
    }
    text(ctx, 'each dash is a distinct address in that bank · a full red column is one serialised pass per dash', ml, 328, { size: 11, color: C.mute })
  })

  return (
    <>
      <canvas {...cv} />
      <Controls>
        <Select label="Access" value={mode} onChange={setMode} options={[['row', 'row  tile[r][lane]'], ['col', 'column  tile[lane][c]'], ['stride', 'stride  s[lane * L]']]} />
        <Select label={mode === 'stride' ? 'Stride' : 'Width L'} value={String(width)} onChange={v => setWidth(+v)} options={[['8', '8'], ['16', '16'], ['32', '32'], ['33', '33']]} />
        <Select label="Padding" value={String(pad)} onChange={v => setPad(+v)} options={[['0', '+0'], ['1', '+1 (odd)'], ['2', '+2 (even)']]} />
        {mode !== 'stride' && <Slider label={mode === 'row' ? 'Row' : 'Column'} min={0} max={31} value={col} onChange={setCol} />}
      </Controls>
      <Legend items={[[C.e, 'one address in this bank'], [C.r, 'several addresses → serialised']]} />
      <Readout>
        Leading dimension <b>{ld}</b>. Conflict degree <b className={map.degree === 1 ? 'g' : 'r'}>{map.broadcast ? '1 (broadcast)' : map.degree}</b>
        {map.degree > 1 && <> — this instruction takes <b>{map.degree}×</b> as long as a conflict-free one.</>}
        {mode === 'col' && width === 32 && pad === 0 && <> Every lane hits <b>bank {col % 32}</b> at a different row. Add padding <b>+1</b>.</>}
        {mode === 'col' && pad === 1 && <> <b>L = {ld}</b> is odd, so a column is a permutation of the 32 banks. Degree 1.</>}
        {mode === 'col' && pad === 2 && width % 2 === 0 && <> <b>+2</b> keeps L even. Even padding does not fix a power-of-two width — it leaves a {map.degree}-way conflict.</>}
        {mode === 'row' && <> A row is 32 consecutive words. Padding does not change it, and the column index is only a shift of which bank lane 0 starts on.</>}
        {mode === 'stride' && <> Lane ℓ reads word <b>ℓ·{ld}</b>. {map.degree === 1 ? 'Those land on distinct banks.' : 'Several lanes share a bank at different addresses.'}</>}
      </Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'c', label: 'Count the sectors', render: () => <CoalesceLab /> },
    { id: 'd', label: '3D: a warp over memory', render: () => <Coalesce3D /> },
    { id: 'b', label: 'Shared-memory banks', render: () => <BankLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Count the sectors a warp really moves, then the banks a column access serialises',
  tryIt: [
    'Leave **stride 1**. Four green sectors, 100%. Click **offset +1**: a fifth sector appears and efficiency falls to 80%.',
    'Click **column, stride 32**. Thirty-two red sectors — 128 useful bytes, 1 KiB moved.',
    'Click **float4**. More sectors than the float case, efficiency still 100%. Width is not waste when every byte is used.',
    'Open **Banks**, access = column, width 32, padding +0. One bank stacks 32 deep. Set padding to **+1** and it flattens.',
    'Set padding to **+2** instead. The conflict drops from 32-way to 2-way and then stops. Even padding is not the fix.',
    'Switch access to **row**. Degree stays 1 at every width — rows were never the problem.'
  ],
  theory, math, practice,
  code: [
    { title: 'Stride copy: same useful bytes, s× the traffic', lang: 'cuda', note: 'Prints useful GB/s for power-of-two strides. Stride 32 should move about 8× more bytes than stride 1.', src: strideSrc },
    { title: 'Row vs conflicted column vs padded column', lang: 'cuda', note: 'The three kernels do the same number of shared-memory reads. Only the bank mapping changes.', src: banksSrc },
    { title: 'Transpose: naive, tiled, and tiled with tile[32][33]', lang: 'cuda', note: 'Global bytes are identical for all three. The gap is coalescing on the store, then bank conflicts on the way out of shared memory.', src: transposeSrc }
  ],
  quiz: [
    { q: 'A coalesced warp load of 32 aligned floats moves:', options: ['32 separate 4-byte transactions', '4 sectors (128 bytes), all of them useful', 'One 32-byte sector', '1024 bytes'], answer: 1, why: '32 × 4 B = 128 B = four 32-byte sectors, which is one cache line.' },
    { q: 'Stride-2 aligned float loads have an efficiency of about:', options: ['100%', '50%', '12.5%', '2%'], answer: 1, why: 'The warp spans 256 B and uses 128 B, so $\\eta = 1/s = 1/2$. Sectors go from 4 to 8.' },
    { q: 'Why does `struct { float x, y, z, w; } p[n]` loaded as `p[i].x` coalesce badly?', options: ['Structs cannot live in global memory', 'Consecutive lanes are 16 bytes apart: stride 4', 'The compiler refuses struct loads', 'x is a register'], answer: 1, why: 'Each lane skips y, z and w. SoA (four separate arrays) makes `.x` stride 1.' },
    { q: 'A shared-memory broadcast is:', options: ['A 32-way bank conflict', 'Many lanes reading one address, served in a single access', 'A global-memory sector', 'Illegal'], answer: 1, why: 'Conflict requires *different* addresses in the same bank. The same address is a broadcast, degree 1.' },
    { q: '`float s[32][32];` and a warp reads `s[threadIdx.x][0]`. The conflict degree is:', options: ['1', '2', '16', '32'], answer: 3, why: 'Word index $= \\ell \\cdot 32$, so every lane maps to bank 0 at a different row.' },
    { q: 'Which padding removes that conflict?', options: ['`s[32][34]`', '`s[32][33]`', '`s[32][32]` with `__syncthreads`', '`s[31][32]`'], answer: 1, why: 'The leading dimension must be odd. 33 is coprime with 32; 34 still collides on even banks (degree 2).' },
    { q: 'The unpadded tiled GEMM reads `As[ty][k]` from a warp with one `ty`. That shared load is:', options: ['A 32-way conflict', 'A broadcast', 'Uncoalesced global traffic', 'A stride-32 sector walk'], answer: 1, why: 'Every lane reads the same address. The conflict shows up when the read is a *column*, as in the transpose.' }
  ]
}
