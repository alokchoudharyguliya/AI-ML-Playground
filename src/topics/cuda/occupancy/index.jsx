import React, { useMemo, useState } from 'react'
import Stage3D from '../../../components/Stage3D.jsx'
import { Controls, Slider, Select, Readout, SubViews, Btn, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, text, rr, alpha, fmtBytes } from '../../../lib/viz.js'
import { GPUS, GPU_OPTIONS } from '../memory/specs.js'
import { SM, occupancy, warpsToHide, bytesInFlight } from './model.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import querySrc from './code/occupancy_query.cu?raw'
import boundsSrc from './code/launch_bounds.cu?raw'
import hideSrc from './code/latency_hide.cu?raw'

const LIM_COLOR = { regs: C.c, smem: C.d, warps: C.b, blocks: C.a }
const BLOCK_COLORS = [C.e, C.b, C.d, C.c, C.a, '#86efac', '#67e8f9', '#f9a8d4']
const pct = x => {
  const p = Math.round(1000 * x) / 10
  return p >= 99.95 ? '100' : String(p)
}
const names = ls => ls.map(l => l.name).join(' and ')

const PRESETS = [
  [128, 32, 0, '128 thr · 32 regs'],
  [128, 37, 0, '37 regs (rounds to 40)'],
  [1024, 37, 0, '1024 thr · 37 regs'],
  [128, 32, 32, '32 KB shared']
]

function useShape(initial) {
  const [gpu, setGpu] = useState('a100')
  const [threads, setThreads] = useState(initial.threads)
  const [regs, setRegs] = useState(initial.regs)
  const [smemKB, setSmemKB] = useState(initial.smemKB)
  const capKB = Math.floor((SM[gpu].smem - 1024) / 1024)
  const smem = Math.min(smemKB, capKB) * 1024
  const o = useMemo(
    () => occupancy({ threads, regs, smem, sm: SM[gpu] }),
    [threads, regs, smem, gpu]
  )
  const applyGpu = id => {
    setGpu(id)
    const cap = Math.floor((SM[id].smem - 1024) / 1024)
    setSmemKB(k => Math.min(k, cap))
  }
  return { gpu, applyGpu, threads, setThreads, regs, setRegs, smemKB: Math.min(smemKB, capKB), setSmemKB, capKB, o }
}

function ShapeControls({ s, extra }) {
  return (
    <Controls>
      <Select label="GPU" value={s.gpu} onChange={s.applyGpu} options={GPU_OPTIONS} />
      <Slider label="Threads / block" min={32} max={1024} step={32} value={s.threads} onChange={s.setThreads} />
      <Slider label="Registers / thread" min={8} max={128} value={s.regs} onChange={s.setRegs} />
      <Slider label="Shared / block" min={0} max={s.capKB} value={s.smemKB} onChange={s.setSmemKB} fmt={v => v + ' KB'} />
      {extra}
    </Controls>
  )
}

/* =====================================================================
   TAB A — calculator
   ===================================================================== */
function OccupancyLab() {
  const s = useShape({ threads: 128, regs: 32, smemKB: 0 })
  const [grid, setGrid] = useState(108)
  const { o, gpu } = s
  const g = GPUS[gpu]
  const sm = SM[gpu]
  const wave = o.fit ? g.sms * o.activeBlocks : 0
  const waves = wave ? Math.ceil(grid / wave) : 0
  const last = wave ? (grid % wave || wave) : 0

  const [cv] = useCanvas(400, (ctx, W) => {
    const ml = 118, mr = 16
    const barW = W - ml - mr
    text(ctx, o.fit ? pct(o.occ) + '% occupancy' : 'does not fit', ml, 18, {
      size: 13, color: !o.fit ? C.r : o.occ > 0.9 ? C.e : o.occ > 0.4 ? C.d : C.r, weight: 700
    })
    text(ctx, o.fit ? `${o.activeWarps} / ${sm.maxWarps} warps` : o.reason, W - mr, 18, {
      size: 12, color: C.mute, align: 'right'
    })

    const rows = o.limits.map(l => ({
      ...l,
      shown: l.blocks === Infinity ? sm.maxBlocks : l.blocks,
      unused: l.blocks === Infinity
    }))
    const scale = Math.max(sm.maxBlocks, ...rows.map(r => r.shown), 1)
    rows.forEach((l, i) => {
      const y = 40 + i * 52
      text(ctx, l.name, 12, y + 14, { size: 12, color: C.ink })
      ctx.fillStyle = '#1a1e32'
      rr(ctx, ml, y, barW, 28, 6); ctx.fill()
      const bind = o.fit && o.limiters.some(x => x.id === l.id)
      ctx.fillStyle = bind ? LIM_COLOR[l.id] : alpha(C.ink, 0.35)
      const w = Math.max(l.shown > 0 ? 4 : 0, barW * l.shown / scale)
      if (w > 0) { rr(ctx, ml, y, Math.min(barW, w), 28, 6); ctx.fill() }
      const label = l.unused ? 'not used' : l.shown + (l.shown === 1 ? ' block' : ' blocks')
      text(ctx, label, ml + 10, y + 14, { size: 12, color: bind ? '#0a0c14' : C.ink, weight: 700, mono: true })
    })

    const foot = 40 + rows.length * 52 + 8
    if (o.fit) {
      text(ctx, `one wave = ${g.sms} SMs × ${o.activeBlocks} blocks = ${wave} blocks`, 12, foot, { size: 12, color: C.mute })
      text(ctx, `grid ${grid} → ${waves} wave${waves === 1 ? '' : 's'}, last wave ${Math.round(100 * last / wave)}% full`, 12, foot + 18, { size: 12, color: C.ink, mono: true })
    }
  })

  return (
    <>
      <canvas {...cv} />
      <ShapeControls s={s} extra={<Slider label="Blocks in the grid" min={1} max={4096} value={grid} onChange={setGrid} />} />
      <div className="controls">
        {PRESETS.map(([th, rg, kb, label]) => (
          <Btn key={label} onClick={() => { s.setThreads(th); s.setRegs(rg); s.setSmemKB(kb) }}>{label}</Btn>
        ))}
      </div>
      <Legend items={[[C.c, 'registers bind'], [C.d, 'shared memory binds'], [C.b, 'warp slots bind'], [C.a, 'block slots bind']]} />
      <Readout>
        {o.fit ? <>
          <b className={o.occ > 0.9 ? 'g' : 'w'}>{o.activeBlocks}</b> blocks/SM · <b>{o.activeWarps}</b> warps · occupancy <b>{pct(o.occ)}%</b>.
          Limited by <b>{names(o.limiters)}</b>.
          {o.regsAlloc !== s.regs && <> The compiler's {s.regs} registers allocate as <b>{o.regsAlloc}</b> (multiples of 8).</>}
          {o.smemCharged > 0 && <> Shared memory is charged as <b>{fmtBytes(o.smemCharged)}</b>, including the 1 KB reserve.</>}
        </> : <b className="r">{o.reason}</b>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB B — 3D slots
   ===================================================================== */
function SlotScene({ o, sm }) {
  const cols = 8
  const rows = Math.ceil(sm.maxWarps / cols)
  return (
    <group>
      {Array.from({ length: sm.maxWarps }, (_, i) => {
        const on = o.fit && i < o.activeWarps
        const block = on ? Math.floor(i / o.warpsPerBlock) : -1
        const c = i % cols
        const r = Math.floor(i / cols)
        const x = (c - (cols - 1) / 2) * 0.46
        const z = (r - (rows - 1) / 2) * 0.46
        const color = on ? BLOCK_COLORS[block % BLOCK_COLORS.length] : '#3d4660'
        return (
          <mesh key={i} position={[x, on ? 0.22 : 0, z]}>
            <boxGeometry args={[0.36, on ? 0.55 : 0.16, 0.36]} />
            <meshStandardMaterial color={color} emissive={on ? color : '#000'} emissiveIntensity={on ? 0.4 : 0} roughness={0.45} />
          </mesh>
        )
      })}
    </group>
  )
}

function Occupancy3D() {
  const s = useShape({ threads: 128, regs: 37, smemKB: 0 })
  const { o, gpu } = s
  const sm = SM[gpu]
  const overlay = o.fit ? (
    <>
      <b style={{ color: C.e }}>{pct(o.occ)}%</b> · {o.activeWarps}/{sm.maxWarps} warp slots filled
      <br />{o.activeBlocks} blocks · limited by {names(o.limiters)}
    </>
  ) : <>does not fit<br />{o.reason}</>
  return (
    <>
      <Stage3D height={420} camera={[0, 5.4, 6.8]} target={[0, 0, 0]} fov={42} overlay={overlay} hint="drag to orbit · one box is a warp slot, colour groups a block">
        <SlotScene o={o} sm={sm} />
      </Stage3D>
      <ShapeControls s={s} />
      <Readout>
        {o.fit
          ? <>Tall boxes are resident warps. A short grey box is a slot this block shape cannot fill. Each colour is one block ({o.warpsPerBlock} warps).</>
          : <>Nothing is resident. {o.reason}</>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB C — latency hiding
   ===================================================================== */
const STALLS = [
  ['reg', 'dependent FMA'],
  ['smem', 'shared memory'],
  ['l2', 'L2'],
  ['hbm', 'HBM']
]

function HideLab() {
  const [gpu, setGpu] = useState('a100')
  const [stall, setStall] = useState('hbm')
  const [ilp, setIlp] = useState(1)
  const [resident, setResident] = useState(32)
  const g = GPUS[gpu]
  const sm = SM[gpu]
  const lat = g.lat[stall]
  const live = Math.min(resident, sm.maxWarps)
  const need = warpsToHide(lat, ilp, sm.schedulers)
  const cover = Math.min(1, live / need)
  const bytes = bytesInFlight(g.bw, g.lat.hbm, g.clk, g.sms)
  const streamWarps = bytes / 128

  const [cv] = useCanvas(340, (ctx, W) => {
    const ml = 16, mr = 16
    const barW = W - ml - mr
    text(ctx, 'schedulers kept busy', ml, 18, { size: 12, color: C.mute, weight: 600 })
    text(ctx, pct(cover) + '%', W - mr, 18, {
      size: 14, align: 'right', weight: 700, color: cover > 0.95 ? C.e : cover > 0.4 ? C.d : C.r
    })
    ctx.fillStyle = '#1a1e32'
    rr(ctx, ml, 32, barW, 22, 6); ctx.fill()
    ctx.fillStyle = cover > 0.95 ? C.e : cover > 0.4 ? C.d : C.r
    rr(ctx, ml, 32, Math.max(4, barW * cover), 22, 6); ctx.fill()
    text(ctx, `${live} resident  /  ${need} needed  ·  L = ${lat} cycles, I = ${ilp}`, ml, 70, { size: 12, color: C.ink, mono: true })

    text(ctx, 'the ' + sm.maxWarps + ' warp slots on one SM', ml, 100, { size: 12, color: C.mute })
    const cols = sm.maxWarps > 48 ? 16 : 12
    const gap = 3
    const cw = Math.min(18, (barW - (cols - 1) * gap) / cols)
    for (let i = 0; i < sm.maxWarps; i++) {
      const c = i % cols
      const r = Math.floor(i / cols)
      const x = ml + c * (cw + gap)
      const y = 114 + r * (cw + gap)
      ctx.fillStyle = i < live ? (i < need ? C.e : C.b) : '#1a1e32'
      rr(ctx, x, y, cw, cw, 3); ctx.fill()
    }
    const rows = Math.ceil(sm.maxWarps / cols)
    const foot = 114 + rows * (cw + gap) + 14
    text(ctx, 'green: resident warps this stall can actually use    cyan: extra, with nothing left to hide', ml, Math.min(326, foot), { size: 11, color: C.mute })
  })

  return (
    <>
      <canvas {...cv} />
      <Controls>
        <Select label="GPU" value={gpu} onChange={setGpu} options={GPU_OPTIONS} />
        <Select label="Stall" value={stall} onChange={setStall} options={STALLS} />
        <Slider label="Independent instructions" min={1} max={32} value={ilp} onChange={setIlp} />
        <Slider label="Resident warps" min={1} max={sm.maxWarps} value={live} onChange={setResident} />
      </Controls>
      <Legend items={[[C.e, 'covers the stall'], [C.b, 'resident but idle for this stall']]} />
      <Readout>
        {g.name}: this stall is ≈ <b>{lat}</b> cycles. With <b>{ilp}</b> independent instruction{ilp > 1 ? 's' : ''} before the result is used, four schedulers want <b className={need > sm.maxWarps ? 'r' : 'g'}>{need}</b> resident warps.
        {need > sm.maxWarps
          ? <> The SM only has {sm.maxWarps}. Occupancy cannot cover it — raise the independent work per thread.</>
          : <> {live >= need ? 'The resident warps cover it.' : <>{need - live} more warps would still find a stall to hide.</>}</>}
        {stall === 'hbm' && <> Filling the HBM pipe is a different number: <b>{fmtBytes(bytes)}</b> in flight per SM, about <b>{Math.round(streamWarps)}</b> coalesced 128-byte loads. A streaming kernel saturates there; a pointer chase does not.</>}
      </Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 'o', label: 'What limits the SM', render: () => <OccupancyLab /> },
    { id: 'd', label: '3D: warp slots', render: () => <Occupancy3D /> },
    { id: 'h', label: 'Hiding the stall', render: () => <HideLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'See which resource caps the SM, then which stalls those warps can actually hide',
  tryIt: [
    'Leave **128 threads, 32 registers**. Both register and warp bars stop at 16 blocks: 100% on an A100.',
    'Click **37 regs**. The count rounds to 40 and occupancy falls to 75%. Nothing else in the kernel changed.',
    'Click **1024 thr · 37 regs**. One block fills 32 of the 48 warps the register file allows. Sixteen slots stay empty.',
    'Click **32 KB shared**. Shared memory becomes the short bar: 4 blocks, 25%.',
    'Open **3D** (it starts at 37 registers). Tall boxes are resident warps; each colour is one block. Switch the GPU to the **4090** — the same kernel fills every slot, because the ceiling is 48.',
    'Open **Hiding the stall**. A dependent FMA needs 16 warps. Switch the stall to **HBM** with one independent instruction: the SM would need 2000 warps, and it has 64.'
  ],
  theory, math, practice,
  code: [
    { title: 'Ask the runtime how many blocks fit', lang: 'cuda', note: 'cudaOccupancyMaxActiveBlocksPerMultiprocessor counts compiled registers and static shared memory. --ptxas-options=-v prints the register count the percentage came from.', src: querySrc },
    { title: 'Ask the compiler for 8 resident blocks', lang: 'cuda', note: '__launch_bounds__(256, 8) caps registers so eight blocks fit. If ptxas then reports spill stores, the extra occupancy is local-memory traffic.', src: boundsSrc },
    { title: 'One dependent chain versus eight independent accumulators', lang: 'cuda', note: 'Same FMAs per accumulator. The chain has I = 1, so only extra warps hide the 4-cycle wait. The wide kernel has other work in the same warp.', src: hideSrc }
  ],
  quiz: [
    { q: 'Occupancy is:', options: ['Resident warps per SM, divided by the SM\'s maximum', 'Achieved bandwidth divided by peak', 'Threads per block divided by 32', 'The fraction of the grid that has finished'], answer: 0, why: 'It is a resident-warp ratio. Bandwidth and elapsed time are what you measure afterwards; occupancy is only the budget of warps that can hide stalls.' },
    { q: 'A thread that "uses 37 registers" is charged for how many, on these GPUs?', options: ['37', '40', '64', '256'], answer: 1, why: 'Register allocation rounds up to a multiple of 8 per thread. 37 becomes 40, which is 1280 registers per warp.' },
    { q: '128 threads, 37 registers, no shared memory, A100. Occupancy is:', options: ['100%', '75%', '50%', '25%'], answer: 1, why: '40 registers/thread allow 48 warps. A 128-thread block is 4 warps, so 12 blocks. $48/64 = 75\\%$.' },
    { q: 'Why can a 1024-thread block leave warp slots empty even though the register file has some left?', options: ['Blocks cannot be split across the leftover slots', '1024 exceeds the thread limit', 'Shared memory is always reserved', 'The warp size changes'], answer: 0, why: 'Slots are taken in whole blocks. A 32-warp block does not fit into 16 leftover warps, so those slots stay empty.' },
    { q: 'CUDA reserves how much shared memory per block, on top of what you allocate?', options: ['0', '128 B', '1 KB', '48 KB'], answer: 2, why: '1 KB per block. That is why the per-block maximum is 163 KB on A100 when the SM pool is 164 KB.' },
    { q: 'A dependent FMA takes about 4 cycles and the kernel has no other independent instruction. Warps that keep four schedulers busy:', options: ['4', '16', '64', '2000'], answer: 1, why: '$4 \\times 4 / 1 = 16$. Full occupancy does not speed a pure FMA loop past that.' },
    { q: 'A dependent HBM load needs far more than 64 warps to keep the schedulers busy. The useful lever is:', options: ['Raising occupancy from 90% to 100%', 'More independent misses per thread', 'A smaller grid', 'Switching the block to 32 threads'], answer: 1, why: 'The SM cannot hold the ~2000 warps the stall would take. Independent loads per thread (and, for streaming, enough bytes in flight) are what fill the wait.' }
  ]
}
