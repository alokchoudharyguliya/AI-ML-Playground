import React, { useMemo, useState } from 'react'
import Stage3D from '../../../components/Stage3D.jsx'
import { Controls, Slider, Select, Toggle, Readout, SubViews, Btn, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, text, rr } from '../../../lib/viz.js'
import { schedule, inFlight } from './model.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import pipeSrc from './code/pipeline.cu?raw'
import eventSrc from './code/events.cu?raw'
import pinSrc from './code/pinned.cu?raw'

const KIND = { h: 'H2D', k: 'kernel', d: 'D2H' }
const KIND_COLOR = { h: C.b, k: C.e, d: C.d }
const PAL = ['#4ade80', '#22d3ee', '#fbbf24', '#f472b6', '#a78bfa', '#fb7185', '#86efac', '#67e8f9']

function usePipe(initial) {
  const [n, setN] = useState(initial.n)
  const [streams, setStreams] = useState(initial.streams)
  const [h, setH] = useState(initial.h)
  const [k, setK] = useState(initial.k)
  const [d, setD] = useState(initial.d)
  const [pinned, setPinned] = useState(true)
  const [split, setSplit] = useState(true)
  const sched = useMemo(
    () => schedule({ n, streams, h, k, d, pinned, splitEngines: split }),
    [n, streams, h, k, d, pinned, split]
  )
  return { n, setN, streams, setStreams, h, setH, k, setK, d, setD, pinned, setPinned, split, setSplit, sched }
}

function PipeControls({ p, extra }) {
  return (
    <Controls>
      <Slider label="Chunks" min={1} max={8} value={p.n} onChange={p.setN} />
      <Slider label="Streams" min={1} max={4} value={p.streams} onChange={p.setStreams} />
      <Slider label="H2D" min={1} max={8} value={p.h} onChange={p.setH} />
      <Slider label="Kernel" min={1} max={8} value={p.k} onChange={p.setK} />
      <Slider label="D2H" min={1} max={8} value={p.d} onChange={p.setD} />
      <Toggle label="Pinned memory" value={p.pinned} onChange={p.setPinned} />
      <Toggle label="Two copy engines" value={p.split} onChange={p.setSplit} />
      {extra}
    </Controls>
  )
}

/* =====================================================================
   TAB A — gantt
   ===================================================================== */
function TimelineLab() {
  const p = usePipe({ n: 8, streams: 1, h: 2, k: 2, d: 2 })
  const { sched } = p
  const speedup = sched.serialTotal / sched.total

  const [cv] = useCanvas(360, (ctx, W) => {
    const ml = 62, mr = 14
    const plot = W - ml - mr
    const rows = ['h', 'k', 'd']
    text(ctx, sched.total + ' units', ml, 16, { size: 13, color: speedup > 1.05 ? C.e : C.d, weight: 700 })
    text(ctx, 'serial ' + sched.serialTotal, W - mr, 16, { size: 12, color: C.mute, align: 'right', mono: true })
    const scale = plot / sched.total
    rows.forEach((kind, r) => {
      const y = 36 + r * 78
      text(ctx, KIND[kind], 10, y + 22, { size: 12, color: KIND_COLOR[kind], weight: 700 })
      ctx.fillStyle = '#14182a'
      rr(ctx, ml, y, plot, 44, 6); ctx.fill()
      sched.ops.filter(o => o.kind === kind).forEach(o => {
        const x = ml + o.start * scale
        const w = Math.max(2, (o.end - o.start) * scale - 1)
        ctx.fillStyle = PAL[o.i % PAL.length]
        rr(ctx, x, y + 6, w, 32, 4); ctx.fill()
        if (w > 16) text(ctx, String(o.i), x + w / 2, y + 22, { size: 12, color: '#0a0c14', align: 'center', weight: 800, mono: true })
      })
    })
    text(ctx, 'each bar is one chunk  ·  a gap on a row is that unit sitting idle', ml, 280, { size: 11, color: C.mute })
  })

  return (
    <>
      <canvas {...cv} />
      <PipeControls p={p} />
      <div className="controls">
        <Btn onClick={() => { p.setN(8); p.setStreams(1); p.setH(2); p.setK(2); p.setD(2) }}>1 stream</Btn>
        <Btn onClick={() => { p.setN(8); p.setStreams(3); p.setH(2); p.setK(2); p.setD(2) }}>3 streams, equal</Btn>
        <Btn onClick={() => { p.setN(6); p.setStreams(2); p.setH(4); p.setK(2); p.setD(1) }}>copy bound</Btn>
      </div>
      <Legend items={[[C.b, 'H2D row'], [C.e, 'kernel row'], [C.d, 'D2H row']]} />
      <Readout>
        Pipeline <b>{sched.total}</b> · one stream <b>{sched.serialTotal}</b> · speedup <b className={speedup > 1.05 ? 'g' : 'w'}>{speedup.toFixed(2)}×</b>.
        {!p.pinned && <> Pinned is off, so every copy blocks the host and the streams never get ahead. The run is serial.</>}
        {p.pinned && !p.split && <> One copy engine: H2D and D2H take turns. Only the kernel can overlap a copy.</>}
        {p.pinned && p.split && p.h === p.k && p.k === p.d && p.streams === 2 && p.n > 2 && <> Two streams can only fill two of the three units, so this run sits above the three-stream line.</>}
        {p.pinned && p.split && p.streams >= 3 && p.h === p.k && p.k === p.d && <> Three equal stages and three busy units. T = N·B + (H+K+D − B) = {sched.total}.</>}
        {p.pinned && p.split && p.streams >= 2 && p.h > p.k && p.h > p.d && <> H2D is the bottleneck. Streams past two do not shorten a copy engine that is already full.</>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB B — 3D stations
   ===================================================================== */
function StationScene({ ops, t }) {
  const live = inFlight(ops, t)
  const xOf = { h: -3, k: 0, d: 3 }
  return (
    <group>
      {['h', 'k', 'd'].map(kind => (
        <mesh key={kind} position={[xOf[kind], -0.05, 0]}>
          <boxGeometry args={[2.2, 0.12, 1.6]} />
          <meshStandardMaterial color="#1a2033" roughness={0.6} />
        </mesh>
      ))}
      {live.map(o => (
        <mesh key={o.kind + o.i} position={[xOf[o.kind], 0.55, 0]}>
          <boxGeometry args={[1.15, 0.9, 0.9]} />
          <meshStandardMaterial color={PAL[o.i % PAL.length]} emissive={PAL[o.i % PAL.length]} emissiveIntensity={0.35} roughness={0.4} />
        </mesh>
      ))}
    </group>
  )
}

function Pipeline3D() {
  const p = usePipe({ n: 8, streams: 3, h: 2, k: 2, d: 2 })
  const [time, setTime] = useState(4)
  const t = Math.min(time, p.sched.total)
  const live = ['h', 'k', 'd'].map(kind => inFlight(p.sched.ops, t).find(o => o.kind === kind)).filter(Boolean)
  const line = live.length
    ? live.map(o => KIND[o.kind] + ' chunk ' + o.i).join('   ')
    : 'pipeline empty'
  const overlay = (
    <>
      t = <b style={{ color: C.b }}>{t}</b> / {p.sched.total}
      <br />{line}
    </>
  )
  return (
    <>
      <Stage3D height={420} camera={[0, 3.6, 8.2]} target={[0, 0.3, 0]} fov={42} overlay={overlay} hint="drag to orbit · left platform is H2D, middle is the kernel, right is D2H">
        <StationScene ops={p.sched.ops} t={t} />
      </Stage3D>
      <PipeControls p={p} extra={<Slider label="Time" min={0} max={Math.max(1, p.sched.total)} value={t} onChange={setTime} />} />
      <Readout>
        {p.streams === 1
          ? <>One stream: only one platform is ever occupied. Step time and the chunk walks left to right.</>
          : <>With {p.streams} streams, two or three platforms can hold a chunk at the same moment. That is the overlap.</>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB C — what is allowed to overlap
   ===================================================================== */
const CASES = [
  {
    id: 'same', label: 'two kernels, one stream', lanes: ['stream 1'], overlap: false,
    ops: [{ lane: 0, kind: 'k', start: 0, end: 4, tag: 'A' }, { lane: 0, kind: 'k', start: 4, end: 8, tag: 'B' }],
    why: 'A stream is ordered. B does not reach the SMs until A has finished.'
  },
  {
    id: 'pin', label: 'pinned H2D beside a kernel', lanes: ['stream 1', 'stream 2'], overlap: true,
    ops: [{ lane: 0, kind: 'k', start: 0, end: 5, tag: 'kernel' }, { lane: 1, kind: 'h', start: 0, end: 5, tag: 'H2D' }],
    why: 'Different streams, pinned host memory, copy engine free. The DMA and the SMs run together.'
  },
  {
    id: 'page', label: 'pageable H2D beside a kernel', lanes: ['stream 1', 'stream 2'], overlap: false,
    ops: [{ lane: 0, kind: 'k', start: 0, end: 4, tag: 'kernel' }, { lane: 1, kind: 'h', start: 4, end: 8, tag: 'H2D' }],
    why: 'cudaMemcpyAsync from pageable memory can block the host while the driver stages the bytes. The kernel is finished before the copy starts.'
  },
  {
    id: 'def', label: 'legacy default stream', lanes: ['stream 1', 'default'], overlap: false,
    ops: [{ lane: 0, kind: 'k', start: 0, end: 4, tag: 'kernel' }, { lane: 1, kind: 'h', start: 4, end: 8, tag: 'H2D' }],
    why: 'Work in the legacy default stream waits for every blocking stream, then blocks them. Launching the copy with no stream argument puts it here.'
  },
  {
    id: 'evt', label: 'wait on one event', lanes: ['stream A', 'stream B'], overlap: false,
    ops: [{ lane: 0, kind: 'k', start: 0, end: 4, tag: 'kernel' }, { lane: 1, kind: 'd', start: 4, end: 7, tag: 'D2H' }],
    why: 'cudaStreamWaitEvent holds B’s copy until A’s kernel finishes. The host is not held: it can keep enqueueing. cudaDeviceSynchronize would have stopped the host and every stream.'
  }
]

function RulesLab() {
  const [id, setId] = useState('pin')
  const c = CASES.find(x => x.id === id)
  const total = Math.max(...c.ops.map(o => o.end))

  const [cv] = useCanvas(260, (ctx, W) => {
    const ml = 78, mr = 16
    const plot = W - ml - mr
    const scale = plot / total
    text(ctx, c.overlap ? 'overlaps' : 'does not overlap', ml, 16, {
      size: 13, color: c.overlap ? C.e : C.d, weight: 700
    })
    c.lanes.forEach((name, r) => {
      const y = 40 + r * 78
      text(ctx, name, 8, y + 20, { size: 11, color: C.mute })
      ctx.fillStyle = '#14182a'
      rr(ctx, ml, y, plot, 40, 6); ctx.fill()
      c.ops.filter(o => o.lane === r).forEach(o => {
        const x = ml + o.start * scale
        const w = Math.max(8, (o.end - o.start) * scale - 2)
        ctx.fillStyle = KIND_COLOR[o.kind]
        rr(ctx, x, y + 6, w, 28, 4); ctx.fill()
        text(ctx, o.tag, x + w / 2, y + 20, { size: 12, color: '#0a0c14', align: 'center', weight: 700 })
      })
    })
  })

  return (
    <>
      <canvas {...cv} />
      <Controls>
        <Select label="Launch" value={id} onChange={setId} options={CASES.map(x => [x.id, x.label])} />
      </Controls>
      <Readout>{c.why}</Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 't', label: 'Timeline', render: () => <TimelineLab /> },
    { id: 'd', label: '3D: three platforms', render: () => <Pipeline3D /> },
    { id: 'r', label: 'What may overlap', render: () => <RulesLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'Queue chunks across streams and watch the copy engines overlap the kernel',
  tryIt: [
    'Leave **1 stream**, 8 chunks, every stage 2. The three rows never run together. The run is 48.',
    'Click **3 streams, equal**. A copy-in, a kernel and a copy-out are busy at once. The run drops to 20, which is 2.40×.',
    'Drag **Streams** to 2, then to 4. Two streams finish at 26 — one unit is always idle. Four streams stay at 20. There is no fourth unit to fill.',
    'Turn **Two copy engines** off. H2D and D2H stop sharing the timeline and the run goes to 32.',
    'Turn **Pinned memory** off. Every stream count collapses back to 48. The host cannot enqueue ahead of a pageable copy.',
    'Open **3D** (it starts at 3 streams). Step **Time**. Two or three platforms hold a chunk at once. Switch to 1 stream and only one platform is ever lit.',
    'Open **What may overlap**. Pinned H2D beside a kernel shares the clock. The legacy default stream and pageable memory do not.'
  ],
  theory, math, practice,
  code: [
    { title: 'Chunked saxpy on 1 stream and on 3', lang: 'cuda', note: 'Pinned buffers, one device buffer per stream, every memcpy and the kernel launched into that stream. One event wait after the whole loop.', src: pipeSrc },
    { title: 'Order stream B behind stream A without stopping the host', lang: 'cuda', note: 'cudaEventRecord on A, cudaStreamWaitEvent on B. The host returns immediately and can enqueue the next chunks.', src: eventSrc },
    { title: 'Pinned allocation, and a stream the default stream cannot see', lang: 'cuda', note: 'cudaMallocHost is the overlap path. cudaStreamNonBlocking does not join the legacy default-stream barrier.', src: pinSrc }
  ],
  quiz: [
    { q: 'Inside one stream, a kernel and the copy that follows it:', options: ['Run at the same time', 'Run in order', 'Run only if the memory is pinned', 'Require an event'], answer: 1, why: 'A stream is a queue. Overlap happens across streams, not inside one.' },
    { q: 'cudaMemcpyAsync from memory returned by malloc:', options: ['Always overlaps the previous kernel', 'Can block the host while the driver stages the bytes', 'Is a compile error', 'Uses both copy engines'], answer: 1, why: 'Pageable memory is not DMA-able. The driver may copy it into a pinned bounce buffer before the call returns, so the kernel you meant to overlap is already done.' },
    { q: 'Eight chunks, each stage length 2, three streams, two copy engines, pinned. The run takes:', options: ['48', '26', '20', '16'], answer: 2, why: '$T = NB + (H+K+D-B) = 16 + 4 = 20$. One stream would take 48.' },
    { q: 'Why does a fourth stream not improve that 20?', options: ['Streams are capped at 3', 'All three units are already busy', 'The fourth stream deadlocks', 'Pinned memory allows only three copies'], answer: 1, why: 'The H2D engine, the SMs and the D2H engine are the whole machine. A fourth queue has nothing new to occupy.' },
    { q: 'A kernel launched with no stream argument, beside an H2D in a blocking stream:', options: ['Overlaps the copy', 'Goes to the legacy default stream, which serializes with blocking streams', 'Is pinned automatically', 'Runs on the copy engine'], answer: 1, why: 'The omitted stream is the legacy default stream. It waits for blocking streams and then blocks them.' },
    { q: 'cudaStreamWaitEvent, compared with cudaDeviceSynchronize:', options: ['Stops the host until every stream finishes', 'Orders one stream behind one event and lets the host keep enqueueing', 'Pins the buffer', 'Starts a copy engine'], answer: 1, why: 'The wait is on the GPU queue. The host thread is not in it, so the rest of the pipeline can still be queued.' },
    { q: 'Two fat kernels, each of which already fills the SMs, in two streams. They:', options: ['Always overlap', 'Overlap only if both fit at once, which they do not', 'Require pinned memory to overlap', 'Share a warp'], answer: 1, why: 'Different streams are necessary and not sufficient. Concurrent kernels need leftover SMs. Copy/compute overlap is the one that does not.' }
  ]
}
