import React, { useMemo, useState } from 'react'
import Stage3D from '../../../components/Stage3D.jsx'
import { Controls, Slider, Select, Readout, SubViews, Btn, Legend } from '../../../components/ui.jsx'
import { useCanvas } from '../../../lib/hooks.js'
import { C, text, rr, alpha, fmtBytes } from '../../../lib/viz.js'
import { GPUS, GPU_OPTIONS, ridge } from '../memory/specs.js'
import { gemmIntensity, achieved, attentionIO, onlineSteps, SCORES, TILE } from './model.js'
import theory from './theory.md?raw'
import math from './math.md?raw'
import practice from './practice.md?raw'
import wmmaSrc from './code/wmma_gemm.cu?raw'
import naiveSrc from './code/naive_attention.cu?raw'
import flashSrc from './code/flash_online.cu?raw'

const PREC = [
  ['fp32', 'FP32 CUDA cores'],
  ['tf32', 'TF32 tensor cores'],
  ['fp16', 'FP16 tensor cores']
]
const PREC_BYTES = { fp32: 4, tf32: 4, fp16: 2 }
const PREC_COLOR = { fp32: C.a, tf32: C.d, fp16: C.e }
const one = n => Math.round(n * 10) / 10

/* =====================================================================
   TAB A — tensor-core roofline
   ===================================================================== */
function TensorLab() {
  const [gpu, setGpu] = useState('a100')
  const [prec, setPrec] = useState('fp16')
  const [N, setN] = useState(256)
  const g = GPUS[gpu]
  const bytes = PREC_BYTES[prec]
  const peak = g.peak[prec]
  const I = gemmIntensity(N, bytes).intensity
  const rate = achieved(I, g.bw, peak)
  const onRoof = rate >= peak - 0.05
  const star = ridge(peak, g.bw)

  const [cv] = useCanvas(380, (ctx, W, H) => {
    const ml = 48, mr = 16, mt = 28, mb = 32
    const x0 = ml, x1 = W - mr, y0 = mt, y1 = H - mb
    const I0 = 1, I1 = 2048, P0 = 1, P1 = 2048
    const X = v => x0 + (Math.log2(v) - Math.log2(I0)) / (Math.log2(I1) - Math.log2(I0)) * (x1 - x0)
    const Y = v => y1 - (Math.log2(Math.max(v, P0)) - Math.log2(P0)) / (Math.log2(P1) - Math.log2(P0)) * (y1 - y0)
    ctx.strokeStyle = '#1c2136'
    ctx.lineWidth = 1
    for (const v of [1, 4, 16, 64, 256, 1024]) {
      ctx.beginPath(); ctx.moveTo(X(v), y0); ctx.lineTo(X(v), y1); ctx.stroke()
      text(ctx, String(v), X(v), y1 + 14, { size: 10, color: C.dim, align: 'center', mono: true })
    }
    for (const v of [4, 16, 64, 256, 1024]) {
      text(ctx, String(v), x0 - 6, Y(v) + 3, { size: 10, color: C.dim, align: 'right', mono: true })
    }
    text(ctx, 'FLOP / byte', (x0 + x1) / 2, H - 6, { size: 11, color: C.mute, align: 'center' })
    text(ctx, 'TF/s', 8, 14, { size: 11, color: C.mute })
    text(ctx, 'N = ' + N, x1, 14, { size: 12, color: C.ink, align: 'right', weight: 700 })

    const drawRoof = (key, width) => {
      const pk = g.peak[key]
      const b = PREC_BYTES[key]
      ctx.strokeStyle = PREC_COLOR[key]
      ctx.lineWidth = width
      ctx.beginPath()
      let started = false
      for (let i = 0; i <= 64; i++) {
        const inten = I0 * Math.pow(I1 / I0, i / 64)
        const perf = Math.min(pk, g.bw * inten / 1000)
        const x = X(inten), y = Y(perf)
        if (!started) { ctx.moveTo(x, y); started = true } else ctx.lineTo(x, y)
      }
      ctx.stroke()
    }
    PREC.forEach(([key]) => drawRoof(key, key === prec ? 2.5 : 1))

    PREC.forEach(([key]) => {
      const inten = gemmIntensity(N, PREC_BYTES[key]).intensity
      const perf = achieved(inten, g.bw, g.peak[key])
      const x = X(inten), y = Y(perf)
      ctx.fillStyle = PREC_COLOR[key]
      ctx.beginPath()
      ctx.arc(x, y, key === prec ? 6 : 3.5, 0, Math.PI * 2)
      ctx.fill()
    })
  })

  return (
    <>
      <canvas {...cv} />
      <Controls>
        <Select label="GPU" value={gpu} onChange={setGpu} options={GPU_OPTIONS} />
        <Select label="Format" value={prec} onChange={setPrec} options={PREC} />
        <Slider label="N" min={128} max={4096} step={128} value={N} onChange={setN} />
      </Controls>
      <div className="controls">
        <Btn onClick={() => setN(256)}>N = 256</Btn>
        <Btn onClick={() => setN(1024)}>N = 1024</Btn>
      </div>
      <Legend items={PREC.map(([k, lab]) => [PREC_COLOR[k], lab])} />
      <Readout>
        {g.name}, {PREC.find(p => p[0] === prec)[1]}. Intensity <b>{one(I)}</b> FLOP/byte, ridge <b>{one(star)}</b>.
        {onRoof
          ? <> The GEMM is on the compute roof at <b className="g">{one(rate)}</b> TFLOP/s.</>
          : <> The GEMM is on the memory slope at <b className="w">{one(rate)}</b> TFLOP/s, short of the <b>{one(peak)}</b> peak.</>}
        {prec === 'tf32' && g.peak.tf32 === g.peak.fp32 && <> On this GPU the TF32 column equals FP32. The tensor-core gap shows up at FP16.</>}
      </Readout>
    </>
  )
}

const tileLabel = v => (v === 0 ? 'start' : 'tile ' + v)

/* =====================================================================
   TAB B — online softmax
   ===================================================================== */
function SoftmaxLab() {
  const steps = useMemo(() => onlineSteps(), [])
  const [step, setStep] = useState(0)
  const cur = steps[step]
  const tiles = SCORES.length / TILE

  const [cv] = useCanvas(210, (ctx, W) => {
    const ml = 16, mr = 16
    const gap = 4
    const cw = Math.min(36, (W - ml - mr - 15 * gap) / 16)
    const y = 78
    text(ctx, step === 0 ? 'no keys yet' : 'after tile ' + step + ' of ' + tiles, ml, 18, { size: 13, color: C.b, weight: 700 })
    text(ctx, cur.m == null ? 'm = −∞' : 'm = ' + cur.m, W - mr, 18, { size: 13, color: C.e, align: 'right', weight: 700, mono: true })
    SCORES.forEach((s, i) => {
      const tile = Math.floor(i / TILE)
      const seen = step > 0 && tile < step
      const hot = step > 0 && tile === step - 1
      const x = ml + i * (cw + gap)
      ctx.fillStyle = hot ? C.d : seen ? alpha(C.e, 0.85) : '#1a1e32'
      rr(ctx, x, y, cw, 48, 4); ctx.fill()
      text(ctx, String(s), x + cw / 2, y + 24, {
        size: 14, align: 'center', weight: 800, mono: true, color: seen || hot ? '#0a0c14' : C.dim
      })
      if (i % TILE === 0) text(ctx, 't' + tile, x, y - 12, { size: 10, color: C.mute, mono: true })
    })
    text(ctx, 'ℓ = ' + (step === 0 ? '0' : cur.l.toFixed(3)), ml, y + 78, { size: 13, color: C.ink, mono: true })
    if (step > 0 && cur.alpha < 0.999) {
      text(ctx, 'earlier mass × ' + cur.alpha.toFixed(4), ml, y + 100, { size: 12, color: C.c, mono: true })
    }
  })

  const raised = step > 0 && cur.alpha < 0.999
  return (
    <>
      <canvas {...cv} />
      <Controls>
        <Slider label="Tile" min={0} max={tiles} value={step} onChange={setStep} fmt={tileLabel} />
      </Controls>
      <Legend items={[[C.d, 'tile just absorbed'], [C.e, 'already in the running softmax']]} />
      <Readout>
        {step === 0 && <> Sixteen integer scores, tiles of 4. V = 1, so the normalized output has to come back as 1.</>}
        {step > 0 && !raised && <> Running max <b>{cur.m}</b>. This tile did not raise it, so the rescale factor is <b>1</b>.</>}
        {raised && <> Running max moved to <b>{cur.m}</b>. Mass from earlier tiles is multiplied by <b className="p">{cur.alpha.toFixed(4)}</b>{step === 2 && <>, which is e<sup>−3</sup></>}.</>}
        {step === tiles && <> ℓ covers every key. Dividing the accumulator by ℓ is the softmax. With V = 1 that quotient is <b className="g">1</b>.</>}
      </Readout>
    </>
  )
}

/* =====================================================================
   TAB C — the score matrix that is never stored
   ===================================================================== */
function ScoreScene({ rows, cols, step }) {
  const col = step % cols
  const row = Math.floor(step / cols) % rows
  return (
    <group>
      {Array.from({ length: rows * cols }, (_, i) => {
        const c = i % cols
        const r = Math.floor(i / cols)
        const current = r === row && c === col
        const done = r < row || (r === row && c < col)
        const h = current ? 0.55 : done ? 0.18 : 0.06
        const x = (c - (cols - 1) / 2) * 0.52
        const z = (r - (rows - 1) / 2) * 0.52
        const color = current ? '#fbbf24' : done ? '#4ade80' : '#2a3148'
        return (
          <mesh key={i} position={[x, h / 2, z]}>
            <boxGeometry args={[0.4, h, 0.4]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={current ? 0.45 : done ? 0.2 : 0} roughness={0.45} />
          </mesh>
        )
      })}
    </group>
  )
}

function FlashLab() {
  const [gpu, setGpu] = useState('a100')
  const [N, setN] = useState(2048)
  const [d, setD] = useState(64)
  const [Br, setBr] = useState(128)
  const [Bc, setBc] = useState(128)
  const [step, setStep] = useState(0)
  const g = GPUS[gpu]
  const io = useMemo(() => attentionIO({ N, d, Br, Bc, bytes: 2 }), [N, d, Br, Bc])
  const rows = Math.min(8, io.Tr)
  const cols = Math.min(8, io.Tc)
  const span = rows * cols
  const safe = ((step % span) + span) % span
  const sramKB = io.sram / 1024
  const fits = sramKB <= g.smemKB
  const r16 = ridge(g.peak.fp16, g.bw)
  const flashRoof = io.iFlash >= r16
  const stdRoof = io.iStd >= r16

  const overlay = (
    <>
      Q tile {Math.floor(safe / cols)} · K tile {safe % cols}
      <br />amber tile is the only scores alive · green already folded into the output
    </>
  )

  return (
    <>
      <Stage3D height={420} camera={[0, 4.4, 6.6]} target={[0, 0, 0]} fov={40} overlay={overlay} hint="drag to orbit · the flat tiles are the N×N matrix that HBM never receives">
        <ScoreScene rows={rows} cols={cols} step={safe} />
      </Stage3D>
      <Controls>
        <Select label="GPU" value={gpu} onChange={setGpu} options={GPU_OPTIONS} />
        <Select label="N" value={String(N)} onChange={v => setN(+v)} options={[['512', '512'], ['1024', '1024'], ['2048', '2048'], ['4096', '4096']]} />
        <Select label="d" value={String(d)} onChange={v => setD(+v)} options={[['32', '32'], ['64', '64'], ['128', '128']]} />
        <Select label="Bᵣ" value={String(Br)} onChange={v => setBr(+v)} options={[['64', '64'], ['128', '128'], ['256', '256']]} />
        <Select label="Bᶜ" value={String(Bc)} onChange={v => setBc(+v)} options={[['64', '64'], ['128', '128']]} />
        <Slider label="Tile step" min={0} max={span - 1} value={safe} onChange={setStep} />
      </Controls>
      <Legend items={[['#fbbf24', 'score tile in SRAM'], ['#4ade80', 'already accumulated into O'], ['#2a3148', 'never stored']]} />
      <Readout>
        FP16, one head. Standard attention moves <b>{fmtBytes(io.standard)}</b>. Flash moves <b className="g">{fmtBytes(io.flash)}</b> ({(io.standard / io.flash).toFixed(1)}× less).
        Intensity <b>{one(io.iStd)}</b> vs <b>{one(io.iFlash)}</b> FLOP/byte. {g.name} FP16 ridge is <b>{one(r16)}</b>, so standard is {stdRoof ? 'on the roof' : 'memory-bound'} and flash is {flashRoof ? 'on the roof' : 'memory-bound'}.
        SRAM for the live tiles is <b className={fits ? 'g' : 'r'}>{sramKB.toFixed(0)} KB</b> against {g.smemKB} KB on the SM{fits ? '' : ' — this tile does not fit, so the scores spill and the intensity collapses'}.
        {rows < io.Tr || cols < io.Tc ? <> The picture shows the first {rows}×{cols} of {io.Tr}×{io.Tc} tiles.</> : null}
      </Readout>
    </>
  )
}

function Lab() {
  return <SubViews views={[
    { id: 't', label: 'Tensor-core roofline', render: () => <TensorLab /> },
    { id: 's', label: 'Online softmax', render: () => <SoftmaxLab /> },
    { id: 'f', label: '3D: the score tile', render: () => <FlashLab /> }
  ]} />
}

export default {
  Lab,
  vizTitle: 'See the tensor-core roof move, then the score matrix that never lands in HBM',
  tryIt: [
    'Leave **A100**, **FP16**, **N = 256**. Intensity is 85, the ridge is 153, and the dot sits on the memory slope at about 174 TFLOP/s — not on the 312 peak.',
    'Switch the format to **FP32**. The same matrix is already on the 19.5 TFLOP/s roof. A faster CUDA core would not help; a tensor core is not fed yet.',
    'Click **N = 1024**. FP16 intensity passes the ridge and the dot lands on 312.',
    'Open **Online softmax** and step to **tile 2**. The max moves from 2 to 5, and the first tile’s mass is multiplied by e⁻³.',
    'Step to the last tile. V = 1, so the normalized output is exactly 1. The N-long score row was never stored.',
    'Open **3D**. Standard attention at N = 2048, d = 64, FP16 moves 33 MB. The tiled kernel moves 8.5 MB, and the amber block is the only score tile that exists.'
  ],
  theory, math, practice,
  code: [
    { title: '16×16×16 FP16 tile, FP32 accumulator', lang: 'cuda', note: 'One warp per output tile. All-ones inputs make every entry N, which is the check. Tensor-core rounding stays in the accumulator.', src: wmmaSrc },
    { title: 'Attention that allocates the N×N matrices', lang: 'cuda', note: 'S and P are real allocations. The byte count printed is the 4N² + 4Nd model. Correct, and the reason the next file exists.', src: naiveSrc },
    { title: 'Online softmax, one warp per query, no score row', lang: 'cuda', note: 'Running max, running sum, rescale. V = 1, so every output is exactly 1. Tile size is one key; the SRAM blocking is the lab.', src: flashSrc }
  ],
  quiz: [
    { q: 'A tensor-core WMMA tile of FP16 values accumulates into:', options: ['FP16', 'FP32', 'INT8', 'The register file as a single FMA'], answer: 1, why: 'The products are low precision. The sum is kept in FP32 so a long K reduction does not wash out.' },
    { q: 'A100, N = 256, TF32. The GEMM runs at about:', options: ['156 TFLOP/s, the TF32 peak', '19.5 TFLOP/s, the FP32 peak', '87 TFLOP/s, on the memory slope', '312 TFLOP/s'], answer: 2, why: 'Intensity is 42.7 FLOP/byte and the TF32 ridge is 76.5. $\\beta I$ is about 87 TFLOP/s. The peak is not the rate of this matrix.' },
    { q: 'Standard attention’s intensity, as N grows, tends to:', options: ['Something proportional to N', 'd / (element size)', 'The tensor-core peak', 'Zero'], answer: 1, why: 'FLOPs and the N×N score traffic grow together. FP16 with d = 64 tends to 32 FLOP/byte, under every tensor-core ridge in the lab.' },
    { q: 'FlashAttention’s asymptotic intensity depends on:', options: ['N only', 'The Q-tile rows Bᵣ and the element size', 'The batch size', 'How many streams you launch'], answer: 1, why: '$I \\to 2 B_r / s$. A bigger tile re-reads K and V fewer times. N cancels out.' },
    { q: 'The online-softmax factor e^(m − m′) is less than 1 when:', options: ['The new tile raises the running max', 'The new tile is empty', 'You use FP16', 'The kernel is memory-bound'], answer: 0, why: 'Exponentials computed against the old max are too large once a bigger max appears. Multiplying by e^(m−m′) corrects them. If the max does not move, the factor is 1.' },
    { q: 'N = 2048, d = 64, tiles 128, FP16. Flash moves:', options: ['The same 33 MB as standard attention', '8.5 MB, about 3.9× less', 'Nothing; Q and K stay in registers', 'N² bytes'], answer: 1, why: 'K and V are re-read once per Q tile (16 times). S and P are not written. 33.0 / 8.50 = 3.9.' },
    { q: 'Why does the backward pass recompute score tiles?', options: ['Softmax is not differentiable', 'Storing P would write the N×N matrix the forward pass avoided', 'Tensor cores cannot multiply V', 'Recomputing uses less SRAM than one tile'], answer: 1, why: 'The bytes were the bottleneck. Recomputing from Q and K, which are N×d, is cheaper in HBM than rereading N×N.' }
  ]
}
