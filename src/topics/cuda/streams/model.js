/**
 * Copy/compute pipeline.
 * Each chunk is H2D, then kernel, then D2H, in that order, on one stream.
 * Chunks are dealt round-robin across streams.
 * One kernel runs at a time. H2D and D2H share an engine unless splitEngines.
 * Without pinned memory the host cannot enqueue ahead, so nothing overlaps.
 */
export function schedule({ n, streams, h, k, d, pinned, splitEngines }) {
  const serial = []
  let t = 0
  for (let i = 0; i < n; i++) {
    serial.push({ i, kind: 'h', stream: 0, start: t, end: t += h })
    serial.push({ i, kind: 'k', stream: 0, start: t, end: t += k })
    serial.push({ i, kind: 'd', stream: 0, start: t, end: t += d })
  }
  const serialTotal = t
  if (!pinned || streams < 1) return { ops: serial, total: serialTotal, serialTotal }

  const queues = Array.from({ length: streams }, () => [])
  for (let i = 0; i < n; i++) queues[i % streams].push(i)
  const idx = Array(streams).fill(0)
  const phase = Array(streams).fill(0)
  const streamReady = Array(streams).fill(0)
  const res = { h: 0, k: 0, d: 0 }
  const ops = []

  for (let step = 0; step < n * 3; step++) {
    let best = null
    for (let s = 0; s < streams; s++) {
      if (idx[s] >= queues[s].length) continue
      const kind = 'hkd'[phase[s]]
      const resource = kind === 'd' && !splitEngines ? 'h' : kind
      const start = Math.max(streamReady[s], res[resource])
      const chunk = queues[s][idx[s]]
      if (!best || start < best.start || (start === best.start && chunk < best.i)) {
        best = { s, kind, resource, start, i: chunk }
      }
    }
    const dur = best.kind === 'h' ? h : best.kind === 'k' ? k : d
    const end = best.start + dur
    ops.push({ i: queues[best.s][idx[best.s]], kind: best.kind, stream: best.s, start: best.start, end })
    streamReady[best.s] = end
    res[best.resource] = end
    phase[best.s] += 1
    if (phase[best.s] === 3) { phase[best.s] = 0; idx[best.s] += 1 }
  }
  return { ops, total: Math.max(...ops.map(o => o.end)), serialTotal }
}

/** Depth of the pipeline: how many chunks are in flight at t (half-open). */
export function inFlight(ops, t) {
  return ops.filter(o => o.start <= t && t < o.end)
}
