/* Shared numeric + drawing helpers for the interactive labs. */

export const C = {
  bg: '#0a0c14', panel: '#10131f', panel2: '#161a2b', line: '#262c44', grid: '#1c2136',
  ink: '#e7e9f4', mute: '#8d93ad', dim: '#5b617b',
  a: '#8b7bff', b: '#22d3ee', c: '#f472b6', d: '#fbbf24', e: '#4ade80', r: '#fb7185', g: '#76d12a'
}

export const clamp = (x, a, b) => Math.max(a, Math.min(b, x))
export const lerp = (a, b, t) => a + (b - a) * t
export const sigmoid = x => 1 / (1 + Math.exp(-x))

export function rng(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
export function randn(r) {
  const u = Math.max(r(), 1e-9), v = r()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}
export function softmax(arr, T = 1) {
  const m = Math.max(...arr)
  const e = arr.map(x => Math.exp((x - m) / T))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map(x => x / s)
}
export const entropy = p => -p.reduce((s, x) => s + (x > 1e-12 ? x * Math.log2(x) : 0), 0)

export const hex2rgb = h => { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255] }
export function mix(c1, c2, t) {
  const a = hex2rgb(c1), b = hex2rgb(c2)
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`
}
export const alpha = (hex, a) => { const [r, g, b] = hex2rgb(hex); return `rgba(${r},${g},${b},${a})` }

const RAMP = [[12, 10, 40], [58, 28, 112], [124, 58, 160], [196, 84, 140], [250, 150, 90], [253, 224, 120]]
export function heatRGB(t) {
  t = clamp(t, 0, 1) * (RAMP.length - 1)
  const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i
  return RAMP[i].map((v, k) => Math.round(v + (RAMP[i + 1][k] - v) * f))
}
export const heat = t => `rgb(${heatRGB(t).join(',')})`
/** diverging ramp, v in [-1,1]: cyan (-) / dark / pink (+) */
export function div(v) {
  v = clamp(v, -1, 1)
  const base = [18, 21, 36], tgt = v >= 0 ? [244, 114, 182] : [34, 211, 238], f = Math.abs(v)
  return `rgb(${base.map((b, i) => Math.round(b + (tgt[i] - b) * f)).join(',')})`
}

/* ---- canvas drawing ---- */
export function text(ctx, s, x, y, o = {}) {
  ctx.font = `${o.weight || 500} ${o.size || 12}px ${o.mono ? "'JetBrains Mono Variable',monospace" : "'Inter Variable',system-ui,sans-serif"}`
  ctx.fillStyle = o.color || C.ink
  ctx.textAlign = o.align || 'left'
  ctx.textBaseline = o.base || 'middle'
  ctx.fillText(s, x, y)
}
export function rr(ctx, x, y, w, h, r = 6) {
  r = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}
export function arrow(ctx, x1, y1, x2, y2, color = C.mute, w = 1.5, head = 7) {
  const a = Math.atan2(y2 - y1, x2 - x1)
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - Math.cos(a) * head * 0.6, y2 - Math.sin(a) * head * 0.6); ctx.stroke()
  ctx.beginPath(); ctx.moveTo(x2, y2)
  ctx.lineTo(x2 - head * Math.cos(a - 0.4), y2 - head * Math.sin(a - 0.4))
  ctx.lineTo(x2 - head * Math.cos(a + 0.4), y2 - head * Math.sin(a + 0.4))
  ctx.closePath(); ctx.fill()
}
export function polyline(ctx, pts, color, w = 2, dash) {
  ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash || [])
  ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke(); ctx.setLineDash([])
}
export const pt = e => {
  const r = e.currentTarget.getBoundingClientRect()
  return { x: e.clientX - r.left, y: e.clientY - r.top }
}
export const fmtInt = n => n.toLocaleString('en-US')
export function fmtBytes(b) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++ }
  return `${b < 10 ? b.toFixed(2) : b < 100 ? b.toFixed(1) : Math.round(b)} ${u[i]}`
}
