import { useCallback, useEffect, useRef } from 'react'

/**
 * Canvas hook with DPR scaling, resize handling and optional animation loop.
 * draw(ctx, w, h, t, dt) is always the latest closure (so it sees current React state).
 * Static canvases repaint after every render; animated ones repaint every frame
 * while visible on screen.
 */
export function useCanvas(height, draw, { animate = false } = {}) {
  const ref = useRef(null)
  const drawRef = useRef(draw)
  drawRef.current = draw
  const size = useRef({ w: 0, h: height })
  const visible = useRef(true)

  const paint = useCallback((t = 0, dt = 0) => {
    const cv = ref.current
    if (!cv || !size.current.w) return
    const ctx = cv.getContext('2d')
    ctx.clearRect(0, 0, size.current.w, size.current.h)
    drawRef.current(ctx, size.current.w, size.current.h, t, dt)
  }, [])

  useEffect(() => {
    const cv = ref.current
    const fit = () => {
      const r = cv.getBoundingClientRect()
      if (!r.width) return
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      size.current = { w: r.width, h: r.height }
      cv.width = Math.round(r.width * dpr)
      cv.height = Math.round(r.height * dpr)
      cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0)
      paint(performance.now() / 1000, 0)
    }
    const ro = new ResizeObserver(fit)
    ro.observe(cv)
    const io = new IntersectionObserver(es => { visible.current = es[0].isIntersecting }, { threshold: 0 })
    io.observe(cv)
    return () => { ro.disconnect(); io.disconnect() }
  }, [paint])

  useEffect(() => {
    if (!animate) return
    let id, last = performance.now()
    const f = now => {
      const dt = Math.min(0.05, (now - last) / 1000); last = now
      if (visible.current) paint(now / 1000, dt)
      id = requestAnimationFrame(f)
    }
    id = requestAnimationFrame(f)
    return () => cancelAnimationFrame(id)
  }, [animate, paint])

  useEffect(() => { if (!animate) paint(performance.now() / 1000, 0) })

  return [{ ref, className: 'viz-cv', style: { height } }, paint]
}

/** Keep a ref that always points at the latest value (for use inside animation callbacks). */
export function useLatest(v) {
  const r = useRef(v)
  r.current = v
  return r
}

/** Interval-driven stepper: calls fn every `ms` while `on` is true. */
export function useTicker(on, ms, fn) {
  const f = useLatest(fn)
  useEffect(() => {
    if (!on) return
    const id = setInterval(() => f.current(), ms)
    return () => clearInterval(id)
  }, [on, ms, f])
}
