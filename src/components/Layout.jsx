import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { tracks, topics, byTrack } from '../topics/registry.js'
import { useProgress } from '../store.js'

function Logo() {
  return (
    <svg viewBox="0 0 32 32" width="30" height="30" aria-hidden="true">
      <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#8b7bff" /><stop offset="1" stopColor="#22d3ee" /></linearGradient></defs>
      <rect x="1" y="1" width="30" height="30" rx="8" fill="#10131f" stroke="url(#lg)" strokeWidth="1.5" />
      <path d="M8 10 L16 16 L24 10 M8 22 L16 16 L24 22 M8 10 L8 22 M24 10 L24 22" stroke="url(#lg)" strokeWidth="1.4" fill="none" />
      <g fill="#e7e9f4"><circle cx="8" cy="10" r="2" /><circle cx="8" cy="22" r="2" /><circle cx="16" cy="16" r="2.4" /><circle cx="24" cy="10" r="2" /><circle cx="24" cy="22" r="2" /></g>
    </svg>
  )
}

function Palette({ open, onClose }) {
  const [q, setQ] = useState('')
  const [i, setI] = useState(0)
  const nav = useNavigate()
  const inp = useRef(null)
  const items = useMemo(() => {
    const all = [
      ...topics.map(x => ({ to: '/t/' + x.id, label: x.title, sub: tracks[x.track].name, hay: `${x.title} ${x.blurb} ${(x.tags || []).join(' ')}` })),
      { to: '/glossary', label: 'Glossary', sub: 'Reference', hay: 'glossary terms definitions' },
      { to: '/', label: 'Home', sub: 'Overview', hay: 'home overview' }
    ]
    const s = q.trim().toLowerCase()
    return s ? all.filter(x => x.hay.toLowerCase().includes(s)) : all
  }, [q])
  useEffect(() => { if (open) { setQ(''); setI(0); setTimeout(() => inp.current && inp.current.focus(), 30) } }, [open])
  useEffect(() => setI(0), [q])
  const go = it => { if (it) { nav(it.to); onClose() } }
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="pal-back" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}>
          <motion.div className="pal" initial={{ y: -16, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -10, opacity: 0 }} onClick={e => e.stopPropagation()}>
            <input ref={inp} value={q} placeholder="Jump to a topic…  (try “warp”, “lstm”, “flash”)" onChange={e => setQ(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'ArrowDown') { e.preventDefault(); setI(v => Math.min(items.length - 1, v + 1)) }
                else if (e.key === 'ArrowUp') { e.preventDefault(); setI(v => Math.max(0, v - 1)) }
                else if (e.key === 'Enter') go(items[i])
                else if (e.key === 'Escape') onClose()
              }} />
            <div className="pal-list">
              {items.slice(0, 12).map((it, k) => (
                <div key={it.to} className={'pal-i' + (k === i ? ' on' : '')} onMouseEnter={() => setI(k)} onClick={() => go(it)}>
                  <span>{it.label}</span><small>{it.sub}</small>
                </div>
              ))}
              {!items.length && <div className="pal-empty">No matches</div>}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default function Layout({ children }) {
  const [navOpen, setNavOpen] = useState(false)
  const [pal, setPal] = useState(false)
  const loc = useLocation()
  const done = useProgress(s => s.done)

  useEffect(() => { setNavOpen(false); window.scrollTo(0, 0) }, [loc.pathname])
  useEffect(() => {
    const h = e => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPal(p => !p) }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [])

  // accent follows the current track
  useEffect(() => {
    const m = loc.pathname.match(/^\/t\/(.+)$/)
    const tp = m && topics.find(x => x.id === m[1])
    const tk = tp ? tracks[tp.track] : tracks.dl
    document.documentElement.style.setProperty('--acc', tk.acc)
    document.documentElement.style.setProperty('--acc2', tk.acc2)
  }, [loc.pathname])

  return (
    <div className={'app' + (navOpen ? ' nav-open' : '')}>
      <header className="topbar">
        <button className="burger" aria-label="Toggle navigation" onClick={() => setNavOpen(o => !o)}><span /><span /><span /></button>
        <Link to="/" className="brand-sm">Neural Atlas</Link>
        <button className="kbd-btn" onClick={() => setPal(true)}>Search</button>
      </header>
      <aside className="nav">
        <Link to="/" className="brand">
          <Logo />
          <span>Neural Atlas<small>DL × CUDA</small></span>
        </Link>
        <button className="searchbtn" onClick={() => setPal(true)}>
          <span>Search topics…</span><kbd>⌘K</kbd>
        </button>
        <nav className="navlist">
          {['dl', 'cuda'].map(tr => (
            <div key={tr} style={{ '--acc': tracks[tr].acc }}>
              <div className="nav-h"><i />{tracks[tr].name}</div>
              {byTrack(tr).map(x => (
                <NavLink key={x.id} to={'/t/' + x.id} className={({ isActive }) => 'nav-i' + (isActive ? ' on' : '')}>
                  <b>{String(x.order).padStart(2, '0')}</b><span>{x.title}</span>{done[x.id] && <em>✓</em>}
                </NavLink>
              ))}
            </div>
          ))}
          <div className="nav-h" style={{ '--acc': '#fbbf24' }}><i />Reference</div>
          <NavLink to="/glossary" className={({ isActive }) => 'nav-i' + (isActive ? ' on' : '')} style={{ '--acc': '#fbbf24' }}><b>Aa</b><span>Glossary</span></NavLink>
        </nav>
        <div className="navfoot">Static site · no backend<br />React · Three.js · KaTeX</div>
      </aside>
      {navOpen && <div className="scrim" onClick={() => setNavOpen(false)} />}
      <main className="main">{children}</main>
      <Palette open={pal} onClose={() => setPal(false)} />
    </div>
  )
}
