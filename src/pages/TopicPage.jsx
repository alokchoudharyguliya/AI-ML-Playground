import React, { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { motion, useScroll, useSpring, AnimatePresence } from 'framer-motion'
import { find, ordered, tracks } from '../topics/registry.js'
import { useProgress } from '../store.js'
import Md from '../components/Md.jsx'
import CodeBlock from '../components/CodeBlock.jsx'
import Quiz from '../components/Quiz.jsx'

class Boundary extends React.Component {
  state = { err: null }
  static getDerivedStateFromError(err) { return { err } }
  componentDidCatch(e) { console.error(e) }
  render() {
    if (this.state.err) return <div className="err">Visualization failed: {String(this.state.err.message || this.state.err)}</div>
    return this.props.children
  }
}

const TABS = [['theory', 'Theory'], ['math', 'Math'], ['code', 'Code'], ['practice', 'Practice'], ['quiz', 'Quiz']]

export default function TopicPage() {
  const { id } = useParams()
  const meta = find(id)
  const [mod, setMod] = useState(null)
  const [tab, setTab] = useState('theory')
  const [loadErr, setLoadErr] = useState(null)
  const done = useProgress(s => s.done[id])
  const toggle = useProgress(s => s.toggle)
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 24 })

  useEffect(() => {
    setMod(null); setTab('theory'); setLoadErr(null)
    if (!meta) return
    let live = true
    meta.load().then(m => live && setMod(m.default)).catch(e => live && setLoadErr(e))
    document.title = meta.title + ' — Neural Atlas'
    return () => { live = false }
  }, [id])

  if (!meta) return <div className="page"><h1>Topic not found</h1><Link to="/">Back home</Link></div>

  const tk = tracks[meta.track]
  const all = ordered()
  const idx = all.findIndex(x => x.id === id)
  const prev = all[idx - 1], next = all[idx + 1]

  return (
    <motion.article className="page" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.22 }}>
      <motion.div className="readbar" style={{ scaleX }} />
      <div className="crumbs"><Link to="/">Atlas</Link> / {tk.name} / {String(meta.order).padStart(2, '0')}</div>
      <h1>{meta.title}</h1>
      <p className="lede">{meta.blurb}</p>
      <div className="chips">
        <span className="chip lvl">{meta.level}</span><span className="chip">{meta.time}</span>
        {meta.tags.map(x => <span key={x} className="chip ghost">{x}</span>)}
      </div>

      <section className="lab">
        <div className="lab-head"><span className="badge">● Interactive lab</span><h2>{mod?.vizTitle || 'Playground'}</h2></div>
        <div className="lab-body">
          {mod ? <Boundary key={id}><mod.Lab /></Boundary>
            : loadErr ? <div className="loading">{meta.ready ? 'This chapter failed to load: ' + String(loadErr.message || loadErr) : 'Coming soon: this chapter has not been written yet.'}</div>
            : <div className="loading">Loading lab…</div>}
        </div>
        {mod?.tryIt && (
          <ul className="try">
            <li className="try-h">Things to try</li>
            {mod.tryIt.map((x, i) => <li key={i}><Md inline>{x}</Md></li>)}
          </ul>
        )}
      </section>

      <div className="tabs" role="tablist">
        {TABS.map(([k, l]) => (
          <button key={k} role="tab" className={'tab' + (tab === k ? ' on' : '')} onClick={() => setTab(k)}>{l}</button>
        ))}
      </div>
      <div className="pane prose">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={tab + (mod ? 'y' : 'n')} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
            {!mod ? <div className="loading">{loadErr ? '—' : 'Loading…'}</div> : (
              <>
                {tab === 'theory' && <Md>{mod.theory}</Md>}
                {tab === 'math' && <Md>{mod.math}</Md>}
                {tab === 'code' && mod.code.map((c, i) => <CodeBlock key={i} {...c} />)}
                {tab === 'practice' && <Md>{mod.practice}</Md>}
                {tab === 'quiz' && <Quiz id={id} items={mod.quiz} />}
              </>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="pager">
        {prev ? <Link to={'/t/' + prev.id} className="pg"><small>← Previous</small>{prev.title}</Link> : <span />}
        <button className={'btn done' + (done ? ' on' : '')} onClick={() => toggle(id)}>{done ? '✓ Completed' : 'Mark as completed'}</button>
        {next ? <Link to={'/t/' + next.id} className="pg r"><small>Next →</small>{next.title}</Link> : <span />}
      </div>
    </motion.article>
  )
}
