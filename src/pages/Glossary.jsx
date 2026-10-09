import React, { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { GLOSSARY } from './glossaryData.js'

export default function Glossary() {
  const [q, setQ] = useState('')
  const [cat, setCat] = useState('all')
  const list = useMemo(() => {
    const s = q.trim().toLowerCase()
    return GLOSSARY
      .filter(g => (cat === 'all' || g.cat === cat) && (!s || (g.term + ' ' + g.def).toLowerCase().includes(s)))
      .sort((a, b) => a.term.localeCompare(b.term))
  }, [q, cat])
  return (
    <motion.div className="page" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
      <div className="crumbs"><Link to="/">Atlas</Link> / Reference</div>
      <h1>Glossary</h1>
      <p className="lede">Short, precise definitions of the terms used throughout the atlas. Each links to the chapter that develops it.</p>
      <div className="controls" style={{ marginBottom: 18 }}>
        <input className="gsearch" placeholder="Filter terms…" value={q} onChange={e => setQ(e.target.value)} />
        {[['all', 'All'], ['dl', 'Deep learning'], ['cuda', 'CUDA']].map(([k, l]) => (
          <button key={k} className={'seg' + (cat === k ? ' on' : '')} onClick={() => setCat(k)}>{l}</button>
        ))}
      </div>
      <div className="gloss">
        {list.map(g => (
          <div className="gitem" key={g.term}>
            <div className="gterm">{g.term} <span className={'gcat ' + g.cat}>{g.cat === 'dl' ? 'DL' : 'CUDA'}</span></div>
            <p>{g.def}</p>
            {g.see && <Link to={'/t/' + g.see}>Read more →</Link>}
          </div>
        ))}
        {!list.length && <p>No terms match.</p>}
      </div>
    </motion.div>
  )
}
