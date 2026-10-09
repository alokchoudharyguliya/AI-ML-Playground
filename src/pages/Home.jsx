import React, { Suspense, lazy } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { tracks, byTrack, ordered } from '../topics/registry.js'
import { useProgress } from '../store.js'

const HeroScene = lazy(() => import('../components/HeroScene.jsx'))

const fade = i => ({ initial: { opacity: 0, y: 18 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true, margin: '-40px' }, transition: { delay: i * 0.07, duration: 0.4 } })

export default function Home() {
  const done = useProgress(s => s.done)
  const all = ordered().filter(x => x.ready)          // only chapters that actually exist
  const resume = all.find(x => !done[x.id]) || all[0]
  const nDone = all.filter(x => done[x.id]).length

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="home">
      <section className="hero">
        <div className="hero-3d"><Suspense fallback={null}><HeroScene /></Suspense></div>
        <div className="hero-in">
          <div className="eyebrow">Interactive · 3D · Math · Code · No backend</div>
          <h1>Understand deep learning <br />down to the <span className="grad">warp</span>.</h1>
          <p>
            A hands-on atlas that runs from convolutions and recurrent nets through Transformers, BERT and modern LLMs —
            and from CUDA thread blocks and shared memory down to Tensor Cores and FlashAttention. Every chapter pairs a
            live simulation with theory, derivations, runnable code and practice.
          </p>
          <div className="cta">
            <Link className="btn primary" to={'/t/' + resume.id}>{nDone ? `Continue: ${resume.title} →` : 'Start with CNNs →'}</Link>
            <Link className="btn" to="/t/cuda-exec">Jump to CUDA</Link>
          </div>
          <div className="stats">
            <div><b>{all.length}</b><span>chapters</span></div>
            <div><b>{all.length}</b><span>live labs</span></div>
            <div><b>{nDone}/{all.length}</b><span>completed</span></div>
          </div>
        </div>
      </section>

      <section className="tracks">
        {['dl', 'cuda'].map((tr, k) => {
          const ts = byTrack(tr), tk = tracks[tr], n = ts.filter(x => x.ready && done[x.id]).length
          return (
            <motion.div key={tr} className="tcard" style={{ '--acc': tk.acc }} {...fade(k)}>
              <div className="tc-h"><h2>{tk.name}</h2><span>{n}/{ts.length} done</span></div>
              <p className="tc-s">{tk.sub}</p>
              <div className="bar"><i style={{ width: (100 * n / ts.length) + '%' }} /></div>
              <ol className="path">
                {ts.map(x => (
                  <li key={x.id}>
                    {x.ready
                      ? <Link to={'/t/' + x.id}><b>{x.title}</b><span>{x.blurb}</span></Link>
                      : <div className="soon"><b>{x.title}<em>soon</em></b><span>{x.blurb}</span></div>}
                  </li>
                ))}
              </ol>
            </motion.div>
          )
        })}
      </section>

      <section className="how">
        {[
          ['Live labs, real numbers', 'Every simulation computes the actual math in your browser — convolve, backprop through time, count memory transactions, schedule blocks on SMs.'],
          ['3D where it matters', 'Orbit a loss landscape with four optimizers racing, or fly through a CUDA grid and click a single thread to see its warp and global index.'],
          ['Theory → math → code → practice', 'Intuition first, then the derivation, then compact PyTorch / CUDA you can paste and run, then worked exercises and a quiz.'],
          ['Skips the baby steps', 'Assumes tensors, matmul and MLPs. The story starts where the complexity does: convolutions, sequence models, attention, GPU architecture.']
        ].map(([h, p], i) => (
          <motion.div key={h} {...fade(i)}><h3>{h}</h3><p>{p}</p></motion.div>
        ))}
      </section>
    </motion.div>
  )
}
