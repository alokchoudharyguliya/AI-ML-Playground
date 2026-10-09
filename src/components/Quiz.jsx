import React, { useState } from 'react'
import { useProgress } from '../store.js'
import Md from './Md.jsx'

/** items: [{ q, options:[...], answer:index, why }] */
export default function Quiz({ id, items }) {
  const [picked, setPicked] = useState({})
  const setQuiz = useProgress(s => s.setQuiz)
  const best = useProgress(s => s.quiz[id])
  const answered = Object.keys(picked).length
  const score = items.filter((it, i) => picked[i] === it.answer).length

  const pick = (i, j) => {
    if (picked[i] !== undefined) return
    const next = { ...picked, [i]: j }
    setPicked(next)
    if (Object.keys(next).length === items.length) {
      setQuiz(id, items.filter((it, k) => next[k] === it.answer).length)
    }
  }

  return (
    <div className="quiz">
      <div className="quiz-head">
        <span>Check your understanding — {items.length} questions</span>
        <span className="quiz-score">{answered ? `${score}/${answered} correct` : best != null ? `best ${best}/${items.length}` : ''}</span>
      </div>
      {items.map((it, i) => {
        const p = picked[i]
        return (
          <div className="qitem" key={i}>
            <div className="qq"><b>{i + 1}.</b> <Md inline>{it.q}</Md></div>
            <div className="qopts">
              {it.options.map((o, j) => {
                let cls = 'qopt'
                if (p !== undefined) cls += j === it.answer ? ' ok' : j === p ? ' bad' : ' off'
                return <button key={j} className={cls} onClick={() => pick(i, j)}><Md inline>{o}</Md></button>
              })}
            </div>
            {p !== undefined && <div className="qwhy"><b>{p === it.answer ? 'Correct. ' : 'Not quite. '}</b><Md inline>{it.why}</Md></div>}
          </div>
        )
      })}
      {answered === items.length && (
        <div className="quiz-end">
          Final score: <b>{score}/{items.length}</b>
          <button className="btn" onClick={() => setPicked({})}>Retry</button>
        </div>
      )}
    </div>
  )
}
