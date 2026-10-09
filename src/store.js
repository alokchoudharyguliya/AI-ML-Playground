import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/* Learning progress, persisted in localStorage (no backend). */
export const useProgress = create(
  persist(
    (set, get) => ({
      done: {},
      quiz: {},
      toggle: id => set(s => ({ done: { ...s.done, [id]: !s.done[id] } })),
      setQuiz: (id, score) => set(s => ({ quiz: { ...s.quiz, [id]: Math.max(score, s.quiz[id] ?? 0) } }))
    }),
    { name: 'neural-atlas-progress' }
  )
)
