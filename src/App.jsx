import React, { Suspense, lazy } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Layout from './components/Layout.jsx'

const Home = lazy(() => import('./pages/Home.jsx'))
const TopicPage = lazy(() => import('./pages/TopicPage.jsx'))
const Glossary = lazy(() => import('./pages/Glossary.jsx'))

export default function App() {
  const loc = useLocation()
  return (
    <Layout>
      <Suspense fallback={<div className="loading">Loading…</div>}>
        <AnimatePresence mode="wait" initial={false}>
          <Routes location={loc} key={loc.pathname}>
            <Route path="/" element={<Home />} />
            <Route path="/t/:id" element={<TopicPage />} />
            <Route path="/glossary" element={<Glossary />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </AnimatePresence>
      </Suspense>
    </Layout>
  )
}
