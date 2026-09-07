import React, { useEffect, useState } from 'react'
import Overview from './pages/Overview.jsx'
import Scenarios from './pages/Scenarios.jsx'
import TracePage from './pages/TracePage.jsx'
import CostPage from './pages/CostPage.jsx'
import BreakIt from './pages/BreakIt.jsx'

// Five pages, in the order somebody should meet them. No router dependency: the
// page is one piece of state and the URL hash keeps it linkable and reloadable.

const PAGES = [
  { id: 'overview', num: 1, label: 'Start here' },
  { id: 'scenarios', num: 2, label: 'Pick a failure' },
  { id: 'trace', num: 3, label: 'Read the trace' },
  { id: 'cost', num: 4, label: 'Cost' },
  { id: 'break', num: 5, label: 'Break it yourself' },
]

// Light is the default. A stored preference wins over it; the system setting is
// deliberately not consulted, because the default here is a design decision
// rather than a guess at what the reader wants.
function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('theme') === 'dark' ? 'dark' : 'light'
    } catch {
      return 'light'
    }
  })
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem('theme', theme)
    } catch {
      /* private mode, and a theme is not worth failing over */
    }
  }, [theme])
  return [theme, setTheme]
}

export default function App() {
  const [theme, setTheme] = useTheme()
  const [page, setPage] = useState(() => {
    const h = window.location.hash.replace('#', '')
    return PAGES.some((p) => p.id === h) ? h : 'overview'
  })
  const [config, setConfig] = useState(null)
  const [replays, setReplays] = useState([])
  const [result, setResult] = useState(null)
  const [activeId, setActiveId] = useState(null)
  const [failures, setFailures] = useState({})
  const [question, setQuestion] = useState('Where is my order ORD-4412 and can I still return it?')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    fetch('/api/config').then((r) => r.json()).then(setConfig).catch(() => {})
    fetch('/api/replay').then((r) => r.json()).then(setReplays).catch(() => {})
  }, [])

  function go(next) {
    setPage(next)
    window.location.hash = next
    window.scrollTo({ top: 0 })
  }

  async function loadReplay(id) {
    setBusy(true)
    const data = await fetch(`/api/replay/${id}`).then((r) => r.json())
    setResult(data)
    setActiveId(id)
    setFailures(data.failures || {})
    if (data.question) setQuestion(data.question)
    setBusy(false)
    go('trace')
  }

  async function run() {
    setBusy(true)
    const data = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, failures }),
    }).then((r) => r.json())
    data.title = 'Your run'
    data.teaches = 'A question you asked, with the failures you switched on.'
    data.failures = failures
    setResult(data)
    setActiveId(null)
    setBusy(false)
    go('trace')
  }

  const mode = config?.mode || 'replay'

  return (
    <div className="app">
      <nav className="topnav">
        <div className="brand">
          <div className="brandname">Multi-Agent Anatomy</div>
          <div className="brandsub">8 stages, 5 agents, breakable on purpose</div>
        </div>
        <div className="navtabs">
          {PAGES.map((p) => (
            <button
              key={p.id}
              className={`navtab ${page === p.id ? 'on' : ''}`}
              onClick={() => go(p.id)}
            >
              <span className="navnum">{p.num}</span>
              {p.label}
            </button>
          ))}
        </div>
        <div className="navright">
          <span className={`modechip ${mode}`}>{mode === 'live' ? 'live model' : 'no API key'}</span>
          <button
            className="themetoggle"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
      </nav>

      {busy && <div className="loadingbar" />}

      <main>
        {page === 'overview' && (
          <Overview go={go} mode={mode} modeReason={config?.mode_reason || ''} />
        )}
        {page === 'scenarios' && (
          <Scenarios replays={replays} onPick={loadReplay} activeId={activeId} />
        )}
        {page === 'trace' && <TracePage result={result} go={go} />}
        {page === 'cost' && <CostPage result={result} config={config} go={go} />}
        {page === 'break' && config && (
          <BreakIt
            catalog={config.failure_catalog}
            failures={failures}
            setFailures={setFailures}
            question={question}
            setQuestion={setQuestion}
            onRun={run}
            busy={busy}
            mode={mode}
          />
        )}
      </main>
    </div>
  )
}
