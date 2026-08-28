import React from 'react'

// Page 4. Your own question, your own switches. Separated from the trace so the
// act of breaking something is a deliberate step rather than a checkbox you
// brush past while reading a waterfall.

export default function BreakIt({ catalog, failures, setFailures, question, setQuestion, onRun, busy, mode }) {
  const set = (key, value) => setFailures({ ...failures, [key]: value })
  const activeCount = Object.values(failures).filter(Boolean).length

  return (
    <div className="page">
      <h1>Break it yourself</h1>
      <p className="lede">
        Ask anything, switch on any combination of failures, and run it. The trace opens when it
        finishes.
      </p>

      <section className="card">
        <h2>1. The question</h2>
        <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={2} />
        <div className="hintrow">
          Try: <button className="linkbtn" onClick={() => setQuestion('Where is my order ORD-4412 and can I still return it?')}>order status and returns</button>
          <button className="linkbtn" onClick={() => setQuestion('Please book the return for ORD-4412 and refund it')}>book a return (runs the saga)</button>
          <button className="linkbtn" onClick={() => setQuestion('hi')}>just say hi (skips the fan-out)</button>
        </div>
      </section>

      <section className="card">
        <h2>2. What to break</h2>
        <p className="cardsub">Leave them all off for a clean run.</p>
        <div className="switchgrid">
          {catalog.map((f) => {
            const on = f.type === 'seconds' ? (failures.slow_tool_seconds || 0) > 0 : !!failures[f.key]
            return (
              <label key={f.key} className={`switch ${on ? 'on' : ''}`}>
                <input
                  type="checkbox"
                  checked={on}
                  disabled={busy}
                  onChange={(e) =>
                    f.type === 'seconds'
                      ? set('slow_tool_seconds', e.target.checked ? f.default_on : 0)
                      : set(f.key, e.target.checked)
                  }
                />
                <div>
                  <div className="switch-title">
                    {f.label} <span className="stage-chip">stage {f.stage}</span>
                  </div>
                  <div className="switch-teaches">{f.teaches}</div>
                  {on && <div className="switch-watch">Watch: {f.watch}</div>}
                </div>
              </label>
            )
          })}
        </div>
      </section>

      <section className="card">
        <h2>3. Run it</h2>
        <p className="cardsub">
          {activeCount === 0 ? 'Nothing broken. This will be a clean run.' : `${activeCount} failure${activeCount > 1 ? 's' : ''} switched on.`}
          {mode === 'live'
            ? ' Live mode: this calls a real model and costs about $0.008.'
            : ' Recorded mode: no API key, nothing billed.'}
        </p>
        <button className="primary big" onClick={onRun} disabled={busy}>
          {busy ? 'Running, this takes a few seconds' : 'Run it and show me the trace'}
        </button>
      </section>
    </div>
  )
}
