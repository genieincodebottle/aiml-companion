import React from 'react'
import StageFlow from '../components/StageFlow.jsx'

// Page 1. Nobody should meet a trace waterfall before they know what the system
// is and why anyone would break it. This page exists to answer both, then hand
// the reader to a specific next click.

export default function Overview({ go, mode, modeReason }) {
  return (
    <div className="page">
      <div className="hero">
        <h1>An order-support assistant, built to be broken</h1>
        <p className="lede">
          A customer asks <i>"where is my order, and can I still return it?"</i> Five AI agents
          answer it across eight steps. You are not here to admire the answer. You are here to
          break the system on purpose and watch exactly what happens.
        </p>
        <div className="herobtns">
          <button className="primary big" onClick={() => go('scenarios')}>
            Start the tour
          </button>
          <button className="ghost big" onClick={() => go('trace')}>
            Skip to a trace
          </button>
        </div>
        <div className={`modenote ${mode}`}>
          <div className="modenote-title">
            {mode === 'live' ? 'Live mode' : 'Recorded mode'}
            <span className="modenote-cost">
              {mode === 'live' ? 'about $0.008 per question you run' : 'free, nothing is called'}
            </span>
          </div>
          {/* Say which variable put us here and how to change it. A mode a
              reader cannot explain is a mode they cannot trust. */}
          <p>{modeReason}</p>
          <p className="modenote-sub">
            The seven recorded scenarios never call anything in either mode. Only the
            <b> Break it yourself </b> page runs a live request.
          </p>
        </div>
      </div>

      <div className="twocol">
        <section className="card">
          <h2>Why this exists</h2>
          <p>
            Every multi-agent tutorial shows the happy path. Production is the other cases: one
            agent dies, one tool hangs, a payment succeeds while the order update fails, or the
            worst one, everything reports success and the answer is still wrong.
          </p>
          <p>
            This project makes those cases into buttons. You turn one on, run the same question,
            and read the trace.
          </p>
        </section>

        <section className="card">
          <h2>The four things you will see</h2>
          <ol className="numlist">
            <li><b>Partial failure.</b> An agent dies and the reply still goes out, honestly.</li>
            <li><b>A slow tool.</b> One hanging call and the whole fan-out waits for it.</li>
            <li><b>A rollback.</b> Three systems changed, one fails, the undos run backwards.</li>
            <li><b>A silent wrong answer.</b> Every check passes. The answer is still wrong. This is the one that matters.</li>
          </ol>
        </section>
      </div>

      <section className="card">
        <h2>How one question flows through the system</h2>
        <p className="cardsub">
          Read this once. Every trace you look at later is this shape, one row per box.
        </p>
        <StageFlow />
        <div className="agentcount">
          <b>Five agents.</b> Three lookups, one writer, and the orchestrator, which is an agent
          itself rather than a scheduler. Only the orchestrator uses the expensive model, and only
          twice per question, which is what keeps the bill sane.
        </div>
      </section>

      <div className="nextbar">
        <span>Ready?</span>
        <button className="primary" onClick={() => go('scenarios')}>
          Pick something to break
        </button>
      </div>
    </div>
  )
}
