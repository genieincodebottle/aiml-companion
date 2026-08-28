import React from 'react'

// Page 2. Seven recorded runs, each with a plain statement of what it teaches
// and what to look for. Clicking one loads it and moves the reader straight to
// the trace, so the card and the waterfall are never on screen at the same time
// competing for attention.

const WATCH = {
  '01-happy-path': 'Nothing is broken. Learn the shape here first.',
  '02-shipping-agent-killed': 'One row turns red. The reply still goes out and says what it could not check.',
  '03-slow-tool': 'One row is 8 seconds long. Everything after it waited.',
  '04-saga-rollback': 'Read the last rows bottom-up: the undos run in reverse order.',
  '05-green-and-wrong': 'Every row is green. The refund window in the answer is wrong anyway.',
  '06-cache-off': 'The cached column is all zeros and the total is higher.',
  '07-cross-tenant-attempt': 'The search is asked for another retailer and returns nothing of theirs.',
}

export default function Scenarios({ replays, onPick, activeId }) {
  return (
    <div className="page">
      <h1>Pick something to break</h1>
      <p className="lede">
        Seven recorded runs of the same system. Each one is the same customer question with one
        thing deliberately broken. They are recordings, so they open instantly and cost nothing.
      </p>

      <div className="cardgrid">
        {replays.map((r, i) => {
          const key = r.id === '05-green-and-wrong'
          return (
            <button
              key={r.id}
              className={`scenariocard ${key ? 'starred' : ''} ${activeId === r.id ? 'active' : ''}`}
              onClick={() => onPick(r.id)}
            >
              <div className="scnum">{String(i + 1).padStart(2, '0')}</div>
              <div className="scbody">
                <div className="sctitle">
                  {r.title.replace(/^\d+\.\s*/, '')}
                  {key && <span className="starchip">the important one</span>}
                </div>
                <div className="scteaches">{r.teaches}</div>
                <div className="scwatch">Watch for: {WATCH[r.id] || ''}</div>
              </div>
              <div className="scgo">View trace &rarr;</div>
            </button>
          )
        })}
      </div>

      <div className="nextbar">
        <span>Want to break it yourself instead, with your own question?</span>
      </div>
    </div>
  )
}
