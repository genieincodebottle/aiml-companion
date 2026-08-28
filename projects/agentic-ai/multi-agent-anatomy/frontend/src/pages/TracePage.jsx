import React from 'react'
import TraceWaterfall from '../components/TraceWaterfall.jsx'

// Page 3. The trace gets the whole screen. The answer the customer would read
// sits underneath it, deliberately second: the answer is the least interesting
// output of this system.

export default function TracePage({ result, go }) {
  if (!result)
    return (
      <div className="page">
        <h1>Trace</h1>
        <div className="emptystate">
          <p>Nothing loaded yet.</p>
          <button className="primary" onClick={() => go('scenarios')}>
            Pick a scenario
          </button>
        </div>
      </div>
    )

  return (
    <div className="page">
      <h1>Trace</h1>
      <p className="lede">
        Every step the system took, in order, with what it cost. This is the product. The reply is
        at the bottom.
      </p>

      <TraceWaterfall result={result} />

      <section className="card answercard">
        <h2>What the customer would read</h2>
        <p className="answer">{result.reply}</p>
        {result.gaps?.length > 0 && (
          <>
            <div className="gapslabel">Stated gaps, rather than invented facts:</div>
            <ul className="gaps">
              {result.gaps.map((g) => (
                <li key={g}>{g}</li>
              ))}
            </ul>
          </>
        )}
        {result.semantic_warnings?.length > 0 && (
          <div className="wrongnote">
            This answer passed every check and may still be wrong. {result.semantic_warnings[0]}
          </div>
        )}
      </section>

      <div className="nextbar">
        <button className="ghost" onClick={() => go('scenarios')}>
          Back to scenarios
        </button>
        <button className="primary" onClick={() => go('cost')}>
          See what this cost
        </button>
      </div>
    </div>
  )
}
