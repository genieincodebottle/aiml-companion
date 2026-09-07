import React from 'react'

// Page 5. The bill, and the one lever that moves it most.

export default function CostPage({ result, config, go }) {
  if (!result)
    return (
      <div className="page">
        <h1>Cost</h1>
        <div className="emptystate">
          <p>Run something first and its cost breakdown appears here.</p>
          <button className="primary" onClick={() => go('scenarios')}>Pick a scenario</button>
        </div>
      </div>
    )

  const s = result.trace.summary
  const byModel = result.cost_by_model || {}
  const cachePct = s.total_tokens ? Math.round((100 * s.cached_tokens) / s.total_tokens) : 0

  return (
    <div className="page">
      <h1>Cost</h1>
      <p className="lede">
        One customer question, {s.model_calls} calls to a language model. Small numbers here become
        large numbers at a million questions a month.
      </p>

      <div className="statrow">
        <div className="stat">
          <div className="statnum">${s.total_cost_usd.toFixed(5)}</div>
          <div className="statlabel">this question</div>
        </div>
        <div className="stat">
          <div className="statnum">${(s.total_cost_usd * 1_000_000).toFixed(0)}</div>
          <div className="statlabel">at 1M questions</div>
        </div>
        <div className="stat">
          <div className="statnum">{s.total_tokens.toLocaleString()}</div>
          <div className="statlabel">tokens</div>
        </div>
        <div className="stat">
          <div className="statnum">{cachePct}%</div>
          <div className="statlabel">served from cache</div>
        </div>
      </div>

      <section className="card">
        <h2>Where the money went</h2>
        <table className="costtable">
          <thead>
            <tr>
              <th>model</th>
              <th>calls</th>
              <th>tokens in</th>
              <th>cached</th>
              <th>out</th>
              <th>cost</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(byModel).map(([m, v]) => (
              <tr key={m}>
                <td className="mono">{m}</td>
                <td>{v.calls}</td>
                <td>{v.input.toLocaleString()}</td>
                <td className="cached">{v.cached.toLocaleString()}</td>
                <td>{v.output.toLocaleString()}</td>
                <td>${v.usd.toFixed(5)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="cardsub">
          The expensive model appears twice and the cheap one five times. That split is the design,
          not an accident.
        </p>
      </section>

      <section className="card">
        <h2>The prompt cache</h2>
        <div className="cachebar">
          <div className="cachefill" style={{ width: `${cachePct}%` }} />
          <span>{cachePct}% of input tokens were served from the prompt cache</span>
        </div>
        <p className="cardsub">
          Each agent's instructions never change between questions, so they are sent once and
          reused. Cached tokens are billed at roughly a tenth of the normal rate. Run the
          <b> prompt caching off </b> scenario to see the same question cost more.
        </p>
        {config && (
          <p className="pricenote">
            Prices per 1M tokens, checked {config.prices_last_checked}. They live in one file,
            backend/app/config.py, because they go stale.
          </p>
        )}
      </section>

      <div className="nextbar">
        <button className="ghost" onClick={() => go('trace')}>Back to the trace</button>
        <button className="primary" onClick={() => go('break')}>Break something else</button>
      </div>
    </div>
  )
}
