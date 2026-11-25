import React from 'react'

// Page 4. Where the tokens went, and the one lever that moves them most.
// The file name and the page id stay 'cost' so an existing #cost link and the
// imports keep working; the page itself is titled Tokens.

// Group the model spans by stage. A stage's label comes from its own
// "stage-N:label" span, which exists for every stage after the first.
function byStage(spans) {
  const rows = new Map()
  for (const sp of spans) {
    if (sp.kind !== 'model') continue
    const r = rows.get(sp.stage) || { stage: sp.stage, label: '', calls: 0, input: 0, cached: 0, output: 0 }
    r.calls += 1
    r.input += sp.input_tokens
    r.cached += sp.cached_tokens
    r.output += sp.output_tokens
    rows.set(sp.stage, r)
  }
  for (const sp of spans) {
    const m = /^stage-\d+:(.+)$/.exec(sp.name || '')
    if (m && rows.has(sp.stage)) rows.get(sp.stage).label = m[1]
  }
  return [...rows.values()].sort((a, b) => a.stage - b.stage)
}

function TokenTable({ head, rows }) {
  return (
    <table className="costtable">
      <thead>
        <tr>
          <th>{head}</th>
          <th>calls</th>
          <th>tokens in</th>
          <th>cached</th>
          <th>out</th>
          <th>total</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key}>
            <td className="mono">{r.name}</td>
            <td>{r.calls}</td>
            <td>{r.input.toLocaleString()}</td>
            <td className="cached">{r.cached.toLocaleString()}</td>
            <td>{r.output.toLocaleString()}</td>
            <td>{r.total.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function TokensPage({ result, go }) {
  if (!result)
    return (
      <div className="page">
        <h1>Tokens</h1>
        <div className="emptystate">
          <p>Run something first and its token breakdown appears here.</p>
          <button className="primary" onClick={() => go('scenarios')}>Pick a scenario</button>
        </div>
      </div>
    )

  const s = result.trace.summary
  const byModel = result.tokens_by_model || {}
  const modelRows = Object.entries(byModel).map(([m, v]) => ({
    key: m, name: m, calls: v.calls, input: v.input, cached: v.cached, output: v.output, total: v.total,
  }))
  const stageRows = byStage(result.trace.spans).map((r) => ({
    key: r.stage,
    name: `${r.stage}${r.label ? ` ${r.label}` : ''}`,
    calls: r.calls, input: r.input, cached: r.cached, output: r.output, total: r.input + r.output,
  }))
  const cachePct = s.input_tokens ? Math.round((100 * s.cached_tokens) / s.input_tokens) : 0

  return (
    <div className="page">
      <h1>Tokens</h1>
      <p className="lede">
        One customer question, {s.model_calls} calls to a language model. Every call is counted in
        tokens, so the numbers here do not depend on any provider's price list.
      </p>

      <div className="statrow">
        <div className="stat">
          <div className="statnum">{s.total_tokens.toLocaleString()}</div>
          <div className="statlabel">tokens this question</div>
        </div>
        <div className="stat">
          <div className="statnum">{cachePct}%</div>
          <div className="statlabel">of input from cache</div>
        </div>
        <div className="stat">
          <div className="statnum">{s.uncached_input_tokens.toLocaleString()}</div>
          <div className="statlabel">uncached input tokens</div>
        </div>
        <div className="stat">
          <div className="statnum">{s.output_tokens.toLocaleString()}</div>
          <div className="statlabel">output tokens</div>
        </div>
      </div>

      <section className="card">
        <h2>Tokens by model</h2>
        <TokenTable head="model" rows={modelRows} />
        <p className="cardsub">
          The large model appears twice and the small one five times. That split is the design,
          not an accident.
        </p>
      </section>

      <section className="card">
        <h2>Tokens by stage</h2>
        <TokenTable head="stage" rows={stageRows} />
        <p className="cardsub">
          Stage 4 is the fan-out, so it is where the token count grows with the number of agents.
          The request budget is a token budget for exactly this reason.
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
          reused. Cached input tokens are a subset of the input tokens, not extra ones, and the
          provider processes them more cheaply. Run the <b>prompt caching off</b> scenario: the
          total stays the same, but every input token is sent uncached on every call.
        </p>
      </section>

      <div className="nextbar">
        <button className="ghost" onClick={() => go('trace')}>Back to the trace</button>
        <button className="primary" onClick={() => go('break')}>Break something else</button>
      </div>
    </div>
  )
}
