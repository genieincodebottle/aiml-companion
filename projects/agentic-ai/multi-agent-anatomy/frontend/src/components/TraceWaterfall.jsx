import React, { useState } from 'react'

// The trace is the product. Everything else on the page is a control for it.
//
// This component is written for somebody reading a trace for the first time.
// Raw span names like `agent:shipping-agent` mean nothing until you already
// know the system, so every row carries a plain-English label first and the
// technical span name second. The header stays pinned while you scroll, because
// a column of unlabelled numbers is worse than no column at all.

const STAGE_NAMES = {
  0: 'request',
  1: 'input guardrail',
  2: 'classify and route',
  3: 'orchestrator plan',
  4: 'fan-out',
  5: 'writer',
  6: 'orchestrator merge',
  7: 'output guardrail',
  8: 'respond',
}

// Plain-English label for each span, and one line saying what it did.
function describe(span) {
  const n = span.name
  if (n === 'request') return ['The whole request', 'Everything below happened inside this span.']
  if (n.startsWith('stage-1')) return ['Checking the incoming question', 'Rate limits, retry detection, prompt-injection screening.']
  if (n.startsWith('stage-2')) return ['Deciding how much work to do', 'Full fan-out, a single lookup, or a canned reply.']
  if (n.startsWith('stage-3')) return ['Orchestrator writes the plan', 'Picks which agents to run. Uses the expensive model.']
  if (n.startsWith('stage-4')) return ['Three agents run at the same time', 'This stage takes as long as its slowest agent.']
  if (n.startsWith('stage-5')) return ['Writer composes the reply', 'Can only use what the agents above returned.']
  if (n.startsWith('stage-6')) return ['Orchestrator merges and checks', 'Second and last call to the expensive model.']
  if (n.startsWith('stage-7')) return ['Redacting the outgoing reply', 'Emails and card-like numbers are stripped here.']
  if (n.startsWith('stage-8')) return ['Sending the reply', 'The customer sees this.']
  if (n.startsWith('agent:')) {
    const who = n.split(':')[1].replace('-agent', '')
    return [`${who[0].toUpperCase()}${who.slice(1)} agent`, 'One delegation, with its own timeout and its own share of the budget.']
  }
  if (n.startsWith('model:')) return ['Call to the language model', 'This is the row that costs money. Tokens and price are on the right.']
  if (n.startsWith('tool:')) return [`Database lookup (${n.split(':')[1]})`, 'A real query against the seeded database.']
  if (n.startsWith('retrieval:')) return ['Searching the policy documents', 'Scoped to this tenant inside the query, never filtered afterwards.']
  if (n.startsWith('saga:')) return ['Multi-step change with undos', 'Three systems, three compensating actions.']
  if (n.startsWith('undo:')) return [`Undoing ${n.slice(5)}`, 'A compensating action, running in reverse order.']
  if (['book_courier', 'charge_fee', 'update_order'].includes(n)) return [`Saga step: ${n.replace(/_/g, ' ')}`, 'A real side effect in one system.']
  return [n, '']
}

const STATUS_HELP = {
  ok: 'finished normally',
  error: 'failed',
  timeout: 'ran out of its time budget',
  killed: 'stopped by a failure toggle',
  skipped: 'never ran',
  running: 'still running',
}

function tree(spans) {
  const byId = new Map(spans.map((s) => [s.span_id, { ...s, children: [] }]))
  const roots = []
  for (const s of byId.values()) {
    const parent = s.parent_id ? byId.get(s.parent_id) : null
    if (parent) parent.children.push(s)
    else roots.push(s)
  }
  return roots
}

function Row({ span, depth, total, onPick, picked }) {
  const left = total ? ((span.started_offset_ms || 0) / total) * 100 : 0
  const width = total ? Math.max((span.duration_ms / total) * 100, 0.6) : 0
  const warn = span.warnings && span.warnings.length > 0
  const [label, blurb] = describe(span)
  const isPicked = picked === span.span_id

  return (
    <>
      <div
        className={`row ${isPicked ? 'picked' : ''}`}
        onClick={() => onPick(isPicked ? null : span.span_id)}
        title={blurb}
      >
        <div className="cell name" style={{ paddingLeft: 8 + depth * 18 }}>
          <span className={`dot ${span.status}`} title={`${span.status}: ${STATUS_HELP[span.status] || ''}`} />
          <span className="label">{label}</span>
          {span.agent_id !== 'gateway' && <span className="who">{span.agent_id}</span>}
          {span.prompt_version && <span className="pv" title="prompt version that produced this">{span.prompt_version}</span>}
          {warn && (
            <span className="warnflag" title={span.warnings.join('\n')}>
              looks fine, may be wrong
            </span>
          )}
          <span className="rawname">{span.name}</span>
        </div>
        <div className="cell bar">
          <div className="track">
            <div className={`fill ${span.status}`} style={{ left: `${left}%`, width: `${width}%` }} />
          </div>
          <span className="ms">{Math.round(span.duration_ms).toLocaleString()} ms</span>
        </div>
        <div className="cell model">{span.model || ''}</div>
        <div className="cell num">{span.input_tokens || ''}</div>
        <div className="cell num cached">{span.cached_tokens || ''}</div>
        <div className="cell num">{span.output_tokens || ''}</div>
        <div className="cell num">{span.cost_usd ? `$${span.cost_usd.toFixed(5)}` : ''}</div>
        <div className="cell num budget">
          {span.deadline_remaining_s != null ? `${span.deadline_remaining_s}s` : ''}
          {span.tokens_remaining != null ? ` / ${span.tokens_remaining.toLocaleString()}t` : ''}
        </div>
      </div>

      {isPicked && (
        <div className="rowdetail">
          <div className="rowdetail-head">
            <b>{label}</b> <span className="rawname">{span.name}</span>
            <span className={`statuschip ${span.status}`}>
              {span.status}: {STATUS_HELP[span.status]}
            </span>
          </div>
          <p>{blurb}</p>
          {span.stage != null && (
            <p className="muted">
              Stage {span.stage}, {STAGE_NAMES[span.stage]}. Ran by <b>{span.agent_id}</b>
              {span.timeout_s != null && <> with a {span.timeout_s}s timeout</>}.
            </p>
          )}
          {span.error && <p className="err">Error: {span.error}</p>}
          {warn && <p className="warn">{span.warnings.join(' ')}</p>}
          <details>
            <summary>Raw span data</summary>
            <pre>{JSON.stringify(span.detail, null, 2)}</pre>
          </details>
        </div>
      )}

      {span.children.map((c) => (
        <Row key={c.span_id} span={c} depth={depth + 1} total={total} onPick={onPick} picked={picked} />
      ))}
    </>
  )
}

export default function TraceWaterfall({ result }) {
  const [picked, setPicked] = useState(null)
  if (!result)
    return (
      <div className="empty">
        Pick a recorded trace on the right, or ask a question. Each row below will be one step
        the system took.
      </div>
    )

  const spans = result.trace.spans
  const roots = tree(spans)
  const total = Math.max(...spans.map((s) => s.duration_ms), 1)
  const summary = result.trace.summary
  const broken = Object.entries(result.failures || {}).filter(([, v]) => v)

  return (
    <div className="waterfall">
      {/* What am I looking at. Answered before any numbers appear. */}
      <div className="explain">
        <b>One row per step.</b> Indented rows happened inside the row above them. The bar shows
        when each step started and how long it took, so steps that overlap ran at the same time.
        Click any row for a plain explanation.
      </div>

      {result.title && (
        <div className="scenario">
          <div className="scenario-title">{result.title}</div>
          {result.teaches && <div className="scenario-teaches">{result.teaches}</div>}
        </div>
      )}

      {broken.length > 0 && (
        <div className="brokenbar">
          Broken on purpose: {broken.map(([k, v]) => (typeof v === 'number' ? `${k} (${v}s)` : k)).join(', ')}
        </div>
      )}

      {result.all_spans_green && result.semantic_warnings?.length > 0 && (
        // The whole point of the project, said out loud.
        <div className="banner">
          <b>Every span is green and the answer is still wrong.</b> {result.semantic_warnings[0]} There
          is no status code for wrong.
        </div>
      )}

      <div className="totals">
        <span><b>{summary.spans_total}</b> steps</span>
        <span><b>{summary.model_calls}</b> model calls</span>
        <span><b>{summary.tool_calls}</b> tool calls</span>
        <span><b>{summary.total_tokens.toLocaleString()}</b> tokens, <b>{summary.cached_tokens.toLocaleString()}</b> from cache</span>
        <span className="totalcost">total <b>${summary.total_cost_usd.toFixed(5)}</b></span>
        <span className={`mode ${summary.mode}`} title={summary.mode === 'replay' ? 'recorded, no API key used' : 'called the real model'}>
          {summary.mode === 'replay' ? 'recorded' : 'live'}
        </span>
      </div>

      <div className="legend">
        <span><i className="dot ok" /> ok</span>
        <span><i className="dot timeout" /> out of time</span>
        <span><i className="dot error" /> failed</span>
        <span><i className="dot killed" /> killed by a toggle</span>
        <span className="legend-note">green means it finished, not that it was right</span>
      </div>

      <div className="tablescroll">
        <div className="head row">
          <div className="cell name">step</div>
          <div className="cell bar">when it ran, and for how long</div>
          <div className="cell model" title="which model was called">model</div>
          <div className="cell num" title="input tokens sent to the model">tokens in</div>
          <div className="cell num cached" title="input tokens that were served from the prompt cache, billed cheaper">cached</div>
          <div className="cell num" title="tokens the model generated">out</div>
          <div className="cell num" title="cost of this single step">cost</div>
          <div className="cell num budget" title="time and tokens the whole request had left when this step started">budget left</div>
        </div>

        {roots.map((r) => (
          <Row key={r.span_id} span={r} depth={0} total={total} onPick={setPicked} picked={picked} />
        ))}
      </div>
    </div>
  )
}
