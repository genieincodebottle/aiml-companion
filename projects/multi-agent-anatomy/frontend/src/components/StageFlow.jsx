import React from 'react'

// The eight stages, drawn. This is the first thing a new reader sees, so it has
// to carry the whole shape of the system on its own: what runs in sequence,
// what runs in parallel, and which two calls use the expensive model.

const SEQ = [
  { n: 1, label: 'Input guardrail', who: 'gateway', note: 'rate limits, injection screen' },
  { n: 2, label: 'Classify and route', who: 'gateway', note: 'is the full fan-out needed?' },
  { n: 3, label: 'Orchestrator plan', who: 'orchestrator', note: 'expensive model, call 1 of 2', big: true },
]

const FAN = [
  { label: 'Order agent', note: 'orders database', timeout: '20s' },
  { label: 'Shipping agent', note: 'shipments database', timeout: '20s' },
  { label: 'Policy agent', note: 'document search', timeout: '25s' },
]

const TAIL = [
  { n: 5, label: 'Writer agent', who: 'writer', note: 'composes from what returned' },
  { n: 6, label: 'Orchestrator merge', who: 'orchestrator', note: 'expensive model, call 2 of 2', big: true },
  { n: 7, label: 'Output guardrail', who: 'gateway', note: 'redaction' },
  { n: 8, label: 'Respond', who: 'gateway', note: 'the customer sees this' },
]

function Node({ n, label, note, big, tone = 'gateway' }) {
  return (
    <div className={`flownode ${tone} ${big ? 'big' : ''}`}>
      {n != null && <span className="flownum">{n}</span>}
      <div>
        <div className="flowlabel">{label}</div>
        <div className="flownote">{note}</div>
      </div>
    </div>
  )
}

export default function StageFlow({ compact = false }) {
  return (
    <div className={`stageflow ${compact ? 'compact' : ''}`}>
      <div className="flowband">
        <span className="bandlabel">budget set once here: 60s deadline, 60k tokens, tenant id</span>
      </div>

      {SEQ.map((s) => (
        <React.Fragment key={s.n}>
          <Node {...s} tone={s.who === 'orchestrator' ? 'orch' : 'gateway'} />
          <div className="flowarrow" />
        </React.Fragment>
      ))}

      <div className="fanwrap">
        <div className="fanhead">
          <span className="flownum">4</span> Fan-out: three agents at once
        </div>
        <div className="fanrow">
          {FAN.map((f) => (
            <div key={f.label} className="flownode agent">
              <div>
                <div className="flowlabel">{f.label}</div>
                <div className="flownote">{f.note}</div>
                <div className="flowtimeout">{f.timeout} timeout</div>
              </div>
            </div>
          ))}
        </div>
        <div className="fanfoot">this stage takes as long as its slowest agent</div>
      </div>

      <div className="flowarrow" />

      {TAIL.map((s, i) => (
        <React.Fragment key={s.n}>
          <Node {...s} tone={s.who === 'orchestrator' ? 'orch' : s.who === 'writer' ? 'agent' : 'gateway'} />
          {i < TAIL.length - 1 && <div className="flowarrow" />}
        </React.Fragment>
      ))}
    </div>
  )
}
