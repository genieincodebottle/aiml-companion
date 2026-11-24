import { eventLabel, eventPlane } from '../format.js';

const PLANE_LABEL = {
  gateway: 'gateway',
  policy: 'policy',
  human: 'human',
  reliability: 'reliability',
  runtime: 'runtime',
};

function summary(payload) {
  const bits = [];
  if (payload.step) bits.push(payload.step);
  if (payload.agent) bits.push(payload.agent);
  if (payload.tool) bits.push(payload.tool);
  if (payload.model) bits.push(payload.model);
  if (payload.rule) bits.push(payload.rule);
  if (payload.reason) bits.push(payload.reason);
  if (payload.cost_usd) bits.push(`$${payload.cost_usd}`);
  if (payload.disposition) bits.push(`→ ${payload.disposition}`);
  return bits.join(' · ');
}

export default function Timeline({ events }) {
  return (
    <div className="timeline">
      <div className="timeline-head">
        <h3>Trace timeline</h3>
        <span className="hint">one span per step, across every plane</span>
      </div>
      <ol className="events">
        {events.map((e) => {
          const plane = eventPlane(e.type);
          return (
            <li key={e.seq} className="event">
              <span className={`dot dot-${plane}`} title={PLANE_LABEL[plane]} />
              <span className="event-seq">{e.seq}</span>
              <span className="event-type">{eventLabel(e.type)}</span>
              <span className="event-summary">{summary(e.payload)}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
