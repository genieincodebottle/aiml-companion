import { statusMeta } from '../format.js';

export default function CaseHistory({ cases, activeId, onOpen }) {
  if (!cases.length) return null;
  return (
    <section className="panel">
      <h2>Recent cases</h2>
      <p className="hint">Read from disk, so a case parked at the gate survives a restart.</p>
      <ul className="case-list">
        {cases.slice(0, 12).map((c) => {
          const meta = statusMeta(c.status);
          return (
            <li key={c.workflow_id}>
              <button
                className={`case-row ${activeId === c.workflow_id ? 'case-active' : ''}`}
                onClick={() => onOpen(c.workflow_id)}
              >
                <span className="case-id">{c.alert_id}</span>
                <span className={`status status-${meta.kind}`}>{meta.label}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
