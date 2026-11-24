import { decidedBy, statusMeta, usd } from '../format.js';
import Timeline from './Timeline.jsx';
import ApprovalGate from './ApprovalGate.jsx';

export default function Investigation({ view, events, me, onDecide, busy }) {
  if (!view) {
    return (
      <div className="empty">
        <h2>No case open</h2>
        <p>Pick an alert from the queue to run it through the platform.</p>
        <p className="empty-sub">
          Watch each call cross the planes: gateway, policy, runtime, then the reviewer gate, with
          the model spend per case.
        </p>
      </div>
    );
  }
  const meta = statusMeta(view.status);
  return (
    <div className="investigation">
      <div className="inv-header">
        <div>
          <h2>{view.workflow_id}</h2>
          <div className="inv-sub">
            tenant <b>{view.tenant}</b> · alert <b>{view.alert_id}</b>
          </div>
        </div>
        <div className="inv-stats">
          <span className={`status status-${meta.kind}`}>{meta.label}</span>
          <span className="stat">steps {view.step_count}</span>
          <span className="stat">spend {usd(view.cost_usd)}</span>
          {view.disposition && <span className="stat stat-disp">→ {view.disposition}</span>}
        </div>
      </div>

      <dl className="audit">
        <dt>Opened by</dt>
        <dd>{view.started_by || '-'}</dd>
        <dt>Decision</dt>
        <dd>{decidedBy(view)}</dd>
        {view.decision_note && (
          <>
            <dt>Reviewer note</dt>
            <dd>{view.decision_note}</dd>
          </>
        )}
        {view.dead_letter_reason && (
          <>
            <dt>Dead-lettered</dt>
            <dd className="audit-bad">{view.dead_letter_reason}</dd>
          </>
        )}
      </dl>

      {view.status === 'awaiting_approval' && (
        <ApprovalGate
          pending={view.pending_action}
          startedBy={view.started_by}
          me={me}
          onDecide={onDecide}
          busy={busy}
        />
      )}

      <Timeline events={events} />
    </div>
  );
}
