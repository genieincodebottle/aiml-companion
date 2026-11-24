import { useState } from 'react';

export default function ApprovalGate({ pending, startedBy, me, onDecide, busy }) {
  const [note, setNote] = useState('');
  const who = me?.operator?.name;
  const ownCase = who && who === startedBy;
  return (
    <div className="gate">
      <div className="gate-icon" aria-hidden="true">
        ✋
      </div>
      <div className="gate-body">
        <h3>Reviewer sign-off required</h3>
        <p>
          The policy engine marked this disposition as a regulatory action. Nothing is filed until a
          reviewer other than <b>{startedBy}</b> decides.
        </p>
        <div className="gate-action">
          proposed: <code>{pending?.tool}</code>
          {pending?.args && <span className="gate-args"> {JSON.stringify(pending.args)}</span>}
        </div>
        {ownCase && (
          <p className="gate-warn" role="note">
            You opened this case, so you cannot decide it (maker-checker). Switch to another person
            above.
          </p>
        )}
        <label className="gate-note">
          Reviewer note
          <input
            value={note}
            maxLength={500}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Why you decided this way (kept on the case record)"
          />
        </label>
        <div className="gate-buttons">
          <button
            className="btn-approve"
            disabled={busy || ownCase}
            onClick={() => onDecide('approve', note)}
          >
            Approve
          </button>
          <button
            className="btn-reject"
            disabled={busy || ownCase}
            onClick={() => onDecide('reject', note)}
          >
            Reject &amp; escalate
          </button>
        </div>
      </div>
    </div>
  );
}
