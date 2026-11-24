export default function AlertCatalog({ alerts, busy, activeId, onStart }) {
  return (
    <section className="panel">
      <h2>Alert queue</h2>
      <p className="hint">Trigger an investigation. Risky dispositions pause for human sign-off.</p>
      <ul className="alert-list">
        {alerts.map((a) => (
          <li key={a.id} className={`alert-card ${activeId === a.id ? 'alert-active' : ''}`}>
            <div className="alert-head">
              <span className="alert-id">{a.id}</span>
              <span className="alert-amount">${a.amount_usd.toLocaleString()}</span>
            </div>
            <div className="alert-type">{a.alert_type}</div>
            <div className="alert-desc">{a.description}</div>
            <div className="alert-actions">
              <button disabled={busy} onClick={() => onStart(a.id, false)}>
                Investigate
              </button>
              <button
                className="btn-ghost"
                disabled={busy}
                title="Inject a commit fault to watch the saga compensate"
                onClick={() => onStart(a.id, true)}
              >
                + fault
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
