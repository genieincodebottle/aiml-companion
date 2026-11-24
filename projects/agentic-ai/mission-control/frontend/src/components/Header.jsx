export default function Header({ health }) {
  const mode = health?.mode || '...';
  const ok = health?.status === 'ok';
  return (
    <header className="header">
      <div className="brand">
        <span className="logo">◈</span>
        <div>
          <h1>Mission Control</h1>
          <p className="tagline">Enterprise Agent Control Plane</p>
        </div>
      </div>
      <div className="chips">
        <span className={`chip ${ok ? 'chip-ok' : 'chip-bad'}`}>
          {ok ? 'healthy' : health?.status === 'degraded' ? 'degraded' : 'backend down'}
        </span>
        <span className="chip chip-mode">
          {mode === 'offline' ? 'offline mode' : `live: ${health?.premium_model}`}
        </span>
        {health && (
          <span className={`chip ${health.policy_enabled ? 'chip-ok' : 'chip-bad'}`}>
            policy: {health.policy_enabled ? 'on' : 'OFF'}
          </span>
        )}
        {health?.auto_approve && <span className="chip chip-warn">auto-approve</span>}
      </div>
    </header>
  );
}
