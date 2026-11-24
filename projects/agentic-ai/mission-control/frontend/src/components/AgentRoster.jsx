export default function AgentRoster({ agents }) {
  return (
    <section className="panel">
      <h2>Squad (A2A cards)</h2>
      <p className="hint">Each agent runs ON the platform, bound by its declared card.</p>
      <ul className="agent-list">
        {agents.map((a) => (
          <li key={a.name} className="agent-row">
            <div className="agent-name">
              {a.role}
              {a.side_effecting && (
                <span className="agent-sfx" title="side-effecting">
                  ⚑
                </span>
              )}
            </div>
            <div className="agent-meta">
              <span className={`tier tier-${a.model_tier}`}>{a.model_tier}</span>
              <span className="agent-tools">{a.allowed_tools.join(', ') || 'no tools'}</span>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
