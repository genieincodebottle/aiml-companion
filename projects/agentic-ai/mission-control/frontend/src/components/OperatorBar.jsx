import { useState } from 'react';
import { getOperatorName, getToken, setOperatorName, setToken } from '../services/api.js';

// Two demo people, so maker-checker can be tried without configuring logins: Ana opens
// cases, Rex reviews them. Anyone can type another name.
const DEMO = ['ana.analyst', 'rex.reviewer'];

export default function OperatorBar({ me, onChange }) {
  const loginRequired = me?.login_required;
  const [name, setName] = useState(getOperatorName());
  const [token, setTokenInput] = useState('');

  const pick = (n) => {
    setName(n);
    setOperatorName(n);
    onChange();
  };

  if (loginRequired === undefined) return null;

  if (loginRequired) {
    const signedIn = Boolean(getToken()) && me?.operator;
    return (
      <div className="operator-bar">
        {signedIn ? (
          <>
            <span>
              Signed in as <b>{me.operator.name}</b> ({me.operator.roles.join(', ')})
            </span>
            <button
              className="btn-ghost"
              onClick={() => {
                setToken('');
                onChange();
              }}
            >
              Sign out
            </button>
          </>
        ) : (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              setToken(token.trim());
              setTokenInput('');
              onChange();
            }}
          >
            <label htmlFor="token">Operator token</label>
            <input
              id="token"
              type="password"
              value={token}
              onChange={(e) => setTokenInput(e.target.value)}
              autoComplete="off"
            />
            <button type="submit" disabled={!token.trim()}>
              Sign in
            </button>
          </form>
        )}
      </div>
    );
  }

  return (
    <div className="operator-bar">
      <label htmlFor="operator">Acting as</label>
      <select
        id="operator"
        value={DEMO.includes(name) ? name : 'other'}
        onChange={(e) => e.target.value !== 'other' && pick(e.target.value)}
      >
        {DEMO.map((n) => (
          <option key={n} value={n}>
            {n}
          </option>
        ))}
        {!DEMO.includes(name) && <option value="other">{name}</option>}
      </select>
      <span className="hint">
        Login is off, so anyone can claim a name. Maker-checker still applies: the person who opens
        a case cannot decide it.
      </span>
    </div>
  );
}
