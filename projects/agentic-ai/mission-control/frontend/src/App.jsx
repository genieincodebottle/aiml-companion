import { useCallback, useEffect, useRef, useState } from 'react';
import {
  decide,
  errorText,
  getAgents,
  getAlerts,
  getCases,
  getEvents,
  getHealth,
  getMe,
  getView,
  startInvestigation,
  streamUrl,
} from './services/api.js';
import Header from './components/Header.jsx';
import OperatorBar from './components/OperatorBar.jsx';
import AlertCatalog from './components/AlertCatalog.jsx';
import AgentRoster from './components/AgentRoster.jsx';
import CaseHistory from './components/CaseHistory.jsx';
import Investigation from './components/Investigation.jsx';

export default function App() {
  const [health, setHealth] = useState(null);
  const [me, setMe] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [agents, setAgents] = useState([]);
  const [cases, setCases] = useState([]);
  const [workflowId, setWorkflowId] = useState(null);
  const [view, setView] = useState(null);
  const [events, setEvents] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const stream = useRef(null);

  const loadCases = useCallback(
    () =>
      getCases()
        .then(setCases)
        .catch(() => {}),
    []
  );
  const loadMe = useCallback(
    () =>
      getMe()
        .then(setMe)
        .catch(() => setMe(null)),
    []
  );

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'down' }));
    getAlerts()
      .then(setAlerts)
      .catch(() => {});
    getAgents()
      .then(setAgents)
      .catch(() => {});
    loadMe();
    loadCases();
  }, [loadMe, loadCases]);

  const refresh = useCallback(async (wid) => {
    const [v, e] = await Promise.all([getView(wid), getEvents(wid)]);
    setView(v);
    setEvents(e);
    return v;
  }, []);

  // Follow a case on the server's event stream. The stream closes itself with a
  // `done` frame when the case parks at the gate or finishes.
  const follow = useCallback(
    (wid) => {
      stream.current?.close();
      const es = new EventSource(streamUrl(wid));
      stream.current = es;
      es.onmessage = null;
      const onEvent = (msg) => {
        if (msg.type === 'done') {
          es.close();
          refresh(wid).catch(() => {});
          loadCases();
          return;
        }
        const data = JSON.parse(msg.data);
        setEvents((prev) =>
          prev.some((e) => e.seq === data.seq)
            ? prev
            : [...prev, { seq: data.seq, type: msg.type, payload: data.payload }]
        );
        getView(wid)
          .then(setView)
          .catch(() => {});
      };
      // EventSource only delivers named events to named listeners.
      [...EVENT_TYPES, 'done'].forEach((t) => es.addEventListener(t, onEvent));
      es.onerror = () => {
        es.close();
        refresh(wid).catch(() => {});
      };
    },
    [refresh, loadCases]
  );

  useEffect(() => () => stream.current?.close(), []);

  const onStart = async (alertId, fault) => {
    setBusy(true);
    setError('');
    setView(null);
    setEvents([]);
    try {
      const res = await startInvestigation(alertId, fault);
      setWorkflowId(res.workflow_id);
      follow(res.workflow_id);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const onDecide = async (decision, note) => {
    setBusy(true);
    setError('');
    try {
      await decide(workflowId, decision, note);
    } catch (err) {
      setError(errorText(err));
    } finally {
      await refresh(workflowId).catch(() => {});
      loadCases();
      setBusy(false);
    }
  };

  const onOpen = async (wid) => {
    setError('');
    stream.current?.close();
    setWorkflowId(wid);
    try {
      const v = await refresh(wid);
      if (v.status === 'running') follow(wid);
    } catch (err) {
      setError(errorText(err));
    }
  };

  return (
    <div className="app">
      <Header health={health} />
      {health?.mode === 'offline' && (
        <div className="banner banner-offline" role="status">
          <b>Offline mode.</b> A deterministic stand-in answers every model call: no key, no
          network, no cost. Set <code>GOOGLE_API_KEY</code> in <code>backend/.env</code> to use
          Gemini.
        </div>
      )}
      {health && health.can_run === false && (
        <div className="banner banner-bad" role="alert">
          {health.detail}
        </div>
      )}
      <OperatorBar me={me} onChange={loadMe} />
      <div className="layout">
        <aside className="sidebar">
          <AlertCatalog alerts={alerts} busy={busy} activeId={view?.alert_id} onStart={onStart} />
          <CaseHistory cases={cases} activeId={workflowId} onOpen={onOpen} />
          <AgentRoster agents={agents} />
        </aside>
        <main className="main">
          {error && (
            <div className="banner banner-bad" role="alert">
              {error}
            </div>
          )}
          <Investigation view={view} events={events} me={me} onDecide={onDecide} busy={busy} />
        </main>
      </div>
    </div>
  );
}

// Every event type the backend can emit (src/models/events.py).
const EVENT_TYPES = [
  'workflow_started',
  'step_started',
  'step_completed',
  'checkpoint_written',
  'workflow_resumed',
  'workflow_completed',
  'workflow_failed',
  'gateway_call',
  'cache_hit',
  'budget_exceeded',
  'provider_failover',
  'policy_allowed',
  'policy_denied',
  'injection_blocked',
  'approval_requested',
  'approval_granted',
  'approval_rejected',
  'maker_checker_blocked',
  'saga_compensated',
  'dead_lettered',
  'retry',
];
