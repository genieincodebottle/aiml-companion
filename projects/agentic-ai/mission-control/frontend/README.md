# Mission Control - console

React + Vite operator console. It opens cases, follows each one live on the event
stream, shows the audit trail and drives the reviewer gate.

```bash
npm ci
npm run dev        # http://localhost:5173 (proxies /api to http://localhost:8000)
npm run lint
npm run test       # unit tests for the formatting helpers
npm run test:e2e   # browser journeys against a real backend (offline; starts its own servers)
npm run build      # static bundle, served by nginx in the Docker image
```

`MC_BACKEND_URL` changes where the dev server proxies `/api`. Components:
[`Header`](src/components/Header.jsx), [`OperatorBar`](src/components/OperatorBar.jsx),
[`AlertCatalog`](src/components/AlertCatalog.jsx), [`CaseHistory`](src/components/CaseHistory.jsx),
[`AgentRoster`](src/components/AgentRoster.jsx), [`Investigation`](src/components/Investigation.jsx),
[`ApprovalGate`](src/components/ApprovalGate.jsx) and [`Timeline`](src/components/Timeline.jsx).
