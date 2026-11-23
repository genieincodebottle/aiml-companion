# Incident Commander - Frontend

The React console. Start with the project [README](../README.md).

You pick a simulated production fault, then watch the agent loop work it live:
triage, diagnosis, evidence requests, the proposed fix and its prediction, the
Critic's objection, the human approval gate, the action, verification, and the
postmortem.

## Run

Needs **Node.js 22.12+** (24 LTS recommended) and the backend running on port 8000.

```bash
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/api` to `http://localhost:8000`. If the backend runs
elsewhere, start with `IC_BACKEND_URL=http://localhost:8010 npm run dev`, or set
`VITE_API_URL` in a `.env` file (see `.env.example`).

## Scripts

| Script | Does |
|---|---|
| `npm run dev` | dev server on port 5173 |
| `npm run build` | production build into `dist/` |
| `npm run preview` | serve the production build |
| `npm run lint` | eslint |
| `npm test` | unit tests (Vitest) |
| `npm run test:e2e` | end-to-end in a real browser: starts its own offline backend + console (run `uv sync` in `backend/` first; once: `npx playwright install chromium`) |
| `npm run format` / `npm run format:check` | prettier |

## Stack

React 19, Vite 8, MUI 9, zustand 5 for state, axios for REST, the browser's
`EventSource` for the live stream, Recharts 3 for the cost chart, React Router 7,
Vitest 5, ESLint 9 (flat config in `eslint.config.js`; ESLint 10 is not yet supported
by the React lint plugins).

## Layout

```
src/
  App.jsx                    routes: Home, Console, Workflow trace, History, Login
  services/api.js            REST client + the live event stream
  store/incidentStore.js     incident state, built from the event stream
  components/Console/        the live incident view, approval gate, timeline
  components/Flow/           the loop diagram and step-by-step trace
  components/Home/           landing page and live numbers
  components/Layout/         navigation, connection status, offline-mode banner
```

`package-lock.json` is committed so every install gets the tested versions. If you
ever regenerate it, do it from a clean tree (delete `node_modules` and the lockfile,
then `npm install`) so it records native binaries for every platform.

Setup problems: [docs/troubleshooting.md](../docs/troubleshooting.md).
