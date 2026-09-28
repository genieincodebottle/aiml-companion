# Autonomous Migration Engineer - Frontend

React console for an autonomous, multi-agent code-migration system. An operator picks a
codemod job (a rule applied across a fleet of repositories), runs it, and watches an agent
fleet migrate every repo in parallel: edit files, run tests, self-review, hit a human
approval gate for risky changes, and open PRs. A guardrail layer stops the agent from doing
unsafe things (for example editing a test to make it pass), and those blocks are surfaced
prominently in the UI.

This package is the frontend only. It expects a backend on port 8000.

## Tech stack

- React 18 + Vite 5
- MUI v5 (`@mui/material`, `@mui/icons-material`, Emotion) - dark theme by default
- zustand for state
- axios for REST, native `EventSource` for the live SSE stream
- recharts is available for charts (dependency parity with the platform)
- react-router-dom for routing (Console and History)

## Getting started

```bash
npm install
npm run dev
```

The dev server runs on `http://localhost:5173`. Vite proxies `/api/*` to
`http://localhost:8000`, so the backend should be running there.

To point at a backend hosted elsewhere, copy `.env.example` to `.env` and set
`VITE_API_URL` to its base URL.

## Scripts

- `npm run dev` - start the Vite dev server on port 5173
- `npm run build` - production build
- `npm run preview` - preview the production build
- `npm run lint` - eslint
- `npm run format` / `npm run format:check` - prettier
- `npm run test` - vitest unit tests

## Troubleshooting

**`Error: Cannot find module @rollup/rollup-<platform>` on `build`/`test`.**
This is a known npm bug with optional native dependencies
([npm/cli#4828](https://github.com/npm/cli/issues/4828)). It is most often triggered by a
`package-lock.json` generated on a different OS. Fix:

```bash
rm -rf node_modules package-lock.json
npm install
```

For that reason this project does **not** commit `package-lock.json` (see `.gitignore`) so
each machine and CI resolves the correct native binary for its own platform.

## How it works

1. Console page (`/console`)
   - Left: Job Launcher lists jobs from `GET /api/jobs`. Pick one (title, repo count, rule,
     description, repo ids), optionally toggle Auto-approve PRs, then Run migration
     (`POST /api/runs`).
   - Right: the fleet board. Each repository gets its own lane, derived live from the SSE
     stream (`GET /api/runs/{id}/stream`) keyed by `repo_id`. A lane shows the repo status
     chip, step / tool / file counters, cost, a tests pass/fail chip, review verdict,
     guardrail blocks (warning-tinted), and a scrollable per-repo activity feed.
   - When any repository reaches `awaiting_approval`, an Approval Gate card appears with the
     changed files, the unified diff (additions green, deletions red), the agent summary,
     and Approve / Reject buttons (`POST /api/runs/{id}/approve` with that `repo_id`). Many
     repositories can await approval at once - each has its own independent gate.
   - When `job_summary` arrives, a results banner shows repos merged / total, the outcome
     breakdown by status, total tokens, total cost, and the execution mode.

2. History page (`/history`)
   - Table of past runs from `GET /api/runs`. Click a row to replay that run
     (`GET /api/runs/{id}`) - the same fleet board and summary render read-only, with each
     lane exposing its final diff on demand.

The AppBar shows the backend execution mode from `GET /api/health`: `live-sdk` or
`live-gemini` (green) when a real model drives the worker, or `stub` (grey) when running
the deterministic simulator.

## Backend API contract

- `GET  /api/health` -> `{ status, mode, model, sdk_available, github_configured, detail }`
- `GET  /api/jobs` -> `{ jobs: [...] }`
- `POST /api/runs` -> `{ run_id }`
- `GET  /api/runs` -> `{ runs: [...] }`
- `GET  /api/runs/{id}` -> full run object (repos + events)
- `GET  /api/runs/{id}/stream` -> SSE stream of migration events
- `POST /api/runs/{id}/approve` -> `{ ok: true }` (409 if no pending approval for that repo)

MigrationEvent shape: `{ type, ts, job_id, repo_id, agent, payload }`. Handled types:
`job_started`, `plan`, `repo_started`, `step`, `tool_call`, `edit_applied`, `tests_run`,
`review`, `awaiting_approval`, `pr_opened`, `repo_done`, `guardrail_block`, `usage`,
`job_summary`, `error`.
