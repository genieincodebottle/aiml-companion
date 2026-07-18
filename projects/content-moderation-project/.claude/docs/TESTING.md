# Testing (load this when you touch tests or ship a change)

This file is loaded on demand, not every session. Keeping it out of `CLAUDE.md`
keeps the always-on context small, which keeps every turn cheaper.

## Backend
- Run the end-to-end tests from `backend/`: `pytest`
- Tests live in `backend/tests/`. Add a case for every new agent behaviour or endpoint.
- Never point tests at real API keys. Mock the LLM calls.

## Frontend
- Build must pass before shipping: `npm run build` from `frontend/`
- Lint: `npm run lint`

## Before you call a change done
- The affected tests pass, and you ran them.
- A failing test is a stop. Fix the cause, do not skip it.