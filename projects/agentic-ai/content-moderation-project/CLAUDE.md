# Content Moderation Platform - Project Rules

Loaded every session. These override default behaviour. Adapt them to your own repo when you copy this setup.

## Stack
- Backend: Python 3.12+, FastAPI, LangGraph, Google Gemini. Code in `backend/src/`.
- Frontend: React + Vite + Material UI. Code in `frontend/src/`.
- Data: SQLite and Chroma under `backend/databases/`.

## Secrets and data (never commit)
- Never commit `.env` files. Copy from `.env.example` and keep real keys local.
- Never commit anything under `backend/databases/`. It is local state, not source.
- Never paste API keys or user data into any external tool or website.

## Git (protect concurrent work)
- Never run: stash, checkout <branch>, reset --hard, switch, restore.
- Read other branches with: git show <branch>:<file>

## Before any code change, answer four questions
- Is it necessary? Is the scope minimal?
- What breaks if it ships? What is the rollback?

## Verify before you call it done
- Backend: run `pytest` from `backend/`. See `.claude/docs/TESTING.md` for the details.
- Frontend: run `npm run build` from `frontend/`.
- A failing test is a stop. Fix the cause, do not skip it.

## Enforced layer (not just prose)
- The dangerous operations above are also denied in `.claude/settings.json`, so they are blocked even if this file is skipped.
- A PreToolUse hook blocks writes to `.env` and `backend/databases/`. A PostToolUse hook formats every edited file.

## Docs (load on demand, keep this file small)
- Testing: `.claude/docs/TESTING.md`
- Add a doc under `.claude/docs/` for any topic that would otherwise bloat this file, and point to it from here.
