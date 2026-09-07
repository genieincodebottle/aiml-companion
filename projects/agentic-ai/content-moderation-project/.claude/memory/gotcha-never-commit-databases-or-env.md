---
name: gotcha-never-commit-databases-or-env
description: Local SQLite/Chroma state and .env secrets are gitignored; never stage or commit them.
metadata:
  type: feedback
---

The SQLite and Chroma files under `backend/databases/` are local state, and the `.env` files hold real API keys.

**This note used to claim both were gitignored. Only `.env` was.** The
databases were tracked, and `.gitignore` listed nothing but `Thumbs.db`, so
the rule read as satisfied while every `git add` re-staged a 256KB binary
diff. Adding a path to `.gitignore` does not untrack a file git already
follows. Fixed 2026-09-04: `git rm --cached` on the four DB paths plus a real
`backend/databases/` rule.

Worth knowing what was in them: 18 users, of which 7 were `e2e_test_*`
throwaways from test runs, and only `chroma.sqlite3` was tracked without its
`.bin` segment files, so a clone got a vector index with no vectors.

**Why:** Committing the databases pollutes the repo with machine-specific state, and committing `.env` leaks secrets.

**How to apply:** Both are genuinely ignored now, so this should not recur.
If you ever add a new artefact directory, check `git check-ignore -v <path>`
rather than trusting that a `.gitignore` line is doing anything - it does
nothing for a file that is already tracked. Copy config from `.env.example`.
Regenerate the databases with `python scripts/initialize_users.py`. Pairs with [[decision-model-tiering]].