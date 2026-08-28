---
name: gotcha-never-commit-databases-or-env
description: Local SQLite/Chroma state and .env secrets are gitignored; never stage or commit them.
metadata:
  type: feedback
---

The SQLite and Chroma files under `backend/databases/` are local state, and the `.env` files hold real API keys. Both are gitignored.

**Why:** Committing the databases pollutes the repo with machine-specific state, and committing `.env` leaks secrets.

**How to apply:** Before any commit, check `git status` and confirm nothing under `backend/databases/` or any `.env` is staged. Copy config from `.env.example` instead. Pairs with [[decision-model-tiering]].