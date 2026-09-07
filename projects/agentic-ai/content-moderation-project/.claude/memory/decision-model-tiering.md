---
name: decision-model-tiering
description: Why this repo tiers models, cheap for search and verify, top only for generation.
metadata:
  type: project
---

Search and verification run on a cheap model (haiku). Generation runs on the top model (opus). Review runs on a mid model (sonnet).

**Why:** Most spend came from search and checking done by the top model. Tiering moves the bulk to cheap models and keeps the top model for the work that needs it.

**How to apply:** Every agent in `.claude/agents/` pins its own `model`. Name the agent type when you delegate, so hard work never falls back to the session model. See [[gotcha-never-commit-databases-or-env]].