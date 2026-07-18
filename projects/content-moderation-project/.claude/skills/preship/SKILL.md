---
name: preship
description: Pre-ship safety gate. Reviews the current diff before shipping. Use when asked to "review before ship" or "is this safe to ship".
allowed-tools: [Read, Glob, Grep, Bash, Agent]
---

# preship

Run this before shipping any change.

1. Collect the diff with `git diff` (staged and unstaged).
2. Spawn the `reviewer` agent on the diff. For a large diff, spawn one reviewer per area (backend, frontend) and run them in parallel.
3. Collect their findings.
4. Return a single verdict: ship, or do-not-ship with the blocking issues listed.

Keep this thin. The review rules live in the `reviewer` agent and in `CLAUDE.md`, not here. Update those and this skill follows.