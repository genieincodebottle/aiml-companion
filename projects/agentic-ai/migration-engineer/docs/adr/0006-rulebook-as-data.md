# 0006. The migration rulebook is data the agent reads through a tool

**Status:** Accepted

## Context

The system performs mechanical migrations - `datetime.utcnow()` to
`datetime.now(timezone.utc)`, and its siblings. That knowledge has to live
somewhere. It could live in the prompt, in Python, or in data the agent looks up.

## Options

1. **In the prompt.** Every new migration means a prompt edit and a re-test of
   every existing one, because prompts do not compose.
2. **In code** - a function per migration that rewrites the AST. Reliable and
   fast, but then the agent is not doing the migration, the code is; the agent is
   a wrapper around a script.
3. **As data in a rulebook**, retrieved by the agent through a
   `search_migration_rules` tool, with the actual edit left to the agent.

## Decision

Option 3. `rulebook/rules.py` holds `MigrationRule` records; `tools/rules.py`
exposes them for lookup. The rule is explicitly **guidance, not a transformation** -
it tells the agent what to look for and what good looks like, and the agent decides
how to apply it to the code in front of it.

## Consequences

**Good**

- Adding a migration is adding a record, not editing a prompt or writing a
  transformer. The existing migrations are untouched, so they do not need
  re-verifying.
- The rulebook is versionable and testable on its own, separately from any agent.
- It models the realistic shape of this task: a real migration is never purely
  mechanical, and an agent that can read guidance and adapt it is the point.

**Bad**

- Guidance is weaker than a transformer. A deterministic AST rewrite would migrate
  correctly every time; an agent following guidance sometimes will not, and the
  only thing catching that is the test suite.
- **The stub worker applies the rule mechanically**, so stub runs demonstrate the
  plumbing but not the judgement. The interesting part of this decision - can the
  agent adapt guidance to unfamiliar code? - is only exercised in live mode.
- Retrieval is lexical over a small set. At rulebook scale it would need real
  search, and the tool interface would have to change with it.
