# Architecture Decision Records

Short records of decisions that had a **real alternative**. Each one states the
context, the options considered, what was chosen, and - most importantly - what it
costs.

A decision with no downside is not a decision, it is a default. These are the
places where something was genuinely traded away, and knowing what was given up is
the difference between understanding a system and memorizing it.

| # | Decision | Status |
|---|---|---|
| [0001](0001-real-git-worktrees.md) | Edit real git worktrees, not a simulated filesystem | Accepted |
| [0002](0002-stub-worker-identical-contract.md) | Ship a stub worker that honours the identical contract | Accepted |
| [0003](0003-deterministic-tamper-gate.md) | Detect reward hacking with git, not with a model's opinion | Accepted |
| [0004](0004-score-passing-and-honest-jointly.md) | Score "passed" and "did not cheat" as one number | Accepted |
| [0005](0005-budgets-at-the-tool-boundary.md) | Enforce budgets at the tool boundary, never in the prompt | Accepted |
| [0006](0006-rulebook-as-data.md) | The migration rulebook is data the agent reads through a tool | Accepted |
| [0007](0007-vcs-provider-seam.md) | One VCS seam, so the demo and GitHub are the same code path | Accepted |
| [0008](0008-configure-before-import.md) | Set configuration before importing anything that memoizes it | Accepted |

## Format

Kept deliberately short. If an ADR needs more than a page, the decision probably
needs splitting.

```
# NNNN. Title
Status | Context | Options | Decision | Consequences (including the bad ones)
```
