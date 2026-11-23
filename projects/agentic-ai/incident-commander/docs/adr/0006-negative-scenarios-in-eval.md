# 0006. Score "escalated correctly" as strictly as "resolved"

**Status:** Accepted

## Context

The original evaluation ran six scenarios, all solvable, once each, and reported
resolution rate, mean iterations and correct-first-action.

That harness cannot fail. Every scenario has a fix, the harness approves every
action, and resolution rate is ~100% by construction. Worse, it cannot detect the
most dangerous behaviour an SRE agent has: confidently "fixing" something that was
never broken, or thrashing through actions on a fault it cannot fix.

## Options

1. **Keep outcome-only metrics on solvable faults.** Flattering and uninformative.
2. **Add negative scenarios but still score "resolved = good".** Now the agent is
   *penalized* for the correct behaviour of escalating. Worse than nothing.
3. **Give every scenario an `expected_outcome` and score against it.**

## Decision

Option 3, plus the scenarios that make it meaningful:

- `Scenario.expected_outcome` is `"resolved"` or `"escalated"`.
  `unfixable_ca_expiry` expects escalation; `false_alarm` expects resolution with
  no action taken.
- The headline metric is `correct_outcome_rate`, not `resolution_rate`.
  `resolution_rate_positive_only` is reported separately so the two are never
  conflated.
- `score_trajectory` adds a `declined_to_act` check on negative scenarios, and a
  `no_harmful_action` check that fails when an action hits a healthy service.
- The sandbox marks such actions `harmful: true` rather than merely ineffective.

## Consequences

**Good**

- "Escalated when it should have" is now a measurable success, so an agent can be
  rewarded for knowing its limits - which is the behaviour that actually matters
  when it runs unattended.
- `harmful_action_rate` gives a safety number that only reaches zero if the agent
  never touches a healthy service.
- The eval can now fail, which is the entire point of an eval.

**Bad**

- The fault catalog is more work to extend: every new scenario needs its expected
  outcome thought through, and negative scenarios need the sandbox to model "no
  action helps".
- Aggregate rates now mix two populations (positive and negative scenarios), so the
  per-scenario table matters more than the headline number. The report prints
  per-scenario first for this reason.
- An agent could game `correct_outcome_rate` by escalating everything. Nothing
  currently catches that beyond `resolution_rate_positive_only` - worth watching.
