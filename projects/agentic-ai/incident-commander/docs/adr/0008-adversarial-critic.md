# 0008. A second agent argues against every proposal

**Status:** Accepted

## Context

The Remediation agent's objective is "produce a plausible corrective action".
Nothing in that objective rewards noticing that the action is wrong. Left alone, it
will confidently propose a rollback for a fault no deploy caused, and the only
thing standing between that and production is a human who has been given a
persuasive rationale and no counter-argument.

## Options

1. **Nothing.** Rely on the human gate. But an operator reading a fluent rationale
   with no dissent is being asked to spot a flaw they have no help finding.
2. **Self-critique.** Ask the Remediation agent to check its own work. Cheap, and
   models are poor at finding faults in reasoning they just produced.
3. **A separate Critic agent with the opposite objective.**

## Decision

Option 3, with three specific design choices that separate a real reviewer from a
rubber stamp:

- **The Critic is not shown the planner's confidence.** Confidence is contagious; a
  reviewer told "the planner is 0.9 sure" finds reasons to agree.
- **`strongest_objection` is required even when endorsing.** Without it a critic
  discharges its duty with "looks good", which is worth nothing. Forcing out the
  best counter-argument is what makes the review real.
- **The prompt states the asymmetry.** A wrong veto costs one loop iteration; a
  wrong endorsement costs a production outage. The default failure mode of an LLM
  reviewer is agreeableness, and it has to be argued out of it explicitly.

A veto does not end the incident. It sends the loop back to re-diagnose with the
objection recorded in history. `IC_MAX_CRITIC_VETOES` bounds that, so a stubborn
Critic and a stubborn Planner cannot argue until the budget is gone.

## Consequences

**Good**

- Every proposal reaches the human with a stated counter-argument, which changes
  the approval from "does this sound right?" to "which of these two do I believe?"
- Vague `expected_signal` values get objected to, applying pressure toward
  falsifiable predictions.
- `IC_CRITIC_ENABLED=false` makes "what does the critic actually catch?" a
  measurable experiment rather than an assertion.

**Bad**

- **One extra strong-tier model call per proposal.** On a single-iteration incident
  that is roughly +20% cost, for a benefit that only shows up on the runs where the
  planner was wrong. Whether that trade is worth it depends on the blast radius of
  a wrong action - which is exactly why it is a config flag.
- A Critic that vetoes too readily burns iterations and pushes runs toward
  escalation. `IC_MAX_CRITIC_VETOES` bounds the damage but does not fix a
  miscalibrated critic.
- Two agents can be confidently wrong in the same direction. A critic sharing the
  planner's blind spot adds cost and no safety, and nothing here detects that
  correlation.
