# 0002. Policy is data, evaluated per call, failing closed

**Status:** Accepted

## Context

Agents call tools. Some tools file regulatory reports, close cases, or move money.
Something has to decide, per call, whether this agent may do this thing right now -
and "the prompt said not to" is not a control.

## Options

1. **Trust the prompt.** Free. Not a control: an agent that is confused, jailbroken,
   or reading attacker-authored data is not bound by it.
2. **Hard-code checks in each tool.** Enforced, but scattered - the policy exists
   only as the union of a dozen `if` statements, and nobody can read it.
3. **A policy engine over rules-as-data**, consulted before every dispatch.

## Decision

Option 3, OPA-shaped but in-process. `policy/engine.py` holds an ordered rule list;
the runtime asks it before each tool call and gets ALLOW, REQUIRE_APPROVAL or DENY.
Three properties are deliberate:

- **Least privilege.** A tool absent from the agent card's allowlist is DENY.
- **Fail closed.** Unknown tools are DENY, never a silent allow. Side-effecting
  tools (`case.file_sar`, `case.escalate`, `case.close`) are REQUIRE_APPROVAL.
- **Auditable.** Every decision names the rule that fired, so "why was this
  allowed?" has an answer.

## Consequences

**Good**

- The policy is one readable list, versionable and testable exactly like prompts
  and agent cards. You can diff a policy change.
- Adding an agent cannot silently widen permissions - it gets a card, and the card
  is the ceiling.
- REQUIRE_APPROVAL turns the human gate into a policy outcome rather than a
  hard-coded step, so which actions need a human is configuration.

**Bad**

- **First match wins**, so rule order is load-bearing and a misordered rule is a
  silent permission change. There is no conflict detection.
- Fail-closed means false denials, and a denied agent looks broken to a user who
  cannot see the policy. That is the correct direction to err and it is still a
  support burden.
- The engine sees the tool name and its arguments, not intent. `case.close` with
  plausible arguments is indistinguishable from `case.close` for a bad reason -
  policy constrains the action space, it does not evaluate judgement.
- In-process means it governs only agents that go through this runtime. See
  [ADR 0001](0001-one-gateway-seam.md) for the same limitation on the model path.
