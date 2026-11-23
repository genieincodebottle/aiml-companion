# 0003. Bind agent outputs to closed enums, not strings

**Status:** Accepted

## Context

The original schemas typed `RemediationProposal.action_type` as `str` with the
valid options listed in the field description, then defensively coerced anything
unrecognized to `no_op` after the fact. `VerificationResult.next_step` was likewise
a free string documented as "resolve | loop | escalate".

That works until it does not. A model that returns `"rollback"` instead of
`"rollback_deploy"`, or `"retry"` instead of `"loop"`, produces a silently wrong
decision rather than a loud failure.

## Options

1. **Free strings + validation in the loop.** Works, but the check lives far from
   the contract and is easy to forget in a new call site.
2. **Free strings + post-hoc coercion.** What the original did. Turns a decoding
   error into a silent behaviour change.
3. **Closed enums bound into the structured-output schema**, plus a validator on
   fields that must stay strings.

## Decision

Option 3.

- `ActionType` and `TelemetryTool` are `str` enums referenced directly in the
  Pydantic schemas, so `with_structured_output` constrains the decoder itself.
- `VerificationResult.next_step` and `SupervisorDecision.next_agent` keep a string
  type (for readability in the API payload) but carry a `field_validator` that
  raises on anything outside the allowed set.
- A raise is the *right* outcome: `llm.structured` retries on a validation error,
  and structured decoding is stochastic enough that a retry usually succeeds.

## Consequences

**Good**

- "Hallucinate a new action" stops being a failure mode. It becomes a decoding
  error, retried once, then surfaced.
- The enum is the single source of truth. `ROLE_ALLOWED`, `ACTION_RISK` and the
  sandbox's `VALID_ACTIONS` are all built from it, and a test asserts every role's
  allowlist is a subset of the menu - so adding an action cannot leave a gap.
- The tool registry gets the same treatment: the Diagnostician can only request a
  tool that exists, so `run_tools` never needs a "no such tool" branch in practice.

**Bad**

- Adding an action means touching the enum, the risk table, every role's allowlist
  and the sandbox. That is more friction than editing a docstring - deliberately,
  because each of those is a decision someone should make consciously.
- Enum-constrained decoding is slightly more likely to fail outright on a weak
  model than free text is, trading a silent wrong answer for a visible retry. That
  is the correct trade for a system that takes production actions, but it is a
  trade.
