"""The policy engine - an OPA-style, in-process decision point for tool calls.

Before any agent tool call executes, the runtime asks the engine: may this agent,
in this role, call this tool with these args? The engine returns ALLOW,
REQUIRE_APPROVAL, or DENY. It is:

  - Least privilege: a tool must be on the agent's card allowlist, else DENY.
  - Fail closed: risky/side-effecting actions default to REQUIRE_APPROVAL or DENY;
    an unknown tool is DENY, never a silent allow.
  - Auditable: every decision names the rule that fired.

Policies are data (a list of rules), so they are versionable and testable exactly
like the prompts and agent cards. This is the "governance" the industry now treats
as the line between a demo and a production system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..models.schemas import AgentCard, PolicyDecision, PolicyEffect


@dataclass(frozen=True)
class Rule:
    """One policy rule. `matches` decides applicability; `effect` is the outcome."""

    name: str
    effect: PolicyEffect
    reason: str
    matches: Callable[[AgentCard, str, dict[str, Any]], bool]


def _default_rules() -> list[Rule]:
    """The platform's baseline policy. Ordered: first match wins."""
    # Tools that mutate the world or file regulatory reports always need a human.
    RISKY = {"case.file_sar", "case.escalate", "case.close"}

    return [
        Rule(
            name="deny-tool-not-on-card",
            effect=PolicyEffect.DENY,
            reason="tool is not on the agent's declared allowlist (least privilege)",
            matches=lambda card, tool, args: tool not in card.allowed_tools,
        ),
        # More specific before more general (first match wins): the high-value rule
        # must precede the blanket risky-tool rule or it can never fire - every
        # side-effecting call would be claimed by the generic rule and the $100k
        # threshold would be dead policy.
        Rule(
            name="approval-for-large-amounts",
            effect=PolicyEffect.REQUIRE_APPROVAL,
            reason="high-value case exceeds autonomous threshold",
            matches=lambda card, tool, args: bool(args.get("amount_usd", 0) >= 100_000),
        ),
        Rule(
            name="approval-for-risky-actions",
            effect=PolicyEffect.REQUIRE_APPROVAL,
            reason="side-effecting / regulatory action requires human sign-off",
            matches=lambda card, tool, args: tool in RISKY,
        ),
        Rule(
            name="allow-read-only",
            effect=PolicyEffect.ALLOW,
            reason="read-only investigative tool on allowlist",
            matches=lambda card, tool, args: not card.side_effecting,
        ),
    ]


class PolicyEngine:
    def __init__(self, rules: list[Rule] | None = None, disabled: bool = False) -> None:
        self.rules = rules if rules is not None else _default_rules()
        self.disabled = disabled

    def evaluate(self, card: AgentCard, tool: str, args: dict[str, Any] | None = None) -> PolicyDecision:
        args = args or {}
        if self.disabled:  # TEACHING FLAG path - ungoverned fleet
            return PolicyDecision(effect=PolicyEffect.ALLOW, rule="policy-disabled", reason="MC_DISABLE_POLICY=1")
        for rule in self.rules:
            if rule.matches(card, tool, args):
                return PolicyDecision(effect=rule.effect, rule=rule.name, reason=rule.reason)
        # Fail closed: nothing matched -> deny.
        return PolicyDecision(effect=PolicyEffect.DENY, rule="default-deny", reason="no rule allowed this action")
