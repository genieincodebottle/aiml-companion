"""Budget propagation.

The post's rule: a deadline and a token budget for the whole request are decided
at the edge and carried down with it, and every component below reads them
rather than inventing its own.

The implementation of that rule is this one object. It travels with the trace id
in the same envelope, so all three move together. An agent that knows it has six
seconds left can return its best partial answer instead of being killed with
nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised when a delegation is asked for more than the request has left."""


@dataclass
class RequestBudget:
    deadline_at: float
    tokens_total: int
    tokens_spent: int = 0
    depth: int = 0
    request_id: str = ""
    tenant_id: str = ""
    # Every delegation that read this budget, for the trace.
    delegations: list[str] = field(default_factory=list)

    @classmethod
    def start(
        cls, *, deadline_s: float, tokens: int, request_id: str, tenant_id: str
    ) -> "RequestBudget":
        return cls(
            deadline_at=time.monotonic() + deadline_s,
            tokens_total=tokens,
            request_id=request_id,
            tenant_id=tenant_id,
        )

    @property
    def remaining_s(self) -> float:
        return max(self.deadline_at - time.monotonic(), 0.0)

    @property
    def tokens_remaining(self) -> int:
        return max(self.tokens_total - self.tokens_spent, 0)

    @property
    def expired(self) -> bool:
        return self.remaining_s <= 0.0 or self.tokens_remaining <= 0

    def spend(self, tokens: int) -> None:
        """Token spend is shared across the fan-out, so three agents running in
        parallel draw down the same pool."""
        self.tokens_spent += tokens

    def timeout_for(self, agent_timeout_s: float) -> float:
        """A per-agent timeout is never allowed to exceed what the request has
        left. This is the line that makes the whole-request deadline real."""
        return max(min(agent_timeout_s, self.remaining_s), 0.0)

    def delegate(self, agent_id: str, *, agent_timeout_s: float) -> "Delegation":
        """Hand a child agent a view of what is left, not a fresh budget."""
        if self.depth >= 2:
            # Hard cap on depth. Without it, an orchestrator that can delegate
            # to an orchestrator recurses until the deadline kills it.
            raise BudgetExceeded(f"max delegation depth reached at {agent_id}")
        self.delegations.append(agent_id)
        return Delegation(
            parent=self,
            agent_id=agent_id,
            timeout_s=self.timeout_for(agent_timeout_s),
            tokens_at_entry=self.tokens_remaining,
            deadline_at_entry=self.remaining_s,
        )


@dataclass
class Delegation:
    """What a sub-agent actually receives. It is a view over the request budget,
    not a copy, so spending here is visible to everyone else immediately."""

    parent: RequestBudget
    agent_id: str
    timeout_s: float
    tokens_at_entry: int
    deadline_at_entry: float

    @property
    def remaining_s(self) -> float:
        return min(self.parent.remaining_s, self.timeout_s)

    @property
    def tokens_remaining(self) -> int:
        return self.parent.tokens_remaining

    def spend(self, tokens: int) -> None:
        self.parent.spend(tokens)

    def tool_timeout(self, requested_s: float) -> float:
        """Each tool gets its own budget, derived from the time the agent has
        left, not from a constant. A tool that cannot answer inside it returns a
        timeout the agent can reason about."""
        return max(min(requested_s, self.remaining_s), 0.0)

    def as_span_fields(self) -> dict[str, float | int]:
        return {
            "deadline_remaining_s": self.remaining_s,
            "tokens_remaining": self.tokens_remaining,
            "timeout_s": self.timeout_s,
        }
