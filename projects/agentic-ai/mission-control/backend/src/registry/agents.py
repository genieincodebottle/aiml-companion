"""The agent registry - A2A-style capability cards.

Every agent that runs on the platform declares itself with an `AgentCard`: its role,
the model tier it needs, the tools it may request, and whether it is side-effecting.
The registry is the single place the platform learns what agents exist and what
they're permitted to do. The policy engine reads these cards; the identity broker
mints tokens scoped to them; the runtime enforces them.
"""

from __future__ import annotations

from ..models.schemas import AgentCard, ModelTier

# The fincrime tenant's squad, declared to the platform. Read-only investigators are
# cheap-tier and side-effect-free; the narrative + critic are premium; only the
# supervisor may request the risky, human-gated disposition tools.
_FINCRIME_SQUAD: list[AgentCard] = [
    AgentCard(
        name="fincrime.triage",
        role="triage",
        description="Classifies alert severity and actionability.",
        model_tier=ModelTier.CHEAP,
        allowed_tools=["alert.read"],
    ),
    AgentCard(
        name="fincrime.graph",
        role="graph_investigator",
        description="Traverses the entity network for suspicious structure.",
        model_tier=ModelTier.PREMIUM,
        allowed_tools=["graph.expand", "alert.read"],
    ),
    AgentCard(
        name="fincrime.screening",
        role="screening",
        description="Screens entities against sanctions / PEP watchlists.",
        model_tier=ModelTier.CHEAP,
        allowed_tools=["screening.check"],
    ),
    AgentCard(
        name="fincrime.narrative",
        role="narrative",
        description="Drafts the evidence-cited SAR narrative.",
        model_tier=ModelTier.PREMIUM,
        allowed_tools=["graph.expand", "screening.check"],
    ),
    AgentCard(
        name="fincrime.critic",
        role="critic",
        description="QA-reviews the narrative for citation completeness.",
        model_tier=ModelTier.PREMIUM,
        allowed_tools=[],
    ),
    AgentCard(
        name="fincrime.supervisor",
        role="supervisor",
        description="Plans the investigation and requests final disposition.",
        model_tier=ModelTier.PREMIUM,
        allowed_tools=["case.file_sar", "case.escalate", "case.close"],
        side_effecting=True,
    ),
]


class AgentRegistry:
    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}
        for card in _FINCRIME_SQUAD:
            self.register(card)

    def register(self, card: AgentCard) -> None:
        self._cards[card.name] = card

    def get(self, name: str) -> AgentCard:
        if name not in self._cards:
            raise KeyError(f"no agent card registered for '{name}'")
        return self._cards[name]

    def list_cards(self) -> list[AgentCard]:
        return list(self._cards.values())
