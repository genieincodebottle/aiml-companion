"""MCP-style tools the fincrime squad calls.

Each tool is a plain function with a name matching the agent-card allowlist and the
policy engine's rules. In a live deployment these are MCP tool servers wrapping a
graph DB, a watchlist API, and a case-management system; here they read the synthetic
dataset. The important part is the *contract*: tools are the only way an agent touches
the world, they are named, and every call is authorized before it runs.

`ToolResult.evidence_source` is what lets the narrative agent cite its claims: every
fact an agent uses is traceable to the tool call that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import EntityGraph, get_alert


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any]
    evidence_source: str  # e.g. "graph.expand:E100" - cited by the narrative agent


class FincrimeTools:
    """The tool surface. Instantiated per workflow so it can hold graph state."""

    def __init__(self) -> None:
        self.graph = EntityGraph()

    def alert_read(self, alert_id: str) -> ToolResult:
        alert = get_alert(alert_id)
        if not alert:
            return ToolResult(ok=False, data={"error": "unknown alert"}, evidence_source=f"alert.read:{alert_id}")
        return ToolResult(ok=True, data=alert.model_dump(), evidence_source=f"alert.read:{alert_id}")

    def graph_expand(self, entity_id: str, depth: int = 2) -> ToolResult:
        root = self.graph.get(entity_id)
        paths = self.graph.paths(entity_id, depth=depth)
        neighbors = [ent for ent, _, _ in paths]
        return ToolResult(
            ok=True,
            data={
                "root": root.name if root else entity_id,
                # The customer's own file notes. Investigators read them, and they are
                # free text a customer or an upstream system can influence, which is
                # exactly why the runtime scans tool output before it reaches a prompt.
                "root_notes": root.notes if root else "",
                "related": [
                    {
                        "id": n.id,
                        "name": n.name,
                        "country": n.country,
                        "notes": n.notes,
                        # 1 = the customer sent funds to it directly; 2 = through `via`.
                        "hops": hops,
                        "via": via,
                    }
                    for n, hops, via in paths
                ],
                "sanctioned_in_network": [n.id for n in neighbors if n.on_sanctions_list],
                "pep_in_network": [n.id for n in neighbors if n.is_pep],
            },
            evidence_source=f"graph.expand:{entity_id}",
        )

    def screening_check(self, entity_id: str) -> ToolResult:
        ent = self.graph.get(entity_id)
        if not ent:
            return ToolResult(
                ok=False, data={"error": "unknown entity"}, evidence_source=f"screening.check:{entity_id}"
            )
        matched: list[str] = []
        if ent.on_sanctions_list:
            matched.append("OFAC-SDN")
        if ent.is_pep:
            matched.append("PEP-Register")
        return ToolResult(
            ok=True,
            data={
                "entity": ent.name,
                "sanctions_hit": ent.on_sanctions_list,
                "pep_hit": ent.is_pep,
                "matched_lists": matched,
            },
            evidence_source=f"screening.check:{entity_id}",
        )
