"""The squad's agent step functions.

Each function is a thin agent: it gathers evidence via governed tool calls, then makes
one bounded, schema-bound model call through the gateway. No agent talks to a provider
or a tool directly - everything goes through the `WorkflowContext`, so the platform's
guarantees (budget, policy, identity, injection scan, tracing) hold for every step.
"""

from __future__ import annotations

import json

from ...models.schemas import (
    CritiqueResult,
    GraphFinding,
    NarrativeDraft,
    ScreeningResult,
    TriageResult,
)
from ...runtime.workflow import WorkflowContext
from .tools import FincrimeTools


def triage(ctx: WorkflowContext) -> None:
    alert = ctx.state["alert"]
    user = f"Alert: {json.dumps(alert)}"
    result = ctx.call_model("fincrime.triage", user, TriageResult, "fincrime.triage")
    ctx.state["triage"] = result.model_dump()


def graph_investigate(ctx: WorkflowContext) -> None:
    tools: FincrimeTools = ctx.state["_tools"]
    entity_id = ctx.state["alert"]["entity_id"]
    expansion = ctx.call_tool("fincrime.graph", "graph.expand", tools.graph_expand, entity_id=entity_id, depth=2)
    ctx.state["graph_raw"] = expansion.data
    user = f"Entity network for {entity_id}: {json.dumps(expansion.data)}"
    finding = ctx.call_model("fincrime.graph", user, GraphFinding, "fincrime.graph")
    ctx.state["graph_finding"] = finding.model_dump()


def screening(ctx: WorkflowContext) -> None:
    tools: FincrimeTools = ctx.state["_tools"]
    entity_id = ctx.state["alert"]["entity_id"]
    # Screen the alerted entity and any sanctioned/PEP entities the graph surfaced.
    to_screen = [entity_id]
    graph_raw = ctx.state.get("graph_raw", {})
    to_screen += graph_raw.get("sanctioned_in_network", [])
    to_screen += graph_raw.get("pep_in_network", [])
    checks = []
    for eid in dict.fromkeys(to_screen):  # de-dup, keep order
        res = ctx.call_tool("fincrime.screening", "screening.check", tools.screening_check, entity_id=eid)
        checks.append(res.data)
    user = f"Screening evidence: {json.dumps(checks)}"
    result = ctx.call_model("fincrime.screening", user, ScreeningResult, "fincrime.screening")
    ctx.state["screening"] = result.model_dump()
    ctx.state["screening_raw"] = checks


def narrative(ctx: WorkflowContext) -> None:
    # Derive the load-bearing signals from real evidence (graph + screening), then
    # state them explicitly so the drafter's grounding is unambiguous and auditable.
    graph_raw = ctx.state.get("graph_raw", {})
    screen = ctx.state.get("screening", {})
    sanctioned = bool(graph_raw.get("sanctioned_in_network")) or bool(screen.get("sanctions_hit"))
    pep = bool(graph_raw.get("pep_in_network")) or bool(screen.get("pep_hit"))
    structuring = ctx.state["alert"].get("alert_type") == "structuring" or "fan-out" in json.dumps(graph_raw)
    findings = {
        "alert": ctx.state["alert"],
        "triage": ctx.state.get("triage"),
        "graph_finding": ctx.state.get("graph_finding"),
        "screening": screen,
    }
    signals = (
        f"SIGNALS sanctions={str(sanctioned).lower()} pep={str(pep).lower()} structuring={str(structuring).lower()}"
    )
    user = f"{signals}\nFINDINGS: {json.dumps(findings)}"
    draft = ctx.call_model("fincrime.narrative", user, NarrativeDraft, "fincrime.narrative")
    ctx.state["narrative"] = draft.model_dump()


def critic(ctx: WorkflowContext) -> None:
    draft = ctx.state["narrative"]
    # A structural check the critic's judgement must agree with: every claim citable.
    every_cited = len(draft.get("evidence", [])) >= len(draft.get("claims", []))
    user = f"Narrative under review: {json.dumps(draft)}\nStructural check: every_claim_cited={every_cited}"
    result = ctx.call_model("fincrime.critic", user, CritiqueResult, "fincrime.critic")
    ctx.state["critique"] = result.model_dump()
    if not result.approved:
        # A failed QA review does not autonomously dispose - it escalates to a human.
        ctx.state["force_escalate"] = True
