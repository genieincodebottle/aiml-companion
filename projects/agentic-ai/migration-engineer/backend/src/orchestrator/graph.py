r"""The LangGraph macro-orchestrator (batch / non-interactive).

The live `engine.py` streams the same pipeline for the UI and opens the real pull
requests; THIS module expresses it declaratively as a LangGraph `StateGraph` for the
durable, auditable, fleet-level control flow used by the CLI batch and the eval gate:

        START -> plan -> migrate -> review --(any passed?)--> gate -> aggregate -> END
                                          \--(none)--------------------/

The crucial architectural point (and a strong interview answer): the graph NODES do not
contain an agent loop. Each `migrate` step calls `migrate_repo_core`, which checks out a
real git worktree and runs a Claude Agent SDK worker whose loop is owned by the SDK
harness. LangGraph owns the "org chart" (which repo, in what order, gated how); the Agent
SDK owns "the work".

This batch path intentionally does NOT open pull requests - opening a real PR is a
human-gated action that belongs to the live engine. The graph reviews and gates so it is
safe to run in CI and evals; `outcome == "pr_open"` here means "approved, ready to open".
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from ..harness.reviewer import review as review_diff
from ..models.schemas import MigrationResult
from ..rulebook.rules import MigrationJob, get_rule
from ..tools.memory import MemoryStore
from ..vcs.base import RepoTarget
from .pipeline import migrate_repo_core


class BatchState(TypedDict, total=False):
    job_id: str
    rule_id: str
    targets: list[RepoTarget]
    approval_policy: str         # "auto" | "dry_run"
    results: list[dict]
    summary: dict


async def _plan(state: BatchState) -> BatchState:
    return {"results": []}


async def _migrate(state: BatchState) -> BatchState:
    settings = get_settings()
    rule = get_rule(state["rule_id"])
    memory = MemoryStore(settings.data_dir, state["job_id"])
    results: list[dict] = []
    for target in state["targets"]:
        result, budget, handle, blocks = await migrate_repo_core(
            settings, state["job_id"], target, rule, memory,
        )
        results.append(
            {
                "repo_id": target.slug,
                "name": target.name,
                "result": result,
                "worktree": str(handle.worktree),
                "tokens": budget.tokens,
                "cost_usd": round(budget.cost_usd, 6),
                "steps": budget.steps,
                "guardrail_blocks": blocks,
            }
        )
    return {"results": results}


async def _review(state: BatchState) -> BatchState:
    for r in state["results"]:
        result: MigrationResult = r["result"]
        r["verdict"] = review_diff(Path(r["worktree"]), result)
    return {"results": state["results"]}


def _route_after_review(state: BatchState) -> str:
    return "gate" if any(r["verdict"].approve for r in state["results"]) else "aggregate"


async def _gate(state: BatchState) -> BatchState:
    policy = state.get("approval_policy", "auto")
    for r in state["results"]:
        verdict = r["verdict"]
        result: MigrationResult = r["result"]
        if result.escalate or not verdict.approve:
            r["outcome"] = "escalated"
        elif policy == "dry_run":
            r["outcome"] = "reviewed_not_merged"
        else:  # "auto": approved, ready for a PR (the live engine opens it)
            r["outcome"] = "pr_open"
    return {"results": state["results"]}


async def _aggregate(state: BatchState) -> BatchState:
    by_outcome: dict[str, int] = {}
    for r in state["results"]:
        o = r.get("outcome", "escalated")
        by_outcome[o] = by_outcome.get(o, 0) + 1
    summary = {
        "job_id": state["job_id"],
        "rule_id": state["rule_id"],
        "repos_total": len(state["results"]),
        "by_outcome": by_outcome,
        "total_cost_usd": round(sum(r["cost_usd"] for r in state["results"]), 6),
        "repos": [
            {
                "name": r["name"],
                # When review rejects everything, the gate node is skipped entirely and
                # no outcome was stamped - default to the only outcome that can mean.
                "outcome": r.get("outcome", "escalated"),
                "tests_passing": r["result"].tests_passing,
                "files_changed": r["result"].files_changed,
                "tampered_with_tests": r["verdict"].tampered_with_tests if "verdict" in r else None,
                "steps": r["steps"],
            }
            for r in state["results"]
        ],
    }
    return {"summary": summary}


def build_graph():
    g = StateGraph(BatchState)
    g.add_node("plan", _plan)
    g.add_node("migrate", _migrate)
    g.add_node("review", _review)
    g.add_node("gate", _gate)
    g.add_node("aggregate", _aggregate)

    g.add_edge(START, "plan")
    g.add_edge("plan", "migrate")
    g.add_edge("migrate", "review")
    g.add_conditional_edges("review", _route_after_review, {"gate": "gate", "aggregate": "aggregate"})
    g.add_edge("gate", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


async def run_batch(job: MigrationJob, approval_policy: str = "auto") -> dict[str, Any]:
    graph = build_graph()
    initial: BatchState = {
        "job_id": f"batch_{job.id}_{uuid.uuid4().hex[:6]}",
        "rule_id": job.rule_id,
        "targets": list(job.targets),
        "approval_policy": approval_policy,
    }
    final = await graph.ainvoke(initial)
    return final["summary"]
