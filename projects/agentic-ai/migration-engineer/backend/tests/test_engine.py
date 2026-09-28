"""The live streaming engine: SSE replay + fan-out + human approval gate."""

from __future__ import annotations

import pytest

from src.models.state import EventType
from src.orchestrator.engine import MigrationManager


@pytest.mark.asyncio
async def test_stream_fanout_with_human_approval():
    mgr = MigrationManager()
    state = mgr.trigger("datetime-fleet", auto_approve=False)

    seen: set[str] = set()
    merged = 0
    async for ev in mgr.subscribe(state.job_id):
        seen.add(ev.type.value)
        if ev.type == EventType.AWAITING_APPROVAL:
            # a human approves each repo's PR
            assert mgr.resolve_approval(state.job_id, ev.repo_id, "approve", "lgtm") is True
        if ev.type == EventType.JOB_SUMMARY:
            merged = ev.payload["repos_merged"]

    assert "guardrail_block" in seen        # the tests were protected from the worker
    assert "awaiting_approval" in seen        # the human gate fired
    assert merged == 4
    assert mgr.get(state.job_id).status.value == "done"


@pytest.mark.asyncio
async def test_rejecting_a_pr_marks_repo_rejected():
    mgr = MigrationManager()
    state = mgr.trigger("datetime-fleet", auto_approve=False)

    rejected_repo = None
    async for ev in mgr.subscribe(state.job_id):
        if ev.type == EventType.AWAITING_APPROVAL:
            if rejected_repo is None:
                rejected_repo = ev.repo_id
                mgr.resolve_approval(state.job_id, ev.repo_id, "reject", "not now")
            else:
                mgr.resolve_approval(state.job_id, ev.repo_id, "approve", "")

    repo = mgr.get(state.job_id).repo(rejected_repo)
    assert repo.status.value == "rejected"
