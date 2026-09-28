"""The per-repo pipeline core and the LangGraph batch orchestrator."""

from __future__ import annotations

import pytest

from src.config import get_settings
from src.orchestrator.graph import run_batch
from src.orchestrator.pipeline import migrate_repo_core
from src.rulebook.rules import get_job, get_rule
from src.tools.memory import MemoryStore
from src.vcs.base import RepoTarget


@pytest.mark.asyncio
async def test_single_repo_core_migrates_and_iterates():
    settings = get_settings()
    rule = get_rule("modernize-datetime")
    memory = MemoryStore(settings.data_dir, "core-job")
    # auth-service needs the two-pass fix (add the timezone import).
    target = RepoTarget(kind="fixture", ref="auth-service", name="auth-service")
    result, budget, handle, _blocks = await migrate_repo_core(
        settings, "core-job", target, rule, memory,
    )
    assert result.tests_passing is True
    assert "auth.py" in result.files_changed
    assert result.diff.strip()
    assert (handle.worktree / ".git").exists()   # a REAL git worktree
    assert budget.steps >= 4  # it iterated: replace -> fail -> add import -> pass


@pytest.mark.asyncio
async def test_batch_migrates_whole_fleet():
    summary = await run_batch(get_job("datetime-fleet"), approval_policy="auto")
    assert summary["repos_total"] == 4
    assert summary["by_outcome"].get("pr_open") == 4
    assert all(r["tests_passing"] for r in summary["repos"])
    assert all(r["tampered_with_tests"] is False for r in summary["repos"])


@pytest.mark.asyncio
async def test_dry_run_policy_does_not_open_prs():
    summary = await run_batch(get_job("datetime-fleet"), approval_policy="dry_run")
    assert summary["by_outcome"].get("pr_open") is None
    assert summary["by_outcome"].get("reviewed_not_merged") == 4
