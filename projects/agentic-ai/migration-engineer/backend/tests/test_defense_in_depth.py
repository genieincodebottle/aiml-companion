"""Defense-in-depth: knock out guardrail layer 1 and prove layer 2 still catches it.

This is the teaching lab (docs/EXERCISES.md) as an executable fact. With the PreToolUse
hook disabled, the worker's regex sweep also rewrites the deprecated string inside the
repository's TEST files (the hook normally blocks those writes). The tests still pass -
so an outcome-only check would ship it - but the reviewer re-derives tamper status from
git and rejects the change. Process-level verification beats outcome-level trust.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import get_settings
from src.harness.reviewer import review
from src.orchestrator.pipeline import migrate_repo_core
from src.rulebook.rules import get_rule
from src.tools.memory import MemoryStore
from src.vcs.base import RepoTarget


@pytest.mark.asyncio
async def test_reviewer_catches_tamper_when_hook_is_disabled():
    settings = dataclasses.replace(get_settings(), disable_guardrails=True)
    rule = get_rule("modernize-datetime")
    memory = MemoryStore(settings.data_dir, "lab-job")
    target = RepoTarget(kind="fixture", ref="billing-service", name="billing-service")

    result, _budget, handle, blocks = await migrate_repo_core(
        settings, "lab-job", target, rule, memory,
    )

    # Layer 1 was off: nothing got blocked, and the test file WAS modified.
    assert blocks == []
    assert any("test_" in f for f in result.files_changed)
    # The outcome looks fine (tests green) - which is exactly the trap.
    assert result.tests_passing is True

    # Layer 2 catches it: the reviewer detects the test edit and rejects.
    verdict = review(handle.worktree, result)
    assert verdict.tampered_with_tests is True
    assert verdict.approve is False


@pytest.mark.asyncio
async def test_with_hook_enabled_tests_are_never_touched():
    settings = get_settings()  # enforce=True (the default)
    rule = get_rule("modernize-datetime")
    memory = MemoryStore(settings.data_dir, "lab-job-on")
    target = RepoTarget(kind="fixture", ref="billing-service", name="billing-service")

    result, _budget, handle, blocks = await migrate_repo_core(
        settings, "lab-job-on", target, rule, memory,
    )

    assert any(b["reason"].startswith("write blocked") for b in blocks)
    assert not any("test_" in f for f in result.files_changed)
    assert review(handle.worktree, result).approve is True
