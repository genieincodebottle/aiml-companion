"""Reviewer agent: the anti-reward-hacking tamper gate (over real git)."""

from __future__ import annotations

from pathlib import Path

from src.config import get_settings
from src.harness.diffing import compute_diff
from src.harness.reviewer import review
from src.models.schemas import MigrationResult
from src.vcs.base import RepoTarget
from src.vcs.local import LocalGitProvider


def _git_worktree(job: str) -> Path:
    provider = LocalGitProvider(get_settings().data_dir)
    target = RepoTarget(kind="fixture", ref="billing-service", name="billing-service")
    return provider.checkout(target, job, "migration/test").worktree


def test_approves_clean_migration():
    wt = _git_worktree("rev-clean")
    src = wt / "billing.py"
    src.write_text(src.read_text().replace("datetime.utcnow()", "datetime.now(timezone.utc)"), encoding="utf-8")
    diff = compute_diff(wt)  # stages the change, real git diff
    verdict = review(wt, MigrationResult(tests_passing=True, files_changed=["billing.py"], diff=diff))
    assert verdict.approve is True
    assert verdict.tampered_with_tests is False


def test_rejects_when_tests_were_tampered():
    wt = _git_worktree("rev-tamper")
    # Simulate the classic reward hack: weaken the test so it trivially passes.
    (wt / "test_billing.py").write_text("print('PASS')\n", encoding="utf-8")
    diff = compute_diff(wt)
    verdict = review(wt, MigrationResult(tests_passing=True, files_changed=["test_billing.py"], diff=diff))
    assert verdict.tampered_with_tests is True
    assert verdict.approve is False


def test_rejects_failing_tests():
    wt = _git_worktree("rev-fail")
    result = MigrationResult(tests_passing=False, files_changed=["billing.py"], diff="+x")
    assert review(wt, result).approve is False


def test_rejects_when_tests_were_deleted_outright():
    """Deletion is the strongest form of the reward hack, and nothing tested it.

    Rewriting a test file to `print('PASS')` (above) leaves the path in the
    diff, so a name-based detector catches it. Deleting the file is what an
    agent optimising purely for "tests pass" would actually do, and whether that
    is caught depends on something subtler: `stage_all` runs `git add -A`, which
    stages deletions, so `git diff --cached --name-only HEAD` still lists the
    path. Had it been `git add .` on an older git, the deletion would have been
    invisible and the hack would have sailed through as a clean run with no
    files changed.
    """
    wt = _git_worktree("rev-delete")
    (wt / "test_billing.py").unlink()
    diff = compute_diff(wt)

    verdict = review(wt, MigrationResult(tests_passing=True, files_changed=[], diff=diff))
    assert verdict.tampered_with_tests is True
    assert verdict.approve is False


def test_rejects_when_a_test_is_deleted_alongside_a_real_change():
    """The realistic shape: a genuine edit plus a quietly removed failing test."""
    wt = _git_worktree("rev-delete-mixed")
    src = wt / "billing.py"
    src.write_text(src.read_text().replace("datetime.utcnow()", "datetime.now(timezone.utc)"), encoding="utf-8")
    (wt / "test_billing.py").unlink()
    diff = compute_diff(wt)

    verdict = review(wt, MigrationResult(tests_passing=True, files_changed=["billing.py"], diff=diff))
    assert verdict.tampered_with_tests is True
    assert verdict.approve is False


def test_a_no_op_repo_is_approved_without_being_called_tampered():
    """Control: the detector must not fire on a repo that needed no change."""
    wt = _git_worktree("rev-noop")
    verdict = review(wt, MigrationResult(tests_passing=True, files_changed=[], diff=""))
    assert verdict.tampered_with_tests is False
    assert verdict.approve is True
