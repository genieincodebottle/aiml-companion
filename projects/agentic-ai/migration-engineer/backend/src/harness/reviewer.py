"""The reviewer agent - a second, independent opinion before the human gate.

Multi-agent by design: the Migrator proposes a diff, the Reviewer critiques it. Its most
important job is adversarial - catch the classic reward hack where an agent "passes" the
tests by weakening or deleting them. It does not trust the Migrator's self-report; it
re-derives the truth from **real git**, asking "did any test file change?".

This deterministic verification is the hard gate and runs in both execution modes. In
`live-sdk` mode it can be augmented with an LLM critique for richer prose reasons; the
tamper check below is the safety net that always runs.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from ..models.schemas import MigrationResult, ReviewVerdict
from .diffing import changed_files

# A changed file matching any of these (by basename or path) is treated as a test edit.
_TEST_GLOBS = ("test_*.py", "*_test.py", "conftest.py", "*.spec.js", "*.test.js", "*_test.go")
_TEST_DIRS = ("tests/", "test/", "__tests__/")


def review(worktree: Path, result: MigrationResult) -> ReviewVerdict:
    reasons: list[str] = []

    tampered = _tests_were_modified(worktree)
    if tampered:
        reasons.append("REJECTED: the repository's tests were modified - possible reward hacking.")

    if not result.tests_passing:
        reasons.append("Tests are not passing.")

    no_change = not result.files_changed
    if no_change and result.tests_passing and not tampered:
        reasons.append("No changes were required; repository already compliant.")
        return ReviewVerdict(approve=True, confidence=0.9, reasons=reasons, tampered_with_tests=False)

    if not no_change and not result.diff.strip():
        reasons.append("Worker claimed changes but produced an empty diff.")

    approve = result.tests_passing and not tampered and bool(result.diff.strip())
    if approve:
        reasons.append("Change is minimal, tests pass, tests untouched.")
    return ReviewVerdict(
        approve=approve,
        confidence=0.9 if approve else 0.3,
        reasons=reasons,
        tampered_with_tests=tampered,
    )


def _tests_were_modified(worktree: Path) -> bool:
    for path in changed_files(worktree):
        norm = path.replace("\\", "/")
        base = norm.rsplit("/", 1)[-1]
        if any(fnmatch.fnmatch(base, g) for g in _TEST_GLOBS):
            return True
        if any(seg in f"/{norm}" for seg in (f"/{d}" for d in _TEST_DIRS)):
            return True
    return False
