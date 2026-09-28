"""The unified diff of the worker's changes - straight from real git.

The worker edits a real git worktree, so the diff is exactly what `git diff` reports.
We stage everything first (so newly created files are included) and diff against HEAD;
that staged state is what the provider then commits when the PR is approved.
"""

from __future__ import annotations

from pathlib import Path

from ..vcs import gitcmd


def compute_diff(worktree: Path) -> str:
    gitcmd.stage_all(worktree)  # excludes __pycache__/*.pyc dropped by test runs
    return gitcmd.git(worktree, "diff", "--cached", "HEAD", check=False).stdout.strip()


def changed_files(worktree: Path) -> list[str]:
    out = gitcmd.git(worktree, "diff", "--cached", "--name-only", "HEAD", check=False).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]
