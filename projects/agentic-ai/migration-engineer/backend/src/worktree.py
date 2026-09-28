"""Fixture helpers + a robust tree remover.

The real migration path uses the `vcs` providers, which clone each repo into a REAL git
worktree. `make_worktree` here is a plain directory copy of a fixture, kept only as a
lightweight helper for low-level tool/guardrail unit tests that need some files to
operate on but do not need git. `fixtures_root` is also used by the LocalGitProvider to
find the source it seeds its bare remote from.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "repos"


def force_rmtree(path: Path) -> None:
    """Remove a directory tree, clearing read-only bits (git pack files on Windows)."""
    def _on_error(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:  # pragma: no cover - best effort
            pass

    shutil.rmtree(path, onerror=_on_error)


def fixtures_root() -> Path:
    return _FIXTURES


def make_worktree(repo_id: str, job_id: str, data_dir: Path) -> Path:
    """Copy fixture repo `repo_id` into an isolated worktree and return its path."""
    src = _FIXTURES / repo_id
    if not src.is_dir():
        raise FileNotFoundError(f"no fixture repo named '{repo_id}' under {_FIXTURES}")
    dest = data_dir / "worktrees" / job_id / repo_id
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest


def cleanup_worktree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
