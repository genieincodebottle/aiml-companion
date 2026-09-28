"""Regression test for the .env search order in `src/config.py`.

The real checkout is `<monorepo root>/projects/<category>/migration-engineer/backend`.
`_env_search_paths` is supposed to try, in order: `backend/.env`, `migration-engineer/.env`
("project .env"), and the true monorepo root's `.env` - four hops up from `backend`.

An earlier version stopped one hop short (three hops, landing on `projects/.env`
instead of the monorepo root), so a `GEMINI_API_KEY` placed only in the real root
`.env` - which the README and ADR 0008 both document as a supported location - was
silently never loaded. This test pins the fix by walking a synthetic directory tree
shaped exactly like the real one, rather than asserting on the index constant alone.
"""

from __future__ import annotations

from pathlib import Path

from src.config import _env_search_paths


def test_env_search_reaches_the_real_monorepo_root(tmp_path: Path) -> None:
    # <tmp>/aiml-companion/projects/agentic-ai/migration-engineer/backend
    monorepo_root = tmp_path / "aiml-companion"
    backend_dir = monorepo_root / "projects" / "agentic-ai" / "migration-engineer" / "backend"
    backend_dir.mkdir(parents=True)

    searched = _env_search_paths(backend_dir)

    assert backend_dir / ".env" in searched
    assert backend_dir.parent / ".env" in searched  # migration-engineer (project root)
    assert monorepo_root / ".env" in searched  # the actual monorepo root

    # The one-hop-short regression searched "projects/.env" (one level below the real
    # root) instead - pin that it is gone.
    projects_dir = monorepo_root / "projects"
    assert projects_dir / ".env" not in searched


def test_env_search_stays_in_bounds_near_filesystem_root(tmp_path: Path) -> None:
    # Docker-shaped case: backend_dir has fewer than 3 parents (e.g. /app -> /).
    # Must not raise IndexError.
    shallow = tmp_path / "app"
    shallow.mkdir()
    searched = _env_search_paths(shallow)
    assert searched  # at least backend_dir/.env itself
    assert all(isinstance(p, Path) for p in searched)
