"""The version-control abstraction.

The single change that turns this from a demo into a real tool: the worker no longer
edits a throwaway directory copy - it edits a REAL git worktree, on a REAL feature
branch, cloned from a REAL remote, and its work is published as a REAL pull request.

Two providers implement the same contract:

  * `GitHubProvider`  - clones github.com over a token, pushes a branch, opens a PR via
                        the REST API. This is the production path.
  * `LocalGitProvider`- does the identical git operations against a local *bare* remote
                        seeded from a fixture. Real git, zero credentials, so CI and the
                        offline demo exercise the same clone/commit/push/PR code path.

The orchestrator only knows this interface, so "run against my GitHub repos" and "run
the offline demo" are the same code with a different provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class VCSError(RuntimeError):
    """A git or provider-API operation failed."""


@dataclass(frozen=True)
class RepoTarget:
    """What to migrate. `ref` meaning depends on `kind`."""

    kind: str                       # "github" | "fixture" | "local"
    ref: str                        # github: "owner/name"; fixture: folder; local: path
    name: str                       # display name (usually the repo name)
    base_branch: str = "main"
    # The command that runs the repo's tests, e.g. ["python","-m","pytest","-q"].
    # None -> the test-runner discovers and runs the repo's test_*.py scripts.
    test_command: tuple[str, ...] | None = None

    @property
    def slug(self) -> str:
        """Filesystem/id-safe identity, derived from the full ref so two targets can
        never collide (orgA/api vs orgB/api) and no path characters survive."""
        source = self.ref if self.kind == "github" else self.name
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", source).strip("-.")
        return slug or "repo"


@dataclass
class CheckoutHandle:
    """A live checkout the worker edits, plus what the provider needs to publish it."""

    target: RepoTarget
    worktree: Path
    branch: str                     # the feature branch created for this migration
    provider: "RepoProvider"
    remote_url: str = ""            # token-free display URL
    meta: dict = field(default_factory=dict)


@dataclass
class PullRequestResult:
    """The published artifact. `number`/`url` are real for GitHub; local uses a
    compare-style summary and a synthetic number."""

    branch: str
    title: str
    body: str
    kind: str                       # "github" | "local"
    number: int | None = None
    url: str = ""


class RepoProvider(Protocol):
    def checkout(self, target: RepoTarget, job_id: str, feature_branch: str) -> CheckoutHandle:
        """Clone/prepare a real git worktree on a fresh feature branch."""
        ...

    def publish(
        self, handle: CheckoutHandle, title: str, body: str, files_changed: list[str]
    ) -> PullRequestResult:
        """Commit the worker's changes, push the branch, and open a pull request."""
        ...
