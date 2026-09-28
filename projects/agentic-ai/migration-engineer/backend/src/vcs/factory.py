"""Provider selection - the one place we choose GitHub vs local git.

`github`-kind targets require a token and use the real GitHub provider; everything else
uses the local-git provider (real git against a bare remote seeded from a fixture). So
the same orchestrator code migrates your GitHub repos or runs the offline demo, decided
only by the target.
"""

from __future__ import annotations

from ..config import Settings
from .base import RepoProvider, RepoTarget, VCSError
from .github import GitHubProvider
from .local import LocalGitProvider


def get_provider(settings: Settings, target: RepoTarget) -> RepoProvider:
    if target.kind == "github":
        if not settings.github_token:
            raise VCSError(
                f"target '{target.name}' is a GitHub repo but GITHUB_TOKEN is not set. "
                "Set a fine-grained PAT (contents + pull_requests: write) to run against real repos."
            )
        return GitHubProvider(
            settings.data_dir, settings.github_token, api_base=settings.github_api_base, host=settings.github_host
        )
    return LocalGitProvider(settings.data_dir)
