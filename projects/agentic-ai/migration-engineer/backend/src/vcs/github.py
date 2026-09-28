"""GitHubProvider - the production path.

Clones a real github.com repository over a token, creates a feature branch, and after
the worker's changes are approved, commits, pushes, and opens a REAL pull request via
the GitHub REST API. Uses the stdlib for the API call so there is no extra runtime
dependency.

Auth: a fine-grained PAT or a GitHub App installation token with `contents:write` and
`pull_requests:write` on the target repos. The token is injected into the clone URL and
kept out of logs (git errors are redacted). Hardening (Phase 2): use short-lived App
installation tokens and a git credential helper instead of a URL-embedded token.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from ..worktree import force_rmtree
from . import gitcmd
from .base import CheckoutHandle, PullRequestResult, RepoTarget, VCSError

_API = "https://api.github.com"
_GIT = "github.com"


class GitHubProvider:
    def __init__(self, data_dir: Path, token: str, api_base: str = _API, host: str = _GIT) -> None:
        if not token:
            raise VCSError("GitHubProvider requires a token")
        self._data = data_dir
        self._token = token
        self._api = api_base.rstrip("/")
        self._host = host

    # --- provider API -------------------------------------------------------

    def checkout(self, target: RepoTarget, job_id: str, feature_branch: str) -> CheckoutHandle:
        owner, repo = self._split(target.ref)
        auth_url = f"https://x-access-token:{self._token}@{self._host}/{owner}/{repo}.git"
        worktree = self._data / "worktrees" / job_id / target.slug
        if worktree.exists():
            force_rmtree(worktree)
        worktree.parent.mkdir(parents=True, exist_ok=True)
        gitcmd.git(
            self._data, "clone", "--depth", "1", "--branch", target.base_branch, auth_url, str(worktree)
        )
        gitcmd.git(worktree, "checkout", "-b", feature_branch)
        return CheckoutHandle(
            target=target, worktree=worktree, branch=feature_branch, provider=self,
            remote_url=f"https://{self._host}/{owner}/{repo}",
            meta={"owner": owner, "repo": repo},
        )

    def publish(
        self, handle: CheckoutHandle, title: str, body: str, files_changed: list[str]
    ) -> PullRequestResult:
        wt = handle.worktree
        gitcmd.stage_all(wt)
        if not gitcmd.has_changes(wt):
            return PullRequestResult(
                branch=handle.branch, title=title, body=body, kind="github",
                url=f"{handle.remote_url} (no changes; no PR opened)",
            )
        gitcmd.git(wt, "commit", "-m", title)
        gitcmd.git(wt, "push", "origin", handle.branch)
        pr = self._open_pr(handle, title, body)
        return PullRequestResult(
            branch=handle.branch, title=title, body=body, kind="github",
            number=pr.get("number"), url=pr.get("html_url", handle.remote_url),
        )

    # --- GitHub REST --------------------------------------------------------

    def _open_pr(self, handle: CheckoutHandle, title: str, body: str) -> dict:
        owner, repo = handle.meta["owner"], handle.meta["repo"]
        payload = json.dumps(
            {"title": title, "head": handle.branch, "base": handle.target.base_branch, "body": body}
        ).encode()
        req = urllib.request.Request(
            f"{self._api}/repos/{owner}/{repo}/pulls",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "migration-engineer",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise VCSError(f"GitHub PR creation failed ({exc.code}): {detail}") from None
        except urllib.error.URLError as exc:
            raise VCSError(f"GitHub API unreachable: {exc.reason}") from None

    @staticmethod
    def _split(ref: str) -> tuple[str, str]:
        parts = ref.strip("/").split("/")
        if len(parts) != 2:
            raise VCSError(f"github target must be 'owner/name', got '{ref}'")
        return parts[0], parts[1]
