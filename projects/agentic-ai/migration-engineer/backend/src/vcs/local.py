"""LocalGitProvider - real git, no credentials.

It does the exact same operations as the GitHub provider (clone a remote, branch,
commit, push), but the "remote" is a local *bare* repository seeded from a fixture. So
the offline demo and CI exercise the genuine clone -> edit -> commit -> push -> "PR"
path with zero network and zero secrets. A "pull request" here is a branch pushed to the
bare remote plus a compare-style summary.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

from ..worktree import fixtures_root, force_rmtree
from . import gitcmd
from .base import CheckoutHandle, PullRequestResult, RepoTarget, VCSError


class LocalGitProvider:
    def __init__(self, data_dir: Path) -> None:
        self._data = data_dir
        self._remotes = data_dir / "remotes"
        self._remotes.mkdir(parents=True, exist_ok=True)

    # --- provider API -------------------------------------------------------

    def checkout(self, target: RepoTarget, job_id: str, feature_branch: str) -> CheckoutHandle:
        remote = self._ensure_remote(target)
        worktree = self._data / "worktrees" / job_id / target.slug
        if worktree.exists():
            force_rmtree(worktree)
        worktree.parent.mkdir(parents=True, exist_ok=True)
        gitcmd.git(self._data, "clone", str(remote), str(worktree))
        gitcmd.git(worktree, "checkout", "-b", feature_branch)
        return CheckoutHandle(
            target=target, worktree=worktree, branch=feature_branch, provider=self,
            remote_url=f"local://{remote.name}",
        )

    def publish(
        self, handle: CheckoutHandle, title: str, body: str, files_changed: list[str]
    ) -> PullRequestResult:
        wt = handle.worktree
        gitcmd.stage_all(wt)
        if gitcmd.has_changes(wt):  # something staged to commit
            gitcmd.git(wt, "commit", "-m", title)
            gitcmd.git(wt, "push", "origin", handle.branch)
        number = self._next_number(handle)
        url = f"local://{handle.target.slug}/pull/{number} (branch {handle.branch})"
        return PullRequestResult(
            branch=handle.branch, title=title, body=body, kind="local", number=number, url=url,
        )

    # --- seeding ------------------------------------------------------------

    # Seeding runs in worker threads (checkout is dispatched via asyncio.to_thread),
    # and concurrent jobs share the same bare remote + seed staging directory. The
    # lock makes creation atomic; the ref check catches a half-initialized bare repo
    # (directory exists but `main` was never pushed) left by a crashed earlier run.
    _seed_lock = threading.Lock()

    def _ensure_remote(self, target: RepoTarget) -> Path:
        """Return a bare remote for the target, creating it once from the source."""
        bare = self._remotes / f"{target.slug}.git"
        with self._seed_lock:
            if self._is_seeded(bare):
                return bare
            if bare.exists():  # half-initialized leftover: rebuild it
                force_rmtree(bare)
            source = self._source_dir(target)
            gitcmd.init_bare(bare)
            seed = self._data / "seeds" / target.slug
            if seed.exists():
                force_rmtree(seed)
            seed.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, seed, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
            gitcmd.git(seed, "init", "--initial-branch=main")
            gitcmd.git(seed, "add", "-A")
            gitcmd.git(seed, "commit", "-m", "seed: initial import")
            gitcmd.git(seed, "push", str(bare), "main")
            return bare

    @staticmethod
    def _is_seeded(bare: Path) -> bool:
        if not bare.exists():
            return False
        proc = gitcmd.git(bare, "rev-parse", "--verify", "refs/heads/main", check=False)
        return proc.returncode == 0

    def _source_dir(self, target: RepoTarget) -> Path:
        if target.kind == "fixture":
            src = fixtures_root() / target.ref
        else:  # "local": ref is a filesystem path to a working tree
            src = Path(target.ref)
        if not src.is_dir():
            raise VCSError(f"local source not found for '{target.name}': {src}")
        return src

    def _next_number(self, handle: CheckoutHandle) -> int:
        bare = self._remotes / f"{handle.target.slug}.git"
        heads = gitcmd.git(bare, "for-each-ref", "--format=%(refname)", "refs/heads", check=False).stdout
        # main + one branch per published migration; number them from 1.
        return max(1, len([ln for ln in heads.splitlines() if ln.strip()]) - 1)
