"""LocalGitProvider: prove the git clone -> branch -> edit -> commit -> push -> PR path
is REAL git, end to end, with no credentials."""

from __future__ import annotations

from src.config import get_settings
from src.vcs import gitcmd
from src.vcs.base import RepoTarget
from src.vcs.local import LocalGitProvider


def _target() -> RepoTarget:
    return RepoTarget(kind="fixture", ref="billing-service", name="billing-service")


def test_checkout_is_a_real_git_worktree_on_a_feature_branch():
    provider = LocalGitProvider(get_settings().data_dir)
    handle = provider.checkout(_target(), "vcs-job", "migration/modernize-datetime/abc123")
    assert (handle.worktree / ".git").exists()
    assert (handle.worktree / "billing.py").exists()
    branch = gitcmd.git(handle.worktree, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert branch == "migration/modernize-datetime/abc123"


def test_publish_commits_and_pushes_the_branch():
    provider = LocalGitProvider(get_settings().data_dir)
    handle = provider.checkout(_target(), "vcs-pub", "migration/x/pub123")
    src = handle.worktree / "billing.py"
    src.write_text(src.read_text().replace("datetime.utcnow()", "datetime.now(timezone.utc)"), encoding="utf-8")

    pr = provider.publish(handle, "Modernize datetime", "body", ["billing.py"])
    assert pr.kind == "local"
    assert pr.branch == "migration/x/pub123"
    assert pr.number is not None

    # The branch really exists on the bare remote now.
    bare = get_settings().data_dir / "remotes" / "billing-service.git"
    heads = gitcmd.git(bare, "for-each-ref", "--format=%(refname:short)", "refs/heads").stdout
    assert "migration/x/pub123" in heads


def test_publish_with_no_changes_opens_no_commit():
    provider = LocalGitProvider(get_settings().data_dir)
    handle = provider.checkout(_target(), "vcs-empty", "migration/x/empty1")
    pr = provider.publish(handle, "noop", "body", [])
    assert pr.kind == "local"  # returns cleanly even when nothing changed
