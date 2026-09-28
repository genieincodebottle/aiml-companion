"""Low-level real git, run as a subprocess.

Thin, safe wrapper: explicit args (never a shell string), a timeout, a fixed committer
identity so commits work in CI, and token redaction in any surfaced error so a
credential in a remote URL never lands in a log or an exception message.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .base import VCSError

_TIMEOUT_S = 120
# Committer identity for automated commits (overridable via env in the provider).
_BOT_NAME = "migration-engineer[bot]"
_BOT_EMAIL = "migration-engineer@users.noreply.github.com"

_TOKEN_IN_URL = re.compile(r"://[^@/]*@")


def _redact(text: str) -> str:
    return _TOKEN_IN_URL.sub("://***@", text or "")


def _git_env() -> dict[str, str]:
    """Environment for every git call, with Windows long paths switched on.

    Worktrees and bare remotes live under the data dir, and git's object paths add
    about 80 characters below it. A clone in a deep folder (Documents, OneDrive, a
    nested workspace) then crosses Windows' 260-character limit and git fails with
    "Filename too long". `core.longpaths` lifts that limit in Git for Windows, and
    other platforms ignore it. It is passed through GIT_CONFIG_* rather than `-c`
    because a local push starts a second git process to receive the objects, and
    only the environment reaches that process.
    """
    env = dict(os.environ)
    try:
        n = int(env.get("GIT_CONFIG_COUNT") or 0)
    except ValueError:
        # Not ours to fix: git reports a malformed count itself, with a clearer message.
        return env
    env[f"GIT_CONFIG_KEY_{n}"] = "core.longpaths"
    env[f"GIT_CONFIG_VALUE_{n}"] = "true"
    env["GIT_CONFIG_COUNT"] = str(n + 1)
    return env


def git(cwd: Path | str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run one git command. Raises VCSError (with tokens redacted) on failure."""
    cmd = [
        "git",
        "-c", f"user.name={_BOT_NAME}",
        "-c", f"user.email={_BOT_EMAIL}",
        "-c", "commit.gpgsign=false",
        "-c", "core.autocrlf=false",
        *args,
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, env=_git_env(),
            # Explicit UTF-8: cp1252 (the Windows default for text=True) raises on
            # non-ASCII bytes in filenames, commit messages, or diff content.
            encoding="utf-8", errors="replace", timeout=_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise VCSError(f"git {args[0]} timed out after {_TIMEOUT_S}s") from exc
    except FileNotFoundError as exc:  # git not installed
        raise VCSError("git is not installed or not on PATH") from exc
    if check and proc.returncode != 0:
        raise VCSError(_redact(f"git {args[0]} failed ({proc.returncode}): {proc.stderr.strip()}"))
    return proc


# Pathspecs that keep build artefacts out of every staged diff and commit: without
# them, running the repo's tests drops __pycache__/*.pyc into the worktree and
# `git add -A` would ship compiled bytecode into the reviewed diff and the real PR.
ADD_EXCLUDES = (":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.pyc")


def stage_all(cwd: Path) -> None:
    """`git add -A` minus bytecode/build artefacts."""
    git(cwd, "add", "-A", "--", ".", *ADD_EXCLUDES)


def init_bare(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "--bare", "--initial-branch=main")


def current_commit(cwd: Path) -> str:
    return git(cwd, "rev-parse", "HEAD").stdout.strip()


def has_changes(cwd: Path) -> bool:
    """True when something is STAGED to commit. `status --porcelain` would also count
    untracked artefacts (e.g. excluded .pyc files) and trigger empty commits."""
    return bool(git(cwd, "diff", "--cached", "--name-only", "HEAD", check=False).stdout.strip())


def diff_stat(cwd: Path, base: str) -> str:
    return git(cwd, "diff", "--stat", f"{base}..HEAD", check=False).stdout.strip()
