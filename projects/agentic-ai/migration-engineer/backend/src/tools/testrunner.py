"""Test-runner tool: actually executes the repository's own tests.

The migration is only "done" when this reports green. Running the real tests (in a
subprocess, in the worktree) is what turns the agent from a text transformer into a
verifier - it must observe test output and iterate, not just claim success.

Each fixture repo ships one or more `test_*.py` scripts that exit 0 on pass and
non-zero on fail. We run each with the current interpreter and aggregate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_TIMEOUT_S = 60


def run_tests(worktree: Path, command: tuple[str, ...] | None = None) -> dict:
    """Run the repository's tests.

    If `command` is given (e.g. ("python","-m","pytest","-q")), run it as the repo's
    real test command. Otherwise discover and run the fixture-style test_*.py scripts.
    """
    if command:
        return _run_command(worktree, list(command))

    test_files = sorted(p.name for p in worktree.glob("test_*.py"))
    if not test_files:
        return {"passed": False, "output": "no test_*.py files found", "test_files": []}

    all_passed = True
    chunks: list[str] = []
    for tf in test_files:
        try:
            proc = subprocess.run(
                [sys.executable, tf],
                cwd=str(worktree),
                capture_output=True,
                # Explicit UTF-8: on Windows text=True decodes with cp1252 and raises
                # UnicodeDecodeError on pytest's box-drawing/unicode output.
                encoding="utf-8",
                errors="replace",
                timeout=_TIMEOUT_S,
            )
            ok = proc.returncode == 0
            all_passed = all_passed and ok
            out = (proc.stdout + proc.stderr).strip()
            chunks.append(f"[{tf}] {'ok' if ok else 'FAIL'}: {out.splitlines()[-1] if out else ''}")
        except subprocess.TimeoutExpired:
            all_passed = False
            chunks.append(f"[{tf}] FAIL: timed out after {_TIMEOUT_S}s")

    return {"passed": all_passed, "output": "\n".join(chunks), "test_files": test_files}


def _run_command(worktree: Path, command: list[str]) -> dict:
    try:
        proc = subprocess.run(
            command, cwd=str(worktree), capture_output=True,
            encoding="utf-8", errors="replace", timeout=_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": f"'{' '.join(command)}' timed out after {_TIMEOUT_S}s", "command": command}
    except FileNotFoundError:
        return {"passed": False, "output": f"test command not found: {command[0]}", "command": command}
    out = (proc.stdout + proc.stderr).strip()
    tail = "\n".join(out.splitlines()[-8:])
    return {"passed": proc.returncode == 0, "output": tail, "command": command}
