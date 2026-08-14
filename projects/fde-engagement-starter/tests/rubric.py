"""`make check` - the definition of done, printed as a list.

Run this on a fresh clone and almost everything is red. That is correct and it is
the point: you are being handed a customer and a specification, not a tutorial.

This is not a replacement for `pytest -v`. It is the view you want at 6pm on day
three when the question is "what is actually left".
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GATES: list[tuple[str, str, str]] = [
    (
        "Environment",
        "tests/test_customer_env.py",
        "the customer environment loads (this one should pass on clone)",
    ),
    (
        "Ingest",
        "tests/test_ingest.py",
        "their exports load with nothing silently dropped -> src/ingest/loader.py",
    ),
    (
        "MCP auth",
        "tests/test_mcp_auth.py",
        "read and write are separate scopes -> src/mcp_server/auth.py",
    ),
    (
        "Audit",
        "tests/test_audit.py",
        "no tool call completes unlogged -> src/mcp_server/audit.py",
    ),
    (
        "Retrieval",
        "tests/test_retrieval.py",
        "hybrid beats the baseline, measured -> src/retrieval/hybrid.py",
    ),
]

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_gate(path: str) -> tuple[bool, int, int]:
    """Run one test file. Returns (passed, n_passed, n_failed)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    passed = failed = 0
    for line in (proc.stdout + proc.stderr).splitlines():
        parts = line.replace(",", " ").split()
        for i, token in enumerate(parts[:-1]):
            if not token.isdigit():
                continue
            label = parts[i + 1]
            if label.startswith("passed"):
                passed = int(token)
            elif label.startswith(("failed", "error")):
                failed += int(token)
    return proc.returncode == 0, passed, failed


def main() -> int:
    print()
    print(f"{BOLD}FDE Engagement Starter - definition of done{RESET}")
    print(f"{DIM}A fresh clone is mostly red. Green is the finish line, not the start.{RESET}")
    print()

    results = []
    for name, path, blurb in GATES:
        ok, passed, failed = run_gate(path)
        results.append(ok)
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}TODO{RESET}"
        counts = f"{DIM}({passed} passing, {failed} failing){RESET}" if (passed or failed) else ""
        print(f"  [{mark}] {BOLD}{name:<12}{RESET} {counts}")
        print(f"         {DIM}{blurb}{RESET}")
    print()

    done = sum(1 for r in results if r)
    total = len(results)
    print(f"  {BOLD}{done}/{total} gates green{RESET}")
    print()
    print(f"  {DIM}Then the evaluation gate: `python run.py eval` (or `make eval`).{RESET}")
    print(f"  {DIM}It needs 30 golden cases across 5 failure modes, built from{RESET}")
    print(f"  {DIM}customer/tickets.jsonl.{RESET}")
    print()

    if done < total:
        print(f"  {DIM}Next: {[n for n, ok in zip([g[0] for g in GATES], results) if not ok][0]}{RESET}")
        print()
    return 0  # informational, never blocks


if __name__ == "__main__":
    raise SystemExit(main())
