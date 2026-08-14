"""Cross-platform task runner.

`make` is not installed on a stock Windows machine, and a good share of the people
using this repo are on Windows. So every task is available two ways:

    python run.py check          make check
    python run.py customer-up    make customer-up

They do the same thing. Use whichever your machine has.

No dependencies beyond the standard library, so this works before you have
installed anything.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASKS: dict[str, tuple[list[str], str]] = {
    "install": (
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Install dependencies",
    ),
    "customer-seed": (
        [sys.executable, "customer/seed.py"],
        "Regenerate the customer's messy data (deterministic)",
    ),
    "customer-up": (
        [
            sys.executable,
            "-m",
            "uvicorn",
            "customer.legacy_api.server:app",
            "--port",
            "8000",
            "--reload",
        ],
        "Start the mock legacy dispatch API on :8000",
    ),
    "check": (
        [sys.executable, "tests/rubric.py"],
        "Run the rubric - the definition of done",
    ),
    "test": (
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        "Run all tests verbosely",
    ),
    "eval": (
        [sys.executable, "eval/run_eval.py"],
        "Run the evaluation gate",
    ),
    "lint": (
        [sys.executable, "-m", "compileall", "-q", "src", "customer", "eval", "tests", "deploy"],
        "Byte-compile everything (no network, no API key)",
    ),
}


def usage() -> int:
    print()
    print("FDE Engagement Starter")
    print()
    print("  python run.py <task>")
    print()
    width = max(len(name) for name in TASKS)
    for name, (_, blurb) in TASKS.items():
        print(f"  {name:<{width}}  {blurb}")
    print()
    print("  Mac and Linux users can use `make <task>` instead. Same tasks.")
    print()
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in {"-h", "--help", "help"}:
        return usage()
    task = argv[1]
    if task not in TASKS:
        print(f"Unknown task: {task}")
        return usage() or 1
    cmd, _ = TASKS[task]
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
