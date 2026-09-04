#!/usr/bin/env python
"""Zero-install entry point.

`make` is not on most Windows machines, and the first thing a reader should be
able to do is watch the pipeline run. `python run.py demo` needs no API key and
no network.

    python run.py demo                  six agents, offline, no key needed
    python run.py analyse --company X   the real thing, needs GOOGLE_API_KEY
    python run.py ui                    the Streamlit app
    python run.py test                  the test suite
"""
from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _main_py(argv):
    sys.argv = ["main.py", *argv]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def cmd_demo(rest):
    """Offline. Deterministic fixtures, a planted contradiction, no key."""
    os.environ["LLM_PROVIDER"] = "offline"
    company = "Northwind Robotics"
    if rest and not rest[0].startswith("-"):
        company = rest[0]
    print(f"Offline demo. No API key and no network are used.\n"
          f"Analysing {company} with six agents.\n")
    _main_py(["--company", company, *[a for a in rest if a.startswith("-")]])


def cmd_analyse(rest):
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("No GOOGLE_API_KEY found. Get a free one at "
              "https://aistudio.google.com/apikey, or run `python run.py demo` "
              "which needs no key at all.", file=sys.stderr)
        raise SystemExit(2)
    _main_py(rest)


def cmd_ui(rest):
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", "app.py", *rest], cwd=ROOT))


def cmd_test(rest):
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", *rest], cwd=ROOT))


COMMANDS = {"demo": cmd_demo, "analyse": cmd_analyse, "analyze": cmd_analyse,
            "ui": cmd_ui, "test": cmd_test}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    COMMANDS[a.command](a.rest)
