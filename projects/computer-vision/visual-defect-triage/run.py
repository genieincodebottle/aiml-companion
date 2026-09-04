#!/usr/bin/env python
"""Zero-install entry point.

`make` is not on most Windows machines, and asking a reader to install the
project before they can run it is friction with no payoff. This script puts
`src/` on the path itself, so every command below works from a fresh clone
with nothing but the dependencies installed.

    python run.py data        generate the synthetic embeddings
    python run.py pipeline    the full triage run
    python run.py demo        both of the above, in order
    python run.py slices      the slice gate, as CI runs it
    python run.py test        the test suite
    python run.py serve       the API (needs the serving extras)
"""
from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _module(name, argv=()):
    sys.argv = [name, *argv]
    runpy.run_module(name, run_name="__main__")


def cmd_data(rest):
    _module("scripts.make_synthetic_data", rest)


def cmd_pipeline(rest):
    _module("src.run_pipeline", rest)


def cmd_demo(rest):
    print(">>> generating synthetic embeddings")
    cmd_data([])
    print("\n>>> running the triage pipeline")
    cmd_pipeline([])
    print("\nartifacts/report.md, slice_report.csv, threshold_sweep.csv, "
          "reliability.csv")


def cmd_slices(rest):
    _module("scripts.check_slices", rest)


def cmd_test(rest):
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", *rest], cwd=ROOT))


def cmd_serve(rest):
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000",
         *rest], cwd=ROOT))


COMMANDS = {
    "data": cmd_data, "pipeline": cmd_pipeline, "demo": cmd_demo,
    "slices": cmd_slices, "test": cmd_test, "serve": cmd_serve,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    COMMANDS[a.command](a.rest)
