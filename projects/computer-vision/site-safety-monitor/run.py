#!/usr/bin/env python
"""Zero-install entry point.

`make` is not on most Windows machines, and asking a reader to install the
project before they can run it is friction with no payoff. This script puts
`src/` on the path itself, so every command below works from a fresh clone.

    python run.py demo        simulate a shift, write artifacts/
    python run.py sweep       the confidence sweep only
    python run.py budget      the frame budget table, no simulation
    python run.py gate        the CI gate, in false alerts per shift
    python run.py test        the test suite
"""
from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_demo(rest):
    sys.argv = ["run_demo", *rest]
    runpy.run_path(str(ROOT / "scripts" / "run_demo.py"), run_name="__main__")


def cmd_sweep(rest):
    cmd_demo(["--sweep-only", *rest])


def cmd_budget(rest):
    from src.budget import modelled_report
    from src.config import settings
    b = modelled_report()
    print(f"{settings.cameras} cameras at {settings.fps} fps "
          f"= {settings.cameras * settings.fps} frames/s")
    print(f"budget            {b['budget_ms']:.2f} ms")
    print()
    for k, v in b["stages"].items():
        print(f"  {k:<10} {v:>5.1f} ms   {v / b['detected_ms'] * 100:>5.1f}%")
    print(f"  {'total':<10} {b['detected_ms']:>5.2f} ms   "
          f"<- a detected frame, which does NOT fit")
    print()
    print(f"a skipped frame   {b['skipped_ms']:.2f} ms")
    print(f"1 in {b['detect_every']} averages   {b['average_ms']:.2f} ms, "
          f"headroom {b['headroom_ms']:.2f} ms")
    print(f"inference share   {b['inference_share'] * 100:.1f}%  "
          f"=> Amdahl ceiling {b['amdahl_ceiling']:.2f}x")


def cmd_gate(rest):
    sys.argv = ["check_alerts", *rest]
    runpy.run_path(str(ROOT / "scripts" / "check_alerts.py"), run_name="__main__")


def cmd_test(rest):
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", *rest], cwd=ROOT))


COMMANDS = {"demo": cmd_demo, "sweep": cmd_sweep, "budget": cmd_budget,
            "gate": cmd_gate, "test": cmd_test}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    COMMANDS[a.command](a.rest)
