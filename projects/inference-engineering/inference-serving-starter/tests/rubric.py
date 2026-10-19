"""`make check` - the definition of done, printed as six gates.

On a fresh clone one gate is green (the given engine) and five are red. That is correct.
This is informational and never blocks; `pytest -v` shows the detail behind each line.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GATES = [
    ("1 Given engine", "tests/test_given_engine.py",
     "engine_sim, incident bundles and gpu/ scripts behave (should pass on clone)"),
    ("2 Harness", "tests/test_bench.py",
     "open and closed loop, TTFT/ITL/E2E records, warm-up -> bench/dataset.py loadgen.py records.py"),
    ("3 Analysis", "tests/test_analysis.py",
     "percentiles, goodput, knee, cost, Little's law -> analysis/tables.py goodput.py knee.py cost.py fleet.py"),
    ("4 Experiments", "tests/test_experiments.py",
     "quantisation and speculation sweeps are physically sensible -> python scripts/run_experiments.py"),
    ("5 Debug lab", "tests/test_debug_lab.py",
     "three incidents diagnosed from data -> debug_lab/diagnoses/*.md"),
    ("6 Report", "tests/test_report.py",
     "one-page report with cost and fleet size -> report/REPORT.md, results/decision.json"),
]

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def run_gate(path: str):
    proc = subprocess.run([sys.executable, "-m", "pytest", path, "-q", "--no-header", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    passed = failed = 0
    for line in (proc.stdout + proc.stderr).splitlines():
        parts = line.replace(",", " ").split()
        for i, tok in enumerate(parts[:-1]):
            if tok.isdigit() and parts[i + 1].startswith("passed"):
                passed = int(tok)
            elif tok.isdigit() and parts[i + 1].startswith(("failed", "error")):
                failed += int(tok)
    return proc.returncode == 0, passed, failed


def main() -> int:
    print()
    print(f"{BOLD}Inference Serving Starter - definition of done{RESET}")
    print(f"{DIM}A fresh clone is mostly red. Green is the finish line, not the start.{RESET}\n")
    results = []
    for name, path, blurb in GATES:
        ok, p, f = run_gate(path)
        results.append(ok)
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}TODO{RESET}"
        print(f"  [{mark}] {BOLD}{name:<14}{RESET} {DIM}({p} passing, {f} failing){RESET}")
        print(f"         {DIM}{blurb}{RESET}")
    done = sum(results)
    print(f"\n  {BOLD}{done}/{len(GATES)} gates green{RESET}\n")
    print(f"  {DIM}Stretch (not gated): k8s/RUNBOOK.md, an HPA that scales engine_sim on queue depth.{RESET}")
    print(f"  {DIM}Real GPU path: gpu/README.md, where the numbers become real.{RESET}\n")
    if done < len(GATES):
        nxt = [g[0] for g, ok in zip(GATES, results) if not ok][0]
        print(f"  {DIM}Next: {nxt}{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
