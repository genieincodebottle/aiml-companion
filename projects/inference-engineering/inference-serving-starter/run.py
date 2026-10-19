"""Cross-platform task runner (`make` is not on a stock Windows machine).

    python run.py check          make check
    python run.py serve          make serve
    python run.py experiments    make experiments

Standard library only, so it works before you have installed anything.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

TASKS = {
    "install": ([PY, "-m", "pip", "install", "-r", "requirements.txt"], "Install dependencies"),
    "serve": ([PY, "-m", "engine_sim", "--gpu", "l4", "--quant", "bf16", "--port", "8100"],
              "Start engine_sim (Llama-3.1-8B bf16 on an L4) on :8100"),
    "check": ([PY, "tests/rubric.py"], "Run the six gates - the definition of done"),
    "test": ([PY, "-m", "pytest", "tests/", "-v", "--tb=short"], "Run all tests verbosely"),
    "experiments": ([PY, "scripts/run_experiments.py"], "Run the quantisation and speculation sweeps"),
    "incidents": ([PY, "debug_lab/make_incidents.py"], "Regenerate the incident bundles (deterministic)"),
    "diagnose": ([PY, "debug_lab/check_diagnosis.py"], "Check your three diagnoses"),
    "lint": ([PY, "-m", "compileall", "-q", "engine_sim", "bench", "analysis", "debug_lab", "tests", "scripts"],
             "Byte-compile everything"),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        print("tasks:")
        for k, (_, d) in TASKS.items():
            print(f"  {k:<12} {d}")
        return 0 if len(sys.argv) < 2 else 2
    return subprocess.call(TASKS[sys.argv[1]][0], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
