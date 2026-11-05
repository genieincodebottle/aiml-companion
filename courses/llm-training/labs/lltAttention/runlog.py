"""Write one Run Library JSON file (schema: LLM Training Lab plan, section 7.6) for lltAttention."""
import datetime
import json
import os
import platform
import sys

RUNS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "runs", "lltAttention"))
CPU_NAME = "AMD Ryzen 9 5900HS (8 cores, 16 threads), 32 GB RAM"


def env_versions():
    out = {"python": platform.python_version()}
    for name in ("torch", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            pass
    out["cuda"] = None
    out["os"] = platform.platform()
    return out


def write_run(run_id, tier, label, config, seed=None, metrics=None, evals=None, samples=None,
              wall_clock_sec=None, notes=None, hardware=None, env=None, stdout=None):
    os.makedirs(RUNS_DIR, exist_ok=True)
    rec = {
        "runId": run_id,
        "module": "lltAttention",
        "tier": tier,
        "label": label,
        "hardware": hardware or {"gpu": None, "count": 0, "vramGB": None, "cpu": CPU_NAME},
        "env": env or env_versions(),
        "notebookCommit": None,
        "config": config,
        "seed": seed,
        "metrics": metrics or [],
        "evals": evals or {},
        "samples": samples or [],
        "wallClockSec": wall_clock_sec,
        "costUSD": 0,
        "priceCheckedOn": None,
        "date": datetime.date.today().isoformat(),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
    }
    if stdout is not None:
        rec["stdout"] = stdout
    if notes:
        rec["notes"] = notes
    path = os.path.join(RUNS_DIR, run_id + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("wrote", path)
    return path
