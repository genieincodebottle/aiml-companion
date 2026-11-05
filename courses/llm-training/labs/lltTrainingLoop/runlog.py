"""Write one Run Library JSON file (schema: LLM Training Lab plan, section 7.6) for lltTrainingLoop."""
import datetime
import json
import os
import platform
import sys

RUNS_DIR = os.environ.get("LLT_RUNS_DIR") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "lltTrainingLoop"))
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


def clean(x):
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def write_run(run_id, tier, label, config, seed, metrics, evals=None, samples=None, wall_clock_sec=None, notes=None):
    os.makedirs(RUNS_DIR, exist_ok=True)
    rec = {
        "runId": run_id, "module": "lltTrainingLoop", "tier": tier, "label": label,
        "hardware": {"gpu": None, "count": 0, "vramGB": None, "cpu": CPU_NAME},
        "env": env_versions(), "notebookCommit": None, "config": config, "seed": seed,
        "metrics": metrics, "evals": evals or {}, "samples": samples or [],
        "wallClockSec": wall_clock_sec, "costUSD": 0, "priceCheckedOn": None,
        "date": datetime.date.today().isoformat(),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
    }
    if notes:
        rec["notes"] = notes
    path = os.path.join(RUNS_DIR, run_id + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(rec), f, indent=1, ensure_ascii=False)
    print("wrote", path, flush=True)
    return path
