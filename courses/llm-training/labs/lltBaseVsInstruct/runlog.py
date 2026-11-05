"""Write one Run Library JSON file (schema: LLM Training Lab plan, section 7.6).

Every figure the lltBaseVsInstruct lab page quotes comes from a file written by this helper.
"""
import datetime
import json
import os
import platform
import sys

RUNS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "runs", "lltBaseVsInstruct"))
CPU_NAME = "AMD Ryzen 9 5900HS (8 cores, 16 threads), 32 GB RAM"
# LLT_RUN_SUFFIX=-v2026-11 writes and reads the re-measured files (current stable stack) next to the originals.
SUFFIX = os.environ.get("LLT_RUN_SUFFIX", "")


def env_versions():
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "tokenizers", "huggingface_hub", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            pass
    out["cuda"] = None
    out["os"] = platform.platform()
    return out


def peak_rss_gb():
    """Peak resident memory of this process in GB (Windows peak working set, else ru_maxrss)."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak:
            return round(peak / 1e9, 2)
    except ImportError:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    except ImportError:
        return None


def read_run(run_id):
    with open(os.path.join(RUNS_DIR, run_id + SUFFIX + ".json"), encoding="utf-8") as f:
        return json.load(f)


def write_run(run_id, tier, label, config, seed, metrics, evals=None, samples=None,
              wall_clock_sec=None, notes=None):
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_id = run_id + SUFFIX
    rec = {
        "runId": run_id,
        "module": "lltBaseVsInstruct",
        "tier": tier,
        "label": label,
        "hardware": {"gpu": None, "count": 0, "vramGB": None, "cpu": CPU_NAME},
        "env": env_versions(),
        "notebookCommit": None,
        "config": config,
        "seed": seed,
        "metrics": metrics,
        "evals": evals or {},
        "samples": samples or [],
        "wallClockSec": wall_clock_sec,
        "wallClockNote": "measured while other jobs shared the CPU; not a timing figure for the page",
        "costUSD": 0,
        "priceCheckedOn": None,
        "date": datetime.date.today().isoformat(),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
    }
    if notes:
        rec["notes"] = notes
    path = os.path.join(RUNS_DIR, run_id + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("wrote", path)
    return path
