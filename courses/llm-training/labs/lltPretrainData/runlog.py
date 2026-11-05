"""Write one Run Library JSON file (schema: LLM Training Lab plan, section 7.6) for lltPretrainData."""
import datetime, json, os, platform, sys

RUNS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "lltPretrainData"))
CPU_NAME = "AMD Ryzen 9 5900HS (8 cores, 16 threads), 32 GB RAM"


def env_versions():
    out = {"python": platform.python_version()}
    for name in ("numpy", "torch", "tokenizers", "trafilatura", "datasketch", "sklearn", "warcio", "pandas", "pyarrow"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            pass
    try:
        from importlib.metadata import version
        for dist in ("fasttext-numpy2", "fasttext-wheel", "fasttext"):
            try:
                out["fasttext"] = dist + " " + version(dist)
                break
            except Exception:
                continue
    except Exception:
        pass
    out["os"] = platform.platform()
    return out


def write_run(run_id, tier, label, config, seed, metrics, evals=None, notes=None, extra=None):
    os.makedirs(RUNS_DIR, exist_ok=True)
    rec = {"runId": run_id, "module": "lltPretrainData", "tier": tier, "label": label,
           "hardware": {"gpu": None, "count": 0, "vramGB": None, "cpu": CPU_NAME},
           "env": env_versions(), "notebookCommit": None, "config": config, "seed": seed,
           "metrics": metrics, "evals": evals or {}, "samples": [], "wallClockSec": None, "costUSD": 0,
           "priceCheckedOn": None, "date": datetime.date.today().isoformat(),
           "command": " ".join([os.path.basename(sys.executable)] + sys.argv)}
    if notes:
        rec["notes"] = notes
    if extra:
        rec.update(extra)
    run_id = run_id + os.environ.get("PTD_RUN_SUFFIX", "")
    rec["runId"] = run_id
    path = os.path.join(RUNS_DIR, run_id + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", path)
    return path
