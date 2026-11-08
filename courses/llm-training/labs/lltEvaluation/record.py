"""Turn the kernel output of llt-eval-main into Run Library files (schema: plan section 7.6) and print every figure
the lltEvaluation page quotes, recomputed from the per-item data.

Usage: python record.py <downloaded kernel output dir> <kernel version>
Writes runs/lltEvaluation/lltEvaluation-T2-*.json. Replies to unsafe prompts were never saved by the kernel.
"""
import json, math, os, sys
import numpy as np

SRC, VER = sys.argv[1], sys.argv[2]
RUNS = os.environ.get("LLT_RUNS_DIR") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "lltEvaluation"))
os.makedirs(RUNS, exist_ok=True)
R = lambda *p: json.load(open(os.path.join(SRC, *p), encoding="utf-8"))
rec = R("eval_record.json")
ENV = dict(rec.get("env", {}), image="Kaggle GPU image 2026-11-07")
HW = {"gpu": "Tesla T4", "count": len(rec.get("gpus", [])), "vramGB": 15.6,
      "note": "page cells on GPU 0; the experiment queue (worker_eval.py) ran on GPU 1 in v1 only, and was off in v2"}
BASE, TUNED = "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B"


def wilson(k, n, z=1.959964):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def paired(a, b, n_boot=10000, seed=0):
    d = np.asarray(b, float) - np.asarray(a, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    bo = d[idx].mean(axis=1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bo, 2.5)), 4), round(float(np.percentile(bo, 97.5)), 4)]


def write(run_id, label, config, evals, metrics=None, seed=None, wall=None, extra_hw=None, kernel=None):
    out = {"runId": run_id, "module": "lltEvaluation", "tier": "T2", "label": label, "hardware": dict(HW, **(extra_hw or {})),
           "env": ENV, "notebookCommit": None, "kernel": kernel or "genieincodebottle/llt-eval-main v" + VER, "config": config, "seed": seed,
           "metrics": metrics or [], "evals": evals, "samples": [], "wallClockSec": wall, "costUSD": 0, "priceCheckedOn": None,
           "date": "2026-11-07"}
    json.dump(out, open(os.path.join(RUNS, run_id + ".json"), "w", encoding="utf-8"), indent=1)
    print("wrote", run_id)


def mean(x):
    return round(float(np.mean(x)), 4)


if __name__ == "__main__":
    print("record.py is driven by record_main.py, written after the run")
