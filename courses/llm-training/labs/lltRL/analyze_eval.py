"""Paired comparisons on the same 200 GSM8K test questions from the evaluation runs (lltRL-T2-eval-*). No new measurement.

  python analyze_eval.py
Reads runs/lltRL/lltRL-T2-eval-*.json (per-question greedy correctness "right", and "nCorrect" of 8 samples where run with
pass@k) and writes lltRL-T2-analysis.json: accuracy of each model, and for each pair the difference in greedy accuracy with
a 95% paired bootstrap interval (10,000 resamples of the 200 questions, seed 0).
"""
import glob, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import write_run  # noqa: E402

RUNS = HERE.parent.parent / "runs" / "lltRL"
models = {}
for f in sorted(RUNS.glob("lltRL-T2-eval-*.json")):
    if f.stem.endswith("-job"):
        continue
    r = json.load(open(f, encoding="utf-8"))
    fin = r["evals"]["final"]
    models[f.stem.replace("lltRL-T2-eval-", "")] = {"right": np.array(fin["right"]), "nCorrect": fin.get("nCorrect"),
                                                    "greedy": fin["greedy"], "runId": f.stem,
                                                    "passk": {k: v for k, v in fin.items() if k.startswith("pass@")}}
rng = np.random.default_rng(0)
idx = rng.integers(0, 200, size=(10000, 200))


def paired(a, b):
    d = a.astype(float) - b.astype(float)
    boots = d[idx].mean(1)
    return {"diff": round(float(d.mean()), 4), "lo": round(float(np.percentile(boots, 2.5)), 4), "hi": round(float(np.percentile(boots, 97.5)), 4),
            "onlyA": int(((a == 1) & (b == 0)).sum()), "onlyB": int(((a == 0) & (b == 1)).sum())}


out = {"models": {k: {"greedy": v["greedy"], "passk": v["passk"], "runId": v["runId"]} for k, v in models.items()}, "pairs": {}}
names = list(models)
ref = [n for n in names if n in ("base", "main-s0", "main-s1")]
for a in names:
    for b in ref:
        if a != b:
            out["pairs"][a + " vs " + b] = paired(models[a]["right"], models[b]["right"])
for k, v in out["pairs"].items():
    print(k, v)
write_run("lltRL-T2-analysis", "T2", "baseline", {"source": "lltRL-T2-eval-*", "bootstrap": "10,000 paired resamples of 200 questions, seed 0"},
          0, [], evals=out, notes="Offline analysis on a CPU of recorded per-question correctness; no model was run.")
