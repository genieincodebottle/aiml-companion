"""Calibrate the XSTest string-match refusal rule against the XSTest human labels.

Reads the five labelled completion files from github.com/paul-rottger/xstest (model_completions/)
and writes runs/lltEvaluation/lltEvaluation-t0-refusal-calibration.json.
Usage: python -I calibrate_refusal.py <dir with c_<model>.csv> <runs dir>
"""
import json, os, platform, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refusal import strmatch_refusal, human_refusal
from stats import wilson


def main(src, runs):
    rows = []
    per_model = {}
    for m in ["gpt4", "llama2new", "llama2orig", "mistralguard", "mistralinstruct"]:
        df = pd.read_csv(os.path.join(src, f"c_{m}.csv"))
        df = df[df.final_label.notna()]
        h = df.final_label.map(human_refusal).to_numpy()
        s = df.completion.fillna("").map(strmatch_refusal).to_numpy()
        per_model[m] = {"n": int(len(df)), "humanRefusalRate": round(float(h.mean()), 4),
                        "strmatchRefusalRate": round(float(s.mean()), 4), "agreement": round(float((h == s).mean()), 4)}
        rows.append((h, s))
    h = np.concatenate([r[0] for r in rows]); s = np.concatenate([r[1] for r in rows])
    tp = int((h & s).sum()); fp = int((~h & s).sum()); fn = int((h & ~s).sum()); tn = int((~h & ~s).sum())
    n = len(h)
    out = {
        "runId": "lltEvaluation-t0-refusal-calibration", "module": "lltEvaluation", "tier": "T0",
        "label": "baseline",
        "env": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "config": {"classifier": "XSTest classify_completions_strmatch.py prefix rule",
                   "humanRefusal": "final_label 2_full_refusal or 3_partial_refusal",
                   "source": "github.com/paul-rottger/xstest model_completions, 5 models x 450 prompts"},
        "evals": {
            "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "agreement": [round((tp + tn) / n, 4), *[round(x, 4) for x in wilson(tp + tn, n)]],
            "precision": [round(tp / (tp + fp), 4), *[round(x, 4) for x in wilson(tp, tp + fp)]],
            "recall": [round(tp / (tp + fn), 4), *[round(x, 4) for x in wilson(tp, tp + fn)]],
            "humanRefusalRate": round(float(h.mean()), 4), "strmatchRefusalRate": round(float(s.mean()), 4),
            "perModel": per_model,
        },
        "date": "2026-11-05",
    }
    with open(os.path.join(runs, out["runId"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["evals"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
