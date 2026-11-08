"""Run Library files that do not come from llt-eval-main v2: the T1 CPU run, LightEval 0.13.0 (from llt-eval-main v1),
the test kernels' failures (llt-eval-smoke v1 and v2, llt-eval-main v1) and the experiment queue of llt-eval-main v1.

Usage: python record_extra.py <T1 work dir> <kaggle out dir>
"""
import glob, json, os, platform, sys
import numpy as np

T1, OUT = sys.argv[1], sys.argv[2]
RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "lltEvaluation"))
mean = lambda x: round(float(np.mean(x)), 4)


def wilson(k, n, z=1.959964):
    import math
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def write(run_id, tier, label, hw, env, config, evals, kernel=None):
    rec = {"runId": run_id, "module": "lltEvaluation", "tier": tier, "label": label, "hardware": hw, "env": env,
           "notebookCommit": None, "kernel": kernel, "config": config, "seed": 0, "metrics": [], "evals": evals,
           "samples": [], "wallClockSec": None, "costUSD": 0, "priceCheckedOn": None, "date": "2026-11-07"}
    json.dump(rec, open(os.path.join(RUNS, run_id + ".json"), "w", encoding="utf-8"), indent=1)
    print("wrote", run_id)


# T1: page cells c01 to c05 on a CPU, SmolLM2-360M pair. Timings are filled by measure_timing.py on a quiet machine.
if os.path.exists(os.path.join(T1, "results", "harness.json")):
    import torch, transformers, lm_eval, datasets
    H = json.load(open(os.path.join(T1, "results", "harness.json"), encoding="utf-8"))
    F = json.load(open(os.path.join(T1, "results", "formats.json"), encoding="utf-8"))
    S = json.load(open(os.path.join(T1, "results", "scorer.json"), encoding="utf-8"))
    P = json.load(open(os.path.join(T1, "results", "paired.json"), encoding="utf-8"))
    panel = {m: {t: {"n": len(d["acc"]), "acc": mean(d["acc"]), "wilson": wilson(int(sum(d["acc"])), len(d["acc"])),
                     "accNorm": mean(d["acc_norm"]) if d.get("acc_norm") else None} for t, d in H[m]["tasks"].items()} for m in H}
    env = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "lm_eval": "0.4.13", "datasets": datasets.__version__, "numpy": np.__version__, "os": platform.platform()}
    write("lltEvaluation-T1-cpu-pair", "T1", "baseline", {"gpu": None, "count": 0, "cpu": "AMD Ryzen 9 5900HS, 4 threads"}, env,
          {"models": list(H), "limit": 200, "mmluPerSubject": 5, "dtype": "float32", "cells": "c01-c05",
           "timing": "machine shared with other jobs; timings come from measure_timing.py"},
          {"panel": panel, "scorer": {k: S[k] for k in ["n", "maxGap", "meanGap"]},
           "formats": {m: {f: mean(v) for f, v in d.items()} for m, d in F.items()}, "paired": P})

# LightEval 0.13.0 and the failures of the test kernels.
v1 = json.load(open(os.path.join(OUT, "llt-eval-main-v1", "eval_record.json"), encoding="utf-8"))
le = {k: ({"rc": v["rc"], "sec": v["sec"], "tail": v["tail"][-700:]} if isinstance(v, dict) and "rc" in v else v) for k, v in v1["lighteval"].items()}
write("lltEvaluation-T2-lighteval", "T2", "broken:lighteval-0.13.0", {"gpu": "Tesla T4", "count": 2, "vramGB": 15.6}, v1.get("env"),
      {"package": "lighteval 0.13.0 (PyPI, released 2025-11-24)", "install": "pip install lighteval==0.13.0 into the lab environment, after all runs",
       "command": "lighteval accelerate model_name=HuggingFaceTB/SmolLM2-360M,dtype=float16 lighteval|arc:easy|0 --max-samples 1000"},
      {"steps": le, "verdict": "installed; import failed without diskcache (litellm disk cache); with diskcache, failed with TypeError: HfApi.list_models() got an unexpected keyword argument 'model_name' under huggingface_hub 2.4.0"},
      kernel="genieincodebottle/llt-eval-main v1")
fails = {}
for tag in ["llt-eval-smoke", "llt-eval-main-v1"]:
    p = os.path.join(OUT, tag, "eval_record.json")
    if os.path.exists(p):
        r = json.load(open(p, encoding="utf-8"))
        fails[tag] = {k: v[-1200:] for k, v in r["cellErrors"].items()}
write("lltEvaluation-T2-test-kernel-failures", "T2", "broken:out-of-memory", {"gpu": "Tesla T4", "count": 2, "vramGB": 15.6}, v1.get("env"),
      {"what": "cell errors of the test kernels before the fixes on the page (batch 8 in step 1, gradient checkpointing on the canary)"},
      {"cellErrors": fails})
