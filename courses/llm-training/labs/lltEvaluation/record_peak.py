"""Run Library file for kernel llt-eval-peak (audit pass): per-cell peak GPU memory and the float32 softmax check.
Also adds the softmax check to lltEvaluation-T2-scorer-match.
Usage: python record_peak.py <downloaded llt-eval-peak output dir>
"""
import json, os, sys

SRC = sys.argv[1]
RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "lltEvaluation"))
P = json.load(open(os.path.join(SRC, "peak_record.json"), encoding="utf-8"))
cells = P["cells"]
peak = max(v["peakAllocatedGB"] for v in cells.values())
rec = {"runId": "lltEvaluation-T2-peak-memory", "module": "lltEvaluation", "tier": "T2", "label": "analysis",
       "hardware": {"gpu": "Tesla T4", "count": 1, "vramGB": 15.6}, "kernel": "genieincodebottle/llt-eval-peak v1",
       "env": {"image": "Kaggle GPU image 2026-11-07", "pins": "same install cell as llt-eval-main v2"},
       "config": {"what": "torch.cuda.max_memory_allocated and max_memory_reserved, reset before each cell; cells c01, c02, c06, c07 rerun unchanged; "
                          "c03's definitions executed with its recorded per-item output from llt-eval-main v2; c10's peak (9.17 GB) is recorded by the cell in llt-eval-main v2",
                  "softmaxCheck": "lm-eval arc_easy on Qwen3-1.7B-Base with HFLM.softmax_dtype = float32, compared with our float32 scorer on the same 1,000 items"},
       "evals": {"cells": cells, "maxPeakAllocatedGB": peak, "c10PeakAllocatedGB_fromV2": 9.17, "softmax": P.get("softmax"),
                 "errors": list(P.get("errors", {})), "skipped": P.get("skipped")},
       "date": "2026-11-07"}
json.dump(rec, open(os.path.join(RUNS, rec["runId"] + ".json"), "w", encoding="utf-8"), indent=1)
d = json.load(open(os.path.join(RUNS, "lltEvaluation-T2-scorer-match.json"), encoding="utf-8"))
d["evals"]["softmaxFp32Check"] = dict(P.get("softmax") or {}, kernel="genieincodebottle/llt-eval-peak v1")
json.dump(d, open(os.path.join(RUNS, "lltEvaluation-T2-scorer-match.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(rec["evals"], indent=1))
