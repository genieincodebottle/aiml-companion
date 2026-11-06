"""Run Library files for step 8 and the notebook record, from kernel llt-insdata-exp v3 (2x T4; v2 was an identical push 16 s earlier).

GPU 0 ran the page's cells end to end (curated-1k and bulk-1k-s0 with seed 0, printed in the log and saved in
insdata_notebook_record.json), then ref-instruct and bulk-4k-s1 through worker_insdata.py. GPU 1 ran curated-1k
and bulk-1k-s1 with seed 1 and bulk-4k-s0 with seed 0 through worker_insdata.py. Seed-0 per-prompt scores for
curated-1k and bulk-1k-s0 come from the notebook record.
Usage: python record_exp.py <kaggle out dir of llt-insdata-exp>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import write_run  # noqa: E402

OUT = Path(sys.argv[1])
rec = json.loads((OUT / "insdata_notebook_record.json").read_text(encoding="utf-8"))
log = "".join(x["data"] for x in json.loads((OUT / "llt-insdata-exp.log").read_text(encoding="utf-8")) if x.get("stream_name") == "stdout")
T4 = {"gpu": rec["gpu"], "count": 2, "vramGB": 15.6, "note": "2x T4; one run per GPU"}
ENV = dict(rec["env"], image="Kaggle GPU image 2026-11-06")

runs = {}
for k, v in rec["ifevalLoose"].items():
    tag = k if k.endswith("-s0") else k + "-s0"
    tl = rec["trainLogs"][k + "-s0"]
    runs[tag] = {"loose": np.array(v, dtype=float), "source": "notebook", "log": tl["loss"], "trainSec": tl["seconds"]}
for f in sorted((OUT / "exp").glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    stem = re.sub(r"^(bulk-[0-9]k-s[0-9])-s[0-9]$", lambda m: m.group(1), f.stem)   # worker tags bulk arms twice, bulk-1k-s1-s1
    runs[stem] = {"loose": np.array(r["ifevalLoose"], dtype=float), "strict": np.array(r["ifevalStrict"], dtype=float),
                    "stopped": r["ifevalStopped"], "trainSec": r["trainSec"], "ifevalSec": r["ifevalSec"], "log": r["log"],
                    "peakMemGB": r["peakMemGB"], "source": "worker", "wallSec": r["wallSec"]}
# Strict scores and stop rates of the two notebook runs come from the printed lines.
for name, loose, strict, stopped in re.findall(r"^(curated-1k|bulk-1k-s0) +\| IFEval prompt-level loose ([0-9.]+) \| strict ([0-9.]+) \| answers that ended with the end token ([0-9.]+)", log, flags=re.M):
    tag = name if name.endswith("-s0") else name + "-s0"
    runs[tag].update(printedLoose=float(loose), strictMean=float(strict), stopped=float(stopped))
    assert abs(runs[tag]["loose"].mean() - float(loose)) < 1e-3
print(sorted(runs))


def paired(a, b, n_boot=10000, seed=0):
    d = np.asarray(b, float) - np.asarray(a, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return round(float(d.mean()), 4), round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)


cur = (runs["curated-1k-s0"]["loose"] + runs["curated-1k-s1"]["loose"]) / 2
b1 = (runs["bulk-1k-s0"]["loose"] + runs["bulk-1k-s1"]["loose"]) / 2
b4 = (runs["bulk-4k-s0"]["loose"] + runs["bulk-4k-s1"]["loose"]) / 2
cmp = {
    "curated_minus_bulk1k_seedmean": paired(b1, cur),
    "curated_minus_bulk4k_seedmean": paired(b4, cur),
    "bulk4k_minus_bulk1k_seedmean": paired(b1, b4),
    "curated_minus_bulk1k_seed0": paired(runs["bulk-1k-s0"]["loose"], runs["curated-1k-s0"]["loose"]),
    "curated_minus_bulk1k_seed1": paired(runs["bulk-1k-s1"]["loose"], runs["curated-1k-s1"]["loose"]),
    "curated_minus_bulk4k_seed0": paired(runs["bulk-4k-s0"]["loose"], runs["curated-1k-s0"]["loose"]),
    "curated_minus_bulk4k_seed1": paired(runs["bulk-4k-s1"]["loose"], runs["curated-1k-s1"]["loose"]),
    "seed_spread_curated": paired(runs["curated-1k-s0"]["loose"], runs["curated-1k-s1"]["loose"]),
}
print(json.dumps(cmp, indent=1))

summary = {}
for tag, r in sorted(runs.items()):
    summary[tag] = {"ifevalLoose": round(float(r["loose"].mean()), 4), "ifevalStrict": round(float(r["strict"].mean()), 4) if "strict" in r else r.get("strictMean"),
                    "stopped": r.get("stopped"), "trainSec": r.get("trainSec"), "ifevalSec": r.get("ifevalSec"), "peakMemGB": r.get("peakMemGB"),
                    "lossFirstLast": [r["log"][0], r["log"][-1]] if r.get("log") else None, "ranIn": r["source"]}
    print(tag, summary[tag])

for tag, r in sorted(runs.items()):
    arm = tag.rsplit("-s", 1)[0] if tag != "ref-instruct" else tag
    write_run("lltInstructionData-T2-sft-" + tag, "T2", "baseline" if tag != "ref-instruct" else "reference",
              {"base": "HuggingFaceTB/SmolLM2-360M" if tag != "ref-instruct" else "HuggingFaceTB/SmolLM2-360M-Instruct (no training)",
               "arm": arm, "steps": 250, "batch": "8 x 2 accumulation = 16", "lr": 5e-5, "warmup": 10, "maxLength": 1024,
               "completionOnlyLoss": True, "fp16": True, "gradientCheckpointing": True,
               "eval": "IFEval 541 prompts, greedy, max_new_tokens 512, batch 32, lm-eval 0.4.13 ifeval checker",
               "kernel": "llt-insdata-exp v3, " + ("notebook cells on GPU 0" if r["source"] == "notebook" else "worker_insdata.py")},
              None if tag == "ref-instruct" else int(tag[-1]),
              [{"step": s, "trainLoss": l} for s, l in (r.get("log") or [])],
              evals=dict(summary[tag], ifevalLoosePerPrompt=r["loose"].astype(int).tolist()), hardware=T4, env=ENV,
              wall_clock_sec=r.get("wallSec"))
write_run("lltInstructionData-T2-lima-compare", "T2", "ablation:data=curated|bulk1k|bulk4k",
          {"metric": "IFEval prompt-level loose accuracy", "ci": "paired bootstrap over the 541 prompts, 10,000 resamples, seed 0",
           "seedMean": "per-prompt mean of the two seeds, then paired"}, None, [summary], evals=cmp, hardware=T4, env=ENV)
nb = {"stepSeconds": rec["stepSeconds"], "env": rec["env"], "gpu": rec["gpu"], "date": rec["date"]}
write_run("lltInstructionData-T2-notebook", "T2", "notebook end to end", {"kernel": "llt-insdata-exp v3, GPU 0, the page cells in order with a timer around each"},
          None, [nb], hardware=T4, env=ENV)
