"""Turn the T4 notebook run (2026-11-11, Google Cloud VM set up like Kaggle's GPU T4 x2) into Run Library files
(courses/llm-training/runs/lltCapstoneReasoner/lltCapstoneReasoner-T2-gcpT4-*.json).

  python make_runs_t4.py <pulled folder with runs/ evals/ nbout/ setup.json>

Per-cell wall time comes from the executed notebooks (nbconvert record_timing), so it is what a learner's notebook shows.
The reference A100 files are read only to put the same cell and seed next to the T4 result. They are not changed.
"""
import csv
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parent.parent.parent / "runs" / "lltCapstoneReasoner"
OUT.mkdir(parents=True, exist_ok=True)
SETUP = json.load(open(SRC / "setup.json", encoding="utf-8"))
ENV = SETUP["env"]
HW = dict(gpu="Tesla T4", count=1, vramGB=15, vm="n1-highmem-4, 2 x T4, 26 GB RAM", driver="580.178.04",
          provider="Google Cloud on-demand VM, us-central1-f, set up like Kaggle's GPU T4 x2 (the notebook uses one T4)")
DATE = "2026-11-11"


def lines(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")] if p.exists() else []


def t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def cell_secs(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    res = {}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        ex = c["metadata"].get("execution", {})
        if ex.get("iopub.execute_input") and ex.get("shell.execute_reply"):
            res[i] = round((t(ex["shell.execute_reply"]) - t(ex["iopub.execute_input"])).total_seconds(), 1)
    return res


# cell numbers in the executed notebook: 2 hardware check, 3 install, 4 get lab folder and tests, 7 base eval, 8 train, 9 evaluate, 10 analyze
LABEL = {2: "hardware check", 3: "install pinned stack", 4: "get lab folder and run test_capstone.py", 7: "base model evaluation (n=16)",
         8: "train.py (one ablation run)", 9: "evaluate.py on the finished run (n=16)", 10: "analyze.py"}

peak_smi = {}
gm = SRC / "nbout" / "gpu_mem.csv"
if gm.exists():
    for row in open(gm):
        parts = row.strip().split(",", 1)
        if len(parts) < 2:
            continue
        for gi, seg in enumerate([x for x in parts[1].split(";") if x.strip()]):
            try:
                mem = int(seg.split(",")[0])
            except ValueError:
                continue
            peak_smi[gi] = max(peak_smi.get(gi, 0), mem)

written, per_run = [], {}
for d in sorted((SRC / "runs").iterdir()):
    cfg_p = d / "config.json"
    if not cfg_p.exists() or not (d / "done.json").exists():
        continue
    cfg = json.load(open(cfg_p, encoding="utf-8"))
    m = lines(d / "metrics.jsonl")
    done = json.load(open(d / "done.json", encoding="utf-8"))
    nbs = sorted(glob.glob(str(SRC / "nbout" / ("executed_%s_pass*.ipynb" % d.name))))
    train_cell = sum(cell_secs(p).get(8, 0) for p in nbs)
    eval_cell = sum(cell_secs(p).get(9, 0) for p in nbs)
    final = SRC / "evals" / (d.name + ".json")
    final_ev = json.load(open(final, encoding="utf-8")) if final.exists() else None
    if final_ev:
        final_ev = {k: v for k, v in final_ev.items() if k not in ("greedyReplies",)}
    ref_p = OUT / ("lltCapstoneReasoner-A100-" + d.name + ".json")
    ref = None
    if ref_p.exists():
        r = json.load(open(ref_p, encoding="utf-8"))
        f = (r.get("evals") or {}).get("final") or {}
        pk = f.get("passAtK") or {}
        ref = dict(runId=r["runId"], passAt1=(pk.get("1") or {}).get("mean"), passAt16=(pk.get("16") or {}).get("mean"),
                   greedy=(f.get("greedy") or {}).get("mean"), meanTokens=(f.get("meanTokens") or {}).get("all"),
                   engine=f.get("engine", "hf"), n=f.get("n"), wallClockSec=r.get("wallClockSec"), runsSideBySide=r.get("runsSideBySide"))
    wall = round(sum(x["sec"] for x in m), 1)
    rid = "lltCapstoneReasoner-T2-gcpT4-" + d.name
    rec = dict(runId=rid, module="lltCapstoneReasoner", tier="T2", label="ablation:agg=%s,scale=%d" % (cfg["agg"], cfg["scale"]), hardware=HW, env=ENV,
               notebookCommit=None, config=cfg, seed=cfg["seed"], metrics=m,
               evals=dict(duringTraining=lines(d / "evals.jsonl"), final=final_ev), samples=lines(d / "groups.jsonl"),
               lltrlFunctions=json.load(open(d / "lltrl_functions.json", encoding="utf-8"))["hashes"] if (d / "lltrl_functions.json").exists() else None,
               peakGB=done.get("peakGB"), wallClockSec=wall, trainCellSec=round(train_cell, 1), evalCellSec=round(eval_cell, 1),
               sessions=len(nbs), runsSideBySide=1, concurrentOtherGpu=True, date=DATE,
               compareToA100Reference=ref,
               note="Measured on a Google Cloud VM set up like Kaggle's GPU T4 x2, by running the learner notebook top to bottom. "
                    "Another ablation cell ran at the same time on the second T4, which did not slow either run (see the notebook summary file).")
    (OUT / (rid + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    written.append(rid)
    per_run[d.name] = dict(trainCellSec=round(train_cell, 1), evalCellSec=round(eval_cell, 1), sessions=len(nbs), peakGB=done.get("peakGB"),
                           steps=done.get("steps"), metricsSecSum=wall, medianStepSec=sorted(x["sec"] for x in m)[len(m) // 2] if m else None,
                           passAt1=final_ev["passAtK"]["1"]["mean"] if final_ev else None, passAt16=final_ev["passAtK"]["16"]["mean"] if final_ev else None,
                           greedy=final_ev["greedy"]["mean"] if final_ev else None, meanTokens=final_ev["meanTokens"]["all"] if final_ev else None, ref=ref)

base_p = SRC / "evals" / "base.json"
if base_p.exists():
    ev = json.load(open(base_p, encoding="utf-8"))
    ev = {k: v for k, v in ev.items() if k != "greedyReplies"}
    rid = "lltCapstoneReasoner-T2-gcpT4-eval-base"
    rec = dict(runId=rid, module="lltCapstoneReasoner", tier="T2", label="evaluation (base model)", hardware=HW, env=ENV, notebookCommit=None,
               config={k: ev.get(k) for k in ("model", "adapter", "n", "nTest", "temperature", "maxNew", "prompt")}, seed=1234, metrics=[], evals=ev,
               samples=ev.get("examples"), wallClockSec=sum(ev["sec"].values()), date=DATE)
    (OUT / (rid + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    written.append(rid)

summary = dict(runId="lltCapstoneReasoner-T2-gcpT4-notebook", module="lltCapstoneReasoner", tier="T2", label="learner notebook, timed",
               hardware=HW, env=ENV, date=DATE, setup=SETUP, perRun=per_run, gpuMemNvidiaSmiPeakMiB=peak_smi,
               note="The first run executed the notebook alone on GPU 0 (hardware check, install, tests, base evaluation, one ablation run, its evaluation). "
                    "Later cells ran with the notebook's parameter cell changed, two at a time, one per T4.")
(OUT / "lltCapstoneReasoner-T2-gcpT4-notebook.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
written.append("lltCapstoneReasoner-T2-gcpT4-notebook")
print(len(written), "run files in", OUT)
for w in written:
    print(" ", w)
