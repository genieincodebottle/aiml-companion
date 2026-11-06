"""Copy the Kaggle outputs of kernel llt-sft-runs (and llt-sft-notebook) into the Run Library.

Usage: python record_gpu.py <kaggle output dir> [<kaggle output dir> ...]
Each lltSFT-T2-*.json written by worker_sft.py becomes runs/lltSFT/<runId>.json with the plan 7.6 keys filled
(costUSD 0 on the free tier, notebookCommit null until the owner pushes). The selection summary and session time
go into lltSFT-T2-session-<kernel>.json.
"""
import glob
import json
import os
import sys

RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "lltSFT"))


def write_notebook_run(d, kernel, res):
    """The learner notebook, end to end on one T4: step 8 figures, the SDPA probe, and cell times from the kernel log."""
    probe_p = os.path.join(d, "sdpa_probe.json")
    log_p = os.path.join(d, kernel + ".log")
    marks = {}
    if os.path.exists(log_p):
        for e in json.load(open(log_p, encoding="utf-8")):
            t, data = e.get("time"), e.get("data", "")
            for key, label in (("Runtime OK", "runtimeCheckDone"), ("3600 training and 300", "step2Done"),
                               ("held-out loss after", "step7Done"), ("base | held-out", "step8BaseScored"),
                               ("packed rows of up to", "step8TrainStart"), ("SFT  | held-out", "step8Done"),
                               ("All checks passed", "challengeDone")):
                if key in data and label not in marks:
                    marks[label] = round(t, 1)
            if "own " in data and "TRL" in data and "step5LossCheck" not in marks:
                marks["step5LossCheck"] = data.strip()
    rec = {"runId": "lltSFT-T2-notebook", "module": "lltSFT", "tier": "T2", "label": "baseline",
           "hardware": {"gpu": res["env"].get("gpu"), "count": 1, "vramGB": 15.6}, "env": res["env"],
           "notebookCommit": None, "config": {"notebook": "labs/lltSFT/lltSFT.ipynb", "kernel": "genieincodebottle/" + kernel,
           "note": "make_sft_notebook_kernel.py swaps the Prove it starter for its solution and appends two recording cells"},
           "seed": 0, "metrics": [{"step": h["step"], "trainLoss": h["loss"]} for h in res.get("lossCurve", [])],
           "evals": {"base": res["base"], "final": res["final"], "train": res["train"],
                     "sdpaProbe": json.load(open(probe_p)) if os.path.exists(probe_p) else None,
                     "logSecondsFromStart": marks},
           "samples": res.get("samplesAfter", [])[:30], "wallClockSec": marks.get("challengeDone"),
           "costUSD": 0, "priceCheckedOn": None, "kaggleKernel": "genieincodebottle/" + kernel}
    out = os.path.join(RUNS, "lltSFT-T2-notebook.json")
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out, marks)
    # The step 10 arms run by the notebook's last cell, with the page's own train_and_score and cross_doc_collator.
    iso_p = os.path.join(d, "iso_runs.json")
    if os.path.exists(iso_p):
        for r in json.load(open(iso_p, encoding="utf-8")):
            arm = "crossdoc-restart" if r["restartPositions"] else "crossdoc-continuous"
            run_id = "lltSFT-T2-" + arm + "-s" + str(r["seed"])
            one = {"runId": run_id, "module": "lltSFT", "tier": "T2",
                   "label": "ablation:packing=" + arm + (", explicit causal mask" if r["explicitMask"] else ""),
                   "hardware": rec["hardware"], "env": res["env"], "notebookCommit": None,
                   "config": {"collator": "cross_doc_collator(restart_positions=%s, explicit_mask=%s)" % (r["restartPositions"], r["explicitMask"]),
                              "base": "make_trainer defaults of step 8 (packing bfd, assistant_only_loss, lr 5e-5 cosine, 1 epoch)",
                              "ranIn": "kernel " + kernel + ", after steps 1 to 8 in the same process"},
                   "seed": r["seed"], "metrics": [], "evals": {"final": {"evalLoss": r["heldOut"], "generation": {
                       "stopRate": r["stopRate"], "meanNewTokens": r["meanNewTokens"]}}},
                   "train": {"steps": r["steps"], "trainSec": r["seconds"], "secPerStep": r["secPerStep"],
                             "firstLoggedLoss": r["trainLoss"][0], "lastLoggedLoss": r["trainLoss"][1],
                             "peakMemGBIncludingScoring": r.get("peakMemGBIncludingScoring")},
                   "costUSD": 0, "priceCheckedOn": None, "kaggleKernel": "genieincodebottle/" + kernel}
            out = os.path.join(RUNS, run_id + ".json")
            json.dump(one, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            print("wrote", out)


def main():
    os.makedirs(RUNS, exist_ok=True)
    for d in sys.argv[1:]:
        kernel = os.path.basename(os.path.normpath(d))
        for p in sorted(glob.glob(os.path.join(d, "lltSFT-T2-*.json"))):
            rec = json.load(open(p, encoding="utf-8"))
            rec.setdefault("module", "lltSFT")
            rec.setdefault("tier", "T2")
            rec["notebookCommit"] = None
            rec["costUSD"] = 0
            rec["priceCheckedOn"] = None
            rec["kaggleKernel"] = "genieincodebottle/" + kernel
            out = os.path.join(RUNS, rec["runId"] + ".json")
            json.dump(rec, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            print("wrote", out, rec.get("ok"))
        nbres = os.path.join(d, "notebook_results.json")
        if os.path.exists(nbres):
            write_notebook_run(d, kernel, json.load(open(nbres, encoding="utf-8")))
            continue
        extra = {}
        for name in ("data_selection.json", "session.json"):
            q = os.path.join(d, name)
            if os.path.exists(q):
                extra[name.replace(".json", "")] = json.load(open(q, encoding="utf-8"))
        if extra:
            out = os.path.join(RUNS, "lltSFT-T2-session-" + kernel + ".json")
            json.dump({"runId": "lltSFT-T2-session-" + kernel, "module": "lltSFT", "tier": "T2",
                       "kaggleKernel": "genieincodebottle/" + kernel, **extra}, open(out, "w", encoding="utf-8"),
                      indent=1, ensure_ascii=False)
            print("wrote", out)


if __name__ == "__main__":
    main()
