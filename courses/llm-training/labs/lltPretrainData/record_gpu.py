"""Copy the Kaggle GPU run files (kaggle/out/llt-ptd-train/lltPretrainData-T2-*.json) into the Run
Library, adding the notebook name and the kernel version, and print a summary of every run.

Usage: python record_gpu.py
"""
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "kaggle", "out", "llt-ptd-train")
DST = os.path.join(HERE, "..", "..", "runs", "lltPretrainData")
os.makedirs(DST, exist_ok=True)
for p in sorted(glob.glob(os.path.join(SRC, "lltPretrainData-T2-*.json"))):
    r = json.load(open(p, encoding="utf-8"))
    r.setdefault("notebookCommit", None)
    r["kernel"] = "genieincodebottle/llt-ptd-train v1 (Kaggle, 2x T4, one run per GPU)"
    r.setdefault("costUSD", 0)
    with open(os.path.join(DST, os.path.basename(p)), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    last = r["metrics"][-1]
    evals = {k: (v["loss"], v["bpb"]) for k, v in last.items() if isinstance(v, dict) and "bpb" in v}
    print(r["runId"], "| params", r["params"], "| trainSec", r.get("trainSec"), "| peak", r.get("peakMemGB"), "| data", json.dumps(r.get("data"))[:160])
    print("    final", json.dumps(evals))
    for k in ("hellaswag", "trainSetLoss", "anneal"):
        if k in r:
            print("   ", k, r[k])
    if "canaryResults" in r:
        for lvl, d in sorted(r["canaryResults"].items(), key=lambda kv: int(kv[0])):
            print("    canary", lvl, "extract", d["extractionRate"], "tokmatch", d["meanTokenMatch"], "suffixLoss", d["meanSuffixLoss"], "seen", sorted(set(d["seen"])))
