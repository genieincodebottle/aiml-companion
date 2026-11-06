"""Convert the lltLoRA Kaggle results (kaggle/out/llt-lora-*/res_*.json, svd_*.json) into Run Library files.

Usage: python make_runs.py   -> writes ../../runs/lltLoRA/lltLoRA-t2-<tag>.json (+ -svd.json for full fine-tunes)
"""
import glob, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "runs", "lltLoRA")
os.makedirs(OUT, exist_ok=True)


def label(tag, job):
    if tag.startswith("broken-"):
        return "broken:" + tag[len("broken-"):]
    if tag == "base-eval":
        return "baseline:base-model"
    return "ablation:" + tag


def main():
    seen = {}
    for res in sorted(glob.glob(os.path.join(ROOT, "kaggle", "out", "llt-lora-*", "res_*.json"))):
        kernel = os.path.basename(os.path.dirname(res))
        r = json.load(open(res, encoding="utf-8"))
        tag, job = r["tag"], r.get("job", {})
        if kernel == "llt-lora-probe":
            run_id = "lltLoRA-t2-probe-" + tag
        else:
            run_id = "lltLoRA-t2-" + tag
        rec = {
            "runId": run_id, "module": "lltLoRA", "tier": "T2", "label": label(tag, job),
            "hardware": {"gpu": (r.get("env") or {}).get("gpu"), "count": 1, "vramGB": 15.6},
            "env": r.get("env"), "kaggleKernel": "genieincodebottle/" + kernel, "notebookCommit": None,
            "config": job, "seed": job.get("seed"), "ok": r.get("ok"), "error": r.get("error"),
            "data": r.get("data"), "trainableParams": r.get("trainableParams"), "totalParams": r.get("totalParams"),
            "peakMemGB": r.get("peakMemGB"), "peakReservedGB": r.get("peakReservedGB"),
            "memAfterLoadGB": r.get("memAfterLoadGB"), "secPerStepMedian": r.get("secPerStepMedian"),
            "trainSec": r.get("trainSec"), "skippedSteps": r.get("skippedSteps"), "nonFinite": r.get("nonFinite"),
            "metrics": r.get("metrics", []), "curve": r.get("curve", []),
            "evals": {"base": r.get("evalBase"), "final": r.get("evalFinal"), "truncation": r.get("truncation"),
                      "dropoutEvalTrainMode": r.get("dropoutEvalTrainMode"),
                      "dropoutEvalEvalMode": r.get("dropoutEvalEvalMode")},
            "adapterBytes": r.get("adapterBytes"), "spectraFile": None,
            "costUSD": 0, "date": "2026-11-06",
        }
        svd = res.replace("res_", "svd_")
        if os.path.exists(svd):
            s = json.load(open(svd, encoding="utf-8"))
            sp_id = run_id + "-svd"
            json.dump({"runId": sp_id, "module": "lltLoRA", "from": run_id, **s},
                      open(os.path.join(OUT, sp_id + ".json"), "w", encoding="utf-8"))
            rec["spectraFile"] = sp_id + ".json"
        json.dump(rec, open(os.path.join(OUT, run_id + ".json"), "w", encoding="utf-8"), indent=1)
        seen[run_id] = r.get("ok")
    print(len(seen), "runs:", seen)


if __name__ == "__main__":
    main()
