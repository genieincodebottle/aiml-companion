"""Convert kaggle/out/llt-da-sweep/res_*.json into Run Library files (plan section 7.6 schema).
Usage: python to_runlib.py <sweep_out_dir> <runs_dir>"""
import json, glob, os, sys
SRC, DST = sys.argv[1:3]
for f in sorted(glob.glob(os.path.join(SRC, "res_*.json"))):
    r = json.load(open(f, encoding="utf-8"))
    j = r["job"]
    label = {"panel": "baseline", "cpt": "ablation", "qlora": "ablation", "extinit": "ablation:init=default"}[j["kind"]]
    if j["kind"] == "cpt":
        label = "ablation:replay=" + str(j["replay"]) + ",peak_lr=" + str(j["peak_lr"]) + (",tokenizer=ext10000,init=" + j["init"] if j.get("tokenizer") else "")
    out = {"runId": "lltDomainAdaptation-t2-" + r["tag"] + "-v2026-11", "module": "lltDomainAdaptation", "tier": "T2", "label": label,
           "hardware": {"gpu": r["env"]["gpu"], "count": 1, "vramGB": 15.6, "note": "one of the two T4s of a Kaggle T4 x2 session; the other GPU ran a second job list"},
           "env": r["env"], "kernel": "genieincodebottle/llt-da-sweep v1 (kaggle/out/llt-da-sweep)", "notebookCommit": None,
           "config": j, "seed": j.get("seed", 0), "date": r["date"], "wallClockSec": r["wallSec"], "costUSD": 0,
           "evalSizes": r["evalSizes"], "qaTestQuestions": r["qaTest"],
           "evalNotes": "hiBpb/enBpb: bits per byte on 200 held-out FineWeb2 hin_Deva test texts (first 1,000 chars) and 200 FineWeb-Edu texts (first 3,000 chars). xscHi/xscEn: XStoryCloze eval split accuracy (1,511 each). hellaswag: acc_norm on the first 2,000 validation items. qa: XQuAD hi, 465 test questions from paragraphs 140-239, SQuAD-style EM and F1, greedy 40 tokens cut at the first newline; closed0/closed3 = no context, 0 or 3 example QA pairs; rag0/rag3 = the top-2 paragraphs retrieved by multilingual-e5-small from an 888-paragraph pool.",
           "metrics": r.get("metrics", [])}
    for k in ("panel", "panel0", "panelFinal", "panelBase4bit", "qa", "qaBase4bit", "interp", "peakMemGB", "trainSec", "tokensPerSec",
              "rows", "tokenizer", "vocabRows", "trainable", "total", "steps", "panelSec", "qaSec", "params", "trainExamples", "epochs", "batch", "lr"):
        if k in r:
            out[k] = r[k]
    p = os.path.join(DST, out["runId"] + ".json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", os.path.basename(p))
