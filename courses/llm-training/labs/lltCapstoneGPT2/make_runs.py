# make_runs.py - turn the two Kaggle measurement outputs into Run Library files (plan section 7.6).
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "runs", "lltCapstoneGPT2")
os.makedirs(OUT, exist_ok=True)
SRC = [("llt-capstone-gpt2-measure", "lltCapstoneGPT2_measure.json"),
       ("llt-capstone-gpt2-measure2", "lltCapstoneGPT2_measure_v2.json")]


def env_of(d):
    e = dict(d["env"])
    e["image"] = "Kaggle GPU image 2026-11-05"
    return e


written = []
for slug, name in SRC:
    path = os.path.join(ROOT, "kaggle", "out", slug, name)
    if not os.path.exists(path):
        print("missing", path)
        continue
    d = json.load(open(path))
    for b in d["benches"]:
        if not b:
            continue
        kind = "resume" if "resume" in b else "throughput"
        run_id = "lltCapstoneGPT2-T2-%s-%s" % (kind, b["tag"].replace("_", "-"))
        rec = {
            "runId": run_id, "module": "lltCapstoneGPT2", "tier": "T2",
            "label": "measure:%s" % kind,
            "hardware": {"gpu": "Tesla T4", "count": b["world"], "vramGB": 15.64, "interconnect": "PCIe, no NVLink"},
            "env": env_of(d), "notebookCommit": None, "kaggleKernel": "genieincodebottle/" + slug,
            "config": {"size": b["size"], "params": b["params"], "micro": b["micro"], "accum": b["accum"],
                       "ctx": 1024, "vocab": 50257, "tokensPerStep": b["tokensPerStep"], "compile": b["compile"],
                       "activationCheckpointing": b["ckpt"], "precision": "fp16 autocast + GradScaler",
                       "parallel": "DDP, no_sync on all but the last micro-batch", "optimizer": "AdamW fused, betas 0.9/0.95, wd 0.1",
                       "lr": "6e-4 peak, linear warmup over 10 steps", "clip": 1.0,
                       "data": "FineWeb-Edu sample-10BT, GPT-2 tokenizer (tiktoken), EOT before each doc"},
            "seed": 0,
            "metrics": [{"step": s["step"], "tokens": s["step"] * b["tokensPerStep"], "trainLoss": s["loss"],
                         "stepSec": s["sec"], "warmup": s.get("warmup", False), "gradNorm": s.get("gradNorm")}
                        for s in b.get("steps", [])],
            "evals": {}, "samples": [],
            "summary": {k: b.get(k) for k in ("tokPerSec", "secPerStep", "peakAllocGB", "peakReservedGB", "flopsPerToken",
                                               "achievedTFLOPS", "mfuVsSpec65", "mfuVsMeasured22p1", "warmupSec", "oom",
                                               "allocAtOomGB", "resume")},
            "mfuNote": "flops per token = 6N + 12*L*d*T (nanoGPT convention); spec peak 65 TFLOPS fp16 per T4; "
                       "measured dense fp16 matmul 22.1 TFLOPS per T4 (run llt-gpu-smoke)",
            "wallClockSec": round(sum(s["sec"] for s in b.get("steps", [])), 1), "costUSD": 0, "priceCheckedOn": None,
            "date": d["started"][:10],
        }
        with open(os.path.join(OUT, run_id + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        written.append(run_id)
    meta = {"runId": "lltCapstoneGPT2-T2-session-" + slug.split("-")[-1], "module": "lltCapstoneGPT2", "tier": "T2",
            "label": "measure:session", "env": env_of(d), "kaggleKernel": "genieincodebottle/" + slug,
            "probes": d["probes"], "picked": d.get("picked"), "data": d.get("data"),
            "steps": {k: (v if isinstance(v, str) else {"rc": v["rc"], "sec": v["sec"]}) for k, v in d["log"].items()},
            "sessionSec": d.get("elapsedSec"), "date": d["started"][:10]}
    with open(os.path.join(OUT, meta["runId"] + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    written.append(meta["runId"])

lm = glob.glob(os.path.join(ROOT, "kaggle", "out", "llt-capstone-gpt2-measure", "lmeval", "*", "*.json"))
if lm:
    d = json.load(open(lm[0]))
    rec = {"runId": "lltCapstoneGPT2-T2-eval-openai-gpt2", "module": "lltCapstoneGPT2", "tier": "T2",
           "label": "baseline:openai-community/gpt2", "hardware": {"gpu": "Tesla T4", "count": 1},
           "env": {"lm_eval": d.get("lm_eval_version"), "transformers": d.get("transformers_version"),
                   "image": "Kaggle GPU image 2026-11-05"},
           "config": {"model_args": d["config"]["model_args"], "batch_size": d["config"]["batch_size"],
                      "tasks": sorted(d["results"]), "num_fewshot": 0, "limit": None},
           "evals": {t: {k: v for k, v in r.items() if k not in ("alias", "name")} for t, r in d["results"].items()},
           "nSamples": d.get("n-samples"), "wallClockSec": round(float(d.get("total_evaluation_time_seconds", 0)), 1),
           "costUSD": 0, "date": "2026-11-05"}
    with open(os.path.join(OUT, rec["runId"] + ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    written.append(rec["runId"])
print("\n".join(written))
