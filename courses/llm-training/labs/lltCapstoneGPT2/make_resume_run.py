# make_resume_run.py - Run Library record for the exact-resume check (kernel llt-capstone-gpt2-resume).
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "kaggle", "out", "llt-capstone-gpt2-resume")
d = json.load(open(os.path.join(SRC, "lltCapstoneGPT2_resume.json")))
A = json.load(open(os.path.join(SRC, "A", "resumeA.json")))
C = json.load(open(os.path.join(SRC, "C", "branchC.json")))
ev = json.load(open(os.path.join(SRC, "evals", "lltCapstoneGPT2-eval-ours-step000012.json")))
rec = {
    "runId": "lltCapstoneGPT2-T2-resume-check", "module": "lltCapstoneGPT2", "tier": "T2", "label": "check:exact-resume",
    "hardware": {"gpu": "Tesla T4", "count": 2, "vramGB": 15.64}, "env": {**d["env"], "image": "Kaggle GPU image 2026-11-05"},
    "notebookCommit": None, "kaggleKernel": "genieincodebottle/llt-capstone-gpt2-resume",
    "config": {k: A["config"][k] for k in ("size", "micro", "tokens_per_step", "max_steps", "warmup", "stable_end", "lr",
                                            "params", "accum")},
    "protocol": "run A trains steps 1-12 and saves full state at step 6; run B is a fresh torchrun process that loads "
                "A's step-6 checkpoint and trains steps 7-12; both use torch.compile, DDP on 2 GPUs, fp16 + GradScaler. "
                "The loader crosses from shard 0 to shard 1 at step 12, after the resume point.",
    "compare": d["resumeCompare"], "maxAbsDiffTrainLoss": d["resumeMaxAbsDiffTrainLoss"],
    "identicalAtLoggedPrecision": d["resumeBitwiseTrainLoss"],
    "loggedPrecision": {"trainLoss": 5, "gradNorm": 4, "valLoss": 5},
    "checkpoint": {"files": d["ckptFiles"], "totalGB": d["ckptGB"],
                   "note": "state.pt holds fp32 weights, AdamW m and v, scaler, loader position, step, metrics; one RNG file per rank"},
    "branchDecay": {"from": 6, "stableEnd": 6, "maxSteps": 9, "metrics": d["branch"]},
    "wrapperCheck": [l for l in d["log"]["eval_wrapper"]["out"].splitlines() if l.startswith("wrapper check")],
    "lmEvalSmoke": {"limit": 50, "note": "12-step model, plumbing test only", "evals": ev["evals"]},
    "stepSec": {k: v["sec"] for k, v in d["log"].items() if isinstance(v, dict)},
    "sessionSec": d["elapsedSec"], "costUSD": 0, "date": d["started"][:10],
}
out = os.path.join(ROOT, "runs", "lltCapstoneGPT2", rec["runId"] + ".json")
with open(out, "w") as f:
    json.dump(rec, f, indent=1)
print(out)
