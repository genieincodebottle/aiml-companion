"""Turn the reference run folders and evaluations (copied from the VM) into Run Library files
(courses/llm-training/runs/lltCapstoneReasoner/<runId>.json, schema in the plan's section 7.6).

  python make_runs.py <pulled folder with runs/ evals/ logs/setup.log>

Price: an A100 40 GB Spot VM in us-central1 at PRICE_PER_H, read on PRICE_DATE from a third-party listing (Spot prices
change; check the cloud console before you rely on it). Runs that shared the GPU are charged their share: wall-clock
divided by the number of runs side by side.
"""
import json
import sys
from pathlib import Path

PRICE_PER_H, PRICE_DATE, PRICE_SRC = 1.84, "2026-11-10", "devzero.io listing of GCP a2-highgpu-1g Spot, us-central1"
SRC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parent.parent.parent / "runs" / "lltCapstoneReasoner"
OUT.mkdir(parents=True, exist_ok=True)
ENV = dict(python="3.12.16", torch="2.11.0+cu130", transformers="5.22.1", trl="1.17.1", peft="0.21.2", accelerate="1.15.0",
           datasets="5.1.0", mathVerify="0.9.0", vllmForEvaluation="0.33.0 (torch 2.14.0+cu130)")
HW = dict(gpu="NVIDIA A100-SXM4-40GB", count=1, vramGB=40, provider="Google Cloud Spot VM, us-central1-a")
SIDE_BY_SIDE = {"t_p": 4, "lr": 3, "agg-": 4, "q17_": 2}


def lines(p):
    return [json.loads(x) for x in open(p)] if p.exists() else []


def share(name):
    for k, v in SIDE_BY_SIDE.items():
        if name.startswith(k):
            return v
    return 1


def label(name, cfg):
    if name.startswith("agg-") or name.startswith("q17_"):
        return "ablation:agg=%s,scale=%d" % (cfg["agg"], cfg["scale"])
    if name.startswith("lr"):
        return "probe:lr=%s" % cfg["lr"]
    if name.startswith("t_"):
        return "timing"
    return "smoke"


written = []
for d in sorted((SRC / "runs").iterdir()):
    cfg_p = d / "config.json"
    if not cfg_p.exists():
        continue
    cfg = json.load(open(cfg_p))
    m = lines(d / "metrics.jsonl")
    done = json.load(open(d / "done.json")) if (d / "done.json").exists() else {}
    wall = round(sum(x["sec"] for x in m), 1)
    n_side = share(d.name)
    gpu_h = wall / 3600 / n_side
    final = SRC / "evals" / (d.name + ".json")
    final_ev = json.load(open(final)) if final.exists() else None
    if final_ev:
        final_ev = {k: v for k, v in final_ev.items() if k not in ("greedyReplies",)}
    tier = "T3" if "1.7B" in cfg["model"] else "T2-config"
    rid = "lltCapstoneReasoner-%s-%s" % ("A100", d.name)
    rec = dict(runId=rid, module="lltCapstoneReasoner", tier=tier, label=label(d.name, cfg), hardware=HW, env=ENV,
               notebookCommit=None, config=cfg, seed=cfg["seed"], metrics=m, evals=dict(duringTraining=lines(d / "evals.jsonl"), final=final_ev),
               samples=lines(d / "groups.jsonl"), lltrlFunctions=json.load(open(d / "lltrl_functions.json"))["hashes"] if (d / "lltrl_functions.json").exists() else None,
               peakGB=done.get("peakGB"), runsSideBySide=n_side, wallClockSec=wall, gpuHours=round(gpu_h, 3),
               costUSD=round(gpu_h * PRICE_PER_H, 2), pricePerHourUSD=PRICE_PER_H, priceCheckedOn=PRICE_DATE, priceSource=PRICE_SRC,
               date="2026-11-10", note=("T2-config: the learner's T2 configuration (Qwen3-0.6B-Base, fp16, LoRA r16) run on an A100; "
                                        "a T4 is slower, see the page's projected T4 times") if tier == "T2-config" else None)
    (OUT / (rid + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    written.append(rid)

for f in sorted((SRC / "evals").glob("*.json")):
    if (SRC / "runs" / f.stem).exists():
        continue                       # already inside its training run's file
    ev = json.load(open(f))
    rid = "lltCapstoneReasoner-A100-eval-" + f.stem
    if f.stem.startswith("replies"):
        rec = dict(runId=rid, module="lltCapstoneReasoner", tier="T3", label="reply analysis (analyze_replies.py)", hardware=HW, env=ENV,
                   notebookCommit=None, config=dict(model="Qwen/Qwen3-0.6B-Base", n=8, temperature=1.0, maxNew=256, engine="vllm"), seed=7,
                   metrics=[], evals={k: v["stats"] for k, v in ev.items()},
                   samples={k: v["replies"][:40] for k, v in ev.items()}, date="2026-11-10")
        (OUT / (rid + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        written.append(rid)
        continue
    rec = dict(runId=rid, module="lltCapstoneReasoner", tier="T3", label="evaluation", hardware=HW, env=ENV, notebookCommit=None,
               config={k: ev.get(k) for k in ("model", "adapter", "engine", "n", "nTest", "temperature", "maxNew", "dtype", "prompt")},
               seed=1234, metrics=[], evals=ev, samples=ev.get("examples"), wallClockSec=sum(ev["sec"].values()),
               costUSD=round(sum(ev["sec"].values()) / 3600 * PRICE_PER_H, 3), priceCheckedOn=PRICE_DATE, date="2026-11-10")
    (OUT / (rid + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    written.append(rid)
print(len(written), "run files in", OUT)
for w in written:
    print(" ", w)
