# make_runs.py - turn the Kaggle measurement outputs into Run Library JSON files (plan section 7.6).
# Usage: python make_runs.py   (reads kaggle/out/llt-memthru-p*/, writes runs/lltMemoryThroughput/)
import json, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "runs", "lltMemoryThroughput")
K = os.path.join(ROOT, "kaggle", "out")
os.makedirs(OUT, exist_ok=True)
HW1 = {"gpu": "Tesla T4", "count": 1, "vramGB": 15.6, "platform": "Kaggle, accelerator NvidiaTeslaT4, CUDA_VISIBLE_DEVICES=0"}
HW2 = {"gpu": "Tesla T4", "count": 2, "vramGB": 15.6, "platform": "Kaggle, accelerator NvidiaTeslaT4 (2x T4, PCIe)"}


def write(run):
    run.setdefault("module", "lltMemoryThroughput")
    run.setdefault("tier", "T2")
    run.setdefault("costUSD", 0)
    path = os.path.join(OUT, run["runId"] + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    print("wrote", os.path.relpath(path, ROOT))


def p1():
    r = json.load(open(os.path.join(K, "llt-memthru-p1", "lltMemoryThroughput_p1.json")))
    v1 = json.load(open(os.path.join(K, "llt-memthru-p1", "lltMemoryThroughput_p1_v1.json")))
    env, d, so = r["env"], r["data"], r["stdout"]
    base = {"hardware": HW1, "env": env, "notebook": "kaggle genieincodebottle/llt-memthru-p1 version 2",
            "date": r["started"][:10], "seed": 0, "model": "GPT-2 small, 124,439,808 parameters (steps/gpt.py)"}
    rows = []
    for b, s, pred, got in d["grid"]:
        rows.append({"batch": b, "seq": s, "tokens": b * s, "predictedGB": round(pred, 3),
                     "allocatedGB": None if got is None else round(got[0], 3),
                     "reservedGB": None if got is None else round(got[1], 3), "oom": got is None})
    write({**base, "runId": "lltMemoryThroughput-T2-memory-grid", "label": "baseline",
           "config": {"precision": "fp16 autocast + GradScaler", "optimizer": "AdamW fused", "measuredStep": 3},
           "metrics": rows, "evals": {"phases_b8_s1024": d["phases"], "bytesPerTokenPerBlock": d["bytes_per_token_per_block"]},
           "stdout": {k: so[k] for k in ("check", "s01", "s02", "s03")}, "wallClockSec": r["stepSec"]["s03"]})
    write({**base, "runId": "lltMemoryThroughput-T2-memory-cuts", "label": "ablation:memory-switches",
           "config": {"batch": 8, "seq": 1024, "chunks": 8},
           "metrics": [{"variant": k, "peakGB": round(v[0], 3), "tokPerSec": round(v[1]), "largestBatchAtSeq1024": v[2]} for k, v in d["cut"].items()],
           "stdout": {"s05": so["s05"]}, "wallClockSec": r["stepSec"]["s05"]})
    write({**base, "runId": "lltMemoryThroughput-T2-precision", "label": "ablation:precision-compile",
           "config": {"batch": 4, "seq": 1024, "timedSteps": 10, "warmupSteps": 3},
           "metrics": [{"variant": k, "tokPerSec": round(v[0]), "firstStepSec": round(v[1], 2)} for k, v in d["speed"].items()],
           "stdout": {"s04": so["s04"], "s06": so["s06"]}, "wallClockSec": r["stepSec"]["s06"]})
    m = re.findall(r"synchronize=(\w+)\s+forward\s+([\d.]+) ms\s+backward\s+([\d.]+) ms", so["s04"])
    write({**base, "runId": "lltMemoryThroughput-T2-broken-no-synchronize", "label": "broken:timing-without-synchronize",
           "config": {"batch": 8, "seq": 1024},
           "metrics": [{"synchronize": a == "True", "forwardMs": float(f), "backwardMs": float(bw)} for a, f, bw in m],
           "stdout": {"s04": so["s04"]}})
    write({**base, "runId": "lltMemoryThroughput-T2-broken-reserved-vs-allocated", "label": "broken:read-reserved-not-allocated",
           "config": {"source": "the same grid as lltMemoryThroughput-T2-memory-grid"},
           "metrics": [x for x in rows if not x["oom"]], "stdout": {"s03": so["s03"]}})
    h = d["half_vs_autocast"]
    write({**base, "runId": "lltMemoryThroughput-T2-broken-fp16-half-adamw", "label": "broken:model.half()-with-AdamW",
           "config": {"batch": 4, "seq": 512, "lr": 6e-4, "optimizer": "AdamW default (foreach), eps 1e-8", "steps": 30,
                      "control": "same seed and batch, fp32 weights, fp16 autocast + GradScaler"},
           "metrics": [{"step": i + 1, "loss": h["half"]["losses"][i], "nonFiniteParams": h["half"]["nonfiniteParams"][i],
                        "controlLoss": h["autocast"]["losses"][i]} for i in range(30)]})
    g1 = {(g[0], g[1]): g[3] for g in v1["data"]["grid"]}
    g2 = {(g[0], g[1]): g[3] for g in d["grid"]}
    write({**base, "runId": "lltMemoryThroughput-T2-broken-leftover-loss", "label": "broken:leftover-loss-keeps-model-alive",
           "notebook": "kaggle genieincodebottle/llt-memthru-p1 version 1 (step 1 without del loss) vs version 2",
           "config": {"bug": "step 1 left loss in scope; its grad_fn kept the step 1 model and its grads alive"},
           "metrics": [{"batch": b, "seq": s, "allocatedGBWithLeftover": g1[(b, s)][0] if g1[(b, s)] else None,
                        "allocatedGBFixed": g2[(b, s)][0] if g2.get((b, s)) else None} for b, s in g1]})


def p2():
    path = os.path.join(K, "llt-memthru-p2", "lltMemoryThroughput_p2.json")
    if not os.path.exists(path):
        return
    r = json.load(open(path))
    base = {"hardware": HW2, "env": r["env"], "notebook": "kaggle genieincodebottle/llt-memthru-p2 version 1", "date": r["started"][:10], "seed": 0}
    so = r["stdout"]
    ar = [{"MiB": int(m[0]), "ms": float(m[1]), "algbwGBs": float(m[2]), "busbwGBs": float(m[3])}
          for m in re.findall(r"(\d+) MiB\s+([\d.]+) ms\s+algbw\s+([\d.]+) GB/s\s+busbw\s+([\d.]+)", so["allreduce"])]
    write({**base, "runId": "lltMemoryThroughput-T2x2-allreduce", "label": "baseline", "config": {"script": "allreduce.py", "dtype": "fp32", "iters": 10},
           "metrics": ar, "stdout": {"allreduce": so["allreduce"]}, "evals": {"topology": r["data"].get("topo")}})
    def ddp(key):
        m = re.search(r"GPUs (\d+)\s+(\S+)\s+([\d,]+) tokens/s\s+last loss per rank \[(.*)\]", so[key])
        return {"run": key, "gpus": int(m[1]), "flags": m[2], "tokPerSec": int(m[3].replace(",", "")), "lastLossPerRank": [float(x) for x in m[4].split(",")]}
    cfg = {"script": "ddp.py", "model": "GPT-2 124M", "microBatch": 8, "seq": 1024, "accum": 4, "timedSteps": 10, "warmupSteps": 3}
    write({**base, "runId": "lltMemoryThroughput-T2x2-ddp", "label": "baseline", "config": cfg,
           "metrics": [ddp(k) for k in ("ddp_1gpu", "ddp_2gpu", "ddp_2gpu_sync_every_micro")],
           "stdout": {k: so[k] for k in ("ddp_1gpu", "ddp_2gpu", "ddp_2gpu_sync_every_micro")}})
    write({**base, "runId": "lltMemoryThroughput-T2x2-broken-same-seed", "label": "broken:identical-data-seed-per-rank", "config": {**cfg, "flag": "--same-seed"},
           "metrics": [ddp(k) for k in ("ddp_2gpu", "ddp_2gpu_same_seed")], "stdout": {k: so[k] for k in ("ddp_2gpu", "ddp_2gpu_same_seed")}})
    write({**base, "runId": "lltMemoryThroughput-T2x2-fsdp2", "label": "baseline",
           "config": {"script": "fsdp.py", "model": "GPT d=2048, 24 layers, 16 heads, 1,313,626,112 parameters", "batchPerGPU": 2, "seq": 1024,
                      "ckpt": True, "chunks": 8, "mp": "param fp16, reduce fp32", "timedSteps": 5},
           "stdout": {k: so[k] for k in ("fsdp_1gpu", "fsdp_2gpu")}})
    write({**base, "runId": "lltMemoryThroughput-T2x2-dcp", "label": "baseline", "config": {"script": "fsdp_ckpt.py", "path": "/tmp/ckpt_fsdp"},
           "stdout": {"fsdp_ckpt": so["fsdp_ckpt"]}, "wallClockSec": r["stepSec"]["fsdp_ckpt"]})
    write({**base, "runId": "lltMemoryThroughput-T2x2-broken-rank0-save", "label": "broken:rank-0-only-save-under-fsdp",
           "config": {"script": "fsdp_ckpt.py, the torch.save branch"}, "stdout": {"fsdp_ckpt": so["fsdp_ckpt"]}})


def p3():
    path = os.path.join(K, "llt-memthru-p3", "lltMemoryThroughput_p3.json")
    if not os.path.exists(path):
        return
    r = json.load(open(path))
    base = {"hardware": HW2, "env": r["env"], "notebook": "kaggle genieincodebottle/llt-memthru-p3", "date": r["started"][:10],
            "config": {"script": "sweep.py", "steps": 600, "warmup": 100, "batch": 32, "seq": 256, "schedule": "linear warmup, cosine to 10%",
                       "optimizer": "AdamW betas (0.9, 0.95), weight decay 0", "clip": None, "precision": "fp16 autocast + GradScaler",
                       "zLoss": 1e-4, "vocab": 4096, "data": "TinyStories, first 200,000 train stories"}}
    for x in r["data"]["sweep"]:
        lbl = {"base": "baseline", "qk": "ablation:qk-norm", "z": "ablation:z-loss"}[x["variant"]]
        write({**base, "runId": f"lltMemoryThroughput-T2-lr-{x['size']}-{x['variant']}-{x['lr']:g}", "label": f"{lbl},lr={x['lr']:g}",
               "seed": x["seed"], "params": x["params"], "metrics": x["metrics"], "evals": {"final": x["final"], "skippedSteps": x["skippedSteps"]},
               "wallClockSec": x.get("wallClockSec")})


def p3_check():
    path = os.path.join(K, "llt-memthru-p3", "lltMemoryThroughput_p3.json")
    if not os.path.exists(path):
        return
    r = json.load(open(path))
    out = r["stdout"]["fsdp_ckpt_check"]
    per_rank = json.loads(out.split(":", 1)[1].strip().replace("'", '"'))
    write({"runId": "lltMemoryThroughput-T2x2-dcp-repeat", "label": "ablation:reload-twice", "hardware": HW2, "env": r["env"],
           "notebook": "kaggle genieincodebottle/llt-memthru-p3 version 1, diag/fsdp_ckpt_check.py", "date": r["started"][:10], "seed": 0,
           "config": {"model": "the 1.31B GPT of fsdp.py", "saveAfterSteps": 3, "thenSteps": 2},
           "metrics": [{"rank": i, **{k: float(v) for k, v in d.items()}} for i, d in enumerate(per_rank)],
           "stdout": {"fsdp_ckpt_check": out}, "wallClockSec": r["stepSec"]["fsdp_ckpt_check"]})


def extras():
    v1 = json.load(open(os.path.join(K, "llt-memthru-p1", "lltMemoryThroughput_p1_v1.json")))
    write({"runId": "lltMemoryThroughput-T2-timing-session1", "label": "baseline", "hardware": HW1, "env": v1["env"],
           "notebook": "kaggle genieincodebottle/llt-memthru-p1 version 1 (an earlier session on another T4 machine)",
           "date": v1["started"][:10], "seed": 0, "config": {"script": "s04_timing.py", "batch": 8, "seq": 1024},
           "metrics": [{"tokPerSec": int(re.search(r"([\d,]+) tokens/s", v1["stdout"]["s04"])[1].replace(",", ""))}],
           "stdout": {"s04": v1["stdout"]["s04"]}})
    smoke = json.load(open(os.path.join(K, "llt-gpu-smoke", "llt_gpu_smoke.json")))
    write({"runId": "lltMemoryThroughput-T2-gpu-smoke", "label": "baseline", "hardware": HW2,
           "env": {"python": smoke["python"], "torch": smoke["torch"], "cuda": smoke["cuda_runtime"]},
           "notebook": "kaggle genieincodebottle/llt-gpu-smoke (kaggle/cells_gpu_smoke.py)", "date": smoke["started"][:10],
           "config": {"matmul": "8192 x 8192 fp16, 4096 x 4096 fp32, 20 timed iterations after 3 warm-up"},
           "metrics": [{"matmulTflopsFp16": smoke["matmul_tflops_fp16"], "matmulTflopsFp32": smoke["matmul_tflops_fp32"],
                        "bf16Supported": smoke["bf16_supported"], "capability": smoke["capability"]}]})
    parts = {}
    for name, f in (("part1", "llt-memthru-p1/lltMemoryThroughput_p1.json"), ("part2", "llt-memthru-p2/lltMemoryThroughput_p2.json"),
                    ("part3", "llt-memthru-p3/lltMemoryThroughput_p3.json")):
        parts[name] = json.load(open(os.path.join(K, f)))["stepSec"]
    sweep = {(x["size"], x["variant"], x["lr"]): x["wallClockSec"] for x in json.load(open(os.path.join(K, "llt-memthru-p3", "lltMemoryThroughput_p3.json")))["data"]["sweep"]}
    learner3 = {"prep": parts["part3"]["prep"],
                "base 3e-3, 1e-2, 3e-2": round(sum(sweep[("10M", "base", lr)] for lr in (0.003, 0.01, 0.03)), 1),
                "qk 3e-3, 1e-2, 3e-2": round(sum(sweep[("10M", "qk", lr)] for lr in (0.003, 0.01, 0.03)), 1)}
    totals = {"part1Sec": round(sum(parts["part1"].values()), 1), "part2Sec": round(sum(parts["part2"].values()), 1),
              "part3LearnerSec": round(learner3["prep"] + max(learner3["base 3e-3, 1e-2, 3e-2"], learner3["qk 3e-3, 1e-2, 3e-2"]), 1)}
    write({"runId": "lltMemoryThroughput-T2-notebook-timing", "label": "baseline", "hardware": HW2, "env": {"python": "3.13.16", "torch": "2.11.0+cu130"},
           "notebook": "step seconds recorded by the kernels llt-memthru-p1 v2, -p2 v1, -p3 v1 (exec time of each shipped cell, GPU session already started)",
           "date": "2026-11-05", "config": {"part3Learner": "prep.py, then the two 3-run sweeps in launch_p3.sh side by side, one per T4"},
           "metrics": [{"part1StepSec": parts["part1"], "part2StepSec": parts["part2"], "part3LearnerSec": learner3, **totals}]})
    print(totals)


def ctl():
    path = os.path.join(K, "llt-memthru-ctl", "lltMemoryThroughput_ctl.json")
    if not os.path.exists(path):
        return
    r = json.load(open(path))
    res = json.loads(r["stdout"]["control"].split("RESULT ", 1)[1].strip().splitlines()[0])
    write({"runId": "lltMemoryThroughput-T2x2-dcp-control", "label": "ablation:reload-without-optimizer-state", "hardware": HW2,
           "env": {**r["env"], **r["data"].get("versions", {})}, "notebook": "kaggle genieincodebottle/llt-memthru-ctl, diag/fsdp_ckpt_control.py",
           "date": r["started"][:10], "seed": 0,
           "config": {"model": "the 1.31B GPT of fsdp.py", "saveAfterSteps": 3, "thenSteps": 2, "data": "random tokens, targets = inputs"},
           "metrics": [{"rank": i, **x} for i, x in enumerate(res)], "evals": {"ramGB": r["data"].get("ramGB")},
           "stdout": {"control": r["stdout"]["control"]}, "wallClockSec": r["stepSec"]["control"]})


p1(); p2(); p3(); p3_check(); extras(); ctl()
