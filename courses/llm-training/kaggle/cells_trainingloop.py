"""Kaggle kernel cells for lltTrainingLoop: the LR x seed sweep on the 10.77M char model, 2x T4.

The training code is the page's own code blocks (labs/lltTrainingLoop/steps/*.py), read at build
time and written to lltloop.py on Kaggle, so the sweep runs exactly what the page shows.
Each GPU runs one worker process; a worker trains its jobs one after another and writes one
Run Library JSON per run to /kaggle/working/runs/.
"""
import os

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs", "lltTrainingLoop", "steps")
DEFS = ["s1_data.py", "s2_model.py", "s3_sched.py", "s5_ckpt.py", "s6_train.py"]
lib = "\n\n".join(open(os.path.join(LAB, f), encoding="utf-8").read() for f in DEFS)
lib += "\n\nchars = prepare()\n"

WORKER = r'''
import json, os, platform, sys, time
WORK = os.environ.get("LLT_WORKDIR", "/kaggle/working")
os.chdir(WORK)
sys.path.insert(0, WORK)
import torch
import lltloop as L
CUDA = torch.cuda.is_available()
jobs = json.load(open(sys.argv[1]))

def clean(x):
    if isinstance(x, float) and (x != x or abs(x) == float("inf")):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x

START, LIMIT = float(os.environ["LLT_START"]), float(os.environ["LLT_LIMIT_SEC"])
for job in jobs:
    if time.time() - START > LIMIT:
        print("time limit reached, skipping", job["runId"], flush=True)
        continue
    cfg = job["cfg"]
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        model, hist = L.train(cfg)
        status = "ok"
    except Exception as e:
        hist, status = [], "error: " + repr(e)
    wall = time.time() - t0
    tokens = cfg["steps"] * cfg["micro_batch"] * cfg["accum"] * cfg["block"]
    nan_steps = sum(1 for r in hist if r["trainLoss"] != r["trainLoss"])
    keep = [dict(r, tokens=r["step"] * cfg["micro_batch"] * cfg["accum"] * cfg["block"]) for r in hist
            if r["step"] % 10 == 0 or "valLoss" in r or r["step"] <= 10]
    vals = [r["valLoss"] for r in hist if "valLoss" in r]
    finite = [v for v in vals if v == v and abs(v) != float("inf")]
    rec = {
        "runId": job["runId"], "module": "lltTrainingLoop", "tier": "T2", "label": job["label"],
        "hardware": {"gpu": torch.cuda.get_device_name(0) if CUDA else None, "count": 1 if CUDA else 0, "vramGB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if CUDA else None,
                     "capability": list(torch.cuda.get_device_capability()) if CUDA else None, "note": "Kaggle 2x T4 session, one run per GPU"},
        "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": L.np.__version__,
                "dtype": str(L.pick_dtype()), "bf16SupportedFlag": torch.cuda.is_bf16_supported() if CUDA else None},
        "notebookCommit": None, "config": cfg, "seed": cfg["seed"], "status": status,
        "params": sum(p.numel() for p in L.GPT(len(L.chars), cfg["block"], cfg["n_layer"], cfg["n_head"], cfg["d"]).parameters()),
        "metrics": keep,
        "evals": {"finalValLoss": vals[-1] if vals else None,
                  "bestValLoss": min(finite) if finite else None,
                  "nanTrainSteps": nan_steps},
        "wallClockSec": round(wall, 1), "tokPerSecInclEval": round(tokens / wall) if hist else None,
        "peakMemGB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if CUDA else None,
        "costUSD": 0, "priceCheckedOn": None, "date": time.strftime("%Y-%m-%d"),
    }
    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", job["runId"] + ".json"), "w") as f:
        json.dump(clean(rec), f, indent=1)
    print(job["runId"], status, "final val", rec["evals"]["finalValLoss"], "wall", rec["wallClockSec"], "tok/s", rec["tokPerSecInclEval"], flush=True)
'''

CELLS = [
("md", "# LLM Training Lab, lltTrainingLoop: learning-rate sweep on a 10.77M char model (T4)\n\nFROM SCRATCH. Char-level Tiny Shakespeare. Learning rates 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 with 3 seeds each, then extra seeds at the best rate. fp16 autocast + GradScaler, dtype picked by compute capability. One run per GPU on Kaggle's 2x T4."),
("code", r'''
# Runtime check. It prints the fix for each problem it finds.
import platform, shutil, sys, time
import torch
START = time.time()
print("Python", platform.python_version(), "| torch", torch.__version__)
if not torch.cuda.is_available():
    print("FIX: no GPU. Kaggle: Settings > Accelerator > GPU T4 x2. Colab: Runtime > Change runtime type > T4 GPU.")
else:
    for i in range(torch.cuda.device_count()):
        print("GPU", i, torch.cuda.get_device_name(i), "capability", torch.cuda.get_device_capability(i))
    if torch.cuda.get_device_capability(0) < (7, 0):
        print("FIX: this GPU is older than the T4 and current PyTorch wheels may not support it. Choose a T4.")
    print("torch.cuda.is_bf16_supported() =", torch.cuda.is_bf16_supported(), "(True on a T4 too, so do not use it to pick the dtype)")
print("Free disk", round(shutil.disk_usage(".").free / 1e9, 1), "GB")
'''),
("code", "LIB = " + repr(lib) + "\nimport os\nWORK = os.environ.get('LLT_WORKDIR', '/kaggle/working')\nopen(os.path.join(WORK, 'lltloop.py'), 'w').write(LIB)\nWORKER = " + repr(WORKER) + "\nopen(os.path.join(WORK, 'worker.py'), 'w').write(WORKER)\nprint('wrote lltloop.py', len(LIB), 'chars and worker.py')"),
("code", r'''
import json, os, subprocess, sys
os.chdir(WORK)
sys.path.insert(0, WORK)
import lltloop as L
print("vocab", len(L.chars), "| device", L.device, "| dtype", L.pick_dtype())
T2 = dict(n_layer=6, n_head=6, d=384, block=256, dropout=0.2, micro_batch=64, accum=1, steps=2000,
          warmup=100, schedule="cosine", wd=0.1, clip=1.0, eval_every=250)
if os.environ.get("LLT_SMOKE"):  # local CPU check of this notebook only
    T2.update(n_layer=2, n_head=2, d=32, block=32, micro_batch=4, steps=20, eval_every=10)
LIMIT = float(os.environ.get("LLT_LIMIT_SEC", str(150 * 60)))

def run_pool(jobs, tag):
    n = max(1, torch.cuda.device_count())
    procs = []
    for g in range(n):
        mine = jobs[g::n]
        if not mine:
            continue
        path = f"jobs_{tag}_{g}.json"
        json.dump(mine, open(path, "w"))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), LLT_START=str(START), LLT_LIMIT_SEC=str(LIMIT))
        procs.append(subprocess.Popen([sys.executable, "worker.py", path], env=env,
                                      stdout=open(f"worker_{tag}_{g}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    for g in range(n):
        if os.path.exists(f"worker_{tag}_{g}.log"):
            print(open(f"worker_{tag}_{g}.log").read())

def name(lr, seed):
    return "lltTrainingLoop-t2-lr" + format(lr, ".0e").replace("e-0", "e-") + "-s" + str(seed)

# Sanity run first: 30 steps. If it fails, nothing else starts.
run_pool([{"runId": "lltTrainingLoop-t2-sanity", "label": "sanity", "cfg": dict(T2, lr=1e-3, seed=0, steps=30, eval_every=30)}], "sanity")
sanity = json.load(open("runs/lltTrainingLoop-t2-sanity.json"))
assert sanity["status"] == "ok", sanity["status"]
print("sanity ok, final val", sanity["evals"]["finalValLoss"], "tok/s", sanity["tokPerSecInclEval"])
'''),
("code", r'''
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
jobs = [{"runId": name(lr, s), "label": "ablation:lr=" + format(lr, ".0e"), "cfg": dict(T2, lr=lr, seed=s)} for s in range(3) for lr in LRS]
run_pool(jobs, "sweep")
res = {}
for j in jobs:
    p = "runs/" + j["runId"] + ".json"
    if os.path.exists(p):
        r = json.load(open(p))
        res.setdefault(j["cfg"]["lr"], []).append(r["evals"]["finalValLoss"] if r["evals"]["finalValLoss"] is not None else float("nan"))
for lr in LRS:
    print(lr, res.get(lr))
pool = [lr for lr in res if all(v == v for v in res[lr])]
best = min(pool, key=lambda lr: sum(res[lr]) / len(res[lr])) if pool else None
print("best lr", best, "elapsed min", round((time.time() - START) / 60, 1))
'''),
("code", r'''
extra = [{"runId": name(best, s), "label": "seeds:lr=" + format(best, ".0e"), "cfg": dict(T2, lr=best, seed=s)} for s in range(3, 10)] if best else []
if extra:
    run_pool(extra, "seeds")
vals = []
for s in range(10):
    p = "runs/" + name(best, s) + ".json" if best else ""
    if best and os.path.exists(p):
        vals.append(json.load(open(p))["evals"]["finalValLoss"])
print("best lr", best, "seeds", len(vals), "final val", vals)
summary = {"best": best, "sweep": {str(k): v for k, v in res.items()}, "bestSeeds": vals,
           "sessionMinutes": round((time.time() - START) / 60, 1)}
json.dump(summary, open("runs/lltTrainingLoop-t2-summary.json", "w"), indent=1)
print(summary)
'''),
]
