"""Run the learner notebook's code cells in order, in one namespace, on this machine (T1, CPU).

Usage: python run_cells.py [steps]   (steps overrides the T1 step count, for a quick check)
Env: LLT_CACHE (token cache folder), LLT_THREADS (torch threads, default 4).
Writes the notebook's own run file plus a Run Library copy in ../../runs/lltTinyStories/.
"""
import io, json, os, platform, sys, time, contextlib, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells_lab import CELLS

if len(sys.argv) > 1:
    os.environ["LLT_STEPS"] = sys.argv[1]
threads = int(os.environ.get("LLT_THREADS", "4"))
ns = {"__name__": "__main__"}
t0 = time.time()
for i, (kind, src) in enumerate(CELLS):
    if kind != "code":
        continue
    if "torch.set_num_threads(max(1" in src:
        src = src.replace("torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))", "torch.set_num_threads(" + str(threads) + ")")
    print("=== cell", i, flush=True)
    exec(compile(src, "cell" + str(i), "exec"), ns)
wall = time.time() - t0
run = json.load(open(ns["run"]["runId"] + ".json"))
steps = ns["CFG"]["steps"]
rid = "lltTinyStories-T1-cpu-" + ("full" if steps == ns["CONFIGS"]["T1"]["steps"] else "steps" + str(steps)) + "-s0" + os.environ.get("LLT_RUN_SUFFIX", "")
rec = {"runId": rid, "module": "lltTinyStories", "tier": "T1", "label": "baseline",
       "hardware": {"gpu": None, "count": 0, "cpu": "AMD Ryzen 9 5900HS (8 cores, 16 threads), 32 GB RAM", "torchThreads": threads,
                    "note": "machine shared with other jobs; timing figures come from measure_timing.py"},
       "env": {"python": platform.python_version(), "torch": ns["torch"].__version__,
               "tokenizers": __import__("tokenizers").__version__, "numpy": ns["np"].__version__,
               "datasets": __import__("datasets").__version__},
       "config": ns["CFG"], "seed": ns["SEED"], "params": ns["N_PARAMS"],
       "data": {"trainTokens": int(ns["train_ids"].size), "valTokens": int(ns["val_ids"].size)},
       "metrics": run["metrics"], "samples": run["samples"], "date": datetime.date.today().isoformat(),
       "trainSec": round(ns["TRAIN_SEC"], 1), "wallClockSec": round(wall, 1),
       "command": "python run_cells.py " + " ".join(sys.argv[1:])}
out = os.path.normpath(os.path.join(HERE, "..", "..", "runs", "lltTinyStories", rid + ".json"))
json.dump(rec, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", out)
