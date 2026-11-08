"""CPU (T1) timings for the lltEvaluation page. Run on a quiet machine; prints one JSON line.

Runs page cells c01 to c05 on the CPU with the T1 pair (SmolLM2-360M base and Instruct, 200 items per task,
5 MMLU questions per subject) in a temporary folder, 4 threads, then reads the cells' own timers.
Keys: t1_index_s (lm-eval task index), t1_harness_base_s, t1_harness_tuned_s, t1_steps1to5_min, t1_peak_gb (peak RAM).
Usage: python measure_timing.py
"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
work = tempfile.mkdtemp(prefix="llt_eval_timing_")
env = dict(os.environ, LLT_THREADS="4", PYTHONIOENCODING="utf-8")
env.pop("LLT_SMOKE", None)
env.pop("LLT_MODELS", None)
t = time.time()
subprocess.run([sys.executable, str(HERE / "run_cells.py"), work, "c01", "c05"], env=env, check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
total = time.time() - t
h = json.load(open(os.path.join(work, "results", "harness.json"), encoding="utf-8"))
cells = json.load(open(os.path.join(work, "cell_seconds.json"), encoding="utf-8"))
base, tuned = list(h)
index_s = cells["c01_harness"] - h[base]["seconds"] - h[tuned]["seconds"]   # model loads are inside this remainder too
print(json.dumps({"t1_index_s": round(index_s), "t1_harness_base_s": round(h[base]["seconds"]),
                  "t1_harness_tuned_s": round(h[tuned]["seconds"]), "t1_steps1to5_min": round(total / 60, 1),
                  "t1_peak_gb": cells.get("peakRssGB")}))
