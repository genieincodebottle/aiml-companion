"""Time the two optional switches on a T4: RUN_GRID = True (step 5, nine more merges) and RUN_VLLM = True (step 12), with the
exit off. Runs steps 1 to 4 for the shared state, then step 5 with the grid, then step 12 with vLLM, in one process.

    python run_extras_t4.py       writes extras_t4.json and stdout_extras.txt in the current folder
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import torch
ns = {"__name__": "__main__"}
times, t_all = {}, time.time()
PATCH = {5: [("RUN_GRID = False", "RUN_GRID = True")], 12: [("RUN_VLLM = False", "RUN_VLLM = True")]}
PATCH_ALL = [("FROM_OUR_CHECKPOINT = True", "FROM_OUR_CHECKPOINT = False")]
for k in (1, 2, 3, 4, 5, 12):
    src = (HERE / "steps" / ("cell%02d_step%d.py" % (k, k))).read_text(encoding="utf-8")
    for a, b in PATCH_ALL + PATCH.get(k, []):
        src = src.replace(a, b)
    print("=====", k, time.strftime("%H:%M:%S"), flush=True)
    t = time.time()
    torch.cuda.reset_peak_memory_stats()
    exec(compile(src, "cell%02d" % k, "exec"), ns)
    times[k] = {"sec": round(time.time() - t, 1), "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print("===== done", k, times[k], flush=True)
    r = ns.get("runs", {})
    json.dump({"cellSec": times, "totalSec": round(time.time() - t_all, 1), "best": ns.get("best"),
               "grid": {n: {kk: v for kk, v in m.items() if kk in ("hiBpb", "enBpb", "replyNll", "kept", "mergeSec", "panelSec")} for n, m in r.items() if isinstance(m, dict) and "kept" in m},
               "vllm": r.get("vllm")}, open("extras_t4.json", "w"), indent=1, default=str)
