"""Run the page cells in order in one namespace, outside a notebook (CPU smoke and the T1 measurement).

Usage: python run_cells.py <work dir> <first cell> <last cell>     e.g. run_cells.py work c01 c05
Env: LLT_SMOKE=1 for a few items; LLT_MODELS="base,tuned" to swap the pair; LLT_THREADS (default 4).
Writes <work dir>/results/*.json (the cells' own outputs) and <work dir>/cell_seconds.json.
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
work, first, last = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(work, exist_ok=True)
os.chdir(work)
import torch
torch.set_num_threads(int(os.environ.get("LLT_THREADS", "4")))
cells = sorted(p for p in (HERE / "cells").glob("c*.py") if first <= p.stem[:3] <= last)
ns = {"__name__": "__main__"}
secs = {}
for p in cells:
    t = time.time()
    print("=" * 20, p.stem, flush=True)
    exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), ns)
    secs[p.stem] = round(time.time() - t, 1)
    print("--", p.stem, secs[p.stem], "s", flush=True)
    json.dump(secs, open("cell_seconds.json", "w"), indent=1)
try:                                             # peak resident memory of this process, for the Lab card
    import psutil
    info = psutil.Process().memory_info()
    secs["peakRssGB"] = round(getattr(info, "peak_wset", 0) / 1e9, 2) or None
except ImportError:
    import resource
    secs["peakRssGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
json.dump(secs, open("cell_seconds.json", "w"), indent=1)
