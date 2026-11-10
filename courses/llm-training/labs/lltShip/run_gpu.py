"""Run every lltShip page cell in order on one GPU machine, in one namespace, exactly as the notebook does, and save
what each cell measured. Used for our recorded runs on the Google Cloud VM (llt-c3, A100) and for timing.

    python run_gpu.py --mode smoke      tiny sizes, every branch on (code path check only, no page numbers)
    python run_gpu.py --mode full       RUN_GRID, RUN_QAT and RUN_VLLM on: the numbers on the page
    python run_gpu.py --mode defaults   the learner defaults: what the notebook does with no flag changed (timing)

Writes ship_<mode>.json (every value the cells kept in `runs`, per-cell seconds, peak GPU memory, versions) and
stdout_<mode>.txt next to it. record_gpu.py turns these into Run Library files.
"""
import argparse, contextlib, io, json, os, platform, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["smoke", "full", "defaults"], required=True)
ap.add_argument("--cells", default="1-12")
args = ap.parse_args()
HERE = Path(__file__).resolve().parent
lo, hi = (int(x) for x in args.cells.split("-"))

PATCH = {
    "full": [("RUN_GRID = False", "RUN_GRID = True"), ("RUN_VLLM = False", "RUN_VLLM = True")],
    "defaults": [],
    "smoke": [("RUN_VLLM = False", "RUN_VLLM = True"),
              ("(200, 300, 100) if", "(8, 8, 4) if"), ("STEPS, MICRO, ACC = 150, 1, 8", "STEPS, MICRO, ACC = 3, 1, 2"),
              ('"--chunks", "40"', '"--chunks", "4"'), ("def smoke(port, n=20)", "def smoke(port, n=3)"),
              ("[:128]", "[:16]"), (".take(64)", ".take(8)"), ("select(range(64))", "select(range(8))")],
}[args.mode]


class Tee(io.TextIOBase):
    def __init__(self, path):
        self.f, self.o = open(path, "w", encoding="utf-8"), sys.__stdout__
    def write(self, x):
        self.o.write(x); self.f.write(x); self.f.flush(); return len(x)
    def flush(self):
        self.o.flush(); self.f.flush()


def to_json(o):
    if isinstance(o, dict):
        return {str(k): to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, float) and o != o:
        return None
    return o


out_dir = Path.cwd()
sys.stdout = Tee(out_dir / ("stdout_" + args.mode + ".txt"))
ns = {"__name__": "__main__"}
cells = sorted((HERE / "steps").glob("cell*_step*.py"))
times, t_all = {}, time.time()
import torch  # noqa: E402
versions = {"python": platform.python_version()}
for name in ("torch", "transformers", "peft", "datasets", "tokenizers", "huggingface_hub", "numpy", "safetensors", "gguf", "mergekit"):
    try:
        versions[name] = __import__(name).__version__
    except Exception:
        pass


def dump(status):
    # Written after every cell, so a preempted or failed run still leaves what the finished cells measured
    rec = {"mode": args.mode, "cells": args.cells, "status": status, "cellSec": times, "totalSec": round(time.time() - t_all, 1),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
           "cpuCount": os.cpu_count(), "versions": versions, "runs": to_json(ns.get("runs", {})), "best": ns.get("best"),
           "date": time.strftime("%Y-%m-%d")}
    tmp = out_dir / ("ship_" + args.mode + ".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_dir / ("ship_" + args.mode + ".json"))
    return rec


for path in cells:
    k = int(path.stem.split("_step")[1])
    if not lo <= k <= hi:
        continue
    src = path.read_text(encoding="utf-8")
    for a, b in PATCH:
        src = src.replace(a, b)
    print("=====", path.name, time.strftime("%H:%M:%S"), flush=True)
    t = time.time()
    try:
        exec(compile(src, str(path), "exec"), ns)
    except BaseException:
        dump("failed in " + path.name)
        raise
    times[path.name] = round(time.time() - t, 1)
    print("===== done", path.name, times[path.name], "s", flush=True)
    dump("partial, last " + path.name)

rec = dump("complete")
print("WROTE ship_" + args.mode + ".json", rec["totalSec"], "s")
