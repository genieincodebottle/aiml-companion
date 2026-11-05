"""Run the page's cells in order (step1 alone, steps 2 to 8 in one namespace, then the challenge)
and record every cell's printed output and the peak RAM to runs/lltGPT/lltGPT-t1-steps.json.
step8b (the 6-variant training cell) is recorded by measure_toggles.py instead.
"""
import contextlib
import io
import os
import runpy
import sys
from pathlib import Path

THREADS = int(os.environ.get("LLT_THREADS", "4"))
os.environ.setdefault("RAYON_NUM_THREADS", str(THREADS))
import psutil  # noqa: E402
import torch  # noqa: E402

from runlog import write_run

HERE = Path(__file__).resolve().parent
torch.set_num_threads(THREADS)


def capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


cells = {}
cells["step1"] = capture(lambda: runpy.run_path(str(HERE / "steps" / "step1.py"), run_name="__main__"))
ns = {"__name__": "__main__"}
for name in ["step2", "step3", "step4", "step5", "step6", "step6b", "step7", "step8"]:
    src = (HERE / "steps" / f"{name}.py").read_text(encoding="utf-8")
    cells[name] = capture(lambda: exec(compile(src, name, "exec"), ns))
    print(f"===== {name}\n{cells[name]}", flush=True)
for name in ["challenge_solution", "challenge_starter"]:
    try:
        cells[name] = capture(lambda: runpy.run_path(str(HERE / "steps" / f"{name}.py"), run_name="__main__"))
    except AssertionError as e:
        cells[name] = "AssertionError: " + str(e)
mi = psutil.Process().memory_info()
peak = getattr(mi, "peak_wset", None) or mi.rss
write_run("lltGPT-t1-steps", "T1", "baseline", {"threads": THREADS, "dtype": "float32", "cells": list(cells)}, 0, [],
          evals={"stdout": cells, "peakRamGB": round(peak / 1e9, 2)})
print("peak RAM GB", round(peak / 1e9, 2))
