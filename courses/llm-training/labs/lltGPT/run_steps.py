"""Run the lab's torch step cells in one namespace, in page order, and print each cell's output.

Usage: python run_steps.py [step names...]   (default: step2 ... step7)
The cells are the ones the page and the notebook show. step1 is NumPy and runs on its own.
"""
import contextlib
import io
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
names = sys.argv[1:] or ["step2", "step3", "step4", "step5", "step6", "step6b", "step7"]
import torch
torch.set_num_threads(4)   # exact figures only; timings come from measure_timing.py
ns = {"__name__": "__main__"}
for name in names:
    path = HERE / "steps" / (name + ".py")
    t = time.time()
    print(f"===== {name}", flush=True)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    print(f"----- {name} {time.time() - t:.1f}s", flush=True)
