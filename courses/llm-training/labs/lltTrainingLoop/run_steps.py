"""Run page code blocks in order in one namespace, the way the notebook runs them.

Usage: python run_steps.py <workdir> <block.py> [<block.py> ...]
Each block prints its own figures; the page quotes those printed lines.
"""
import os
import sys
import time

workdir, files = sys.argv[1], sys.argv[2:]
paths = [os.path.abspath(f) for f in files]
os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)
ns = {"__name__": "__main__"}
for p in paths:
    t = time.time()
    print("=====", os.path.basename(p), flush=True)
    with open(p, encoding="utf-8") as f:
        exec(compile(f.read(), p, "exec"), ns)
    print("----- done in", round(time.time() - t, 1), "s", flush=True)
