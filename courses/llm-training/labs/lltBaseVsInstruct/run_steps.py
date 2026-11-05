"""Run every page cell of lltBaseVsInstruct in order, in one namespace, and record the printed output.

The cells are steps/cellNN_stepK.py, written by `node build_notebook.mjs <module file>` from the page's own
code blocks. The recorded output is Run Library file lltBaseVsInstruct-t1-cells; the page quotes it in
"What you should see". Wall-clock here is not a page figure (measure_timing.py gives those).

Usage: python run_steps.py            (LLT_THREADS sets torch threads inside the cells, default 4)
"""
import contextlib
import io
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)  # the cells import prompts.py from the working directory, as in the notebook
from runlog import peak_rss_gb, write_run  # noqa: E402


def main():
    cells = sorted((HERE / "steps").glob("cell*_step*.py"))
    assert cells, "run node build_notebook.mjs first"
    threads = os.environ.get("LLT_THREADS", "4")
    ns = {"__name__": "__main__"}
    outputs = []
    t0 = time.perf_counter()
    for path in cells:
        src = path.read_text(encoding="utf-8").replace("torch.set_num_threads(4)", f"torch.set_num_threads({threads})")
        buf = io.StringIO()
        c0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, str(path), "exec"), ns)
        outputs.append({"cell": path.name, "stdout": buf.getvalue(), "secShared": round(time.perf_counter() - c0, 1)})
        print("=====", path.name, flush=True)
        print(buf.getvalue(), flush=True)
    write_run("lltBaseVsInstruct-t1-cells", "T1", "baseline",
              {"cells": [p.name for p in cells], "threads": int(threads), "source": "steps/ from build_notebook.mjs"},
              None, [{"step": 0, "peakRssGB": peak_rss_gb()}], samples=outputs,
              wall_clock_sec=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
