"""Run the lab page's code cells in order, in one shared namespace, the way the notebook runs them.

browser/step1.py to step4.py each run alone (the page gives every in-browser cell a fresh
namespace). steps/step5.py to step10.py share one namespace, as notebook cells do.
Writes ../../runs/lltTokenizer/lltTokenizer-t1-cells.json with each cell's output and time.

Usage: python run_steps.py
"""
import contextlib
import io
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure  # noqa: E402  (reuses its run-file writer)


def run(path, ns):
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return {"stdout": buf.getvalue(), "sec": round(time.perf_counter() - t0, 2)}


if __name__ == "__main__":
    t0 = time.perf_counter()
    cells = {}
    for n in (1, 2, 3, 4):
        cells[f"step{n}"] = run(HERE / "browser" / f"step{n}.py", {"__name__": "__main__"})
        print(f"step{n}", cells[f"step{n}"]["sec"], "s")
    shared = {"__name__": "__main__"}
    for n in (5, 6, 7, 8, 9, 10):
        cells[f"step{n}"] = run(HERE / "steps" / f"step{n}.py", shared)
        print(f"step{n}", cells[f"step{n}"]["sec"], "s")
        print(cells[f"step{n}"]["stdout"])
    measure.write_run("lltTokenizer-t1-cells", "T1", "baseline",
                      {"cells": "browser/step1-4.py alone, steps/step5-10.py in one namespace"},
                      [], {"cells": cells}, time.perf_counter() - t0)
