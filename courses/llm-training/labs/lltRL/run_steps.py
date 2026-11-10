"""Run the CPU page cells of lltRL (steps 1, 2, 7 and 8) in order, in one namespace, and record the printed output.

The cells are steps/cellNN_stepK.py, the same files build_notebook.py puts in the notebook and export_page_code.py puts
in the page. Writes Run Library file lltRL-t1-cells; the page quotes it in "What you should see".
Wall-clock here is not a page figure (measure_timing.py gives the page time).

Usage: python run_steps.py     (math-verify must be importable: pip install math-verify==0.9.0)
"""
import contextlib, io, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
from runlog import peak_rss_gb, write_run  # noqa: E402

CPU_STEPS = (1, 2, 7, 8)


def main():
    cells = [p for p in sorted((HERE / "steps").glob("cell*_step*.py")) if int(p.stem.split("_step")[1]) in CPU_STEPS]
    ns = {"__name__": "__main__"}
    outputs = []
    t0 = time.perf_counter()
    for path in cells:
        buf = io.StringIO()
        c0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        outputs.append({"cell": path.name, "stdout": buf.getvalue(), "secShared": round(time.perf_counter() - c0, 1)})
        print("=====", path.name, flush=True)
        print(buf.getvalue(), flush=True)
    write_run("lltRL-t1-cells", "T1", "baseline", {"cells": [p.name for p in cells], "source": "steps/"},
              0, [{"step": 0, "peakRssGB": peak_rss_gb()}], samples=outputs, wall_clock_sec=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
