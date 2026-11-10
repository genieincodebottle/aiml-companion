"""Run the CPU page cells of lltShip (steps 1, 2, 3 and 6) in order, in one namespace, and record the printed output.

The cells are steps/cellNN_stepK.py, the same files build_notebook.py puts in the notebook and export_page_code.py puts
in the page. Writes Run Library files lltShip-t1-cells (steps 1 to 3) and lltShip-t1-mergekit (step 6).
mergekit-yaml must be on PATH (pip install mergekit==0.1.5). Wall-clock here is not a page figure except the two
merge times of step 6, which the page labels as our laptop's.
"""
import contextlib, io, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORK = HERE / "_work"
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
from runlog import peak_rss_gb, write_run  # noqa: E402

ns = {"__name__": "__main__"}
out = {}
for k in (1, 2, 3, 6):
    path = next((HERE / "steps").glob("cell*_step%d.py" % k))
    buf = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    out[k] = {"cell": path.name, "stdout": buf.getvalue(), "secShared": round(time.perf_counter() - t, 1)}
    print("=====", path.name, flush=True)
    print(buf.getvalue(), flush=True)
write_run("lltShip-t1-cells", "T1", "baseline", {"cells": [out[k]["cell"] for k in (1, 2, 3)], "threads": 4, "source": "steps/"},
          0, [{"step": 0, "peakRssGB": peak_rss_gb()}], samples=[out[k] for k in (1, 2, 3)])
write_run("lltShip-t1-mergekit", "T1", "baseline", {"cell": out[6]["cell"], "method": "ties", "density": 0.2, "dtype": "float32"},
          0, [{"step": 0, "peakRssGB": peak_rss_gb()}], samples=[out[6]], wall_clock_sec=out[6]["secShared"])
import shutil  # noqa: E402
shutil.rmtree(WORK / "merged-mergekit", ignore_errors=True)
