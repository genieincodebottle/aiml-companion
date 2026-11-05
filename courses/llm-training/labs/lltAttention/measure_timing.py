"""Timing figures for the lltAttention page. Run on an idle machine; prints ONE JSON line.

Every key fills a [[TIME:key]] placeholder in the module text:
  notebook_min        cells step4 to step7 in one namespace on a laptop CPU, minutes (includes the
                      TinyStories download in step 7; torch.set_num_threads(4) as in step 4)
  step7_min           cell step7 alone (the three 600-step training runs, page step 8), minutes
  notebook_peak_gb    peak resident memory of this process over steps 4 to 7, GB (psutil; not a timing,
                      but measured on the same run)
  pyodide_load_s      Pyodide 0.27.0 start plus numpy load, seconds
  pyodide_slowest_ms  the slowest browser cell (step1, step2, step3, challenge solution), median of 3, ms
The Pyodide keys appear only when LLT_PYODIDE_DIR names a folder with node_modules/pyodide
(npm i pyodide@0.27.0). CPU only. GPU timings on the page come from the Kaggle T4 run
(runs/lltAttention/lltAttention-t2-t4-sweep.json), not from this script.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_cell(name, ns):
    path = HERE / "cells" / (name + ".py")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)


def main():
    out = {}
    ns = {"__name__": "__main__"}
    t = time.perf_counter()
    for name in ("step4", "step5", "step6"):
        run_cell(name, ns)
    t7 = time.perf_counter()
    run_cell("step7", ns)
    end = time.perf_counter()
    out["notebook_min"] = round((end - t) / 60, 1)
    out["step7_min"] = round((end - t7) / 60, 1)
    import psutil
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)
    if peak is None:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    out["notebook_peak_gb"] = round(peak / 1e9, 1)

    pdir = os.environ.get("LLT_PYODIDE_DIR")
    if pdir and (Path(pdir) / "node_modules" / "pyodide").exists():
        shutil.copy(HERE / "pyodide_time.mjs", Path(pdir) / "pyodide_time_attention.mjs")
        subprocess.run(["node", "pyodide_time_attention.mjs", str(HERE)], cwd=pdir, check=True, capture_output=True)
        rec = json.loads((HERE.parent.parent / "runs" / "lltAttention" / "lltAttention-t0-pyodide.json").read_text(encoding="utf-8"))
        out["pyodide_load_s"] = round(rec["evals"]["pyodideLoadMs"] / 1000, 1)
        out["pyodide_slowest_ms"] = max(s["medianMs"] for s in rec["evals"]["steps"].values())
    print(json.dumps(out))


if __name__ == "__main__":
    main()
