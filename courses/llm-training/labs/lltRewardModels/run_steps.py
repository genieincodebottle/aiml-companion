"""Run the CPU page cells of lltRewardModels (steps 1 to 4) in order, in one namespace, and record the printed output.

The cells are steps/cellNN_stepK.py, the same files build_notebook.py puts in the notebook and export_page_code.py puts
in the page. Writes Run Library file lltRewardModels-t1-cells; the page quotes it in "What you should see".
Wall-clock here is not a page figure.

Usage: python run_steps.py            (LLT_THREADS sets torch threads inside the cells, default 4)
"""
import contextlib, io, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
from runlog import peak_rss_gb, write_run  # noqa: E402


def main():
    cells = sorted((HERE / "steps").glob("cell*_step*.py"))
    cpu_cells = [p for p in cells if int(p.stem.split("_step")[1]) <= 4]
    threads = os.environ.get("LLT_THREADS", "4")
    ns = {"__name__": "__main__"}
    outputs = []
    t0 = time.perf_counter()
    for path in cpu_cells:
        src = path.read_text(encoding="utf-8").replace("torch.set_num_threads(4)", "torch.set_num_threads(%s)" % threads)
        buf = io.StringIO()
        c0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, str(path), "exec"), ns)
        outputs.append({"cell": path.name, "stdout": buf.getvalue(), "secShared": round(time.perf_counter() - c0, 1)})
        print("=====", path.name, flush=True)
        print(buf.getvalue(), flush=True)
    write_run("lltRewardModels-t1-cells", "T1", "baseline",
              {"cells": [p.name for p in cpu_cells], "threads": int(threads), "source": "steps/"},
              0, [{"step": 0, "peakRssGB": peak_rss_gb()}], samples=outputs, wall_clock_sec=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
