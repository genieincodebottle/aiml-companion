"""Run the CPU page cells of lltSFT (steps 1 to 7) in order, in one namespace, and record the printed output.

The cells are steps/cellNN_stepK.py, written by `node build_notebook.mjs <module file>` from the page's code blocks.
Writes Run Library file lltSFT-t1-cells; the page quotes it in "What you should see". It also checks that the
page's own selection (step 2) gives the same conversations as sft_common.py (hashes in lltSFT-t1-data).
Wall-clock here is not a page figure (measure_timing.py gives those).

Usage: python run_steps.py           (LLT_THREADS sets torch threads inside the cells, default 4)
"""
import contextlib
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
import sft_common as C  # noqa: E402
from runlog import peak_rss_gb, read_run, write_run  # noqa: E402


def main():
    cells = sorted((HERE / "steps").glob("cell*_step*.py"))
    assert cells, "run node build_notebook.mjs first"
    cpu_cells = [p for p in cells if int(p.stem.split("_step")[1]) <= 7]
    threads = os.environ.get("LLT_THREADS", "4")
    ns = {"__name__": "__main__"}
    outputs = []
    t0 = time.perf_counter()
    for path in cpu_cells:
        src = path.read_text(encoding="utf-8").replace("torch.set_num_threads(4)", f"torch.set_num_threads({threads})")
        buf = io.StringIO()
        c0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, str(path), "exec"), ns)
        outputs.append({"cell": path.name, "stdout": buf.getvalue(), "secShared": round(time.perf_counter() - c0, 1)})
        print("=====", path.name, flush=True)
        print(buf.getvalue(), flush=True)
    data = read_run("lltSFT-t1-data")["evals"]
    hashes = {"train": C.data_hash(ns["train"]), "eval": C.data_hash(ns["held"])}
    same = hashes["train"] == data["trainHash"] and hashes["eval"] == data["evalHash"]
    print("page selection equals sft_common selection:", same)
    write_run("lltSFT-t1-cells", "T1", "baseline",
              {"cells": [p.name for p in cpu_cells], "threads": int(threads), "source": "steps/ from build_notebook.mjs"},
              0, [{"step": 0, "peakRssGB": peak_rss_gb()}], evals={"selectionHashes": hashes, "sameAsSftCommon": same},
              samples=outputs, wall_clock_sec=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
