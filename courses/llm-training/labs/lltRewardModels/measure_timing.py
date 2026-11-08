"""CPU timing for the lltRewardModels page, run on an idle machine (orchestrator pass). Prints one JSON line.

Keys:
  t1_steps1to4_min   wall-clock minutes for page cells 1 to 4 (steps 1 to 4) in one process, 4 torch threads,
                     models and datasets already in the local cache (run run_steps.py once first to fill the cache)
  t1_peak_rss_gb     peak resident memory of that process
"""
import contextlib, io, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import peak_rss_gb  # noqa: E402

cells = [p for p in sorted((HERE / "steps").glob("cell*_step*.py")) if int(p.stem.split("_step")[1]) <= 4]
ns = {"__name__": "__main__"}
t0 = time.perf_counter()
for p in cells:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), ns)
sec = time.perf_counter() - t0
print(json.dumps({"t1_steps1to4_min": round(sec / 60, 1), "t1_peak_rss_gb": peak_rss_gb(), "threads": 4,
                  "python": sys.version.split()[0]}))
