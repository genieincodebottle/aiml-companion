"""CPU timing for the lltShip page, run on an idle laptop (orchestrator pass). Prints one JSON line.

Keys:
  t1_steps123_min   wall-clock minutes for page cells 1 to 3 in one process, 4 torch threads, models already cached
  t1_step6_min      wall-clock minutes for page cell 6 (mergekit and your TIES merge) after cells 1 to 3
  t1_peak_rss_gb    peak resident memory of that process
mergekit-yaml must be on PATH. Run run_steps.py once first to fill the model cache.
"""
import contextlib, io, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Step 1 reads small config files and headers from the Hub, so this needs the network; weights come from the cache
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import peak_rss_gb  # noqa: E402

work = HERE / "_work"
work.mkdir(exist_ok=True)
os.chdir(work)
ns = {"__name__": "__main__"}
t = {}
for k in (1, 2, 3, 6):
    p = next((HERE / "steps").glob("cell*_step%d.py" % k))
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), ns)
    t[k] = time.perf_counter() - t0
print(json.dumps({"t1_steps123_min": round((t[1] + t[2] + t[3]) / 60, 1), "t1_step6_min": round(t[6] / 60, 1),
                  "t1_peak_rss_gb": peak_rss_gb(), "threads": 4, "python": sys.version.split()[0]}))
