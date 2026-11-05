"""Run the notebook steps (steps/step*.py and the in-page cells) in ONE Python namespace, in order,
the way the notebook runs them, and record every printed line in the Run Library file
lltPretrainData-t1-notebook. One process, no pool.

Local shim: on Windows this machine has fasttext-wheel, whose predict() wrapper calls
np.array(copy=False) and fails under NumPy 2. The notebook installs fasttext-numpy2 on Linux,
which does not. The shim below patches predict() to the same call without copy=False.

Usage: python run_steps.py <work_dir>
"""
import contextlib, io, os, runpy, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from runlog import write_run  # noqa: E402

work = sys.argv[1]
os.makedirs(work, exist_ok=True)
os.chdir(work)
sys.path.insert(0, HERE)  # ptd.py

import fasttext.FastText as FT  # noqa: E402


def _predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    pred = self.f.predict(text, k, threshold, on_unicode_error)
    probs, labels = zip(*pred) if pred else ((), ())
    return labels, np.asarray(probs)


FT._FastText.predict = _predict

ns = {"__name__": "__main__"}
out = {}
order = ["steps/step1.py", "steps/step2.py", "browser/rules.py", "steps/step4.py", "steps/step5.py",
         "browser/minhash.py", "steps/step6.py", "steps/step7.py"]
t_all = time.time()
for name in order:
    buf = io.StringIO()
    t = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(os.path.join(HERE, name), encoding="utf-8").read(), name, "exec"), ns)
    out[name] = buf.getvalue().splitlines()
    print("=====", name, round(time.time() - t, 1), "s")
    print(buf.getvalue())
peak = None
try:
    import psutil
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)
except Exception:
    pass
write_run("lltPretrainData-t1-notebook", "T1", "baseline",
          {"source": "first 60 MB (62,914,560 bytes) of CC-MAIN-20240220211055-20240221001055-00000.warc.gz", "process": "single"},
          None, [], evals={"stdout": out}, extra={"peakRamGB": round(peak / 1e9, 2) if peak else None})
