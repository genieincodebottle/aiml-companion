"""lltPretrainData timing figures for the lab page, measured on an idle machine.

Runs the notebook steps 1 to 7 (steps/*.py and the in-page cells) in one process and one namespace,
the way the notebook runs them, in a fresh work directory, and times each step. The WARC slice and
the model files are downloaded inside the timed steps, as a learner downloads them.
Prints one JSON line. Keys match the [[TIME:key]] placeholders in the module:
  t1_notebook_min   steps 1 to 7, minutes, downloads included
  t1_step1_s ... t1_step7_s   each step, seconds

Needs the packages in README.md. On Windows with fasttext-wheel and NumPy 2, the predict() shim
below stands in for fasttext-numpy2 (same call, no copy=False).
Usage: python measure_timing.py [work_dir]
"""
import contextlib, io, json, os, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="ptd_timing_")
os.makedirs(work, exist_ok=True)
os.chdir(work)
sys.path.insert(0, HERE)

try:
    import fasttext.FastText as FT

    def _predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
        pred = self.f.predict(text, k, threshold, on_unicode_error)
        probs, labels = zip(*pred) if pred else ((), ())
        return labels, np.asarray(probs)

    FT._FastText.predict = _predict
except Exception:
    pass

order = [("t1_step1_s", "steps/step1.py"), ("t1_step2_s", "steps/step2.py"), ("t1_step3_s", "browser/rules.py"),
         ("t1_step4_s", "steps/step4.py"), ("t1_step5_s", "steps/step5.py"), ("t1_step6_numpy_s", "browser/minhash.py"),
         ("t1_step6_s", "steps/step6.py"), ("t1_step7_s", "steps/step7.py")]
ns = {"__name__": "__main__"}
out = {}
t_all = time.perf_counter()
for key, rel in order:
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(open(os.path.join(HERE, rel), encoding="utf-8").read(), rel, "exec"), ns)
    out[key] = round(time.perf_counter() - t, 1)
out["t1_notebook_min"] = round((time.perf_counter() - t_all) / 60, 1)
print(json.dumps(out))
