"""Build lltPretrainData.ipynb from the same code files the lab page embeds (steps/, browser/, ptd.py),
so the page and the notebook cannot drift. Band checks come from run lltPretrainData-t1-notebook.

Usage: python build_notebook.py
"""
import json, os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "..", "runs", "lltPretrainData", "lltPretrainData-t1-notebook.json")


def src(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return f.read().rstrip() + "\n"


runtime_check = '''# Runtime check. Run this first. It prints the fix for each problem it finds.
import platform, shutil, sys
print("Python", platform.python_version())
if sys.version_info < (3, 11):
    print("FIX: use Python 3.11 or later. On Colab, Runtime > Change runtime type, keep the default runtime.")
try:
    import psutil
    print(f"RAM {psutil.virtual_memory().total / 1e9:.1f} GB")
except ImportError:
    pass
free = shutil.disk_usage(".").free / 1e9
print(f"Free disk {free:.1f} GB")
if free < 2:
    print("FIX: steps 1 to 7 download about 0.4 GB. Free some disk space.")
print("Steps 1 to 7 run on a CPU. No GPU is needed.")'''

stdout = json.load(open(RUN, encoding="utf-8"))["evals"]["stdout"]


def line_with(step, prefix):
    for ln in stdout[step]:
        if ln.startswith(prefix):
            return ln
    raise KeyError(prefix)


band = f'''# Band checks against our recorded run (lltPretrainData-t1-notebook, one CPU process).
# Counts can move by a page or two with another trafilatura version.
print("pages with text", len(docs), "| English", len(english), "| after rules", len(kept), "| after dedup", len(deduped))
assert abs(len(docs) - {int(line_with("steps/step1.py", "WARC records").split("pages with text")[1])}) <= 3, "step 1 count differs. Check that the slice is the first 62,914,560 bytes."
assert abs(len(english) - {int(line_with("steps/step2.py", "English with score").split("|")[0].split()[-1])}) <= 3, "step 2 count differs. Check the 0.65 threshold."
assert abs(len(kept) - {int(line_with("steps/step4.py", "kept").split()[1])}) <= 3, "step 4 count differs. Check ptd.py."
print("All band checks passed")'''

peak = '''# Peak RAM of this notebook so far
import psutil, sys
mi = psutil.Process().memory_info()
peak_b = getattr(mi, "peak_wset", None)
if peak_b is None:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_b = r * 1024 if sys.platform.startswith("linux") else r
print(f"Peak RAM {peak_b / 1e9:.2f} GB")'''

gpu_md = ("## Steps 8 to 10. Training runs on a T4\n\n"
          "These steps train 18 small GPTs. Our runs used two Kaggle sessions: a CPU session for the data prep "
          "(`prep_ptd.py`, 35 min on 4 cores) and a 2x T4 session for the training queues "
          "(`worker_ptd.py`). Both scripts are in this folder of the repository, with the queues in "
          "`kaggle/cells_ptd_train.py`. Every result is in `runs/lltPretrainData/`.\n\n"
          "The cells below run on a CPU: the extraction test from step 9, the token accounting and the RegMix "
          "regression from step 10.")

# fasttext-numpy2 ships wheels for Linux x86_64 only (PyPI, checked 2026-11-06) and its source build
# fails with MSVC 2019 on Windows. Elsewhere the cell installs fasttext-wheel 0.9.2 (wheels for Windows
# and Intel macOS up to Python 3.12, Apple Silicon up to 3.11) and patches its predict(), which calls
# np.array(copy=False) under NumPy 2.
install_cell = '''# Pinned versions of the recorded runs. pip runs inside this notebook's own Python.
import platform, subprocess, sys
pkgs = ["trafilatura==2.3.1", "warcio==1.8.2"]
linux = platform.system() == "Linux" and platform.machine() == "x86_64"
if linux:
    pkgs.append("fasttext-numpy2==0.10.4")  # the NumPy 2 build of fastText, Linux wheels only
elif sys.version_info >= (3, 13):
    print("FIX: on Windows and macOS, fastText has wheels up to Python 3.12. Use a Python 3.12 environment.")
elif platform.system() == "Darwin" and platform.machine() == "arm64" and sys.version_info >= (3, 12):
    print("FIX: on Apple Silicon, fastText has wheels up to Python 3.11. Use a Python 3.11 environment.")
else:
    pkgs.append("fasttext-wheel==0.9.2")  # prebuilt for Windows and macOS, patched below for NumPy 2
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
if any(p.startswith("fasttext-wheel") for p in pkgs):
    import numpy as np
    import fasttext.FastText as FT

    def _predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
        pred = self.f.predict(text, k, threshold, on_unicode_error)
        probs, labels = zip(*pred) if pred else ((), ())
        return labels, np.asarray(probs)

    FT._FastText.predict = _predict
    print("fasttext-wheel predict() patched for NumPy 2. After a restart, run this cell again.")
print("Installed", ", ".join(pkgs))'''

md = nbf.v4.new_markdown_cell
cc = nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
    md("# lltPretrainData, Turn Raw Web Pages Into Pretraining Data\n\n"
       "LLM Training Lab, module 9. Steps 1 to 7 run on a CPU on the first 60 MB of one Common Crawl WARC file. "
       "Our recorded runs are in `courses/llm-training/runs/lltPretrainData/`."),
    cc(runtime_check),
    md("Install the three libraries this lab adds. NumPy, pandas, scikit-learn and PyTorch come with Colab and Kaggle, and this notebook never installs NumPy."),
    cc(install_cell),
    md("The filters, PII scrub, decontamination and MinHash code, as one file the steps import."),
    cc("%%writefile ptd.py\n" + src("ptd.py")),
    md("## Step 1. Download one Common Crawl file and extract the text"),
    cc(src("steps/step1.py")),
    md("## Step 2. Keep English pages with fastText language ID"),
    cc(src("steps/step2.py")),
    md("## Step 3. Write the Gopher quality rules\n\nThe same cell runs in the lab page."),
    cc(src("browser/rules.py")),
    md("## Step 4. Run the Gopher and C4 rules over every page"),
    cc(src("steps/step4.py")),
    md("## Step 5. Train a quality classifier on pages scored by Llama 3"),
    cc(src("steps/step5.py")),
    md("## Step 6. Remove near-duplicates with MinHash and LSH"),
    cc(src("browser/minhash.py")),
    cc(src("steps/step6.py")),
    md("## Step 7. Decontaminate against benchmarks and scrub personal data\n\n"
       "Predict first. How many of 20 reworded ARC questions does a 13-gram check catch?"),
    cc(src("steps/step7.py")),
    md("## Band checks"),
    cc(band),
    cc(peak),
    md(gpu_md),
    cc(src("steps/step9.py")),
    cc(src("browser/accounting.py")),
    cc(src("browser/regmix.py")),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = os.path.join(HERE, "lltPretrainData.ipynb")
nbf.write(nb, out)
print("wrote", out, len(nb.cells), "cells")
