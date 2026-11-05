"""Kernel llt-ptd-prep: lltPretrainData data prep on a Kaggle CPU session (no GPU quota).

Embeds labs/lltPretrainData/ptd.py and prep_ptd.py, installs the extraction and dedup libraries,
and writes every token file, the funnel counts and the classifier scores to /kaggle/working for the
GPU kernel llt-ptd-train (attached as a kernel source).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "labs", "lltPretrainData")
PTD = open(os.path.join(LAB, "ptd.py"), encoding="utf-8").read()
PREP = open(os.path.join(LAB, "prep_ptd.py"), encoding="utf-8").read()

CELLS = [
("md", "# lltPretrainData data prep (CPU)\n\nThe funnel on one Common Crawl WARC file, the quality classifier, and the training corpora for the GPU kernel."),
("code", r'''
import os, platform, subprocess, sys, time
print("python", platform.python_version(), "| cpus", os.cpu_count())
# One package per call, so one failed build cannot block the rest. fasttext-numpy2 ships cp313 wheels.
for pkg in ["trafilatura", "datasketch", "warcio", "fasttext-numpy2", "datatrove", "spacy"]:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True)
    print(pkg, "exit", r.returncode, r.stderr[-400:])
for m in ["trafilatura", "datasketch", "warcio", "fasttext", "datatrove", "spacy", "tokenizers", "sklearn", "numpy", "pandas"]:
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", repr(e)[:120])
'''),
("code", "PTD_SRC = " + repr(PTD) + "\nopen('/tmp/ptd.py', 'w').write(PTD_SRC)\n"
         "PREP_SRC = " + repr(PREP) + "\nopen('/tmp/prep_ptd.py', 'w').write(PREP_SRC)\n"),
("code", r'''
t = time.time()
env = dict(os.environ, PTD_RAW="/tmp/raw", PTD_PROCS="4")
p = subprocess.Popen([sys.executable, "/tmp/prep_ptd.py", "/kaggle/working"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
p.wait()
print("prep exit", p.returncode, "wall", round(time.time() - t, 1))
assert p.returncode == 0
'''),
("code", r'''
import json
s = json.load(open("/kaggle/working/ptd_prep_stats.json"))
print(json.dumps({k: s[k] for k in ["stages", "c4", "classifier", "corpora", "tokenizer", "timings"] if k in s}, indent=1)[:12000])
for f in sorted(os.listdir("/kaggle/working")):
    print(f, os.path.getsize(os.path.join("/kaggle/working", f)))
'''),
]
