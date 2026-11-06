"""Kaggle kernel llt-insdata-exp (2x T4): the lab notebook end to end on GPU 0, the rest of the experiment on GPU 1.

GPU 0 runs the page's cells in order (labs/lltInstructionData/cells, each wrapped in a timer), which trains
curated-1k and bulk-1k-s0 with seed 0. GPU 1 runs labs/lltInstructionData/worker_insdata.py on the pool and arms
this notebook writes in step 7 (it waits for data/flags.json): curated-1k and bulk-1k-s1 with seed 1, bulk-4k-s0
with seed 0. After the notebook, GPU 0 scores SmolLM2-360M-Instruct on IFEval as a reference and trains bulk-4k-s1.
"""
import json, os, runpy

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs", "lltInstructionData")
head = runpy.run_path(os.path.join(LAB, "notebook_head.py"))
nb = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_insdata.py"))
read = lambda *p: open(os.path.join(LAB, *p), encoding="utf-8").read()

COMMON = '''import collections, json, os
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
device = "cuda"
'''
FILES = {"worker_insdata.py": read("worker_insdata.py"),
         "worker_cells/c00_common.py": COMMON,
         "worker_cells/c08_train.py": read("cells", "c08_train.py"),
         "worker_cells/c09_eval.py": read("cells", "c09_eval.py"),
         "worker_cells/c10_functions.py": "# nothing extra\n"}
Q1 = [{"arm": "curated-1k", "seed": 1}, {"arm": "bulk-1k-s1", "seed": 1}, {"arm": "bulk-4k-s0", "seed": 0}]
Q0_AFTER = [{"arm": "ref-instruct"}, {"arm": "bulk-4k-s1", "seed": 1}]

LAUNCH = "import os, glob, json, shutil, subprocess, sys\n" + "".join(
    "os.makedirs(os.path.dirname(" + repr(k) + ") or '.', exist_ok=True)\nopen(" + repr(k) + ", 'w', encoding='utf-8').write(" + json.dumps(v) + ")\n"
    for k, v in FILES.items()) + r'''
json.dump(''' + json.dumps(Q1) + r''', open("queue1.json", "w"))
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           LLT_WAIT_FOR="data/flags.json")
W1 = subprocess.Popen([sys.executable, "worker_insdata.py", "queue1.json", "data", "exp"], env=env,
                      stdout=open("exp_gpu1.log", "w"), stderr=subprocess.STDOUT)
print("GPU 1 queue waits for data/flags.json, written at the end of step 7")
'''

AFTER = r'''
import time, subprocess, sys, json
json.dump(''' + json.dumps(Q0_AFTER) + r''', open("queue0.json", "w"))
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false")
r0 = subprocess.run([sys.executable, "worker_insdata.py", "queue0.json", "data", "exp"], env=env0, capture_output=True, text=True)
print(r0.stdout[-2000:], r0.stderr[-2000:])
while W1.poll() is None:
    time.sleep(60)
print("GPU 1 exit code", W1.returncode)
print(open("exp_gpu1.log").read()[-6000:])
'''

CELLS = [("md", "# lltInstructionData: notebook on GPU 0, experiment queue on GPU 1"),
         ("code", head["RUNTIME_CHECK"]), ("code", head["INSTALL"]), ("code", LAUNCH)]
CELLS += [c for c in nb["CELLS"][3:]]          # the page cells with timers, then the record cell
CELLS.append(("code", AFTER))
