"""Kaggle kernels llt-eval-smoke and llt-eval-main (lltEvaluation).

The page's cells (labs/lltEvaluation/cells/c01..c11, after the notebook's runtime check and install cells) run in
order on GPU 0, each wrapped in a timer and a try block so one failure does not end the session. The experiment
queue (labs/lltEvaluation/worker_eval.py) runs on GPU 1 when the session has two GPUs, otherwise on GPU 0 after the
notebook. Then LightEval 0.13.0 is tried in its own virtual environment, so it cannot change the pinned stack.
Build: python build_nb.py cells_eval.py <dir> <slug> "<title>" with LLT_KERNEL=smoke or main in the environment.
"""
import json, os, runpy

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "labs", "lltEvaluation")
MODE = os.environ.get("LLT_KERNEL", "main")
head = runpy.run_path(os.path.join(LAB, "notebook_head.py"))
read = lambda *p: open(os.path.join(LAB, *p), encoding="utf-8").read()
CELL_FILES = sorted(f for f in os.listdir(os.path.join(LAB, "cells")) if f.startswith("c") and f.endswith(".py"))
FILES = {"worker_eval.py": read("worker_eval.py")}
FILES.update({"cells/" + f: read("cells", f) for f in CELL_FILES})
JOBS = os.environ.get("LLT_JOBS", "extra,seed1,safety,gsm8k")   # empty: the notebook only
RUN_LE = os.environ.get("LLT_LIGHTEVAL", "1") == "1"

SETUP = "import os, json, time, subprocess, sys, traceback" + chr(10) + (
    "os.environ['LLT_SMOKE'] = '1'" + chr(10) if MODE == "smoke" else "") + "".join(
    "os.makedirs(os.path.dirname(" + repr(k) + ") or '.', exist_ok=True)" + chr(10) +
    "open(" + repr(k) + ", 'w', encoding='utf-8').write(" + json.dumps(v) + ")" + chr(10) for k, v in FILES.items()) + r'''
T0 = time.time()
REC = {"mode": "''' + MODE + r'''", "cellSeconds": {}, "cellErrors": {}}
def save_rec():
    json.dump(REC, open("eval_record.json", "w"), indent=1)
try:
    gl = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
except FileNotFoundError:
    gl = ""
REC["gpus"] = gl.strip().splitlines()
N_GPU = len(REC["gpus"])
print("GPUs:", REC["gpus"], flush=True)
W1 = None
JOBS = "''' + JOBS + r'''"
if N_GPU >= 2 and JOBS:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    W1 = subprocess.Popen([sys.executable, "worker_eval.py", "cells", "exp", "''' + JOBS + r'''"], env=env,
                          stdout=open("worker.log", "w"), stderr=subprocess.STDOUT)
    print("experiment queue started on GPU 1", flush=True)
def run_cell(name, src):
    t = time.time()
    try:
        exec(compile(src, name, "exec"), globals())
    except Exception:
        REC["cellErrors"][name] = traceback.format_exc()[-3000:]
        print("CELL FAILED", name, REC["cellErrors"][name], flush=True)
    REC["cellSeconds"][name] = round(time.time() - t, 1)
    print("--", name, REC["cellSeconds"][name], "s | session", round((time.time() - T0) / 60, 1), "min", flush=True)
    save_rec()
save_rec()
'''

AFTER = r'''
import gc, glob, shutil
from importlib.metadata import version
REC["env"] = {"python": sys.version.split()[0]}
for pkg in ["torch", "numpy", "transformers", "tokenizers", "huggingface_hub", "datasets", "peft", "accelerate", "lm-eval", "nltk", "langdetect"]:
    try:
        REC["env"][pkg] = version(pkg)
    except Exception as e:
        REC["env"][pkg] = "missing"
save_rec()
for k in ["LMS", "FT", "canary"]:
    globals().pop(k, None)
gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
except Exception:
    pass
if W1 is None and JOBS:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    r = subprocess.run([sys.executable, "worker_eval.py", "cells", "exp", "''' + JOBS + r'''"], env=env, stdout=open("worker.log", "w"), stderr=subprocess.STDOUT)
    REC["workerExit"] = r.returncode
elif W1 is not None:
    while W1.poll() is None:
        time.sleep(30)
    REC["workerExit"] = W1.returncode
REC["workerMinutes"] = round((time.time() - T0) / 60, 1)
save_rec()
if JOBS:
    print(open("worker.log").read()[-5000:], flush=True)

if ''' + str(RUN_LE) + r''':
    # LightEval 0.13.0 (released 2025-11-24), installed last into this session so it cannot touch the runs above.
    LE = {}
    def sh(cmd, timeout):
        t = time.time()
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return {"cmd": cmd[-300:], "rc": p.returncode, "sec": round(time.time() - t, 1), "tail": ((p.stdout or "")[-2500:] + (p.stderr or "")[-2500:])}
        except subprocess.TimeoutExpired:
            return {"cmd": cmd[-300:], "rc": -9, "sec": round(time.time() - t, 1), "tail": "TIMEOUT"}
    LE["install"] = sh(sys.executable + " -m pip install lighteval==0.13.0 2>&1 | grep -i -E 'successfully installed|error|conflict' | tail -5", 900)
    LE["versions"] = sh(sys.executable + " -c \"from importlib.metadata import version as v; print('lighteval', v('lighteval'), 'transformers', v('transformers'), 'torch', v('torch'), 'nltk', v('nltk'))\"", 120)
    LE["tasks"] = sh("lighteval tasks list 2>&1 | grep -i -E 'arc' | head -20", 300)
    lim = "16" if os.environ.get("LLT_SMOKE") == "1" else "1000"
    for spec in ["lighteval|arc:easy|0", "lighteval|arc:easy|0|0", "leaderboard|arc:easy|0"]:
        k = "run " + spec
        LE[k] = sh("cd /kaggle/working && lighteval accelerate \"model_name=HuggingFaceTB/SmolLM2-360M,dtype=float16\" \"" + spec + "\" --max-samples " + lim + " --output-dir /kaggle/working/le_out", 900)
        print(k, LE[k]["rc"], LE[k]["sec"], LE[k]["tail"][-600:], flush=True)
        if LE[k]["rc"] == 0:
            break
        if "litellm[caching]" in LE[k]["tail"] and "diskcache" not in LE:
            LE["diskcache"] = sh(sys.executable + " -m pip install -q diskcache 2>&1 | tail -3", 300)
            LE[k + " after diskcache"] = sh("cd /kaggle/working && lighteval accelerate \"model_name=HuggingFaceTB/SmolLM2-360M,dtype=float16\" \"" + spec + "\" --max-samples " + lim + " --output-dir /kaggle/working/le_out", 900)
            print(k, "after diskcache", LE[k + " after diskcache"]["rc"], LE[k + " after diskcache"]["tail"][-600:], flush=True)
            if LE[k + " after diskcache"]["rc"] == 0:
                break
    LE["results"] = [open(f).read()[-4000:] for f in glob.glob("/kaggle/working/le_out/**/*.json", recursive=True)][:3]
    REC["lighteval"] = LE
REC["totalMinutes"] = round((time.time() - T0) / 60, 1)
save_rec()
print("DONE", REC["totalMinutes"], "min", flush=True)
'''

CELLS = [("md", "# lltEvaluation (" + MODE + "): page cells on GPU 0, experiment queue on GPU 1"),
         ("code", head["RUNTIME_CHECK"]), ("code", head["INSTALL"]), ("code", SETUP)]
CELLS += [("code", "run_cell(" + repr(f[:-3]) + ", open(" + repr("cells/" + f) + ", encoding='utf-8').read())") for f in CELL_FILES]
CELLS.append(("code", AFTER))
