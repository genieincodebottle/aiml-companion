"""Build the lltDomainAdaptation sweep kernel for one phase.
Usage: python make_da_kernel.py <phase> <qa_split.json>
Writes kaggle/llt-da-sweep/ (notebook + metadata). Each phase lists the jobs for GPU 0 and GPU 1; the two
GPUs of a Kaggle T4 x2 session run their lists in parallel, in separate processes."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "labs", "lltDomainAdaptation")
phase, split_path = sys.argv[1], sys.argv[2]
src = lambda p: open(p, encoding="utf-8").read()

CPT = dict(kind="cpt", rows_per_step=8, micro=4, eval_every=50, qa=True)
PHASES = {
    "smoke": {
        "0": [dict(kind="panel", tag="smoke-panel-q3", model="Qwen/Qwen3-0.6B-Base", qa=True, smoke=True),
              dict(CPT, tag="smoke-cpt-r5", replay=0.05, peak_lr=1e-4, steps=12, eval_every=6, interp=[0.5], save=True, smoke=True)],
        "1": [dict(kind="panel", tag="smoke-panel-q35", model="Qwen/Qwen3.5-0.8B-Base", qa=True, smoke=True),
              dict(CPT, tag="smoke-cpt-ext", replay=0.05, peak_lr=1e-4, steps=6, eval_every=3, interp=[], tokenizer=True,
                   new_tokens=10000, mark_regex=False, init="mean", micro=2, smoke=True),
              dict(kind="qlora", tag="smoke-qlora", model="Qwen/Qwen3-0.6B-Base", epochs=1, batch=8, lr=2e-4, smoke=True)],
    },
}
STEPS = int(os.environ.get("DA_STEPS", "320"))
I3 = [0.25, 0.5, 0.75]
EXT = dict(tokenizer=True, new_tokens=10000, mark_regex=False)
PHASES["main"] = {
    "0": [dict(kind="panel", tag="panel-qwen3-0.6b-base", model="Qwen/Qwen3-0.6B-Base", qa=True),
          dict(CPT, tag="cpt-r0-lr3e-5", replay=0.0, peak_lr=3e-5, steps=STEPS, interp=I3),
          dict(CPT, tag="cpt-r5-lr3e-5", replay=0.05, peak_lr=3e-5, steps=STEPS, interp=I3, save=True),
          dict(kind="qlora", tag="qlora-cpt-r5-lr3e-5", model="/kaggle/working/ckpt_cpt-r5-lr3e-5", epochs=2, batch=8, lr=2e-4),
          dict(kind="qlora", tag="qlora-qwen3-0.6b-base", model="Qwen/Qwen3-0.6B-Base", epochs=2, batch=8, lr=2e-4)],
    "1": [dict(kind="panel", tag="panel-qwen3.5-0.8b-base", model="Qwen/Qwen3.5-0.8B-Base", qa=True),
          dict(CPT, tag="cpt-r25-lr3e-5", replay=0.25, peak_lr=3e-5, steps=STEPS, interp=I3),
          dict(CPT, tag="cpt-r5-lr1e-4", replay=0.05, peak_lr=1e-4, steps=STEPS, interp=[0.5]),
          dict(kind="extinit", tag="extinit-default", **{k: v for k, v in EXT.items() if k != "tokenizer"}),
          dict(CPT, tag="cpt-ext-mean-r5-lr3e-5", replay=0.05, peak_lr=3e-5, steps=STEPS, interp=[0.5], init="mean", micro=2, **EXT)],
}
jobs = PHASES[phase]
SPLIT = open(split_path, encoding="utf-8").read()

SETUP = r'''
import json, os, subprocess, sys, time, platform, itertools
t_start = time.time()
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "peft==0.21.2", "bitsandbytes==0.50.2",
                "accelerate==1.15.0", "datasets==5.1.0", "tokenizers==0.23.3", "huggingface_hub==2.4.0"], check=False)
W = "/kaggle/working"
os.makedirs(W + "/data", exist_ok=True)
for name, text in FILES.items():
    open(W + "/" + name, "w", encoding="utf-8").write(text)
open(W + "/data/qa_split.json", "w", encoding="utf-8").write(SPLIT)
from datasets import load_dataset
def dump(name, ds, n):
    import hashlib
    h = hashlib.sha256(); k = 0
    with open(W + "/data/" + name + ".jsonl", "w", encoding="utf-8") as f:
        for r in itertools.islice(ds, n):
            f.write(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n"); h.update(r["id"].encode()); k += 1
    print(name, k, h.hexdigest()[:16], flush=True)
dump("hi_test", load_dataset("HuggingFaceFW/fineweb-2", name="hin_Deva", split="test", streaming=True), 1000)
dump("en_edu", load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True), 8000)
dump("hi_train", load_dataset("HuggingFaceFW/fineweb-2", name="hin_Deva", split="train", streaming=True), 20000)
print("setup min", round((time.time() - t_start) / 60, 1), flush=True)
'''

RUN = r'''
import glob
CKPT_DIR = glob.glob("/kaggle/input/**/ckpt_cpt-r5-lr1e-4", recursive=True)
for g, lst in JOBS.items():
    for j in lst:
        if str(j.get("model", "")).startswith("CKPT:"):
            j["model"] = CKPT_DIR[0] if CKPT_DIR else "MISSING"
procs = {}
for g, lst in JOBS.items():
    json.dump(lst, open(W + "/jobs_" + g + ".json", "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
    procs[g] = subprocess.Popen([sys.executable, W + "/worker_da.py", W + "/jobs_" + g + ".json", W],
                                stdout=open(W + "/log_gpu" + g + ".txt", "w"), stderr=subprocess.STDOUT, env=env)
while any(p.poll() is None for p in procs.values()):
    time.sleep(60)
    print(round((time.time() - t_start) / 60, 1), "min", {g: open(W + "/log_gpu" + g + ".txt").read()[-200:].replace("\n", " | ") for g in procs}, flush=True)
ALL = {}
for f in sorted(glob.glob(W + "/res_*.json")):
    r = json.load(open(f)); ALL[r["tag"]] = {"ok": r.get("ok"), "wallSec": r.get("wallSec"), "error": r.get("error")}
ALL["totalMin"] = round((time.time() - t_start) / 60, 1)
json.dump(ALL, open(W + "/summary.json", "w"), indent=1)
print(json.dumps(ALL, indent=1))
'''

files = {"dalab.py": src(os.path.join(LAB, "dalab.py")), "extend_tokenizer.py": src(os.path.join(LAB, "extend_tokenizer.py")),
         "worker_da.py": src(os.path.join(HERE, "worker_da.py"))}
CELLS = [("md", "# lltDomainAdaptation sweep, phase " + phase + "\n\nTwo GPUs, one job list each. Results in `res_*.json`."),
         ("code", "FILES = " + repr(files) + "\nSPLIT = " + repr(SPLIT) + "\nJOBS = " + repr(jobs) + "\n" + SETUP),
         ("code", RUN)]
cells_py = os.path.join(HERE, "cells_da_" + phase + ".py")
with open(cells_py, "w", encoding="utf-8") as f:
    f.write("CELLS = " + repr(CELLS) + "\n")
SLUG = {"smoke": ("llt-da-smoke", "LLT DA Smoke"), "main": ("llt-da-sweep", "LLT DA Sweep"), "ext": ("llt-da-ext", "LLT DA Ext")}[phase]
subprocess.run([sys.executable, os.path.join(HERE, "build_nb.py"), cells_py, os.path.join(HERE, SLUG[0]), SLUG[0], SLUG[1]], check=True)
if phase == "ext":
    meta_p = os.path.join(HERE, SLUG[0], "kernel-metadata.json")
    meta = json.load(open(meta_p)); meta["kernel_sources"] = ["genieincodebottle/llt-da-sweep"]
    json.dump(meta, open(meta_p, "w"), indent=1)
print("jobs", json.dumps({g: [j["tag"] for j in l] for g, l in jobs.items()}))
