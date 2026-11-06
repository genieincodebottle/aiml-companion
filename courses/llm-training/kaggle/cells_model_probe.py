import os
_w = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_worker.py"), encoding="utf8").read()

SETUP = r'''
import json, os, subprocess, sys, time, platform
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "peft==0.21.2", "accelerate", "bitsandbytes", "datasets"], check=False)
WORKER = ''' + repr(_w) + r'''
open("/kaggle/working/probe_worker.py", "w").write(WORKER)
import torch, transformers, peft
import bitsandbytes
try:
    import trl; trlv = trl.__version__
except Exception as e:
    trlv = "n/a"
ALL = {"date": time.strftime("%Y-%m-%d"), "gpu": torch.cuda.get_device_name(0), "gpuCount": torch.cuda.device_count(),
       "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "transformers": transformers.__version__, "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__, "trl": trlv},
       "runs": []}
def fl():
    r = {}
    for m in ("fla", "causal_conv1d"):
        try:
            __import__(m); r[m] = True
        except Exception:
            r[m] = False
    return r
ALL["envFlaBefore"] = fl()
print(ALL, flush=True)
T0 = time.time()
def save():
    json.dump(ALL, open("/kaggle/working/llt_model_probe.json", "w"), indent=1)
def job(tag, model, mode, steps, batch, cap):
    jf = f"/kaggle/working/job_{tag}.json"; rf = f"/kaggle/working/res_{tag}.json"
    json.dump(dict(tag=tag, model=model, mode=mode, steps=steps, batch=batch, time_cap=cap), open(jf, "w"))
    t = time.time()
    p = subprocess.run([sys.executable, "/kaggle/working/probe_worker.py", jf, rf], capture_output=True, text=True)
    tail = (p.stdout or "")[-1500:] + (p.stderr or "")[-1500:]
    try:
        r = json.load(open(rf))
    except Exception:
        r = {"tag": tag, "ok": False, "error": "no result file, worker crashed"}
    r["wallSec"] = round(time.time() - t, 1)
    r["tail"] = tail
    ALL["runs"].append(r); save()
    print(tag, {k: r.get(k) for k in ("ok", "loadedClass", "visionTowerLoaded", "sPerStepMedian", "peakMemGB", "loss1", "lossLast", "error")}, flush=True)
    print("elapsed min", round((time.time() - T0) / 60, 1), flush=True)
'''

PHASE_A = r'''
CAP = 420
job("q3-0.6b-lora", "Qwen/Qwen3-0.6B-Base", "lora", 20, 4, CAP)
job("q35-0.8b-lora", "Qwen/Qwen3.5-0.8B-Base", "lora", 20, 4, CAP)
job("q3-1.7b-lora", "Qwen/Qwen3-1.7B-Base", "lora", 20, 4, CAP)
job("q35-2b-lora", "Qwen/Qwen3.5-2B-Base", "lora", 20, 4, CAP)
job("q3-0.6b-fullft", "Qwen/Qwen3-0.6B-Base", "fullft", 2, 4, 300)
job("q35-0.8b-fullft", "Qwen/Qwen3.5-0.8B-Base", "fullft", 2, 4, 300)
'''

PHASE_9B = r'''
job("q35-9b-qlora", "Qwen/Qwen3.5-9B-Base", "qlora", 5, 2, 900)
'''

PHASE_B = r'''
# If a fast path was missing in phase A: install flash-linear-attention (Triton, no compile) and re-measure Qwen3.5.
flnote = {}
if not ALL["envFlaBefore"].get("fla"):
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flash-linear-attention"], capture_output=True, text=True)
    flnote["pipFlaRc"] = p.returncode; flnote["pipFlaTail"] = (p.stdout + p.stderr)[-600:]
    flnote["after"] = fl()
    ALL["flaInstall"] = flnote; save(); print(flnote, flush=True)
    if flnote["after"].get("fla") and (time.time() - T0) / 60 < 55:
        job("q35-0.8b-lora-fla", "Qwen/Qwen3.5-0.8B-Base", "lora", 20, 4, 300)
        if (time.time() - T0) / 60 < 58:
            job("q35-2b-lora-fla", "Qwen/Qwen3.5-2B-Base", "lora", 20, 4, 300)
ALL["totalMinutes"] = round((time.time() - T0) / 60, 1); save()
print("DONE", ALL["totalMinutes"])
'''
CELLS = [("md", "# LLT model probe: Qwen3.5 vs Qwen3 on a free T4\n\nWrites `llt_model_probe.json`."),
         ("code", SETUP), ("code", PHASE_A), ("code", PHASE_9B), ("code", PHASE_B)]
