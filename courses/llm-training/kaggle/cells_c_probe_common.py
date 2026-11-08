import os
_w = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_c_worker.py"), encoding="utf8").read()

SETUP = r'''
import json, os, subprocess, sys, time, platform
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "trl==1.17.1", "peft==0.21.2", "accelerate==1.15.0",
                "bitsandbytes==0.50.2", "datasets==5.1.0"], check=False)
WORKER = ''' + repr(_w) + r'''
open("/kaggle/working/probe_c_worker.py", "w").write(WORKER)
import torch, transformers, peft, trl, accelerate, bitsandbytes, datasets
ALL = {"date": time.strftime("%Y-%m-%d"), "gpu": torch.cuda.get_device_name(0), "gpuCount": torch.cuda.device_count(),
       "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
               "trl": trl.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__, "datasets": datasets.__version__},
       "runs": []}
ALL["trlSurface"] = {n: hasattr(trl, n) for n in ["PPOTrainer", "RewardTrainer", "DPOTrainer", "KTOTrainer", "GRPOTrainer", "RLOOTrainer", "DistillationTrainer", "SFTTrainer", "ORPOTrainer", "CPOTrainer", "GKDTrainer"]}
import importlib
for mod, cls in [("trl.experimental.orpo", "ORPOTrainer"), ("trl.experimental.cpo", "CPOTrainer"), ("trl.experimental.gkd", "GKDTrainer"), ("trl.experimental.gold", "GOLDTrainer"), ("trl.experimental.online_dpo", "OnlineDPOTrainer")]:
    try:
        ALL["trlSurface"][mod + "." + cls] = hasattr(importlib.import_module(mod), cls)
    except Exception as e:
        ALL["trlSurface"][mod + "." + cls] = "ERR " + type(e).__name__
print(ALL, flush=True)
T0 = time.time()
def save():
    json.dump(ALL, open("/kaggle/working/llt_c_probe_%s.json" % TAG, "w"), indent=1)
def job(tag, kind, model, cap=420, **kw):
    jf = "/kaggle/working/job_%s.json" % tag; rf = "/kaggle/working/res_%s.json" % tag
    json.dump(dict(tag=tag, kind=kind, model=model, **kw), open(jf, "w"))
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "/kaggle/working/probe_c_worker.py", jf, rf], capture_output=True, text=True, timeout=cap)
        tail = (p.stdout or "")[-1200:] + (p.stderr or "")[-1200:]
    except subprocess.TimeoutExpired as e:
        tail = "TIMEOUT after %ss" % cap
    try:
        r = json.load(open(rf))
    except Exception:
        r = {"tag": tag, "ok": False, "error": "no result file / crashed / timeout"}
    r["wallSec"] = round(time.time() - t, 1); r["tail"] = tail
    ALL["runs"].append(r); save()
    print(tag, {k: r.get(k) for k in ("ok", "sPerStepMedian", "sPerStepMean", "peakMemGB", "error")}, flush=True)
    print("elapsed min", round((time.time() - T0) / 60, 1), flush=True)
'''
