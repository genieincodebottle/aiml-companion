"""Shared setup cell for the lltRewardModels Run Library kernels (llt-rm-smoke, llt-rm-pool, llt-rm-train)."""
import os
_w = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_rm.py"), encoding="utf8").read()

SETUP = r'''
import json, os, subprocess, sys, time, platform, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
assert sys.version_info >= (3, 11), "Python 3.11 or newer is needed"
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout)
except FileNotFoundError:
    print("nvidia-smi not found: this session has no GPU")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "trl==1.17.1", "peft==0.21.2", "accelerate==1.15.0",
                "datasets==5.1.0"], check=False)
WORKER = ''' + repr(_w) + r'''
open("/kaggle/working/worker_rm.py", "w").write(WORKER)
import torch, transformers, trl, accelerate, datasets
ALL = {"date": time.strftime("%Y-%m-%d"), "gpu": torch.cuda.get_device_name(0), "gpuCount": torch.cuda.device_count(),
       "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
               "trl": trl.__version__, "accelerate": accelerate.__version__, "datasets": datasets.__version__},
       "runs": []}
print(ALL, flush=True)
T0 = time.time()
POOL = (glob.glob("/kaggle/input/**/pool.json", recursive=True) or ["/kaggle/working/pool.json"])[0]
print("pool file", POOL, os.path.exists(POOL))
def save():
    json.dump(ALL, open("/kaggle/working/llt_rm_%s.json" % TAG, "w"), indent=1)
def job(tag, kind, cap=3600, **kw):
    jf = "/kaggle/working/job_%s.json" % tag; rf = "/kaggle/working/res_%s.json" % tag
    json.dump(dict(tag=tag, kind=kind, pool=POOL, **kw), open(jf, "w"))
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "/kaggle/working/worker_rm.py", jf, rf], capture_output=True, text=True, timeout=cap)
        tail = (p.stdout or "")[-3000:] + (p.stderr or "")[-1500:]
    except subprocess.TimeoutExpired:
        tail = "TIMEOUT after %ss" % cap
    try:
        r = json.load(open(rf))
    except Exception:
        r = {"tag": tag, "ok": False, "error": "no result file / crashed / timeout"}
    r["wallSec"] = round(time.time() - t, 1); r["tail"] = tail
    ALL["runs"].append(r); save()
    print(tag, {k: r.get(k) for k in ("ok", "sPerStepMedian", "peakMemGB", "error", "totalGenSec", "rb2Mean", "lossCheck")}, flush=True)
    print(tail[-1500:], flush=True)
    print("elapsed min", round((time.time() - T0) / 60, 1), flush=True)
'''
