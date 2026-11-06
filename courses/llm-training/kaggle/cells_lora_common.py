# Shared setup cell for the lltLoRA Kaggle kernels: pinned HF stack, the lab's lora_lab.py and the worker.
import os
_here = os.path.dirname(os.path.abspath(__file__))
_lab = open(os.path.join(_here, "..", "labs", "lltLoRA", "steps", "lora_lab.py"), encoding="utf8").read()
_w = open(os.path.join(_here, "lora_worker.py"), encoding="utf8").read()

SETUP = r'''
import json, os, subprocess, sys, time, platform
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "tokenizers==0.23.3",
                "huggingface_hub==2.4.0", "peft==0.21.2", "accelerate==1.15.0", "bitsandbytes==0.50.2",
                "datasets==5.1.0"], check=False)
open("/kaggle/working/lora_lab.py", "w").write(''' + repr(_lab) + r''')
open("/kaggle/working/lora_worker.py", "w").write(''' + repr(_w) + r''')
import torch, transformers, peft, bitsandbytes, datasets
ENV = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
       "transformers": transformers.__version__, "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__,
       "datasets": datasets.__version__, "gpu": torch.cuda.get_device_name(0), "gpuCount": torch.cuda.device_count()}
print(ENV, flush=True)
T0 = time.time()
DONE = []
def job(**kw):
    tag = kw["tag"]
    jf = f"/kaggle/working/job_{tag}.json"; rf = f"/kaggle/working/res_{tag}.json"
    json.dump(kw, open(jf, "w"))
    t = time.time()
    p = subprocess.run([sys.executable, "/kaggle/working/lora_worker.py", jf, rf], capture_output=True, text=True)
    open(f"/kaggle/working/log_{tag}.txt", "w").write((p.stdout or "") + "\n---stderr---\n" + (p.stderr or "")[-6000:])
    try:
        r = json.load(open(rf))
    except Exception:
        r = {"tag": tag, "ok": False, "error": "no result file, worker crashed", "stderr": (p.stderr or "")[-3000:]}
        json.dump(r, open(rf, "w"))
    DONE.append(tag)
    print(tag, {k: r.get(k) for k in ("ok", "peakMemGB", "secPerStepMedian", "trainSec", "skippedSteps", "error")},
          "| final", {k: v for k, v in (r.get("evalFinal") or {}).items() if k != "samples"},
          "| wall min", round((time.time() - t) / 60, 1), "| total min", round((time.time() - T0) / 60, 1), flush=True)
    json.dump({"env": ENV, "done": DONE, "minutes": round((time.time() - T0) / 60, 1)}, open("/kaggle/working/summary.json", "w"))
    return r
'''
