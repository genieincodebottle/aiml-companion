"""Kaggle kernel llt-capstone-gpt2-measure: throughput, memory and MFU for the GPT-2 capstone on 2x T4.
Build: python build_nb.py cells_capstone_gpt2.py llt-capstone-gpt2-measure llt-capstone-gpt2-measure "LLT Capstone GPT2 Measure"
"""
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs", "lltCapstoneGPT2")


def write_cell(name):
    with open(os.path.join(SRC, name), encoding="utf-8") as f:
        src = f.read()
    return ("code", "with open(%r, 'w') as f:\n    f.write(%r)\nprint('wrote', %r)\n" % (name, src, name))


SETUP = r'''
import json, os, platform, subprocess, sys, time
T_START = time.time()
RUN = {"module": "lltCapstoneGPT2", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "probes": [], "benches": [], "log": {}}
os.makedirs("/kaggle/working/data", exist_ok=True)

def save():
    import torch
    RUN["env"] = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    for mod in ("transformers", "datasets", "tiktoken", "numpy", "lm_eval"):
        try:
            RUN["env"][mod] = __import__(mod).__version__
        except Exception:
            pass
    RUN["elapsedSec"] = round(time.time() - T_START, 1)
    RUN["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open("/kaggle/working/lltCapstoneGPT2_measure.json", "w") as f:
        json.dump(RUN, f, indent=1, default=str)

def sh(key, cmd, timeout=900, env=None):
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        out, rc = p.stdout, p.returncode
        if rc != 0:
            out += "\nSTDERR TAIL\n" + p.stderr[-3000:]
    except subprocess.TimeoutExpired as e:
        out, rc = "TIMEOUT " + str(e)[-500:], -9
    RUN["log"][key] = {"rc": rc, "sec": round(time.time() - t, 1), "out": out[-6000:]}
    print(key, "rc", rc, round(time.time() - t, 1), "s")
    print(out[-3000:])
    save()
    return out

def result_line(out):
    for line in out.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return None

import torch
print(platform.python_version(), torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
try:
    import tiktoken
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"], check=True)
save()
'''

PREP = r'''
out = sh("prep", [sys.executable, "prep_data.py", "/kaggle/working/data", "6000000", "500000"], timeout=900)
RUN["data"] = result_line(out)
save()
'''

PROBE = r'''
env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
for key, args in [("probe_124m", ["124m", "0", "8", "12", "16"]),
                  ("probe_124m_ckpt", ["124m", "1", "16", "24", "32"]),
                  ("probe_40m", ["40m", "0", "16", "24", "32", "48"])]:
    r = result_line(sh(key, [sys.executable, "probe.py"] + args, timeout=300, env=env1))
    RUN["probes"].append(r)
save()

def pick(r, ddp_extra_gb):
    ok = [int(b) for b, v in r["peakGB"].items() if v != "OOM" and v + ddp_extra_gb < 14.8]
    return max(ok) if ok else None

P = {r["size"] + ("_ckpt" if r["ckpt"] else ""): r for r in RUN["probes"] if r}
B124 = pick(P["124m"], 4 * P["124m"]["params"] / 1e9 + 0.3)
B124C = pick(P["124m_ckpt"], 4 * P["124m"]["params"] / 1e9 + 0.3)
B40 = pick(P["40m"], 4 * P["40m"]["params"] / 1e9 + 0.3)
RUN["picked"] = {"124m": B124, "124m_ckpt": B124C, "40m": B40}
print(RUN["picked"])
save()
'''


def bench(key, args, budget_min=None):
    guard = "" if budget_min is None else "if time.time() - T_START < %d * 60:\n    " % budget_min
    body = ("r = result_line(sh(%r, [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2', 'bench.py', "
            "'--tag', %r] + %s, timeout=900))\n" % (key, key, args))
    if guard:
        body = guard + body.replace("\n", "\n    ").rstrip() + "\n    RUN['benches'].append(r)\n    save()\n"
    else:
        body += "RUN['benches'].append(r)\nsave()\n"
    return ("code", body)


LMEVAL = r'''
if time.time() - T_START < 26 * 60:
    sh("pip_lmeval", [sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.13"], timeout=300)
    env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    sh("lmeval_gpt2", ["lm_eval", "--model", "hf", "--model_args", "pretrained=openai-community/gpt2,dtype=float16",
                       "--tasks", "hellaswag,arc_easy,piqa,lambada_openai", "--device", "cuda:0", "--batch_size", "64",
                       "--output_path", "/kaggle/working/lmeval"], timeout=900, env=env1)
else:
    RUN["log"]["lmeval_gpt2"] = "skipped: time budget"
save()
import shutil
shutil.rmtree("/kaggle/working/data", ignore_errors=True)
print("total min", round((time.time() - T_START) / 60, 1))
save()
'''

CELLS = [
    ("md", "# lltCapstoneGPT2 measurements (2x T4)\n\nThroughput, peak memory and MFU for the 124M and 40M course GPT "
           "on FineWeb-Edu sample-10BT, DDP across both T4s, fp16 autocast + GradScaler, about 0.5M tokens per step."),
    ("code", SETUP),
    write_cell("gpt.py"), write_cell("prep_data.py"), write_cell("probe.py"), write_cell("bench.py"),
    ("code", PREP),
    ("code", PROBE),
    bench("124m_eager", "['--size', '124m', '--micro', str(B124), '--warmup', '1', '--steps', '3']"),
    bench("124m_compile", "['--size', '124m', '--micro', str(B124), '--warmup', '2', '--steps', '3', '--compile']"),
    bench("40m_eager", "['--size', '40m', '--micro', str(B40), '--warmup', '1', '--steps', '4']"),
    bench("40m_compile", "['--size', '40m', '--micro', str(B40), '--warmup', '2', '--steps', '4', '--compile']"),
    bench("124m_ckpt_eager", "['--size', '124m', '--micro', str(B124C), '--warmup', '1', '--steps', '2', '--ckpt']", 15),
    bench("124m_resume", "['--size', '124m', '--micro', str(B124), '--accum', '2', '--warmup', '1', '--steps', '1', '--resume-check']", 18),
    bench("124m_ckpt_compile", "['--size', '124m', '--micro', str(B124C), '--warmup', '2', '--steps', '2', '--ckpt', '--compile']", 20),
    ("code", LMEVAL),
]
