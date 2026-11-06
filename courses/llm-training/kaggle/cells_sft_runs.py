"""Kernel llt-sft-runs: the lltSFT Run Library on Kaggle's 2x T4.

Selects the SmolTalk2 conversations once (labs/lltSFT/sft_common.py), then runs two queues at the same time, one
per GPU (CUDA_VISIBLE_DEVICES set per worker before torch loads), with labs/lltSFT/worker_sft.py. Every run writes
its JSON to /kaggle/working as soon as it ends.
Build: python build_nb.py cells_sft_runs.py llt-sft-runs llt-sft-runs "LLT SFT Runs"
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "labs", "lltSFT")
COMMON = open(os.path.join(LAB, "sft_common.py"), encoding="utf-8").read()
WORKER = open(os.path.join(LAB, "worker_sft.py"), encoding="utf-8").read()

P = "lltSFT-T2-"
MAIN = dict(assistantOnly=True, packing="bfd", epochs=1)


def cfg(run_id, label, **kw):
    c = dict(MAIN)
    c.update(kw)
    c["runId"] = P + run_id
    c["label"] = label
    return c


GPU0 = [
    cfg("assistant-packed-s0", "baseline", seed=0, genAtStart=True, rawPromptGen=True, lossCheck=True),
    cfg("alltokens-packed-s0", "ablation:loss=all-tokens", seed=0, assistantOnly=False),
    cfg("naive-packing-s0", "ablation:packing=naive", seed=0, packing="naive"),
    cfg("padded-s0", "ablation:packing=none", seed=0, packing="none"),
    cfg("broken-pad-is-eos-s0", "broken:pad-is-eos", seed=0, packing="none", padIsEos=True),
]
GPU1 = [
    {"runId": P + "instruct-reference", "label": "reference", "model": "HuggingFaceTB/SmolLM2-360M-Instruct", "evalOnly": True},
    cfg("assistant-packed-s1", "baseline", seed=1),
    cfg("alltokens-packed-s1", "ablation:loss=all-tokens", seed=1, assistantOnly=False),
    cfg("naive-packing-s1", "ablation:packing=naive", seed=1, packing="naive"),
    cfg("epochs3-s0", "ablation:epochs=3", seed=0, epochs=3, evalEachEpoch=True),
]

CELLS = [
("md", "# lltSFT Run Library\n\nFull SFT of SmolLM2-360M with TRL on SmolTalk2 conversations. Two queues on Kaggle's 2x T4, one per GPU."),
("code", r'''
import json, os, platform, subprocess, sys, time
T_SESSION = time.time()
print("python", platform.python_version())
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"], capture_output=True, text=True).stdout)
# Pin the Hugging Face stack only. torch and numpy stay as the image ships them.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "trl==1.17.1", "accelerate==1.15.0",
                "datasets==5.1.0", "tokenizers==0.23.3", "huggingface_hub==2.4.0"], check=False)
print(subprocess.run([sys.executable, "-c", "import torch, transformers, trl, datasets, accelerate; print(torch.__version__, transformers.__version__, trl.__version__, datasets.__version__, accelerate.__version__)"], capture_output=True, text=True).stdout)
'''),
("code", "os.makedirs('/tmp/lltsft', exist_ok=True)\nCOMMON_SRC = " + repr(COMMON) + "\nopen('/tmp/lltsft/sft_common.py', 'w').write(COMMON_SRC)\n"
         "WORKER_SRC = " + repr(WORKER) + "\nopen('/tmp/lltsft/worker_sft.py', 'w').write(WORKER_SRC)\n"),
("code", r'''
t = time.time()
prep = r"""
import json, sys, time
sys.path.insert(0, "/tmp/lltsft")
import sft_common as C
from transformers import AutoTokenizer
tok = C.setup_tokenizer(AutoTokenizer.from_pretrained(C.BASE_360), AutoTokenizer.from_pretrained(C.INSTRUCT_360))
t = time.time()
train, held, stats = C.select_conversations(tok)
json.dump({"train": train, "eval": held, "stats": stats}, open("/tmp/lltsft/data.json", "w"), ensure_ascii=False)
out = {"stats": stats, "trainHash": C.data_hash(train), "evalHash": C.data_hash(held), "nTrain": len(train), "nEval": len(held), "selectSec": round(time.time() - t, 1)}
json.dump(out, open("/kaggle/working/data_selection.json", "w"), indent=1)
print(out)
"""
r = subprocess.run([sys.executable, "-c", prep], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2000:])
assert r.returncode == 0, "prep failed"
print("prep wall", round(time.time() - t, 1))
'''),
("code", "QUEUES = " + repr([GPU0, GPU1]) + r'''

procs = []
for gpu, q in enumerate(QUEUES):
    qp = f"/tmp/lltsft/queue{gpu}.json"; json.dump(q, open(qp, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    log = open(f"/kaggle/working/worker{gpu}.log", "w")
    procs.append(subprocess.Popen([sys.executable, "/tmp/lltsft/worker_sft.py", qp, "/tmp/lltsft/data.json", "/kaggle/working"],
                                  env=env, stdout=log, stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs):
    time.sleep(300)
    print(round((time.time() - T_SESSION) / 60, 1), "min", [p.poll() for p in procs], flush=True)
    for gpu in range(len(procs)):
        lines = [l for l in open(f"/kaggle/working/worker{gpu}.log").read().splitlines() if l.startswith(("DONE", "lltSFT", "env"))]
        print(gpu, lines[-1][:300] if lines else "", flush=True)
print("exit codes", [p.returncode for p in procs])
json.dump({"sessionSec": round(time.time() - T_SESSION, 1)}, open("/kaggle/working/session.json", "w"))
print(sorted(os.listdir("/kaggle/working")))
'''),
]
