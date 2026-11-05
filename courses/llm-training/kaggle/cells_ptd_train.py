"""Kernel llt-ptd-train: the lltPretrainData training experiments on Kaggle's 2x T4.

Reads the token files written by the CPU kernel llt-ptd-prep (attached as a kernel source) and runs
two queues at the same time, one per GPU (CUDA_VISIBLE_DEVICES set per worker before torch loads).
Every run writes its Run Library JSON to /kaggle/working as soon as it ends.

GPU 0: unfiltered vs filtered vs classifier-filtered (2 seeds for U and F), WSD anneal on F vs E.
GPU 1: canaries at 1/10/100/1,000 copies on a 3M and a 30M model, epochs 1/2/4/8 at a fixed token
budget, five mixtures of web, Wikipedia and stories (one weighted by documents, the broken run).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = open(os.path.join(HERE, "..", "labs", "lltPretrainData", "worker_ptd.py"), encoding="utf-8").read()
STEPS_MAIN = int(os.environ.get("PTD_STEPS_MAIN", "800"))

BASE = dict(vocab=4096, d=320, layers=8, heads=8, ctx=512, batch=64, lr=1e-3, warmup=100, steps=STEPS_MAIN,
            eval_every=200, wd=0.1, clip=1.0, seed=0, evalTokens=131072, evalTokensFinal=600000)
EV_MAIN = ["F_val", "U_val", "U_val_clean", "U_val_leak", "wiki_test", "ts_val"]
EV_MIX = ["F_val", "wiki_test", "ts_val"]
S3M = dict(d=192, layers=5, heads=6)
S30M = dict(d=512, layers=9, heads=8, micro=32)


def cfg(run_id, label, **kw):
    c = dict(BASE); c.update(kw); c["runId"] = "lltPretrainData-T2-" + run_id; c["label"] = label
    return c


def single(corpus, **kw):
    d = {"kind": "single", "corpus": corpus}; d.update(kw); return d


GPU0 = [
    cfg("unfiltered-s0", "ablation:data=unfiltered", data=single("U_train"), evals=EV_MAIN, hellaswag=True),
    cfg("filtered-s0", "baseline", data=single("F_train"), evals=EV_MAIN, hellaswag=True),
    cfg("edu-s0", "ablation:data=classifier-top-half", data=single("E_train"), evals=EV_MAIN, hellaswag=True),
    cfg("unfiltered-s1", "ablation:data=unfiltered", seed=1, data=single("U_train"), evals=EV_MAIN, hellaswag=True),
    cfg("filtered-s1", "baseline", seed=1, data=single("F_train"), evals=EV_MAIN, hellaswag=True),
    cfg("wsd-anneal-F", "ablation:anneal=F", steps=600, schedule="wsd", decayFrac=0.2, data=single("F_train"),
        anneal={"corpus": "F_train", "frac": 0.2}, evals=["F_val", "wiki_test", "U_val"]),
    cfg("wsd-anneal-E", "ablation:anneal=E", steps=600, schedule="wsd", decayFrac=0.2, data=single("F_train"),
        anneal={"corpus": "E_train", "frac": 0.2}, evals=["F_val", "wiki_test", "U_val"]),
]
CAN = {"kind": "canary", "corpus": "F_train", "levels": [0, 1, 10, 100, 1000], "perLevel": 32}
EP = 600 * 64 * 512  # token budget of each epoch run
GPU1 = [
    cfg("canary-3m", "experiment:canary,size=3M", data=CAN, canaryEval=True, evals=["F_val"], **S3M),
    cfg("canary-30m", "experiment:canary,size=30M", data=CAN, canaryEval=True, evals=["F_val"], **S30M),
] + [
    cfg("epochs-%d" % k, "ablation:epochs=%d" % k, steps=600, data=single("F_train", uniqueTokens=EP // k + 1),
        evals=["F_val"], trainSetEval=True)
    for k in (1, 2, 4, 8)
] + [
    cfg("mix-%s" % name, label, steps=300, eval_every=100, data={"kind": "mixture", "by": by, "domains": w}, evals=EV_MIX)
    for name, label, by, w in [
        ("equal-tokens", "ablation:mixture=equal-by-tokens", "tokens", {"F_train": 1 / 3, "wiki_train": 1 / 3, "ts_train": 1 / 3}),
        ("equal-docs", "broken:weights-by-documents", "docs", {"F_train": 1 / 3, "wiki_train": 1 / 3, "ts_train": 1 / 3}),
        ("web60", "ablation:mixture=web60", "tokens", {"F_train": 0.6, "wiki_train": 0.2, "ts_train": 0.2}),
        ("wiki60", "ablation:mixture=wiki60", "tokens", {"F_train": 0.2, "wiki_train": 0.6, "ts_train": 0.2}),
        ("stories60", "ablation:mixture=stories60", "tokens", {"F_train": 0.2, "wiki_train": 0.2, "ts_train": 0.6}),
    ]
]

CELLS = [
("md", "# lltPretrainData training runs (Run Library)\n\nTwo queues on Kaggle's 2x T4, one per GPU, over the token files from llt-ptd-prep."),
("code", r'''
import glob, json, os, platform, subprocess, sys, time
T_SESSION = time.time()
print("python", platform.python_version())
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"], capture_output=True, text=True).stdout)
hits = glob.glob("/kaggle/input/**/ptd_prep_stats.json", recursive=True)
print(hits)
DATA = os.path.dirname(hits[0])
print(sorted(os.listdir(DATA)))
'''),
("code", "WORKER_SRC = " + repr(WORKER) + "\nopen('/tmp/worker.py', 'w').write(WORKER_SRC)\n"),
("code", r'''
# Check the notebook's language-ID call on Linux with fasttext-numpy2 (the lab page's step 2 code path).
import urllib.request
try:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fasttext-numpy2"], capture_output=True, text=True)
    print("pip fasttext-numpy2 exit", r.returncode)
    import fasttext
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", "/tmp/lid.176.bin")
    lid = fasttext.load_model("/tmp/lid.176.bin")
    labels, probs = lid.predict("The library opens at nine every morning and closes at six.", k=1)
    print("fasttext-numpy2 predict", labels, probs, type(probs))
    json.dump({"labels": list(labels), "probs": [float(p) for p in probs]}, open("/kaggle/working/fasttext_numpy2_check.json", "w"))
except Exception as e:
    print("fasttext check failed", repr(e))
'''),
("code", "QUEUES = " + repr([GPU0, GPU1]) + r'''

procs = []
for gpu, q in enumerate(QUEUES):
    qp = f"/tmp/queue{gpu}.json"; json.dump(q, open(qp, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = open(f"/kaggle/working/worker{gpu}.log", "w")
    procs.append((subprocess.Popen([sys.executable, "/tmp/worker.py", qp, DATA, "/kaggle/working"], env=env, stdout=log, stderr=subprocess.STDOUT), log))
while any(p.poll() is None for p, _ in procs):
    time.sleep(60)
    for gpu in range(2):
        lines = open(f"/kaggle/working/worker{gpu}.log").read().splitlines()
        print(gpu, lines[-1][:300] if lines else "")
for p, log in procs:
    log.close(); print("exit", p.returncode)
print("session wall", round(time.time() - T_SESSION, 1))
for gpu in range(2):
    txt = open(f"/kaggle/working/worker{gpu}.log").read()
    print("=== worker", gpu, "tail"); print(txt[-4000:])
'''),
]
