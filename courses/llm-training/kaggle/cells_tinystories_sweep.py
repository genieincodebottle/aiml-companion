"""Kernel llt-tinystories-sweep: the lltTinyStories Run Library sweep on Kaggle's 2x T4.

Encodes TinyStories ONCE, caches the token ids as .npy, then runs two training queues at the same
time, one per GPU (CUDA_VISIBLE_DEVICES set per worker before torch loads). Every run writes its
Run Library JSON to /kaggle/working as soon as it ends.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = open(os.path.join(HERE, "prep_tinystories.py"), encoding="utf-8").read()
WORKER = open(os.path.join(HERE, "worker_tinystories.py"), encoding="utf-8").read()

PROMPTS = ["Once upon a time", "Tom and his dog went to the park. Tom",
           "Lily wanted to bake a cake for her mom, but", "The little bird could not fly. One day,",
           "Max found a shiny key under his bed."]
BASE = dict(vocab=4096, d=320, layers=8, heads=8, ctx=512, batch=64, lr=1e-3, warmup=200, steps=3000,
            eval_every=250, eval_batches=40, wd=0.1, clip=1.0, seed=0, prompts=PROMPTS)


def cfg(run_id, label, **kw):
    c = dict(BASE); c.update(kw); c["runId"] = run_id; c["label"] = label
    return c


P = "lltTinyStories-T2-"
GPU0 = [
    cfg(P + "size30m-s0", "ablation:size=30M", d=512, layers=9, heads=8, micro=32, saveFinal=True),
    cfg(P + "broken-lr1e-2-s0", "broken:lr=1e-2", lr=1e-2, stopIfDiverged=True),
    cfg(P + "broken-lr3e-2-s0", "broken:lr=3e-2", lr=3e-2, stopIfDiverged=True),
]
GPU1 = [
    cfg(P + "size3m-s0", "ablation:size=3M", d=192, layers=5, heads=6, saveFinal=True),
    cfg(P + "baseline-s1", "baseline", seed=1, saveAt=2000, resumeTests=[0, 100], saveFinal=True),
    cfg(P + "deep-s0", "ablation:shape=deep", d=256, layers=13, heads=8, saveFinal=True),
    cfg(P + "wide-s0", "ablation:shape=wide", d=448, layers=4, heads=8, saveFinal=True),
]

CELLS = [
("md", "# lltTinyStories sweep (Run Library)\n\nTwo training queues on Kaggle's 2x T4, one per GPU, over one shared token cache."),
("code", r'''
import json, os, platform, subprocess, sys, time
T_SESSION = time.time()
print("python", platform.python_version())
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "tokenizers"], check=False)
'''),
("code", "PREP_SRC = " + repr(PREP) + "\nopen('/tmp/prep.py', 'w').write(PREP_SRC)\n"
         "WORKER_SRC = " + repr(WORKER) + "\nopen('/tmp/worker.py', 'w').write(WORKER_SRC)\n"),
("code", r'''
t = time.time()
r = subprocess.run([sys.executable, "/tmp/prep.py", "/tmp/cache", "/kaggle/working"], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-3000:])
assert r.returncode == 0, "prep failed"
print("prep wall", round(time.time() - t, 1))
'''),
("code", "QUEUES = " + repr([GPU0, GPU1]) + r'''

procs = []
for gpu, q in enumerate(QUEUES):
    qp = f"/tmp/queue{gpu}.json"; json.dump(q, open(qp, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    log = open(f"/kaggle/working/worker{gpu}.log", "w")
    procs.append(subprocess.Popen([sys.executable, "/tmp/worker.py", qp, "/tmp/cache", "/kaggle/working"], env=env, stdout=log, stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs):
    time.sleep(300)
    print(round((time.time() - T_SESSION) / 60, 1), "min", [p.poll() for p in procs], flush=True)
    for gpu in range(len(procs)):
        lines = open(f"/kaggle/working/worker{gpu}.log").read().splitlines()
        print(gpu, lines[-1][:200] if lines else "", flush=True)
print("exit codes", [p.returncode for p in procs])
json.dump({"sessionSec": round(time.time() - T_SESSION, 1)}, open("/kaggle/working/session.json", "w"))
print(sorted(os.listdir("/kaggle/working")))
'''),
]
