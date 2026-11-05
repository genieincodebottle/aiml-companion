import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_cells_common import SETUP, src, write_file_cell, step_cell

LRS = ["1e-3", "3e-3", "1e-2", "3e-2", "1e-1", "3e-1"]
# Two worker queues, one per T4, balanced by cost (a 27M run costs about 2.5 10M runs).
QUEUES = {
    "0": [("10M", "base"), ("10M", "qk"), ("10M", "z"), ("27M", "base", "first3")],
    "1": [("27M", "qk"), ("27M", "base", "last3")],
}

RUNNER = r'''
import json, glob
LRS = %s
QUEUES = %s
def cmds(q):
    out = []
    for item in q:
        size, variant = item[0], item[1]
        lrs = LRS if len(item) == 2 else (LRS[:3] if item[2] == "first3" else LRS[3:])
        out.append(["python", "sweep.py", size, variant] + lrs)
    return out
script = {g: " ; ".join(" ".join(c) for c in cmds(q)) for g, q in QUEUES.items()}
procs = {}
t0 = time.time()
for g, s in script.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
    procs[g] = subprocess.Popen(s, shell=True, env=env, stdout=open(f"worker{g}.log", "w"), stderr=subprocess.STDOUT)
for g, p in procs.items():
    p.wait()
    print("worker", g, "rc", p.returncode, round(time.time() - t0), "s")
    print(open(f"worker{g}.log").read()[-3000:])
RUN["data"]["sweep"] = [json.load(open(f)) for f in sorted(glob.glob("p3_*.json"))]
RUN["stepSec"]["sweep"] = round(time.time() - t0, 1)
save("lltMemoryThroughput_p3")
''' % (repr(LRS), repr(QUEUES))

CHECK = r'''
t = time.time()
p = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2', 'fsdp_ckpt_check.py'], capture_output=True, text=True)
RUN['stdout']['fsdp_ckpt_check'] = p.stdout + ('' if p.returncode == 0 else p.stderr[-3000:])
RUN['stepSec']['fsdp_ckpt_check'] = round(time.time() - t, 1)
print(RUN['stdout']['fsdp_ckpt_check'])
import shutil
shutil.rmtree('/tmp/ckpt_check', ignore_errors=True)
save('lltMemoryThroughput_p3')
'''

CELLS = [
    ("md", "# lltMemoryThroughput part 3 measurements (2x T4)\n\nLearning-rate sweep on TinyStories, baseline vs QK-norm vs z-loss at 10M, baseline vs QK-norm at 27M. One worker per T4."),
    ("code", SETUP + "\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets', 'tokenizers'], check=False)\n"),
    ("code", write_file_cell("tiny.py", src("tiny.py"))),
    ("code", write_file_cell("sweep.py", src("sweep.py"))),
    ("code", write_file_cell("prep.py", src("prep.py"))),
    ("code", "t = time.time()\np = subprocess.run([sys.executable, 'prep.py'], capture_output=True, text=True)\nprint(p.stdout, p.stderr[-2000:])\nRUN['stdout']['prep'] = p.stdout\nRUN['stepSec'] = {'prep': round(time.time() - t, 1)}\n"),
    ("code", RUNNER),
    ("code", step_cell("read_sweep", "read_sweep.py") + "save('lltMemoryThroughput_p3')"),
    ("code", write_file_cell("gpt.py", src("gpt.py")) + write_file_cell("fsdp.py", src("fsdp.py")) + write_file_cell("fsdp_ckpt_check.py", open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag", "fsdp_ckpt_check.py"), encoding="utf-8").read())),
    ("code", CHECK),
]
