"""Timing figures for lltTrainingLoop on the T1 laptop CPU. Run on an IDLE machine, alone.

Prints one JSON line. Keys match the [[TIME:key]] placeholders on the lab page:
  cpu_train_2000_min   minutes for the small config, 2,000 steps (step 6), 4 threads
  cpu_tok_per_sec      training tokens per second in that run, evaluation included
  cpu_resume_test_min  minutes for the exact-resume test in step 7 (straight, first half, resumed half, 4 near misses)
  cpu_quick_path_min   whole minutes for steps 6 and 7 together, the Lab card quick path (steps 1 to 5 take seconds)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
ns = {"__name__": "lltloop"}
for f in ["s1_data.py", "s2_model.py", "s3_sched.py", "s5_ckpt.py", "s6_train.py"]:
    p = os.path.join(HERE, "steps", f)
    exec(compile(open(p, encoding="utf-8").read(), p, "exec"), ns)
torch = ns["torch"]
ns["chars"] = ns["prepare"]()
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
train = ns["train"]
CPU_CFG = dict(n_layer=4, n_head=4, d=128, block=64, dropout=0.0, micro_batch=12, accum=1, steps=2000, lr=1e-3,
               warmup=100, schedule="cosine", wd=0.1, clip=1.0, seed=0, eval_every=250)

t = time.time()
train(CPU_CFG)
t_train = time.time() - t

R = dict(CPU_CFG, dropout=0.1, steps=200, eval_every=100, ckpt=os.path.join(HERE, "timing-resume.pt"), ckpt_every=100)
t = time.time()
train(R)
train(dict(R, stop_at=100))
train(dict(R, resume=True, ckpt_every=10**9))
for missing in ["rng", "sampler", "scheduler", "optimizer"]:
    train(dict(R, resume=True, ckpt_every=10**9, skip=(missing,)))
t_resume = time.time() - t
os.remove(R["ckpt"])

print(json.dumps({
    "cpu_train_2000_min": round(t_train / 60, 1),
    "cpu_tok_per_sec": int(2000 * 12 * 64 / t_train),
    "cpu_resume_test_min": round(t_resume / 60, 1),
    "cpu_quick_path_min": round((t_train + t_resume) / 60),
    "torch": torch.__version__, "threads": 4, "python": sys.version.split()[0],
}))
