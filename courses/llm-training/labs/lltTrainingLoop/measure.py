"""CPU (T1) measurements for lltTrainingLoop. Exact losses, deterministic mode, 4 threads.

The training code is the page's own code blocks (steps/*.py), executed into one namespace,
so every figure comes from the code the learner runs.
Usage: python measure.py <group> [<group> ...]
Groups: cosine, wsd, lrhigh, accum, resume, adamw
Wall-clock figures in these files are distorted by other jobs on this machine; the page uses
measure_timing.py for every timing figure.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
from runlog import write_run  # noqa: E402

ns = {"__name__": "lltloop"}
for f in ["s1_data.py", "s2_model.py", "s3_sched.py", "s5_ckpt.py", "s6_train.py"]:
    p = os.path.join(HERE, "steps", f)
    exec(compile(open(p, encoding="utf-8").read(), p, "exec"), ns)
torch, np = ns["torch"], ns["np"]
ns["chars"] = ns["prepare"]()
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
train = ns["train"]
SMALL = dict(n_layer=4, n_head=4, d=128, block=64)
CPU_CFG = dict(SMALL, dropout=0.0, micro_batch=12, accum=1, steps=2000, lr=1e-3, warmup=100,
               schedule="cosine", wd=0.1, clip=1.0, seed=0, eval_every=250)


def thin(hist):
    return [r for r in hist if r["step"] % 10 == 0 or "valLoss" in r or r["step"] <= 20]


def run(run_id, label, cfg, notes=None):
    t = time.time()
    _, hist = train(cfg)
    wall = time.time() - t
    vals = [r["valLoss"] for r in hist if "valLoss" in r]
    evals = {"finalValLoss": vals[-1], "bestValLoss": min(vals), "finalTrainLoss": hist[-1]["trainLoss"],
             "maxGradNorm": max(r["gradNorm"] for r in hist), "clippedSteps": sum(1 for r in hist if r["gradNorm"] > cfg["clip"])}
    write_run(run_id, "T1", label, cfg, cfg["seed"], thin(hist), evals=evals, wall_clock_sec=round(wall, 1), notes=notes)
    print(run_id, {k: round(v, 4) if isinstance(v, float) else v for k, v in evals.items()}, flush=True)
    return hist


def g_cosine():
    for s in range(3):
        run(f"lltTrainingLoop-t1-cosine-s{s}", "baseline", dict(CPU_CFG, seed=s))


def g_cosine12():
    # Seed 0 of this config is the step 6 run itself (steps log), converted by from_steps_log.py.
    for s in [1, 2]:
        run(f"lltTrainingLoop-t1-cosine-s{s}", "baseline", dict(CPU_CFG, seed=s))


def g_wsd():
    for s in range(3):
        run(f"lltTrainingLoop-t1-wsd-s{s}", "ablation:schedule=wsd", dict(CPU_CFG, seed=s, schedule="wsd"))


def g_lrhigh():
    for lr in [1e-2, 3e-2]:
        run("lltTrainingLoop-t1-broken-lr" + format(lr, ".0e").replace("e-0", "e-"), "broken:lr-too-high", dict(CPU_CFG, lr=lr))


def g_lr1e1():
    run("lltTrainingLoop-t1-broken-lr1e-1", "broken:lr-too-high", dict(CPU_CFG, lr=1e-1))


def g_accum():
    # 4 micro-batches of 3 = 12 sequences per update, the same as the baseline, at the same cost per step.
    A = dict(CPU_CFG, micro_batch=3, accum=4)
    run("lltTrainingLoop-t1-accum4-s0", "control:accum=4", A)
    run("lltTrainingLoop-t1-broken-accum-nodivide", "broken:accum-no-divide", dict(A, divide_loss=False))
    run("lltTrainingLoop-t1-broken-sched-per-micro", "broken:scheduler-per-micro-batch", dict(A, sched_per_micro=True))


def g_resume():
    R = dict(CPU_CFG, dropout=0.1, steps=200, eval_every=100, ckpt=os.path.join(HERE, "resume.pt"), ckpt_every=100)
    _, straight = train(R)
    _, first = train(dict(R, stop_at=100))
    _, second = train(dict(R, resume=True, ckpt_every=10**9))
    out = {"straight": [r["trainLoss"] for r in straight], "resumed": [r["trainLoss"] for r in first + second]}
    res = {}

    def cmp(a, b):
        return max(abs(x["trainLoss"] - y["trainLoss"]) for x, y in zip(a, b)), b[-1]["valLoss"] - a[-1]["valLoss"]

    d, dv = cmp(straight, first + second)
    res["full"] = {"maxTrainLossDiff": d, "valDiff200": dv, "valStraight200": straight[-1]["valLoss"], "valResumed200": second[-1]["valLoss"]}
    for missing in ["rng", "sampler", "scheduler", "optimizer"]:
        _, part = train(dict(R, resume=True, ckpt_every=10**9, skip=(missing,)))
        d, dv = cmp(straight, first + part)
        res["without_" + missing] = {"maxTrainLossDiff": d, "valDiff200": dv, "loss101": part[0]["trainLoss"],
                                     "loss101Straight": straight[100]["trainLoss"],
                                     "maxLoss101to110": max(r["trainLoss"] for r in part[:10]),
                                     "maxStraight101to110": max(r["trainLoss"] for r in straight[100:110])}
        out["without_" + missing] = [r["trainLoss"] for r in first + part]
        print(missing, res["without_" + missing], flush=True)
    print("full", res["full"], flush=True)
    metrics = [{"step": i + 1, "straight": out["straight"][i], "resumed": out["resumed"][i],
                **{k: out[k][i] for k in out if k.startswith("without_")}} for i in range(200)]
    write_run("lltTrainingLoop-t1-resume", "T1", "test:exact-resume", R, 0, metrics, evals=res,
              notes="Steps 1-100 shared; checkpoint at 100; steps 101-200 after load. Deterministic CPU mode, 4 threads.")
    if os.path.exists(R["ckpt"]):
        os.remove(R["ckpt"])


def g_adamw():
    # Pin the torch.optim.AdamW values the NumPy challenge must match.
    p0 = [0.5, -1.0, 2.0, 0.0]
    grads = [[0.1, -0.2, 0.3, 0.4], [0.05, -0.1, -0.2, 0.3], [-0.3, 0.2, 0.1, 0.0]]
    w = torch.nn.Parameter(torch.tensor(p0, dtype=torch.float64))
    opt = torch.optim.AdamW([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    after = []
    for g in grads:
        w.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
        after.append([round(float(v), 8) for v in w.detach()])
    print("adamw after each step", after, flush=True)
    write_run("lltTrainingLoop-t0-adamw-pin", "T0", "pin:adamw", {"p0": p0, "grads": grads, "lr": 0.1, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.1}, None,
              [{"step": i + 1, "params": a} for i, a in enumerate(after)])


GROUPS = {"cosine": g_cosine, "cosine12": g_cosine12, "wsd": g_wsd, "lrhigh": g_lrhigh, "lr1e1": g_lr1e1, "accum": g_accum, "resume": g_resume, "adamw": g_adamw}
for g in sys.argv[1:]:
    print("=== group", g, flush=True)
    GROUPS[g]()
