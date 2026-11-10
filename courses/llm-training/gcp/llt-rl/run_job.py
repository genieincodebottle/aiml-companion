"""Run the lltRL page cells (steps/cellNN_stepK.py, the same files the notebook and the page use) as one GPU job.

  python run_job.py <mode> <out_dir> [key=value ...]
Modes (each patches only flags and sizes, never the training code):
  smoke      every code path with toy sizes (PPO 3 iterations, GRPO 3 steps, arms 2 steps, TRL 2 steps)
  ppo        PPO main run (seed from seed=..., default 0); with exp=1 also the KL sweep (step 5) and the clip/epoch arms (step 6)
  grpo       one GRPO run from scratch: model=<hub id> lr=<float> steps=<int> seed=<int> arm=<name in ARMS or main>
             eval=1 adds the 200-question greedy evaluation every 50 steps; passk=1 adds pass@k before and after
  trl        30 GRPOTrainer steps (step 13)
  defaults   the learner notebook as shipped (our checkpoints from the Hub), timed cell by cell
Writes <out_dir>/result.json and <out_dir>/stdout.txt; saves models under <out_dir>.
"""
import json, os, platform, sys, time
from pathlib import Path

mode, OUT = sys.argv[1], Path(sys.argv[2]).resolve()
kv = dict(a.split("=", 1) for a in sys.argv[3:])
OUT.mkdir(parents=True, exist_ok=True)
STEPS = Path(__file__).resolve().parent / "steps"


class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "a"), sys.stdout
    def write(self, x):
        self.o.write(x); self.f.write(x); self.f.flush()
    def flush(self):
        self.o.flush(); self.f.flush()
    def __getattr__(self, n):
        return getattr(self.o, n)


sys.stdout = Tee(OUT / "stdout.txt")
print("JOB", mode, kv, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)


def src(name, reps=(), cut=None):
    s = (STEPS / name).read_text(encoding="utf-8")
    if cut:
        assert cut in s, (name, cut)
        s = s.split(cut)[0]
    for a, b in reps:
        assert a in s, (name, a)
        s = s.replace(a, b, 1)
    return s


PPO_DEFS_ONLY = "runs = {}" + chr(10) + "if RUN_PPO:"           # cell 4 up to its definitions
CELL8_DEFS = "# One fixed batch"                                 # cell 8 definitions, without the CPU check
plan = []
if mode == "smoke":
    plan = [("cell02_step2.py", (), None),
            ("cell03_step3.py", [('prefixes("train", 12800, 0), prefixes("test", 128, 0)', 'prefixes("train", 64, 0), prefixes("test", 16, 0)')], None),
            ("cell04_step4.py", [("RUN_PPO = False", "RUN_PPO = True"), ("iters=200, batch=64", "iters=3, batch=16"),
                                 ("def evaluate(policy, ref, rm, n=128)", "def evaluate(policy, ref, rm, n=16)")], None),
            ("cell05_step5.py", [("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True")], None),
            ("cell06_step6.py", [("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True")], None),
            ("cell07_step7.py", (), None), ("cell08_step8.py", (), None),
            ("cell09_step9.py", [("N_TEST, MAX_NEW = 200, 256", "N_TEST, MAX_NEW = 16, 256"), ("PASS_AT_K = False", "PASS_AT_K = True")], None),
            ("cell10_step10.py", [('START_FROM = "lltRL/grpo-s0-step180"', "START_FROM = None"), ("steps=200,", "steps=3,"),
                                  ("eval_every=0, save_at=()", "eval_every=2, save_at=(3,)")], None),
            ("cell11_step11.py", (), None),
            ("cell12_step12.py", [("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True"), ("ARMS = {", "GRPO = dict(GRPO, steps=2)" + chr(10) + "ARMS = {")], None),
            ("cell13_step13.py", [("RUN_TRL = False", "RUN_TRL = True"), ("max_steps=30,", "max_steps=2,")], None)]
elif mode == "ppo":
    seed = kv.get("seed", "0")
    plan = [("cell02_step2.py", (), None), ("cell03_step3.py", (), None),
            ("cell04_step4.py", [("RUN_PPO = False", "RUN_PPO = True"), ("lam=0.95, seed=0)", "lam=0.95, seed=" + seed + ")"),
                                 ('runs["ppo-kl0.05-s0"]', 'runs["ppo-kl0.05-s' + seed + '"]')], None)]
    if kv.get("exp") == "1":
        plan += [("cell05_step5.py", [("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True")], None),
                 ("cell06_step6.py", [("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True")], None)]
elif mode == "grpo":
    model, arm = kv.get("model", "Qwen/Qwen3-0.6B-Base"), kv.get("arm", "main")
    grpo_cfg = "dict(GRPO, steps=%s, lr=%s, seed=%s, eval_every=%s, save_at=%s)" % (
        kv.get("steps", "200"), kv.get("lr", "2e-5"), kv.get("seed", "0"), "50" if kv.get("eval") == "1" else "0", kv.get("save", "()"))
    plan = [("cell02_step2.py", (), None), ("cell03_step3.py", [('prefixes("train", 12800, 0)', 'prefixes("train", 64, 0)')], "policy, ref, value, rm, mem = load_models()"),
            ("cell04_step4.py", (), PPO_DEFS_ONLY), ("cell07_step7.py", (), None), ("cell08_step8.py", (), CELL8_DEFS),
            ("cell09_step9.py", [('GMODEL = "Qwen/Qwen3-0.6B-Base"', "GMODEL = " + repr(model)),
                                 ("PASS_AT_K = False", "PASS_AT_K = " + str(kv.get("passk") == "1"))] +
             ([] if kv.get("base", "1") == "1" else [("base_eval = evaluate_gsm(policy, 8 if PASS_AT_K else 0)", "base_eval = {}"),
                                                    ('print("base model on"', 'if base_eval: print("base model on"'),
                                                    ('print(repr(base_eval["replies"][0][:300]))', 'if base_eval: print(repr(base_eval["replies"][0][:300]))')]), None),
            ("cell10_step10.py", (), "WORK = ")]
    plan.append(("__inline__", "WORK = " + repr(str(OUT)) + chr(10) + "runs = {}" + chr(10) + "del policy; gc.collect()" + chr(10) +
                 "exec(open(" + repr(str(STEPS / "cell12_step12.py")) + ").read().split('if RUN_EXPERIMENT')[0])" + chr(10) +
                 "cfg = " + ("GRPO" if arm == "main" else "ARMS[" + repr(arm) + "]") + chr(10) +
                 "cfg = " + grpo_cfg.replace("GRPO", "cfg", 1) + chr(10) +
                 "grpo_model, runs[" + repr("grpo-" + arm) + "] = train_grpo(cfg, tag=" + repr("grpo-" + arm) + ", resume=WORK)" + chr(10) +
                 "runs[" + repr("grpo-" + arm) + "]['final'] = evaluate_gsm(grpo_model, 8 if PASS_AT_K else 0)" + chr(10) +
                 "print('FINAL', {k: v for k, v in runs[" + repr("grpo-" + arm) + "]['final'].items() if k != 'replies'})" + chr(10) +
                 "grpo_model.save_pretrained(os.path.join(WORK, " + repr("grpo-" + arm + "-final") + "))", None))
elif mode == "trl":
    plan = [("cell02_step2.py", (), None), ("cell03_step3.py", [('prefixes("train", 12800, 0)', 'prefixes("train", 64, 0)')], "policy, ref, value, rm, mem = load_models()"),
            ("cell04_step4.py", (), PPO_DEFS_ONLY), ("cell07_step7.py", (), None), ("cell08_step8.py", (), CELL8_DEFS),
            ("cell09_step9.py", [("base_eval = evaluate_gsm(policy, 8 if PASS_AT_K else 0)", "base_eval = {}"),
                                 ('print("base model on"', 'if base_eval: print("base model on"'),
                                 ('print(repr(base_eval["replies"][0][:300]))', 'if base_eval: print(repr(base_eval["replies"][0][:300]))')], None),
            ("__inline__", "WORK = " + repr(str(OUT)) + chr(10) + "runs = {}" + chr(10) + "GRPO = dict(lr=2e-5)", None),
            ("cell13_step13.py", [("RUN_TRL = False", "RUN_TRL = True")], None)]
elif mode == "evalall":
    # greedy accuracy per question (and pass@k for the base model) for the base model and every saved final adapter,
    # so arms can be compared with paired intervals on the same 200 questions
    adapters = kv.get("adapters", "").split(",")
    plan = [("cell02_step2.py", (), None), ("cell03_step3.py", [('prefixes("train", 12800, 0)', 'prefixes("train", 64, 0)')], "policy, ref, value, rm, mem = load_models()"),
            ("cell04_step4.py", (), PPO_DEFS_ONLY), ("cell07_step7.py", (), None), ("cell08_step8.py", (), CELL8_DEFS),
            ("cell09_step9.py", [("PASS_AT_K = False", "PASS_AT_K = True")] +
             ([] if kv.get("base", "1") == "1" else [("base_eval = evaluate_gsm(policy, 8 if PASS_AT_K else 0)", "base_eval = {}"),
                                                    ('print("base model on"', 'if base_eval: print("base model on"'),
                                                    ('print(repr(base_eval["replies"][0][:300]))', 'if base_eval: print(repr(base_eval["replies"][0][:300]))')]), None),
            ("__inline__", "runs = {'eval-base': dict(final=base_eval)} if base_eval else {}" + chr(10) + "del policy; gc.collect()" + chr(10) +
             "for a in " + repr(adapters) + ":" + chr(10) +
             "    base = AutoModelForCausalLM.from_pretrained(GMODEL, dtype=torch.float16).to(DEV)" + chr(10) +
             "    m = PeftModel.from_pretrained(base, a)" + chr(10) +
             "    key = 'eval-' + a.rstrip('/').split('/')[-2].replace('out2_', '').replace('_', '-')" + chr(10) +
             "    runs[key] = dict(final=evaluate_gsm(m, 8 if " + repr(kv.get("passk") == "1") + " else 0), adapter=a)" + chr(10) +
             "    print('EVAL', a, runs[key]['final']['greedy'], flush=True)" + chr(10) +
             "    del m, base; gc.collect(); torch.cuda.empty_cache()", None)]
elif mode == "defaults":
    plan = [(p.name, (), None) for p in sorted(STEPS.glob("cell*_step*.py"))]
else:
    raise SystemExit("unknown mode " + mode)

ns = {"__name__": "__main__"}
cell_times, t_start = {}, time.time()
for name, reps, cut in plan:
    code = reps if name == "__inline__" else src(name, reps, cut)
    t = time.time()
    print("CELL", name, "start", round(t - t_start, 1), flush=True)
    exec(compile(code, name, "exec"), ns)
    cell_times[name if name != "__inline__" else "inline" + str(len(cell_times))] = round(time.time() - t, 1)
    if ns.get("DEV") is not None and ns["DEV"].type == "cuda":
        import torch
        print("CELL", name, "end", round(time.time() - t_start, 1), "| peak GB so far", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)

import torch, transformers, trl, peft, datasets  # noqa: E402


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "tolist"):
        return o.tolist()
    return o


rec = {"mode": mode, "args": kv, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
               "trl": trl.__version__, "peft": peft.__version__, "datasets": datasets.__version__},
       "cellTimes": cell_times, "totalSec": round(time.time() - t_start, 1), "runs": clean(ns.get("runs", {})),
       "sessionPeakGB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None}
for k in ("mem", "base_eval", "after", "start", "end"):
    if k in ns:
        rec[k] = clean(ns[k])
if mode == "ppo" and "trained" in ns:
    ns["trained"].save_pretrained(OUT / ("ppo-kl0.05-s" + kv.get("seed", "0")))
json.dump(rec, open(OUT / "result.json", "w"), indent=1)
print("RECORD written", rec["totalSec"], flush=True)
