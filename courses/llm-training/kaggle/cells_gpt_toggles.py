"""Kaggle kernel cells for lltGPT step 8: six variants x seeds 0 and 1 plus two broken runs, fp32, 2x T4.

The library is the page's own cells (labs/lltGPT/steps/step2,3,4,6,8.py), read at build time. The only
change is moving tensors to the GPU: the model is built on the CPU (same init as the CPU notebook),
then moved, and each batch is sampled with the same CPU generator, then moved. No autocast, so the
runs are float32 (a T4 has no TF32). One worker process per GPU (CUDA_VISIBLE_DEVICES), each
trains its jobs in turn and writes one Run Library JSON per run to /kaggle/working/runs/.
"""
import os

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs", "lltGPT", "steps")


def read(name):
    return open(os.path.join(LAB, name), encoding="utf-8").read()


step8 = read("step8.py")
EDITS = [
    ("VAL_Y = val_ids[1 : 256 * 128 + 1].view(256, 128)\n",
     "VAL_Y = val_ids[1 : 256 * 128 + 1].view(256, 128)\nVAL_X, VAL_Y = VAL_X.to(DEV), VAL_Y.to(DEV)\n"),
    ("model, gen = GPT(cfg), torch.Generator().manual_seed(seed)",
     "model, gen = GPT(cfg).to(DEV), torch.Generator().manual_seed(seed)"),
    ("        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())",
     "        x, y = x.to(DEV), y.to(DEV)\n        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())"),
]
for a, b in EDITS:
    assert step8.count(a) == 1, a
    step8 = step8.replace(a, b)
lib = "import torch\nDEV = 'cuda' if torch.cuda.is_available() else 'cpu'\n\n"
lib += "\n\n".join(read(f) for f in ["step2.py", "step3.py", "step4.py", "step6.py"]) + "\n\n" + step8

WORKER = r'''
import json, math, os, platform, sys, time
WORK = os.environ.get("LLT_WORKDIR", "/kaggle/working")
os.chdir(WORK)
import torch
import torch.nn as nn
import torch.nn.functional as F
ns = {"__name__": "lltgpt"}
t0 = time.time()
exec(compile(open("lltgpt.py", encoding="utf-8").read(), "lltgpt.py", "exec"), ns)
data_sec = time.time() - t0
GPT, train, numel, replace, TOGGLE = ns["GPT"], ns["train"], ns["numel"], ns["replace"], ns["TOGGLE"]
DEV = ns["DEV"]
CUDA = torch.cuda.is_available()
STEPS = int(os.environ.get("LLT_STEPS", "600"))
jobs = json.load(open(sys.argv[1]))

def attn_rope_v(self, x, cos=None, sin=None):
    B, T, C = x.shape
    q, k, v = self.qkv(x).split(C, dim=2)
    q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
    q, k, v = (ns["apply_rope"](t, cos, sin) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.proj(y.transpose(1, 2).reshape(B, T, C))

def gpt_rope_v(cfg):
    m = GPT(cfg)
    for b in m.blocks:
        b.attn.forward = attn_rope_v.__get__(b.attn)
    return m

def gpt_tie_after(cfg):
    m = GPT.__new__(GPT)
    nn.Module.__init__(m)
    m.cfg = cfg
    m.tok = nn.Embedding(cfg.vocab, cfg.d)
    m.pos = nn.Embedding(cfg.ctx, cfg.d) if cfg.pos == "learned" else None
    m.blocks = nn.ModuleList([ns["Block"](cfg) for _ in range(cfg.layers)])
    m.norm = ns["make_norm"](cfg)
    m.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            std = 0.02 / math.sqrt(2 * cfg.layers) if name.endswith(("proj", "down")) else 0.02
            nn.init.normal_(mod.weight, 0.0, std)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
    m.head.weight = m.tok.weight
    return m

BUILDERS = {"none": GPT, "rope-on-v": gpt_rope_v, "tie-after-init": gpt_tie_after}

def shift_test(builder):
    torch.manual_seed(0)
    m = builder(replace(ns["MODERN"], vocab=1000, ctx=64, d=64, layers=2, heads=4)).eval()
    idx = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        return (m(idx) - m(idx, offset=40)).abs().max().item(), (m(idx) - m(idx)).abs().max().item()

for job in jobs:
    cfg = replace(TOGGLE, **job["overrides"])
    builder = BUILDERS[job["builder"]]
    extra = {}
    if job["builder"] == "rope-on-v":
        extra["shift40MaxLogitChange"], extra["samePassTwiceMaxChange"] = shift_test(gpt_rope_v)
        extra["shift40Correct"], _ = shift_test(GPT)
    torch.manual_seed(job["seed"])
    probe = builder(cfg)
    extra["headWeightStdAtInit"] = round(probe.head.weight.std().item(), 4)
    ns["GPT"] = builder
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        model, curve = train(cfg, steps=STEPS, seed=job["seed"], lr=job.get("lr", 3e-3))
        status = "ok"
    except Exception as e:
        model, curve, status = None, [], "error: " + repr(e)
    if CUDA:
        torch.cuda.synchronize()
    wall = time.time() - t
    ns["GPT"] = GPT
    fin = [v for _, v in curve if v == v]
    rec = {
        "runId": job["runId"], "module": "lltGPT", "tier": "T2", "label": job["label"],
        "hardware": {"gpu": torch.cuda.get_device_name(0) if CUDA else None, "count": 1 if CUDA else 0,
                     "vramGB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if CUDA else None,
                     "capability": list(torch.cuda.get_device_capability()) if CUDA else None, "note": "Kaggle 2x T4 session, one run per GPU"},
        "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "tokenizers": __import__("tokenizers").__version__, "dtype": "float32, no autocast"},
        "notebookCommit": None,
        "config": {**vars(cfg), "steps": STEPS, "batch": 32, "ctx": 128, "lr": job.get("lr", 3e-3), "warmup": 50,
                   "schedule": "linear warmup, cosine to 10%", "optimizer": "AdamW betas (0.9, 0.95), wd 0.1 on 2D params", "clip": 1.0,
                   "params": numel(model) if model is not None else None, "builder": job["builder"],
                   "data": {"dataset": "roneneldan/TinyStories TinyStoriesV2-GPT4-valid.txt", "trainTokens": len(ns["train_ids"]),
                            "valTokens": len(ns["val_ids"]), "valEvalTokens": 256 * 128, "dataPrepSec": round(data_sec, 1)}},
        "seed": job["seed"], "status": status,
        "metrics": [{"step": s, "tokens": s * 32 * 128, "valLoss": round(v, 4)} for s, v in curve],
        "evals": {"finalValLoss": round(curve[-1][1], 4) if curve else None, "bestValLoss": round(min(fin), 4) if fin else None, **extra},
        "wallClockSec": round(wall, 1), "peakMemGB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if CUDA else None,
        "costUSD": 0, "priceCheckedOn": None, "date": time.strftime("%Y-%m-%d"),
    }
    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", job["runId"] + ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(job["runId"], status, "val", [round(v, 3) for _, v in curve], "wall", round(wall, 1), flush=True)
'''

CELLS = [
("md", "# LLM Training Lab, lltGPT step 8: toggle each modern component (T4, fp32)\n\nFROM SCRATCH. A 2.2M-parameter GPT on TinyStories, 600 steps per run. Modern config (RMSNorm, SwiGLU, RoPE, pre-norm) and one swap back to GPT-2's choice per variant, seeds 0 and 1, plus two deliberately broken runs. The code is the lab page's own cells, moved to the GPU, float32."),
("code", r'''
import os, platform, shutil, subprocess, sys, time
START = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers==0.22.1"], check=False)
import torch
print("Python", platform.python_version(), "| torch", torch.__version__)
if not os.environ.get("LLT_SMOKE"):
    assert torch.cuda.is_available(), "No GPU: Settings > Accelerator > GPU T4 x2"
for i in range(torch.cuda.device_count()):
    print("GPU", i, torch.cuda.get_device_name(i), "capability", torch.cuda.get_device_capability(i))
print("Free disk", round(shutil.disk_usage(".").free / 1e9, 1), "GB")
'''),
("code", "LIB = " + repr(lib) + "\nimport os\nWORK = os.environ.get('LLT_WORKDIR', '/kaggle/working')\nopen(os.path.join(WORK, 'lltgpt.py'), 'w').write(LIB)\nWORKER = " + repr(WORKER) + "\nopen(os.path.join(WORK, 'worker.py'), 'w').write(WORKER)\nprint('wrote lltgpt.py', len(LIB), 'chars and worker.py')"),
("code", r'''
import json
os.chdir(WORK)
V = {"modern": {}, "learned-pos": {"pos": "learned"}, "gelu": {"mlp": "gelu"}, "layernorm": {"norm": "layernorm"},
     "postnorm": {"prenorm": False}, "gpt2": {"norm": "layernorm", "mlp": "gelu", "pos": "learned", "bias": True}}
jobs = [{"runId": f"lltGPT-t2-toggle-{k}-s{s}", "label": "baseline" if k == "modern" else f"ablation:{k}", "overrides": o, "seed": s, "builder": "none"}
        for s in (0, 1) for k, o in V.items()]
if os.environ.get("LLT_JOBS") == "lr":
    jobs = [{"runId": f"lltGPT-t2-lr-{k}-{lr}-s0", "label": f"ablation:{k},lr={lr}", "overrides": V[k], "seed": 0, "builder": "none", "lr": float(lr)}
            for k in ("modern", "postnorm") for lr in ("1e-3", "3e-4")]
else:
  jobs += [{"runId": f"lltGPT-t2-broken-{b}", "label": f"broken:{b}", "overrides": {}, "seed": 0, "builder": b} for b in ("rope-on-v", "tie-after-init")]
n = max(1, torch.cuda.device_count())
if os.environ.get("LLT_SMOKE"):
    jobs = [jobs[0], jobs[-2], jobs[-1]]
procs = []
for g in range(n):
    path = f"jobs_{g}.json"
    json.dump(jobs[g::n], open(path, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    procs.append(subprocess.Popen([sys.executable, "worker.py", path], env=env, stdout=open(f"worker_{g}.log", "w"), stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
for g in range(n):
    print(open(f"worker_{g}.log").read())
print("runs", sorted(os.listdir("runs")), "elapsed min", round((time.time() - START) / 60, 1))
'''),
]
