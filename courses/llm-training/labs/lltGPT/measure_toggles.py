"""Step 8 experiment and two broken runs, recorded to the Run Library.

Runs the page's own cells (steps 2, 3, 4, 6, 8) in one namespace, then trains each variant for
600 steps per seed on CPU and writes runs/lltGPT/<runId>.json. Losses are exact for this machine
and thread count; wall-clock is recorded but the page takes timings from measure_timing.py.

Usage: python measure_toggles.py [--seeds 0 1] [--steps 600] [--only name ...] [--broken]
"""
import argparse
import contextlib
import io
import math
import time
from pathlib import Path

from runlog import write_run

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--broken", action="store_true")
ap.add_argument("--no-variants", action="store_true")
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
import os  # noqa: E402
os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))   # caps the tokenizers trainer too
import torch as _torch  # noqa: E402
_torch.set_num_threads(args.threads)

ns = {"__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()):
    for name in ["step2", "step3", "step4", "step6"]:
        exec(compile((HERE / "steps" / f"{name}.py").read_text(encoding="utf-8"), name, "exec"), ns)
t = time.time()
exec(compile((HERE / "steps" / "step8.py").read_text(encoding="utf-8"), "step8", "exec"), ns)
data_sec = round(time.time() - t, 1)
torch, nn, F, replace = ns["torch"], ns["nn"], ns["F"], ns["replace"]
torch.set_num_threads(args.threads)
TOGGLE, GPT, train, numel = ns["TOGGLE"], ns["GPT"], ns["train"], ns["numel"]
data_info = {"dataset": "roneneldan/TinyStories TinyStoriesV2-GPT4-valid.txt", "stories": len(ns["stories"]),
             "trainTokens": len(ns["train_ids"]), "valTokens": len(ns["val_ids"]), "valEvalTokens": 256 * 128,
             "tokenizer": "byte-level BPE, vocab 2048, trained on the 95% train split", "dataPrepSec": data_sec}
VARIANTS = {
    "modern": TOGGLE,
    "learned-pos": replace(TOGGLE, pos="learned"),
    "gelu": replace(TOGGLE, mlp="gelu"),
    "layernorm": replace(TOGGLE, norm="layernorm"),
    "postnorm": replace(TOGGLE, prenorm=False),
    "gpt2": replace(TOGGLE, norm="layernorm", mlp="gelu", pos="learned", bias=True),
}
TRAIN = {"steps": args.steps, "batch": 32, "ctx": 128, "lr": 3e-3, "warmup": 50, "schedule": "linear warmup, cosine to 10%",
         "optimizer": "AdamW betas (0.9, 0.95), wd 0.1 on 2D params", "clip": 1.0, "threads": args.threads}


def record(run_id, label, cfg, seed, model, curve, sec, extra=None):
    cfg_d = dict(vars(cfg))
    write_run(run_id, "T1", label, {**cfg_d, **TRAIN, "params": numel(model), "data": data_info}, seed,
              [{"step": s, "tokens": s * 32 * 128, "valLoss": round(v, 4)} for s, v in curve],
              evals={"finalValLoss": round(curve[-1][1], 4), **(extra or {})}, wall_clock_sec=round(sec, 1))


if not args.no_variants:
    for name, cfg in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        for seed in args.seeds:
            t = time.time()
            model, curve = train(cfg, steps=args.steps, seed=seed)
            sec = time.time() - t
            print(f"{name:12} seed {seed} params {numel(model):,} val {curve[-1][1]:.4f} ({sec:.0f}s)", flush=True)
            record(f"lltGPT-t1-toggle-{name}-s{seed}", "baseline" if name == "modern" else f"ablation:{name}", cfg, seed, model, curve, sec)

if args.broken:
    # Broken 1: RoPE applied to V as well as Q and K.
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

    # Broken 2: init touches nn.Linear only, and the head is tied AFTER the init,
    # so the shared tensor keeps nn.Embedding's default N(0, 1).
    def gpt_tie_after(cfg):
        torch_rng_state = None  # noqa: F841
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
        m.head.weight = m.tok.weight   # tied after init
        return m

    def shift_test(builder):
        torch.manual_seed(0)
        small = replace(ns["MODERN"], vocab=1000, ctx=64, d=64, layers=2, heads=4)
        m = builder(small).eval()
        idx = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            return (m(idx) - m(idx, offset=40)).abs().max().item()

    for tag, builder, extra_fn in [("rope-on-v", gpt_rope_v, lambda: {"shift40MaxLogitChange": shift_test(gpt_rope_v), "shift40Correct": shift_test(GPT)}),
                                   ("tie-after-init", gpt_tie_after, lambda: {})]:
        extra = extra_fn()
        ns["GPT"] = builder
        t = time.time()
        model, curve = train(TOGGLE, steps=args.steps, seed=0)
        sec = time.time() - t
        ns["GPT"] = GPT
        torch.manual_seed(0)
        probe = builder(TOGGLE)
        extra["headWeightStdAtInit"] = round(probe.head.weight.std().item(), 4)
        extra["headIsEmbedding"] = probe.head.weight is probe.tok.weight
        print(f"broken {tag} val {[round(v, 3) for _, v in curve]} {extra} ({sec:.0f}s)", flush=True)
        record(f"lltGPT-t1-broken-{tag}", f"broken:{tag}", TOGGLE, 0, model, curve, sec, extra)
