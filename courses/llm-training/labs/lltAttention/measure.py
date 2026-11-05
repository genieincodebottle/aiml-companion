"""Regenerate every CPU Run Library file for lltAttention (../../runs/lltAttention/).

Usage: python measure.py [browser] [exact] [errors] [gqa] [train] [ablation] [dropout] [seeds]   (no argument = all)

The page code lives in cells/*.py (the notebook cells, in order). This script runs those cells
as shipped, keeps their printed output, and adds what the page quotes but the cells do not print
(loss curves every 25 steps, extra seeds, the dropout and head-ablation runs, exact error text).
Exact figures only. Timing lives in measure_timing.py.
"""
import contextlib
import functools
import io
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import write_run  # noqa: E402

NS = {"__name__": "__main__"}


def run_cell(name, ns=None):
    path = HERE / "cells" / (name + ".py")
    buf = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), NS if ns is None else ns)
    out = buf.getvalue()
    print(f"==== {name} ({time.perf_counter() - t:.1f} s)")
    print(out, end="")
    return out


def browser():
    outs = {}
    for name in ("step1", "step2", "step3"):
        outs[name] = subprocess.run([sys.executable, str(HERE / "cells" / (name + ".py"))], capture_output=True,
                                    text=True, check=True).stdout
        print(outs[name])
    write_run("lltAttention-t0-browser-cells", "T0", "baseline",
              {"cells": ["step1", "step2", "step3"], "runtime": "CPython 3.12.3 + NumPy 2.5.4, fresh process per cell; "
               "the same files print identical output in Pyodide 0.27.0 and 0.27.8 (scratch harness run_pyodide.mjs)"},
              seed=0, evals={"stdout": outs})


def exact():
    outs = {n: run_cell(n) for n in ("step4", "step5", "step6")}
    torch, F = NS["torch"], NS["F"]
    # step 4 figures, recomputed from the shipped classes
    torch.manual_seed(0)
    B, T, C, h = 4, 128, 256, 8
    x = torch.randn(B, T, C)
    errs = {}
    for dtype in (torch.float32, torch.float64):
        ours = NS["CausalSelfAttention"](C, h).to(dtype).eval()
        ref = NS["SDPAAttention"](C, h).to(dtype).eval()
        ref.load_state_dict(ours.state_dict())
        with torch.no_grad():
            errs[str(dtype)] = (ours(x.to(dtype)) - ref(x.to(dtype))).abs().max().item()
    write_run("lltAttention-t1-sdpa-match", "T1", "baseline",
              {"B": B, "T": T, "C": C, "h": h, "threads": 4, "reference": "F.scaled_dot_product_attention(is_causal=True)"},
              seed=0, evals={"maxAbsError": errs, "params": 4 * C * C}, stdout=outs["step4"])
    torch.manual_seed(0)
    base = NS["CausalSelfAttention"](C, h).eval()
    leak = {}
    for cname in ("CausalSelfAttention", "SDPAAttention", "NoMask", "MaskAfterSoftmax"):
        a = NS[cname](C, h).eval()
        a.load_state_dict(base.state_dict())
        leak[cname] = NS["leakage"](a)
    write_run("lltAttention-t1-leakage", "T1", "baseline",
              {"T": 32, "C": C, "h": h, "test": "for every t, perturb token t+1, max abs change at positions 0..t"},
              seed=0, evals={"worstChange": leak}, stdout=outs["step5"])
    mem = {str(k): {"scoresMiB": v[0], "naivePeakMiB": round(v[1], 2), "sdpaPeakMiB": round(v[2], 2)}
           for k, v in NS["memory"].items()}
    write_run("lltAttention-t1-memory-cpu", "T1", "baseline",
              {"B": 1, "h": 12, "d": 64, "dtype": "float32",
               "method": "torch.profiler profile_memory=True; replay [memory] alloc/free events in time order, keep the peak"},
              evals={"memory": mem}, stdout=outs["step6"])


def make_model(attn_cls, seed=0):
    torch = NS["torch"]
    torch.manual_seed(seed)
    return NS["TinyLM"](attn_cls, len(NS["chars"]))


def train_logged(attn_cls, steps=600, seed=0, log_every=25):
    print("training", getattr(attn_cls, "__name__", attn_cls), "steps", steps, "seed", seed, flush=True)
    torch, F = NS["torch"], NS["F"]
    model = make_model(attn_cls, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    metrics = []
    for step in range(steps):
        x, y = NS["batch"](NS["train"], g=g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            m = {"step": step + 1, "tokens": (step + 1) * 32 * 128, "trainLoss": round(loss.item(), 4)}
            if (step + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    m["valLoss"] = round(F.cross_entropy(model(NS["xv"]).flatten(0, 1), NS["yv"].flatten()).item(), 4)
                model.train()
            metrics.append(m)
    model.eval()
    return model, loss.item(), metrics


def sample(model, prompt="Once upon a time", n=120, seed=0):
    torch = NS["torch"]
    chars = NS["chars"]
    g = torch.Generator().manual_seed(seed)
    ids = [chars.index(c) for c in prompt]
    with torch.no_grad():
        for _ in range(n):
            logits = model(torch.tensor([ids[-128:]]))[0, -1] / 0.8
            ids.append(int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g)))
    return "".join(chars[i] for i in ids)


def ensure_data():
    if "TinyLM" not in NS:
        for n in ("step4", "step5"):
            run_cell(n)
        return run_cell("step7")
    return None


def curve_metrics(curve, every=25):
    return [{"step": i + 1, "tokens": (i + 1) * 32 * 128, "trainLoss": round(x, 4)}
            for i, x in enumerate(curve) if i == 0 or (i + 1) % every == 0]


def train():
    """The three step 8 runs come straight from the shipped cell (cells/step7.py): its curves, models
    and printed results. Nothing is retrained here."""
    t = time.perf_counter()
    out7 = ensure_data()
    wall = round(time.perf_counter() - t, 1)
    torch = NS["torch"]
    results, curves, models = NS["results"], NS["curves"], NS["models"]
    cfg = {"model": "TinyLM, 4 layers, C=64, h=4, ctx 128, RMSNorm, GELU MLP", "steps": 600, "batch": "32 x 128",
           "optimizer": "AdamW lr 3e-3", "data": "TinyStoriesV2-GPT4-valid.txt, first 1,000,000 characters, 900k train / 100k val",
           "eval": "64 x 128 validation characters, seed 1; prefix loss scores position t from tokens 0..t only",
           "cell": "labs/lltAttention/cells/step7.py (page step 8), run as shipped"}
    labels = {"CausalSelfAttention": ("lltAttention-t1-mask-control-s0", "baseline"),
              "NoMask": ("lltAttention-t1-broken-no-mask", "broken:no-mask"),
              "MaskAfterSoftmax": ("lltAttention-t1-broken-mask-after-softmax", "broken:mask-after-softmax")}
    for cname, (run_id, label) in labels.items():
        tr, full, pre = results[cname]
        c = curves[cname]
        write_run(run_id, "T1", label, dict(cfg, attention=cname), seed=0, metrics=curve_metrics(c),
                  evals={"finalTrainLoss": tr, "meanTrainLossLast50": sum(c[-50:]) / 50, "valLossFullSequence": full,
                         "valLossPrefixOnly": pre, "params": sum(p.numel() for p in models[cname].parameters()),
                         "vocab": len(NS["chars"])},
                  samples=[{"step": 600, "prompt": "Once upon a time", "completion": sample(models[cname])}],
                  wall_clock_sec=wall, stdout=out7)


def seeds():
    ensure_data()
    torch, F = NS["torch"], NS["F"]
    for seed in (1, 2):
        model, last, metrics = train_logged(NS["CausalSelfAttention"], seed=seed)
        with torch.no_grad():
            full = F.cross_entropy(model(NS["xv"]).flatten(0, 1), NS["yv"].flatten()).item()
        write_run(f"lltAttention-t1-mask-control-s{seed}", "T1", "baseline",
                  {"as": "lltAttention-t1-mask-control-s0 with another seed (model init and batch order)"},
                  seed=seed, metrics=metrics, evals={"finalTrainLoss": last, "valLossFullSequence": full,
                                                     "valLossPrefixOnly": NS["prefix_loss"](model, NS["xv"], NS["yv"])})


def dropout():
    ensure_data()
    torch, F = NS["torch"], NS["F"]
    CSA = NS["CausalSelfAttention"]

    class SDPADropoutAlwaysOn(CSA):              # the bug: dropout_p passed with no training check
        def attend(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=self.p, is_causal=True)

    good = functools.partial(CSA, dropout=0.1)
    model, last, metrics = train_logged(good, steps=300)
    sd = model.state_dict()
    bug = make_model(functools.partial(SDPADropoutAlwaysOn, dropout=0.1))
    bug.load_state_dict(sd)
    bug.eval()
    vals = {}
    with torch.no_grad():
        vals["correct"] = [F.cross_entropy(model(NS["xv"]).flatten(0, 1), NS["yv"].flatten()).item() for _ in range(3)]
        torch.manual_seed(123)
        vals["dropoutAlwaysOn"] = [F.cross_entropy(bug(NS["xv"]).flatten(0, 1), NS["yv"].flatten()).item() for _ in range(3)]
        x = NS["xv"][:4]
        o1, o2 = bug(x), bug(x)
        ref = model(x)
    evals = {"valLossEvalMode": vals, "maxAbsLogitDiffTwoBugPasses": (o1 - o2).abs().max().item(),
             "maxAbsLogitDiffBugVsCorrect": (o1 - ref).abs().max().item(), "finalTrainLoss": last}
    print(evals)
    write_run("lltAttention-t1-broken-dropout-eval", "T1", "broken:attention-dropout-on-at-eval",
              {"model": "TinyLM as in page step 8, attention dropout 0.1", "steps": 300,
               "bug": "F.scaled_dot_product_attention(..., dropout_p=self.p) with no self.training check"},
              seed=0, metrics=metrics, evals=evals)


def ablation():
    ensure_data()
    torch, F = NS["torch"], NS["F"]
    CSA = NS["CausalSelfAttention"]

    import types
    t = time.perf_counter()
    model = NS["models"]["CausalSelfAttention"]
    for b in model.blocks:                       # zero a head's output before the output projection
        b.attn.keep = torch.ones(b.attn.h)
        b.attn.attend = types.MethodType(lambda self, q, k, v: CSA.attend(self, q, k, v) * self.keep[None, :, None, None], b.attn)
    g = torch.Generator().manual_seed(7)
    xe, ye = NS["batch"](NS["val"], B=256, g=g)

    def val():
        with torch.no_grad():
            return F.cross_entropy(model(xe).flatten(0, 1), ye.flatten()).item()

    heads = [(li, hi) for li in range(4) for hi in range(4)]
    attns = [b.attn for b in model.blocks]
    base = val()
    single = {}
    for li, hi in heads:
        attns[li].keep[hi] = 0
        single[f"L{li}H{hi}"] = round(val() - base, 4)
        attns[li].keep[hi] = 1
    removed, path = [], []
    remaining = list(heads)
    while remaining:
        best = None
        for li, hi in remaining:
            attns[li].keep[hi] = 0
            v = val()
            attns[li].keep[hi] = 1
            if best is None or v < best[0]:
                best = (v, (li, hi))
        v, (li, hi) = best
        attns[li].keep[hi] = 0
        remaining.remove((li, hi))
        removed.append(f"L{li}H{hi}")
        path.append({"removed": len(removed), "head": f"L{li}H{hi}", "valLoss": round(v, 4),
                     "risePct": round(100 * (v / base - 1), 3)})
        print(path[-1])
        if v > base * 1.05:
            break
    within1 = max([p["removed"] for p in path if p["risePct"] <= 1.0], default=0)
    write_run("lltAttention-t1-head-ablation", "T1", "ablation:heads",
              {"model": "the page step 8 control model (lltAttention-t1-mask-control-s0), 600 steps", "eval": "256 x 128 validation characters, seed 7",
               "method": "zero one head's output before the output projection; greedy removal of the head whose removal raises val loss least"},
              seed=0, evals={"baseValLoss": base, "params": sum(p.numel() for p in model.parameters()),
                                               "singleHeadRise": single, "greedyPath": path, "headsRemovedWithin1Pct": within1},
              wall_clock_sec=round(time.perf_counter() - t, 1))


def errors():
    import torch
    out = {}
    s = torch.randn(1, 4, 4, dtype=torch.float16)
    try:
        s.masked_fill(torch.triu(torch.ones(4, 4, dtype=torch.bool), 1), -1e9)
    except RuntimeError as ex:
        out["fp16_masked_fill_minus_1e9_cpu"] = str(ex)
    out["fp16_cast_minus_1e9"] = str(torch.tensor(-1e9).half().item())
    out["fp16_finfo_min"] = torch.finfo(torch.float16).min
    out["fp16_softmax_all_minus_inf"] = str(torch.softmax(torch.full((4,), float("-inf"), dtype=torch.float16), -1).tolist())
    a = torch.randn(2, 5, 3, 4).transpose(1, 2)
    try:
        a.view(2, 3, 20)
    except RuntimeError as ex:
        out["view_after_transpose"] = str(ex)
    print(out)
    write_run("lltAttention-t1-error-messages", "T1", "broken:error-text",
              {"what": "exact error text for the fp16 -1e9 mask and .view after .transpose, torch CPU"}, evals=out)


def gqa():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    B, T, hq, hkv, d = 2, 128, 12, 4, 64
    q = torch.randn(B, hq, T, d)
    k, v = torch.randn(B, hkv, T, d), torch.randn(B, hkv, T, d)
    ours = F.scaled_dot_product_attention(q, k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1), is_causal=True)
    lib = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    err = (ours - lib).abs().max().item()
    C = hq * d
    ev = {"maxAbsError": err, "kvProjParamsMHA": 2 * C * C, "kvProjParamsGQA": 2 * C * hkv * d,
          "kvCacheBytesPerTokenPerLayerFp16MHA": 2 * hq * d * 2, "kvCacheBytesPerTokenPerLayerFp16GQA": 2 * hkv * d * 2}
    print(ev)
    write_run("lltAttention-t1-gqa-preview", "T1", "preview:gqa",
              {"B": B, "T": T, "queryHeads": hq, "kvHeads": hkv, "d": d,
               "ours": "repeat_interleave K and V to 12 heads, then SDPA", "library": "SDPA enable_gqa=True"},
              seed=0, evals=ev)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["browser", "exact", "errors", "gqa", "train", "ablation", "dropout", "seeds"]
    import torch
    torch.set_num_threads(4)
    for name in todo:
        globals()[name]()
