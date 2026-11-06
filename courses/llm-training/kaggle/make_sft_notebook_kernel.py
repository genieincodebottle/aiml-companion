"""Build kernel llt-sft-notebook: the lltSFT learner notebook, run end to end on a Kaggle T4.

Takes labs/lltSFT/lltSFT.ipynb as built from the page, swaps the Prove it starter (which raises
NotImplementedError by design) for its solution, and appends one recording cell that writes
/kaggle/working/notebook_results.json with the figures step 8 prints. Nothing else changes.
Usage: python make_sft_notebook_kernel.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "labs", "lltSFT")
SLUG = "llt-sft-notebook"

nb = json.load(open(os.path.join(LAB, "lltSFT.ipynb"), encoding="utf-8"))
starter = open(os.path.join(LAB, "steps", "challenge1_starter.py"), encoding="utf-8").read().rstrip("\n")
solution = open(os.path.join(LAB, "steps", "challenge1_solution.py"), encoding="utf-8").read().rstrip("\n")
swapped = 0
for c in nb["cells"]:
    if c["cell_type"] == "code" and "".join(c["source"]).rstrip("\n") == starter:
        c["source"] = solution
        swapped += 1
assert swapped == 1, "starter cell not found"

RECORD = r'''
# Kaggle validation only: record the figures step 8 printed.
import json, platform, transformers, trl, datasets, accelerate
logs = [h for h in trainer.state.log_history if "loss" in h]
res = {"env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "transformers": transformers.__version__, "trl": trl.__version__, "datasets": datasets.__version__,
               "accelerate": accelerate.__version__, "gpu": torch.cuda.get_device_name(0)},
       "base": {"evalLoss": before, "generation": {k: v for k, v in g0.items() if k != "samples"}},
       "final": {"evalLoss": after, "generation": {k: v for k, v in g1.items() if k != "samples"}},
       "train": {"steps": trainer.state.global_step, "rows": len(trainer.train_dataset), "runtimeSec": out.metrics["train_runtime"],
                 "firstLoss": logs[0]["loss"], "lastLoss": logs[-1]["loss"], "peakMemGB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                 "tokens": logs[-1].get("num_tokens")},
       "lossCurve": [{"step": h["step"], "loss": h["loss"]} for h in logs],
       "samplesAfter": g1["samples"][:30], "samplesBefore": g0["samples"][:10]}
json.dump(res, open("/kaggle/working/notebook_results.json", "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in res.items() if k not in ("lossCurve", "samplesAfter", "samplesBefore")})
'''
ISO = r'''
# Kaggle validation only: the step 10 arms our first session did not have, run with the page's own step 9 and 10 code.
import json, psutil
RUN_EXPERIMENT = True
del trainer, model360
start = psutil.Process().create_time()
iso = []
for seed, name, restart, explicit in [(0, "cross-document, positions restart", True, True),
                                      (1, "cross-document, positions restart", True, True),
                                      (0, "cross-document, positions run on", False, True)]:
    if time.time() - start > 70 * 60:
        print("skipped", name, seed, "to stay inside the GPU budget")
        continue
    t = fresh_trainer(seed=seed, data_seed=seed)
    t.data_collator = cross_doc_collator(t.data_collator, restart, explicit)
    torch.cuda.reset_peak_memory_stats()
    r = train_and_score(t, name + ", seed " + str(seed))
    r.update(seed=seed, restartPositions=restart, explicitMask=explicit, secPerStep=round(r["seconds"] / r["steps"], 3),
             peakMemGBIncludingScoring=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    iso.append(r)
    json.dump(iso, open("/kaggle/working/iso_runs.json", "w"), indent=1)
    del t
print(json.dumps(iso, indent=1))
'''
PROBE = r'''
# Kaggle validation only: why naive packing ran 2.3x slower per step in llt-sft-runs.
# One packed micro-batch, forward + backward under fp16 autocast, positions restarted vs continuous.
import time, collections
from torch.profiler import profile, ProfilerActivity
b = next(iter(trainer.get_train_dataloader()))
x = b["input_ids"].to(GPU); y = b["labels"].to(GPU); pos = b["position_ids"].to(GPU)
model360.train()
def step(p, am=None):
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model360(input_ids=x, position_ids=p, attention_mask=am, use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), y[0, 1:], ignore_index=-100)
    loss.backward()
    model360.zero_grad(set_to_none=True)
probe = {"shape": list(x.shape), "docs": int((pos == 0).sum())}
n = x.shape[1]
cont = torch.arange(n, device=GPU)[None]
full = torch.tril(torch.ones(n, n, dtype=torch.bool, device=GPU))[None, None]
for name, p, am in [("reset", pos, None), ("continuous", cont, None), ("continuousExplicitCausal", cont, full),
                    ("restartExplicitCausal", pos, full)]:
    for _ in range(2):
        step(p, am)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for _ in range(5):
        step(p, am)
    torch.cuda.synchronize()
    sec = (time.time() - t) / 5
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(p, am); torch.cuda.synchronize()
    k = collections.Counter()
    for e in prof.key_averages():
        if e.device_time_total > 0:
            k[e.key[:90]] += e.device_time_total
    probe[name] = {"secPerFwdBwd": round(sec, 3), "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "topKernelsUs": [[n, round(v)] for n, v in k.most_common(8)],
                   "attentionKernels": sorted({n for n in k if any(s in n.lower() for s in ("fmha", "attention", "flash", "efficient", "softmax"))})}
    print(name, probe[name]["secPerFwdBwd"], probe[name]["attentionKernels"])
json.dump(probe, open("/kaggle/working/sdpa_probe.json", "w"), indent=1)
'''
nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": RECORD})
nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": PROBE})
nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ISO})

out_dir = os.path.join(HERE, SLUG)
os.makedirs(out_dir, exist_ok=True)
json.dump(nb, open(os.path.join(out_dir, SLUG + ".ipynb"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
meta = {"id": "genieincodebottle/" + SLUG, "title": "LLT SFT Notebook", "code_file": SLUG + ".ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": [], "keywords": []}
json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)
print("built", out_dir, len(nb["cells"]), "cells")
