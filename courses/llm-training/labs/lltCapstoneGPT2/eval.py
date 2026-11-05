# eval.py - score our checkpoints and openai-community/gpt2 in lm-evaluation-harness 0.4.13, same tasks and dtype.
#
#   python eval.py --weights /workspace/run/weights/weights_step_019073.pt --baseline
#   python eval.py --weights "/workspace/run/weights/weights_step_*.pt" --tasks hellaswag     # a HellaSwag curve
#
# Writes one Run Library eval file per model into --out, and prints a comparison table.
import argparse, glob, importlib.metadata, json, os, time
import numpy as np
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer
from hf_wrapper import load

p = argparse.ArgumentParser()
p.add_argument("--weights", default="")                 # a file or a glob
p.add_argument("--baseline", action="store_true")       # also score openai-community/gpt2
p.add_argument("--tasks", default="hellaswag,arc_easy,piqa,lambada_openai")
p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--limit", type=int, default=0)          # 0 = full tasks; tests only
p.add_argument("--val", default="")                     # val.bin for the wrapper-equals-native check
p.add_argument("--out", default="evals")
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
cap = torch.cuda.get_device_capability(0)
dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.dtype, torch.bfloat16 if cap >= (8, 0) else torch.float16)
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
tasks = a.tasks.split(",")


def score(lm, name, extra):
    t0 = time.time()
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, limit=a.limit or None,
                                  batch_size=a.batch_size, log_samples=False)
    ev = {t: {k: v for k, v in r.items() if k not in ("alias", "name")} for t, r in res["results"].items()}
    rec = {"runId": "lltCapstoneGPT2-eval-" + name, "module": "lltCapstoneGPT2", "label": "eval",
           "hardware": {"gpu": torch.cuda.get_device_name(0)}, "env": {"lm_eval": importlib.metadata.version("lm_eval"),
           "torch": torch.__version__, "dtype": str(dtype)}, "config": {"tasks": tasks, "num_fewshot": 0,
           "limit": a.limit or None, "batch_size": a.batch_size, **extra}, "evals": ev,
           "wallClockSec": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d")}
    with open(os.path.join(a.out, rec["runId"] + ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    return ev


def headline(ev):
    keys = {"hellaswag": "acc_norm,none", "arc_easy": "acc,none", "piqa": "acc,none", "lambada_openai": "acc,none"}
    return {t: round(ev[t][k], 4) for t, k in keys.items() if t in ev}


rows = []
for path in sorted(glob.glob(a.weights)) if a.weights else []:
    model, blob = load(path, dtype=dtype)
    if a.val:                                   # the wrapper must give the same loss as the native model
        w = torch.from_numpy(np.memmap(a.val, dtype=np.uint16, mode="r")[:4 * 1024 + 1].astype(np.int64)).cuda()
        x, y = w[:-1].view(4, 1024), w[1:].view(4, 1024)
        with torch.no_grad():
            native = model.gpt(x, y).item()
            last = torch.nn.functional.cross_entropy(model(input_ids=x).logits.float().flatten(0, 1), y.flatten()).item()
        print(f"wrapper check: native {native:.6f}  wrapper logits {last:.6f}  diff {abs(native - last):.2e}", flush=True)
        assert abs(native - last) < 1e-3, "the wrapper's logits differ from the native model"
    lm = HFLM(pretrained=model, tokenizer=tok, backend="causal", batch_size=a.batch_size, max_length=1024)
    name = "ours-step%06d" % blob["step"]
    ev = score(lm, name, {"weights": os.path.basename(path), "step": blob["step"], "tokens": blob["tokens"]})
    rows.append((name, blob["tokens"], headline(ev)))
    del lm, model
    torch.cuda.empty_cache()

if a.baseline:
    lm = HFLM(pretrained="openai-community/gpt2", dtype="bfloat16" if dtype == torch.bfloat16 else "float16",
              batch_size=a.batch_size, device="cuda")
    rows.append(("openai-community/gpt2", None, headline(score(lm, "openai-gpt2", {"model": "openai-community/gpt2"}))))

for name, tokens, h in rows:
    print(f"{name:<24} tokens {tokens or '':>14}  " + "  ".join(f"{t} {v}" for t, v in h.items()), flush=True)
