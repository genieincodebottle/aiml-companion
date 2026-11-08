import gc, json, os, time
import numpy as np
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

SMOKE = os.environ.get("LLT_SMOKE") == "1"          # 1 = a few items per task, to test the notebook quickly
device = "cuda" if torch.cuda.is_available() else "cpu"
NL = chr(10)
if device == "cuda":     # T2: a base model and the same model after its maker's post-training
    BASE, TUNED, DTYPE, LIMIT = "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B", "float16", 1000
else:                    # T1, a CPU: a smaller pair and fewer items
    BASE, TUNED, DTYPE, LIMIT = "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct", "float32", 200
if os.environ.get("LLT_MODELS"):                    # your own pair, e.g. "base-id,your-finetune-dir"
    BASE, TUNED = os.environ["LLT_MODELS"].split(",")
if SMOKE:
    LIMIT = 16
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]
MMLU_PER_SUBJECT = 2 if SMOKE else 20 if device == "cuda" else 5   # the first questions of each of 57 subjects
os.makedirs("results", exist_ok=True)
TM = TaskManager()                                  # indexes every task YAML once; reuse it for each call

def requests_of(sample):
    # lm-eval logs each loglikelihood request as (context, continuation).
    args = sample["arguments"]
    if isinstance(args, dict):
        args = [[a["arg_0"], a["arg_1"]] for a in args.values()]
    return [list(a) for a in args]

def run_harness(name):
    # Batch 8: a batch of long MMLU items times Qwen3's 151,936-token vocabulary needs about 2.4 GB of fp16 logits.
    lm = HFLM(pretrained=name, dtype=DTYPE, batch_size=8, device=device)
    t = time.time()
    main = lm_eval.simple_evaluate(model=lm, tasks=TASKS, limit=LIMIT, log_samples=True, task_manager=TM)
    mmlu = lm_eval.simple_evaluate(model=lm, tasks=["mmlu"], limit=MMLU_PER_SUBJECT, log_samples=True, task_manager=TM)
    out = {"model": name, "seconds": round(time.time() - t, 1), "tasks": {}}
    for task in TASKS:
        rows = sorted(main["samples"][task], key=lambda s: s["doc_id"])
        out["tasks"][task] = {"acc": [float(s["acc"]) for s in rows],
                              "acc_norm": [float(s["acc_norm"]) for s in rows] if "acc_norm" in rows[0] else None,   # winogrande has acc only
                              "stderr": main["results"][task]["acc_stderr,none"]}
    rows = [s for sub in sorted(mmlu["samples"]) for s in sorted(mmlu["samples"][sub], key=lambda s: s["doc_id"])]
    out["tasks"]["mmlu"] = {"acc": [float(s["acc"]) for s in rows], "acc_norm": None,
                            "stderr": mmlu["results"]["mmlu"]["acc_stderr,none"]}
    # Keep the ARC-Easy requests and the harness's own log-likelihoods for step 2.
    arc = sorted(main["samples"]["arc_easy"], key=lambda s: s["doc_id"])
    out["arcRequests"] = [{"doc_id": s["doc_id"], "target": s["target"], "pairs": requests_of(s),
                           "lls": [float(r[0]) for r in s["filtered_resps"]]} for s in arc]
    return lm, out

LMS, HARNESS = {}, {}
for name in [BASE, TUNED]:
    LMS[name], HARNESS[name] = run_harness(name)
    gc.collect()
    torch.cuda.empty_cache()
    print(name, "| harness time", HARNESS[name]["seconds"], "s")
print("task".ljust(14), "n".rjust(5), "base acc".rjust(9), "tuned acc".rjust(10), "stderr".rjust(7), "base acc_norm".rjust(14), "tuned acc_norm".rjust(15))
for task in TASKS + ["mmlu"]:
    b, t = HARNESS[BASE]["tasks"][task], HARNESS[TUNED]["tasks"][task]
    norm = lambda d: "-" if d["acc_norm"] is None else str(round(np.mean(d["acc_norm"]), 3))
    print(task.ljust(14), str(len(b["acc"])).rjust(5), str(round(np.mean(b["acc"]), 3)).rjust(9),
          str(round(np.mean(t["acc"]), 3)).rjust(10), str(round(t["stderr"], 3)).rjust(7), norm(b).rjust(14), norm(t).rjust(15))
json.dump(HARNESS, open("results/harness.json", "w"))
