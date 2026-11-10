"""pass@k curves, length statistics and failure cases for a base model or a trained adapter, on the held-out test set.

  python evaluate.py --out evals/base-0.6b.json --model Qwen/Qwen3-0.6B-Base --n 32
  python evaluate.py --out evals/x.json --model Qwen/Qwen3-0.6B-Base --adapter runs/x/adapter-final --n 32

n samples per problem at temperature 1.0 (the training temperature), plus one greedy reply. pass@k for every
k = 1, 2, 4, ... n uses the unbiased estimator from lltRL (rl_core.pass_at_k), with bootstrap CIs over problems.
The file keeps the correct count of every problem, so two models can be compared with a paired bootstrap later.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import countdown as cd   # noqa: E402
import passk             # noqa: E402
import rl_core           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
ap.add_argument("--adapter", default=None, help="a saved LoRA folder (local path) or hub:<subfolder> in aimlcompanion/llm-training-lab")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--max_new", type=int, default=256)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--data", default=str(HERE / "data"))
ap.add_argument("--test_file", default="test.jsonl")
ap.add_argument("--prompt", default="oneshot", choices=["zero", "oneshot"])
args = ap.parse_args()
cd.PROMPT_STYLE = args.prompt

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qtok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
ns = rl_core.load(args.model, args.max_new, cd.STOPS, qtok, DEV)
torch.manual_seed(args.seed)
if args.adapter and args.adapter.startswith("hub:"):
    model = ns["load_policy"](args.adapter[4:])
else:
    model = ns["load_policy"]()
    if args.adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, {k: v.to(DEV) for k, v in load_file(str(Path(args.adapter) / "adapter_model.safetensors")).items()})
model.eval()
test = [json.loads(line) for line in open(Path(args.data) / args.test_file, encoding="utf-8")][:args.n_test]
golds = [cd.problem_text(p) for p in test]
prompts = [cd.prompt(g) for g in golds]

t0 = time.time()
greedy_rows = ns["sample"](model, prompts, temperature=0, batch=args.batch)
greedy = [qtok.decode(r) for r in greedy_rows]
t1 = time.time()
rows = ns["sample"](model, [p for p in prompts for _ in range(args.n)], temperature=args.temperature, batch=args.batch)
t2 = time.time()
replies = [qtok.decode(r) for r in rows]

counts, why, len_ok, len_bad, trunc = [], collections.Counter(), [], [], 0
examples = dict(correct=[], wrong=[])
for i, g in enumerate(golds):
    c = 0
    for j in range(args.n):
        k = i * args.n + j
        reason = cd.diagnose(replies[k], g)
        why[reason] += 1
        (len_ok if reason == "correct" else len_bad).append(len(rows[k]))
        trunc += len(rows[k]) >= args.max_new
        c += reason == "correct"
        if j == 0 and len(examples["correct" if reason == "correct" else "wrong"]) < 12:
            examples["correct" if reason == "correct" else "wrong"].append(dict(problem=g, reply=replies[k], reason=reason, tokens=len(rows[k])))
    counts.append(c)
ks = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] if k <= args.n]
g_ok = [cd.correct(r, g) for r, g in zip(greedy, golds)]
by_n = {}
for size in (3, 4):
    idx = [i for i, p in enumerate(test) if len(p["nums"]) == size]
    if not idx:
        continue
    by_n[str(size)] = dict(greedy=round(sum(g_ok[i] for i in idx) / len(idx), 4),
                           passAtK={k: round(v, 4) for k, v in passk.curve([counts[i] for i in idx], args.n, ks).items()})
lo, hi = passk.bootstrap_ci(g_ok)
res = dict(model=args.model, adapter=args.adapter, n=args.n, nTest=len(test), temperature=args.temperature, maxNew=args.max_new,
           greedy=dict(mean=round(sum(g_ok) / len(g_ok), 4), lo=round(lo, 4), hi=round(hi, 4)),
           passAtK=passk.curve_with_ci(counts, args.n, ks), byNumbers=by_n, counts=counts, greedyCorrect=g_ok,
           solvedByAnySample=round(sum(c > 0 for c in counts) / len(counts), 4),
           reasons={k: round(v / len(replies), 4) for k, v in why.most_common()},
           meanTokens=dict(all=round(sum(map(len, rows)) / len(rows), 1), correct=round(sum(len_ok) / len(len_ok), 1) if len_ok else None,
                           wrong=round(sum(len_bad) / len(len_bad), 1) if len_bad else None, greedy=round(sum(map(len, greedy_rows)) / len(greedy_rows), 1)),
           truncated=round(trunc / len(rows), 4), examples=examples, greedyReplies=greedy[:20],
           sec=dict(greedy=round(t1 - t0, 1), sampled=round(t2 - t1, 1)), gpu=torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
           lltrlFunctions=ns["_hashes"], date=time.strftime("%Y-%m-%d"))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("greedy", "passAtK", "byNumbers", "solvedByAnySample", "reasons", "meanTokens", "truncated", "sec")}), flush=True)
