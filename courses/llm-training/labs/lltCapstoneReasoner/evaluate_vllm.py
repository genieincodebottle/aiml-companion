"""pass@k with vLLM (T3 and above): the same test set, prompt, stop strings, checker and estimator as evaluate.py,
about 10x faster for many samples. vLLM pins its own torch, so it lives in its own environment:
  uv venv vllm-venv --python 3.12 && source vllm-venv/bin/activate && uv pip install vllm==0.33.0

  python evaluate_vllm.py --out evals/base.json --model Qwen/Qwen3-0.6B-Base --n 64
  python evaluate_vllm.py --out evals/x.json --model Qwen/Qwen3-0.6B-Base --adapter runs/x/adapter-final --n 64
Several adapters in one call (the base model loads once): --adapter a,b,c --out ea.json,eb.json,ec.json
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import countdown as cd   # noqa: E402
import passk             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
ap.add_argument("--adapter", default="")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--max_new", type=int, default=256)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--prompt", default="oneshot", choices=["zero", "oneshot"])
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--mem", type=float, default=0.35)
ap.add_argument("--data", default=str(HERE / "data"))
ap.add_argument("--test_file", default="test.jsonl")
args = ap.parse_args()
cd.PROMPT_STYLE = args.prompt

from vllm import LLM, SamplingParams          # noqa: E402
from vllm.lora.request import LoRARequest     # noqa: E402

adapters = [a for a in args.adapter.split(",") if a] or [None]
outs = args.out.split(",")
assert len(outs) == len(adapters), "one --out per adapter"
llm = LLM(args.model, dtype=args.dtype, seed=args.seed, gpu_memory_utilization=args.mem, max_model_len=1024 + args.max_new,
          enable_lora=adapters != [None], max_lora_rank=16, enable_prefix_caching=True)
test = [json.loads(line) for line in open(Path(args.data) / args.test_file, encoding="utf-8")][:args.n_test]
golds = [cd.problem_text(p) for p in test]
prompts = [cd.prompt(g) for g in golds]
common = dict(max_tokens=args.max_new, stop=cd.STOPS, include_stop_str_in_output=True)
ks = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] if k <= args.n]

for ai, (adapter, out) in enumerate(zip(adapters, outs)):
    lr = LoRARequest("a" + str(ai), ai + 1, str(Path(adapter).resolve())) if adapter else None
    t0 = time.time()
    g = llm.generate(prompts, SamplingParams(temperature=0, **common), lora_request=lr, use_tqdm=False)
    t1 = time.time()
    s = llm.generate(prompts, SamplingParams(n=args.n, temperature=args.temperature, top_p=1.0, top_k=-1, seed=args.seed, **common),
                     lora_request=lr, use_tqdm=False)
    t2 = time.time()
    greedy = [o.outputs[0].text for o in g]
    g_tok = [len(o.outputs[0].token_ids) for o in g]
    counts, why, len_ok, len_bad, trunc, total = [], collections.Counter(), [], [], 0, 0
    examples = dict(correct=[], wrong=[])
    nonascii = 0.0
    for gi, o in zip(golds, s):
        c = 0
        for j, comp in enumerate(o.outputs):
            reason = cd.diagnose(comp.text, gi)
            why[reason] += 1
            n_tok = len(comp.token_ids)
            (len_ok if reason == "correct" else len_bad).append(n_tok)
            trunc += comp.finish_reason == "length"
            nonascii += sum(ord(ch) > 127 for ch in comp.text) / max(1, len(comp.text))
            total += 1
            c += reason == "correct"
            bucket = "correct" if reason == "correct" else "wrong"
            if j == 0 and len(examples[bucket]) < 12:
                examples[bucket].append(dict(problem=gi, reply=comp.text, reason=reason, tokens=n_tok))
        counts.append(c)
    g_ok = [cd.correct(r, gi) for r, gi in zip(greedy, golds)]
    by_n = {}
    for size in (3, 4):
        idx = [i for i, p in enumerate(test) if len(p["nums"]) == size]
        if not idx:
            continue
        by_n[str(size)] = dict(greedy=round(sum(g_ok[i] for i in idx) / len(idx), 4),
                               passAtK={k: round(v, 4) for k, v in passk.curve([counts[i] for i in idx], args.n, ks).items()})
    lo, hi = passk.bootstrap_ci(g_ok)
    res = dict(model=args.model, adapter=adapter, engine="vllm", n=args.n, nTest=len(test), temperature=args.temperature, maxNew=args.max_new,
               dtype=args.dtype, prompt=args.prompt,
               greedy=dict(mean=round(sum(g_ok) / len(g_ok), 4), lo=round(lo, 4), hi=round(hi, 4)),
               passAtK=passk.curve_with_ci(counts, args.n, ks), byNumbers=by_n, counts=counts, greedyCorrect=g_ok,
               solvedByAnySample=round(sum(c > 0 for c in counts) / len(counts), 4),
               reasons={k: round(v / total, 4) for k, v in why.most_common()},
               meanTokens=dict(all=round((sum(len_ok) + sum(len_bad)) / total, 1), correct=round(sum(len_ok) / len(len_ok), 1) if len_ok else None,
                               wrong=round(sum(len_bad) / len(len_bad), 1) if len_bad else None, greedy=round(sum(g_tok) / len(g_tok), 1)),
               truncated=round(trunc / total, 4), nonAsciiShare=round(nonascii / total, 4), examples=examples, greedyReplies=greedy[:20],
               sec=dict(greedy=round(t1 - t0, 1), sampled=round(t2 - t1, 1)), vllm=__import__("vllm").__version__, date=time.strftime("%Y-%m-%d"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(out, json.dumps({k: res[k] for k in ("greedy", "passAtK", "byNumbers", "solvedByAnySample", "reasons", "meanTokens", "truncated", "nonAsciiShare", "sec")}), flush=True)
