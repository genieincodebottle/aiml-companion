"""What the replies look like, base vs trained: operators used, whether </think> survives, how often the one-shot
example's wording is copied, and whether the arithmetic claimed inside <think> is true.

  python analyze_replies.py --out evals/replies.json --model Qwen/Qwen3-0.6B-Base --adapters runs/a/adapter-final,runs/b/adapter-final
Runs with vLLM: 8 samples per test problem at temperature 1.0, plus the greedy reply. Saves every reply.
"""
import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import countdown as cd   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
ap.add_argument("--adapters", default="")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--max_new", type=int, default=256)
ap.add_argument("--mem", type=float, default=0.5)
args = ap.parse_args()
cd.PROMPT_STYLE = "oneshot"
from vllm import LLM, SamplingParams          # noqa: E402
from vllm.lora.request import LoRARequest     # noqa: E402

CLAIM = re.compile(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)")


def stats(replies):
    n = len(replies)
    ans = [cd.extract_answer(r) or "" for r in replies]
    claims = [m for r in replies for m in CLAIM.findall(r.split("<answer>")[0])]
    true = 0
    for a, op, b, c in claims:
        a, b, c = Fraction(int(a)), Fraction(int(b)), Fraction(int(c))
        v = cd._apply(a, b, op)
        true += v is not None and v == c
    return dict(n=n,
                usesMulOrDiv=round(sum(("*" in x or "/" in x) for x in ans) / n, 4),
                closesThink=round(sum("</think>" in r for r in replies) / n, 4),
                copiesExample=round(sum("fairly large" in r for r in replies) / n, 4),
                claimsPerReply=round(len(claims) / n, 2), claimsTrue=round(true / max(1, len(claims)), 4))


adapters = [a for a in args.adapters.split(",") if a]
llm = LLM(args.model, dtype="float16", seed=7, gpu_memory_utilization=args.mem, max_model_len=1024 + args.max_new,
          enable_lora=bool(adapters), max_lora_rank=16)
test = [json.loads(line) for line in open(HERE / "data" / "test.jsonl")]
golds = [cd.problem_text(p) for p in test]
prompts = [cd.prompt(g) for g in golds]
out = {}
for i, a in enumerate([None] + adapters):
    lr = LoRARequest("a" + str(i), i + 1, str(Path(a).resolve())) if a else None
    s = llm.generate(prompts, SamplingParams(n=args.n, temperature=1.0, top_p=1.0, top_k=-1, seed=7, max_tokens=args.max_new,
                                             stop=cd.STOPS, include_stop_str_in_output=True), lora_request=lr, use_tqdm=False)
    replies = [c.text for o in s for c in o.outputs]
    replies_golds = [(c.text, g) for o, g in zip(s, golds) for c in o.outputs]
    st = stats(replies)
    st["correct"] = round(sum(cd.correct(r, g) for r, g in replies_golds) / len(replies), 4)
    ok = [r for r, g in replies_golds if cd.correct(r, g)]
    bad = [r for r, g in replies_golds if not cd.correct(r, g)]
    st["correctReplies"], st["wrongReplies"] = stats(ok) if ok else None, stats(bad) if bad else None
    key = a or "base"
    out[key] = dict(stats=st, replies=[dict(problem=g, reply=r, reason=cd.diagnose(r, g)) for r, g in replies_golds])
    print(key, json.dumps(st), flush=True)
json.dump(out, open(args.out, "w"), indent=1)
