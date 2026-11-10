"""The ablation table, main effects, interaction and seed spread, from the run folders and the pass@k evaluations.

  python analyze.py --runs runs --evals evals --out results.json
Expects runs/<cell>_s<seed>/ (metrics.jsonl, evals.jsonl, config.json) and evals/<cell>_s<seed>.json (evaluate.py or
evaluate_vllm.py output), cells named agg-<grpo|dr_grpo>_scale-<1|0>. Effects are paired over the same test problems:
for each problem we take the difference in pass@1 (from the sample counts), average it over the matched pairs of runs,
and bootstrap over problems. Seed spread is reported next to every effect, because two seeds per cell show run-to-run
noise only roughly and the bootstrap does not include it.
"""
import argparse
import json
from pathlib import Path

import passk

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--evals", default="evals")
ap.add_argument("--out", default="results.json")
ap.add_argument("--seeds", default="0,1")
ap.add_argument("--k", type=int, default=1, help="effects on pass@k for this k")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
AGG, SCALE = ["grpo", "dr_grpo"], [1, 0]


def name(a, s, seed):
    return "agg-%s_scale-%d_s%d" % (a, s, seed)


def per_problem(ev, k):
    return [passk.pass_at_k(ev["n"], c, k) for c in ev["counts"]]


def lines(p):
    return [json.loads(x) for x in open(p)] if p.exists() else []


runs, table = {}, []
for a in AGG:
    for s in SCALE:
        for seed in SEEDS:
            nm = name(a, s, seed)
            ev = json.load(open(Path(args.evals) / (nm + ".json")))
            m = lines(Path(args.runs) / nm / "metrics.jsonl")
            last = m[-25:]
            runs[nm] = ev
            table.append(dict(run=nm, agg=a, scale=s, seed=seed, greedy=ev["greedy"]["mean"], passAt1=ev["passAtK"]["1"]["mean"] if "1" in ev["passAtK"] else ev["passAtK"][1]["mean"],
                              passAtK={str(k): v["mean"] for k, v in ev["passAtK"].items()}, meanTokens=ev["meanTokens"],
                              rewardLast25=round(sum(x["reward"] for x in last) / len(last), 4) if last else None,
                              lenFirst25=round(sum(x["meanLen"] for x in m[:25]) / 25, 1) if len(m) >= 25 else None,
                              lenLast25=round(sum(x["meanLen"] for x in last) / len(last), 1) if last else None,
                              zeroStdMean=round(sum(x["zeroStd"] for x in m) / len(m), 3) if m else None,
                              trainSec=round(sum(x["sec"] for x in m), 1) if m else None))


def effect(pairs, k):
    """pairs: list of (run_hi, run_lo). Per problem, average over pairs of (hi - lo); paired bootstrap over problems."""
    vals = None
    for hi, lo in pairs:
        d = [x - y for x, y in zip(per_problem(runs[hi], k), per_problem(runs[lo], k))]
        vals = d if vals is None else [v + w for v, w in zip(vals, d)]
    vals = [v / len(pairs) for v in vals]
    lo_, hi_ = passk.bootstrap_ci(vals)
    per_pair = [round(sum(per_problem(runs[h], k)) / len(runs[h]["counts"]) - sum(per_problem(runs[lw], k)) / len(runs[lw]["counts"]), 4) for h, lw in pairs]
    return dict(mean=round(sum(vals) / len(vals), 4), lo=round(lo_, 4), hi=round(hi_, 4), perPair=per_pair)


k = args.k
eff = {
    "agg dr_grpo minus grpo": effect([(name("dr_grpo", s, sd), name("grpo", s, sd)) for s in SCALE for sd in SEEDS], k),
    "scale off minus on": effect([(name(a, 0, sd), name(a, 1, sd)) for a in AGG for sd in SEEDS], k),
}
inter = [((name("dr_grpo", 0, sd), name("dr_grpo", 1, sd)), (name("grpo", 0, sd), name("grpo", 1, sd))) for sd in SEEDS]
vals = None
for (a, b), (c, d) in inter:
    v = [(w - x) - (y - z) for w, x, y, z in zip(per_problem(runs[a], k), per_problem(runs[b], k), per_problem(runs[c], k), per_problem(runs[d], k))]
    vals = v if vals is None else [p + q for p, q in zip(vals, v)]
vals = [v / len(inter) for v in vals]
lo_, hi_ = passk.bootstrap_ci(vals)
eff["interaction"] = dict(mean=round(sum(vals) / len(vals), 4), lo=round(lo_, 4), hi=round(hi_, 4))
spread = {}
for a in AGG:
    for s in SCALE:
        xs = [r["passAt1"] for r in table if r["agg"] == a and r["scale"] == s]
        spread["agg-%s_scale-%d" % (a, s)] = dict(mean=round(sum(xs) / len(xs), 4), seeds=xs, spread=round(max(xs) - min(xs), 4))
res = dict(k=k, table=table, effects=eff, cells=spread)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps(dict(effects=eff, cells=spread), indent=1))
