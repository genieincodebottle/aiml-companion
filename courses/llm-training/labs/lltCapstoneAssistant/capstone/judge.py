"""Stage 5b. A pairwise LLM judge: first check it against human labels, then use it on our models.

Every pair is judged twice, once in each order, because judges favour a position. A model's score on a prompt is
the mean over the two orders of win = 1, tie = 0.5, loss = 0. A verdict the parser cannot read counts as a tie
and is reported.

Calibration set: HelpSteer3 preference (CC-BY-4.0), general-domain English single-turn pairs where the human
annotators' overall preference is at least 2 on their -3..3 scale, so there is a clear human answer.
"""
import collections
import re

from .common import JUDGE, Timer, cli, done, path, read_json, read_jsonl, stage_record, write_json
from .gen import run_multi
from .stats import boot, lc_winrate, wilson

SYSTEM = ("You compare two assistant replies to the same user request. Decide which reply better does what the user asked: "
          "correct, complete, follows every explicit instruction, and is clear. Do not prefer a reply because it is longer. "
          "Do not let the order of the replies influence you. If they are equally good or equally bad, call it a tie. "
          "Reason briefly, then end with exactly one verdict: [[A]] if reply A is better, [[B]] if reply B is better, [[C]] for a tie.")
ASK = "[User request]\n{p}\n\n[Reply A]\n{a}\n\n[Reply B]\n{b}"
VERDICT = re.compile(r"\[\[([ABC])\]\]")
PAIRS = [("dpo", "official"), ("sft", "official"), ("dpo", "sft")]


def judge_kwargs(judge):
    if "gpt-oss" in judge:
        return {"temperature": 0.0, "max_new": 2048, "chat_kwargs": {"reasoning_effort": "low"}, "max_model_len": 16384}
    return {"temperature": 0.0, "max_new": 768, "chat_kwargs": None, "max_model_len": 16384}


def _inputs(items):
    return [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK.format(p=p, a=x, b=y)}] for p, x, y in items]


def run_judge(a, judge, named_items):
    """named_items: {tag: [(prompt, replyA, replyB), ...]}. One model load for all of them.
    Returns {tag: (verdicts, result)} with verdicts 'A' | 'B' | 'C' | None."""
    kw = judge_kwargs(judge)
    margs = {"model": judge, "backend": a.cfg["backend"], "seed": a.seed, "max_model_len": kw["max_model_len"],
             "dtype": "bfloat16" if a.cfg["backend"] == "vllm" else "float16", "quant": "nf4" if a.cfg["backend"] == "hf" else None}
    tags = list(named_items)
    groups = [{"inputs": _inputs(named_items[t]), "out": path(a, "judge", "%s_%s.json" % (t, judge.split("/")[-1])),
               "params": {"n": 1, "temperature": kw["temperature"], "top_p": 1.0, "max_new": kw["max_new"],
                          "chat_kwargs": kw["chat_kwargs"]}} for t in tags]
    out = {}
    for t, res in zip(tags, run_multi(margs, groups)):
        v = []
        for r in res["results"]:
            m = VERDICT.findall(r[0]["text"])
            v.append(m[-1] if m else None)
        out[t] = (v, res)
    return out


def calibrate(judge, calib, V):
    (fwd, r1), (rev, r2) = V["calib_ab"], V["calib_ba"]
    soft, strict, consistent, first, nontie, longer, unread = [], 0, 0, 0, 0, 0, 0
    for c, v1, v2 in zip(calib, fwd, rev):
        w1 = {"A": "A", "B": "B"}.get(v1)               # winner in human terms (A = response1)
        w2 = {"A": "B", "B": "A"}.get(v2)
        unread += (v1 is None) + (v2 is None)
        soft.append(sum(1.0 if w == c["human"] else 0.5 if w is None else 0.0 for w in (w1, w2)) / 2)
        strict += w1 == c["human"] and w2 == c["human"]
        consistent += w1 is not None and w1 == w2
        for v in (v1, v2):
            if v in ("A", "B"):
                nontie += 1
                first += v == "A"
        if w1 is not None and w1 == w2:
            longer += (len(c["a"]) > len(c["b"])) == (w1 == "A")
    n = len(calib)
    human_longer = sum((len(c["a"]) > len(c["b"])) == (c["human"] == "A") for c in calib)
    return {"judge": judge, "n": n, "softAgreement": round(sum(soft) / n, 4), "softCI": boot(soft),
            "strictAgreement": round(strict / n, 4), "strictCI": wilson(strict, n),
            "consistentShare": round(consistent / n, 4), "firstPositionShare": round(first / max(1, nontie), 4),
            "judgePrefersLongerShare": round(longer / max(1, consistent), 4), "humanPrefersLongerShare": round(human_longer / n, 4),
            "unreadable": unread, "sec": r1["wallClockSec"] + r2["wallClockSec"]}


def compare(judge, arena, gens, x, y, V):
    gx, gy = gens[x], gens[y]
    (fwd, r1), (rev, r2) = V["%s_vs_%s_ab" % (x, y)], V["%s_vs_%s_ba" % (x, y)]
    score, dlen, per_cat, unread, consistent = [], [], collections.defaultdict(list), 0, 0
    for i, q in enumerate(arena):
        s1 = {"A": 1.0, "B": 0.0}.get(fwd[i], 0.5)
        s2 = {"B": 1.0, "A": 0.0}.get(rev[i], 0.5)
        unread += (fwd[i] is None) + (rev[i] is None)
        consistent += fwd[i] in ("A", "B") and s1 == s2
        s = (s1 + s2) / 2
        score.append(s)
        dlen.append(gx[i]["tokens"] - gy[i]["tokens"])
        per_cat[q["category"]].append(s)
    n = len(score)
    return {"x": x, "y": y, "judge": judge, "n": n, "winRate": round(sum(score) / n, 4), "ci": boot(score),
            "lengthControlled": lc_winrate(score, dlen),
            "perCategory": {k: {"n": len(v), "winRate": round(sum(v) / len(v), 4), "ci": boot(v)} for k, v in per_cat.items()},
            "meanTokens": {x: round(sum(g["tokens"] for g in gx) / n, 1), y: round(sum(g["tokens"] for g in gy) / n, 1)},
            "consistentShare": round(consistent / n, 4), "unreadable": unread,
            "perPrompt": [{"uid": q["uid"], "score": s, "verdicts": [fwd[i], rev[i]]} for i, (q, s) in enumerate(zip(arena, score))],
            "sec": r1["wallClockSec"] + r2["wallClockSec"]}


def main():
    a = cli("Stage 5b: judge", [(("--judges",), dict(default=None)), (("--pairs",), dict(default=None))])
    judges = (a.judges or JUDGE[a.tier]).split(",")
    calib = read_jsonl(path(a, "data", "calib.jsonl"))
    arena = read_jsonl(path(a, "data", "arena.jsonl"))
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")] if a.pairs else PAIRS
    rec = {"calibration": {}, "comparisons": {}}
    if done(path(a, "records", "judge.json")):
        rec = read_json(path(a, "records", "judge.json"))
    gens = {m: [r[0] for r in read_json(path(a, "eval", "gen_%s_arena.json" % m))["results"]]
            for m in {m for p in pairs for m in p}}
    items = {"calib_ab": [(c["prompt"], c["a"], c["b"]) for c in calib], "calib_ba": [(c["prompt"], c["b"], c["a"]) for c in calib]}
    for x, y in pairs:
        items["%s_vs_%s_ab" % (x, y)] = [(q["prompt"], gens[x][i]["text"], gens[y][i]["text"]) for i, q in enumerate(arena)]
        items["%s_vs_%s_ba" % (x, y)] = [(q["prompt"], gens[y][i]["text"], gens[x][i]["text"]) for i, q in enumerate(arena)]
    with Timer() as t:
        for j in judges:
            V = run_judge(a, j, items)
            rec["calibration"][j] = calibrate(j, calib, V)
            for x, y in pairs:
                c = compare(j, arena, gens, x, y, V)
                write_json(path(a, "judge", "compare_%s_vs_%s_%s.json" % (x, y, j.split("/")[-1])), c)
                rec["comparisons"]["%s|%s vs %s" % (j, x, y)] = {k: v for k, v in c.items() if k != "perPrompt"}
    stage_record(a, "judge", {"wallClockSec": t.sec, **{k: v for k, v in rec.items() if k in ("calibration", "comparisons")},
                              "prompt": {"system": SYSTEM, "user": ASK}})


if __name__ == "__main__":
    main()
