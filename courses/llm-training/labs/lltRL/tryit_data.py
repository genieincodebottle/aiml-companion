"""Build tryit_data.json for the lltRL Try it widget from recorded runs only. No new measurement.

  python tryit_data.py
Reads runs/lltRL/lltRL-T2-grpo-main-s0.json (scored groups and per-step curves), the step 12 arms
(lltRL-T2-grpo-seqmean, -drgrpo, -G4, -G16, -beta0.04, -rloo) and lltRL-T2-ppo-kl0.05-s0 (clip fractions).
Every advantage and per-token weight is recomputed here from the recorded rewards and reply lengths with the
same formulas as page cells 7 and 8, so the widget can show them without running a model.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "runs" / "lltRL"
NL = chr(10)


def load(run_id):
    return json.load(open(RUNS / (run_id + ".json"), encoding="utf-8"))


def advantages(r, kind):
    G = len(r)
    m = sum(r) / G
    if kind == "rloo":
        return [x - (sum(r) - x) / (G - 1) for x in r]
    a = [x - m for x in r]
    if kind == "grpo":
        sd = (sum((x - m) ** 2 for x in r) / (G - 1)) ** 0.5
        a = [v / (sd + 1e-4) for v in a]
    return a


def cut(text):
    ends = [text.find(s) for s in (NL + "Question", NL + "[Question]") if s in text]
    return (text[:min(ends)] if ends else text).replace("<|endoftext|>", "")


main = load("lltRL-T2-grpo-main-s0")
groups = []
for g in main["evals"]["groups"]:
    G = len(g["rewards"]) // 2
    for k in range(2):
        r = g["rewards"][k * G:(k + 1) * G]
        toks = g["tokens"][k * G:(k + 1) * G]
        replies = g["replies"][k * G:(k + 1) * G]
        entry = {"step": g["step"], "question": g["questions"][k], "gold": g["gold"][k], "replies": []}
        adv = {kind: advantages(r, kind) for kind in ("grpo", "drgrpo", "rloo")}
        total_tokens = sum(toks)
        for i in range(G):
            a = adv["grpo"][i]
            entry["replies"].append({
                "text": cut(replies[i])[:600], "reward": r[i], "tokens": toks[i], "truncated": toks[i] >= 256,
                "advantage": {kind: round(adv[kind][i], 4) for kind in adv},
                # weight of one token of this reply in the loss, for the grpo-scaled advantage, per normalization
                "tokenWeight": {"seqmean": round(a / toks[i] / G, 6), "dapo": round(a / total_tokens, 6), "drgrpo": round(a / (G * 256), 6)},
            })
        groups.append(entry)

curves = {}
for name, run_id in [("default", "lltRL-T2-grpo-main-s0"), ("seed 1", "lltRL-T2-grpo-main-s1"), ("seqmean", "lltRL-T2-grpo-seqmean"),
                     ("drgrpo", "lltRL-T2-grpo-drgrpo"), ("G4", "lltRL-T2-grpo-G4"), ("G16", "lltRL-T2-grpo-G16"),
                     ("beta0.04", "lltRL-T2-grpo-beta0.04"), ("rloo", "lltRL-T2-grpo-rloo")]:
    try:
        h = load(run_id)["metrics"]
    except FileNotFoundError:
        continue
    curves[name] = {"runId": run_id, "step": [x["step"] for x in h], "reward": [x["reward"] for x in h], "meanLen": [x["meanLen"] for x in h],
                    "truncated": [x["truncated"] for x in h], "zeroStd": [x["zeroStd"] for x in h], "entropy": [x["entropy"] for x in h]}

ppo = load("lltRL-T2-ppo-kl0.05-s0")
out = {"source": "runs/lltRL (recorded on a Google Cloud A100 with the T4 recipe); advantages and token weights recomputed from recorded rewards and lengths",
       "clip": {"epsilon": 0.2, "ppoClipFracByIter": [[x["iter"], x["clipfrac"]] for x in ppo["metrics"]][::5], "runId": ppo["runId"]},
       "groups": groups, "curves": curves, "maxNew": 256}
json.dump(out, open(HERE / "tryit_data.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("groups", len(groups), "| curves", list(curves), "| bytes", (HERE / "tryit_data.json").stat().st_size)
