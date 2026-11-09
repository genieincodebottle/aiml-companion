"""Offline analysis of the recorded lltDPO sessions (no GPU). Writes runs/lltDPO/lltDPO-T2-analysis.json.

  python analyze_runs.py <record.json> [<record.json> ...]      (llt_dpo_core.json, llt_dpo_exp-gcp.json, ...)
Each record is analysed on its own: every arm is paired against the SFT replies of ITS OWN session, never another
session's. Uses only the recorded replies, judge scores and per-pair log-prob changes. Bootstrap: 10,000 resamples over
prompts or pairs, numpy default_rng(0) per statistic. Per-token gaps use the held-out reply lengths, recomputed from the
page's selection (first 300 UltraFeedback test pairs that fit 512 tokens, Qwen3-0.6B-Base tokenizer, plain template).
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import write_run  # noqa: E402


def ci(x):
    b = np.random.default_rng(0).integers(0, len(x), (10000, len(x)))
    return [round(float(v), 3) for v in np.percentile(x[b].mean(1), [2.5, 97.5])]


def win(a, b):
    w = 0.5 + 0.5 * np.sign(a - b)
    return {"win": round(float(w.mean()), 3), "winCI": ci(w), "n": int(len(w))}


def held_lengths():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    T = "### Instruction:" + chr(10) + "{}" + chr(10) + chr(10) + "### Response:" + chr(10)
    nc, nr = [], []
    for ex in load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs"):
        c, r = ex["chosen"][-1]["content"].strip(), ex["rejected"][-1]["content"].strip()
        if c == r:
            continue
        p = len(tok(T.format(ex["prompt"])).input_ids)
        a, b = len(tok(T.format(ex["prompt"]) + c).input_ids) + 1, len(tok(T.format(ex["prompt"]) + r).input_ids) + 1
        if a <= 512 and b <= 512:
            nc.append(a - p)
            nr.append(b - p)
        if len(nc) == 300:
            return np.array(nc), np.array(nr)


def session(rec, nc, nr):
    out = {"length": {}, "armVsArm": {}, "heldOut": {}, "judge2": {}}
    gens = rec.get("gens", {})
    gold = {k: np.array(v) for k, v in rec.get("gold", {}).items()}
    j2 = {k: np.array(v) for k, v in rec.get("judge2", {}).items()}
    if "sft" in gold:
        sft, st = gold["sft"], np.array([g["tokens"] for g in gens["sft"]])
        se = np.array([g["ended"] for g in gens["sft"]])
        out["sftGoldTokenCorr"] = round(float(np.corrcoef(sft, st)[0, 1]), 3)
        out["sftGoldEnded"], out["sftGoldCapped"], out["sftCapped"] = round(float(sft[se].mean()), 3), round(float(sft[~se].mean()), 3), int((~se).sum())
        names = [k for k in gold if k != "sft" and len(gold[k]) == len(sft)]
        for k in names:
            t = np.array([g["tokens"] for g in gens[k]])
            notlonger = t <= st
            out["length"][k] = {"all": win(gold[k], sft), "notLonger": win(gold[k][notlonger], sft[notlonger]),
                                "goldTokenCorr": round(float(np.corrcoef(gold[k], t)[0, 1]), 3)}
            if k in j2 and "sft" in j2:
                out["judge2"][k] = {"all": win(j2[k], j2["sft"]), "notLonger": win(j2[k][notlonger], j2["sft"][notlonger])}
        out["armVsArm2"] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out["armVsArm"][a + " vs " + b] = win(gold[a], gold[b])
                if a in j2 and b in j2:
                    out["armVsArm2"][a + " vs " + b] = win(j2[a], j2[b])
        if j2:
            both = [k for k in names if k in j2] + ["sft"]
            out["judgeAgreement"] = {k: round(float(np.corrcoef(gold[k], j2[k])[0, 1]), 3) for k in both if k in j2}
    per_pair = {}
    if "finalHeld" in rec:
        per_pair["dpo-b0.1-s0"] = (np.array(rec["finalHeld"]["dc"]), np.array(rec["finalHeld"]["dr"]))
    for k, r in rec["runs"].items():
        if isinstance(r, dict) and "perPair" in r:
            per_pair.setdefault(k, (np.array(r["perPair"]["chosen"]), np.array(r["perPair"]["rejected"])))
    for k, (c, r) in per_pair.items():
        if len(c) != len(nc):
            continue
        gap = c / nc - r / nr
        out["heldOut"][k] = {"accuracy": round(float((c > r).mean()), 3), "accuracyCI": ci((c > r).astype(float)),
                             "chosenMean": round(float(c.mean()), 2), "chosenCI": ci(c), "rejectedMean": round(float(r.mean()), 2),
                             "chosenFell": round(float((c < 0).mean()), 3), "bothFell": round(float(((c < 0) & (r < 0)).mean()), 3),
                             "corr": round(float(np.corrcoef(c, r)[0, 1]), 3), "chosenMedian": round(float(np.median(c)), 2),
                             "gapPerTokenMean": round(float(gap.mean()), 4), "gapPerTokenMedian": round(float(np.median(gap)), 4),
                             "chosenPerToken": round(float((c / nc).mean()), 4), "rejectedPerToken": round(float((r / nr).mean()), 4)}
    return out


nc, nr = held_lengths()
out = {}
for path in sys.argv[1:]:
    rec = json.load(open(path, encoding="utf-8"))
    key = rec.get("host", "kaggle") + ":" + rec["kernel"]
    out[key] = session(rec, nc, nr)
write_run("lltDPO-T2-analysis", "T2", "analysis", {"inputs": [Path(p).name for p in sys.argv[1:]], "script": "labs/lltDPO/analyze_runs.py",
          "rule": "each session analysed on its own; arms paired with their own session's SFT replies"},
          0, [], evals=out, hardware={"gpu": None, "count": 0, "vramGB": None, "cpu": "offline analysis of recorded T4 runs"})
print(json.dumps({k: {kk: (vv if kk != "length" else list(vv)) for kk, vv in v.items() if kk in ("length", "heldOut")} for k, v in out.items()}, indent=1)[:3000])
