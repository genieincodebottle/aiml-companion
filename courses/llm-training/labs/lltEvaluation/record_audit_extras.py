"""Audit pass (2026-11-07): add fields the page quotes to existing Run Library files. Rerun-safe.

- T2-scorer-match, T2-generative, T2-judge-calibration: values printed only in the v2 notebook log (record_main.py parses them).
- T2-refusal-handcheck: counts of opener categories (no text), from the v2 behaviour output.
- T2-canary: which leaked item each of the 3 extra 8-gram flags matches (needs the ARC-Easy test split).
- T1-cpu-pair: items scored differently by our scorer and the harness on the CPU.
Usage: python record_audit_extras.py <v2 output dir> <T1 work dir>
"""
import json, os, sys

SRC, T1 = sys.argv[1], sys.argv[2]
RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "lltEvaluation"))
load = lambda rid: json.load(open(os.path.join(RUNS, rid + ".json"), encoding="utf-8"))
def save(d):
    json.dump(d, open(os.path.join(RUNS, d["runId"] + ".json"), "w", encoding="utf-8"), indent=1)
    print("updated", d["runId"])

F = json.load(open(os.path.join(SRC, "figures_summary.json"), encoding="utf-8"))
log = F["fromLog"]
d = load("lltEvaluation-T2-scorer-match"); d["evals"]["accVsAccNormPickDiffers"] = log["accVsAccNormPickDiffers"]; save(d)
d = load("lltEvaluation-T2-generative"); d["evals"]["ifevalSeconds"] = log["ifevalOriginalSeconds"]; save(d)
d = load("lltEvaluation-T2-judge-calibration"); d["evals"]["judgeSeconds"] = log["judgeSeconds"]; save(d)

# Opener categories of replies to unsafe prompts (counts only).
p = os.path.join(SRC, "results", "behaviour.json")
if os.path.exists(p):
    B = json.load(open(p, encoding="utf-8"))["behaviour"]
    d = load("lltEvaluation-T2-refusal-handcheck")
    for arm, b in B.items():
        ref = dict(zip(b["ids"], b["refusedStrmatch"]))
        ops = {int(k): v.lower() for k, v in b["unsafeOpenings"].items()}
        cats = {"apology": sum(1 for k, v in ops.items() if ref[k] and ("sorry" in v.split(" ")[0:3] or v.startswith(("sorry", "i'm sorry", "i am sorry")))),
                "statesItCannotHelp": sum(1 for k, v in ops.items() if ref[k] and v.startswith(("i cannot", "i can't"))),
                "answerLikeOpening": sum(1 for k, v in ops.items() if not ref[k] and v.startswith(("there are", "there is", "you can ", "the best way")))}
        d["evals"][arm]["openerCategories"] = cats
    save(d)

# The 3 extra 8-gram flags of the canary: which leaked item each matches.
from datasets import load_dataset
a = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", revision="210d026faf9955653af8916fad021475a3f00453")
def grams(s, n=8):
    w = "".join(ch if ch.isalnum() else " " for ch in s.lower()).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
gold = lambda doc: doc["choices"]["text"][doc["choices"]["label"].index(doc["answerKey"])]
item = lambda doc: "Question: " + doc["question"] + chr(10) + "Answer: " + gold(doc)
leaked = {j: grams(item(a[j])) for j in range(1000, 1100)}
d = load("lltEvaluation-T2-canary")
extra = sorted(set(d["evals"]["perItem"]["ngram"]["flagged"]) - set(range(1000, 1100)))
src = {}
for j in extra:
    g = grams(item(a[j]))
    best = max(leaked, key=lambda k: len(g & leaked[k]))
    src[str(j)] = {"matchesLeakedItem": best, "shareOf8grams": round(len(g & leaked[best]) / len(g), 3),
                   "sameQuestionText": a[j]["question"].strip() == a[best]["question"].strip()}
d["evals"]["summary"]["ngram"]["extraFlagSources"] = src
save(d)

# T1: items scored differently on the CPU.
S = json.load(open(os.path.join(T1, "results", "scorer.json"), encoding="utf-8"))
H = json.load(open(os.path.join(T1, "results", "harness.json"), encoding="utf-8"))
base = list(H)[0]
d = load("lltEvaluation-T1-cpu-pair")
d["evals"]["scorer"]["itemsDiffer"] = int(sum(x != y for x, y in zip(S["acc"], H[base]["tasks"]["arc_easy"]["acc"])))
d["evals"]["scorer"]["items"] = len(S["acc"])
save(d)
