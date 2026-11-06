"""Run Library files for steps 1 to 7, from the first notebook run on Kaggle (kernel llt-insdata-notebook v1).

That run executed the page's cells 1 to 7 to completion on one T4 and saved data/ (pool, scores, flags, arms,
prompt embeddings); its log holds every printed line and each cell's seconds. This script recomputes the counts
from data/ and checks each one against the printed log, then writes one run file per step.
Usage: python record_data.py <kaggle out dir of llt-insdata-notebook>
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

from runlog import write_run

OUT = Path(sys.argv[1])
D = OUT / "data"
T4 = {"gpu": "Tesla T4", "count": 1, "vramGB": 15.6}
ENV = {"python": "3.13.16", "image": "Kaggle GPU image 2026-11-06, torch from the image (2.11.0+cu130 in the same-day kernel llt-insdata-teacher)",
       "pinned": "transformers 5.22.1, tokenizers 0.23.3, huggingface_hub 2.4.0, datasets 5.1.0, trl 1.17.1, accelerate 1.15.0, sentence-transformers 6.2.0, lm-eval 0.4.13, langdetect 1.0.9"}
KERNEL = "llt-insdata-notebook v1 (cells 1 to 7 completed; cell 10 stopped on CUDA out of memory, fixed and re-run in llt-insdata-exp)"

log = "".join(x["data"] for x in json.loads((OUT / "llt-insdata-notebook.log").read_text(encoding="utf-8"))
              if x.get("stream_name") == "stdout")
seconds = [float(s) for s in re.findall(r"cell seconds ([0-9.]+)", log)]
names = ["c01_pool", "c02_teacher", "c03_dedup", "c04_decontam", "c05_pii", "c06_reward", "c07_select", "c08_train", "c09_eval"]
cell_sec = dict(zip(names, seconds))
print("cell seconds", cell_sec)

pool = [json.loads(line) for line in open(D / "pool_scored.jsonl", encoding="utf-8")]
F = json.loads((D / "flags.json").read_text(encoding="utf-8"))
arms = json.loads((D / "arms.json").read_text(encoding="utf-8"))
E = np.load(D / "prompt_embeddings.npy")
src = collections.Counter(r["source"] for r in pool)
assert src == {"dolci": 3998, "wildchat": 3992, "synthetic": 299}, src
assert "7990 pool rows | skipped, no user and assistant pair {'dolci': 2, 'wildchat': 8}" in log


def write(step, label, config, metrics, samples=None, sec=None):
    rid = "lltInstructionData-T2-" + step
    config = dict(config, kernel=KERNEL)
    write_run(rid, "T2", label, config, 0, metrics, samples=samples, hardware=T4, env=ENV, wall_clock_sec=sec)


# Step 1, the pool.
dolci_src = collections.Counter(r["sub"] for r in pool if r["source"] == "dolci")
wild_lang = collections.Counter(r["lang"] for r in pool if r["source"] == "wildchat")
wild_model = collections.Counter(r["sub"] for r in pool if r["source"] == "wildchat")
write("pool", "baseline", {"dolci": "allenai/Dolci-Instruct-SFT bd3c8f3a shard 0 of 15 (143,475 rows), 4,000 rows sampled with seed 0",
                           "wildchat": "allenai/WildChat-1M 7d6490e4 shard 0 of 14 (59,857 conversations), 4,000 sampled with seed 1",
                           "rule": "first user message and the assistant reply after it"},
      [{"dolciRowsInShard": 143475, "wildchatRowsInShard": 59857, "poolRows": 7990, "skippedDolci": 2, "skippedWildchat": 8,
        "dolciSources": dict(dolci_src.most_common()), "wildchatLanguages": dict(wild_lang.most_common()),
        "wildchatModels": dict(wild_model.most_common()),
        "englishShareWildchat": round(wild_lang["English"] / sum(wild_lang.values()), 3)}], sec=cell_sec["c01_pool"])

# Step 2, the teacher.
synth = [r for r in pool if r["source"] == "synthetic"]
assert "300 requests sampled | 300 never closed the user turn | 299 pairs kept | pool now 8289" in log
write("synth", "baseline", {"teacher": "Qwen/Qwen3-4B-Instruct-2507", "requests": 300, "requestMaxNew": 96, "answerMaxNew": 512,
                            "rule": "cut each request at its first blank line; keep it if 15 characters or more"},
      [{"requests": 300, "neverClosedUserTurn": 300, "kept": len(synth), "answersEnded": sum(r["ended"] for r in synth),
        "medianRequestChars": int(np.median([len(r["prompt"]) for r in synth])),
        "medianReward": round(float(np.median([r["reward"] for r in synth])), 3)}],
      samples=[{"prompt": r["prompt"][:300], "ended": r["ended"]} for r in synth[:12]], sec=cell_sec["c02_teacher"])

# Step 3, dedup.
exact = set(F["exact_dup"])
keep = [i for i in range(len(pool)) if i not in exact]
S = E[keep] @ E[keep].T
np.fill_diagonal(S, -1.0)
near = S.max(axis=1)
dropped_by_t = {float(t): int(n) for t, n in re.findall(r"threshold ([0-9.]+) \| near duplicates dropped ([0-9]+)", log)}
bands = []
for lo, hi in [(0.9, 0.93), (0.93, 0.95), (0.95, 0.97), (0.97, 0.99), (0.99, 1.01)]:
    m = (near >= lo) & (near < hi)
    bands.append({"band": [lo, hi], "rows": int(m.sum())})
kept_rows = set(F["near_kept_rows"])
assert len(kept_rows) == len(keep) - dropped_by_t[0.95], (len(kept_rows), len(keep), dropped_by_t)
write("dedup", "baseline", {"embedder": "ibm-granite/granite-embedding-311m-multilingual-r2", "maxSeqLength": 512,
                            "rule": "exact match after lower-casing and collapsing whitespace, then greedy in pool order at cosine 0.95"},
      [{"rows": len(pool), "exactDuplicates": len(exact), "nearDroppedByThreshold": dropped_by_t,
        "rowsAfterDedup": len(kept_rows), "nearestNeighbourBands": bands,
        "nearestNeighbourQuantiles": {str(q): round(float(np.quantile(near, q)), 4) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}}],
      sec=cell_sec["c03_dedup"])

# Step 4, decontamination.
cont = {k: len(v) for k, v in F["contaminated"].items()}
emb = {}
for name, t, n, nc in re.findall(r"(IFEval|GSM8K test|HumanEval|MMLU test) +cosine ([0-9.]+) or more to an eval prompt: ([0-9]+) \| not caught by 8-grams: ([0-9]+)", log):
    emb.setdefault(name, {})[t] = {"rows": int(n), "notCaughtBy8grams": int(nc)}
hit_src = {k: dict(collections.Counter(pool[int(i)]["sub"] if pool[int(i)]["source"] == "dolci" else pool[int(i)]["source"] for i in v))
           for k, v in F["contaminated"].items()}
write("decontam", "baseline", {"evals": {"IFEval": 541, "GSM8K test": 1319, "HumanEval": 164, "MMLU test": 14042},
                               "rule8gram": "a pool prompt is flagged when it holds half or more of one eval item's word 8-grams",
                               "ruleEmbedding": "cosine of prompt embeddings; only IFEval matches at 0.95 or more are removed"},
      [{"eightGramHits": cont, "eightGramHitSources": hit_src, "embedding": emb, "ifevalParaphrasesRemoved": len(F["ifeval_near"])}],
      sec=cell_sec["c04_decontam"])

# Step 5, personal data. Hand labels: each of the 21 regex hits read by the page author on 2026-11-06 (labels only, no text).
HAND = {"personal data about a person (an app data export with an email, a public IP and phone-like ids)": ["4679"],
        "public business or support contact": ["4036", "7531", "7515"], "placeholder address at example.com": ["4300"],
        "private or loopback network address": ["5029", "7118", "7931"],
        "not personal data (timestamps, ids, an ISBN, digit runs, map coordinates)": ["4085", "4824", "4866", "5157", "5295", "5518", "5699", "5772", "5800", "5964", "6515", "6743", "6892"]}
assert sorted(sum(HAND.values(), [])) == sorted(F["pii"]), "hand labels must cover every hit"
kinds = collections.Counter(k for h in F["pii"].values() for k in h)
wild = [i for i, r in enumerate(pool) if r["source"] == "wildchat"]
write("pii", "baseline", {"regex": "email, phone, IPv4, card numbers that pass the Luhn check, written without backslashes",
                          "ner": "dslim/bert-base-NER, PER entities with score 0.9 or more, English WildChat prompts, first 1,500 characters",
                          "handling": "drop every row with a regex hit from the curated set; report NER hits, do not drop on a name alone"},
      [{"userPrompts": len(wild), "rowsWithRegexHit": len(F["pii"]), "hitsByKind": dict(kinds),
        "wildchatRedactedFlag": sum(bool(pool[i]["redacted"]) for i in wild),
        "handLabels": {k: len(v) for k, v in HAND.items()},
        "englishPrompts": sum(pool[i]["lang"] == "English" for i in wild), "rowsNamingAPerson": len(F["names"]),
        "nerSampleOf25": {"public figures, authors or fictional characters": 24, "a private person (a student named by first name with questionnaire answers)": 1}}],
      sec=cell_sec["c05_pii"])

# Step 6, reward model.
cal = re.search(r"([0-9]+) single-turn HelpSteer2 responses \| ([0-9]+) pairs with different human helpfulness", log)
agree = float(re.search(r"agrees with the human order on ([0-9.]+) of pairs", log).group(1))
th = [{"keepAtLeast": float(a), "kept": int(b), "keptShareRated3or4": float(c), "droppedShareRated3or4": float(d)}
      for a, b, c, d in re.findall(r"keep score >= ([-0-9.]+) \| kept ([0-9]+) \| share rated 3 or 4: ([0-9.]+) \| dropped share rated 3 or 4: ([0-9.]+)", log)]
med = {s: round(float(np.median([r["reward"] for r in pool if r["source"] == s])), 3) for s in src}
write("reward", "baseline", {"rewardModel": "Skywork/Skywork-Reward-V2-Qwen3-0.6B", "dtype": "float16", "maxLen": 2048,
                             "labels": "nvidia/HelpSteer2 validation, single-turn rows, human helpfulness 0 to 4",
                             "floor": "the median reward of those HelpSteer2 responses"},
      [{"helpsteerResponses": int(cal.group(1)), "pairsWithDifferentHelpfulness": int(cal.group(2)), "pairAgreement": agree,
        "thresholds": th, "floor": F["rm_floor"], "poolMedianReward": med,
        "poolQuantiles": {str(q): round(float(np.quantile([r["reward"] for r in pool], q)), 3) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}}],
      sec=cell_sec["c06_reward"])

# Step 7, selection and diversity.
stages = re.findall(r"^(pool|no exact or near duplicate|not contaminated|no personal data hit|English|answer not cut off|reward score [0-9.]+ or more) +([0-9]+) (\{.*\})$", log, flags=re.M)
div = re.findall(r"^(curated-1k|bulk-1k-s0|bulk-4k-s0|bulk-1k-s1|bulk-4k-s1) +\| mean cosine ([0-9.]+) \| Vendi score ([0-9.]+) \| median reward ([-0-9.]+)", log, flags=re.M)


def vendi(X):
    lam = np.linalg.eigvalsh(X @ X.T / len(X))
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


arm_stats = {}
for k, rows in arms.items():
    X = E[rows[:1000]]
    arm_stats[k] = {"rows": len(rows), "sources": dict(collections.Counter(pool[i]["source"] for i in rows)),
                    "vendi1k": round(vendi(X), 3), "medianReward": round(float(np.median([pool[i]["reward"] for i in rows])), 3),
                    "english": round(float(np.mean([i in set(F["english"]) for i in rows])), 3)}
by_source = {}
for s in src:
    idx = [i for i, r in enumerate(pool) if r["source"] == s][:299]
    by_source[s + "-first299"] = round(vendi(E[idx]), 3)
write("select", "baseline", {"clusters": "k-means, 100 clusters, random_state 0, n_init 3, on the candidates' prompt embeddings",
                             "rule": "take the best-scored row from each cluster in turn until 1,000",
                             "bulk": "random rows from the pool minus decontamination hits, seed 0 and 1"},
      [{"stages": [{"stage": a, "rows": int(b), "sources": c} for a, b, c in stages],
        "printedDiversity": [{"arm": a, "meanCosine": float(b), "vendi": float(c), "medianReward": float(d)} for a, b, c, d in div],
        "arms": arm_stats, "vendiFirst299BySource": by_source}], sec=cell_sec["c07_select"])
print(json.dumps(arm_stats, indent=1)[:1500], by_source)
