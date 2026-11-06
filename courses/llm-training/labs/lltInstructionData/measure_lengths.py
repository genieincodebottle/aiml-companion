"""Why curated lost: answer length, truncation at max_length 1024, and the reward model's length preference.

Tokenizes every training arm with the SmolLM2 chat template (the one train_arm uses) and counts examples longer
than 1,024 tokens, which SFTTrainer truncates, so their end-of-turn token is never trained. Also the Spearman
correlation between reward and answer length on 1,500 random pool rows.
Usage: python measure_lengths.py <notebook data dir> <kaggle out dir of llt-insdata-exp>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from runlog import write_run

D, EXP = Path(sys.argv[1]), Path(sys.argv[2])
pool = [json.loads(line) for line in open(D / "pool_scored.jsonl", encoding="utf-8")]
arms = json.loads((D / "arms.json").read_text(encoding="utf-8"))
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")


def lengths(i):
    r = pool[i]
    total = len(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}],
                                        tokenize=True)["input_ids"])
    return total, len(tok(r["response"])["input_ids"])


metrics = []
for k, rows in arms.items():
    ls = np.array([lengths(i) for i in rows])
    metrics.append({"arm": k, "rows": len(rows), "medianTotalTokens": int(np.median(ls[:, 0])), "medianAnswerTokens": int(np.median(ls[:, 1])),
                    "shareOver1024": round(float((ls[:, 0] > 1024).mean()), 3)})
    print(metrics[-1])
idx = np.random.default_rng(0).choice(len(pool), 1500, replace=False)
la = np.array([lengths(i)[1] for i in idx])
sc = np.array([pool[i]["reward"] for i in idx])
rho = round(float(spearmanr(sc, la).statistic), 3)
ended = {}
for f in sorted((EXP / "exp").glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    ended[f.stem] = round(r["ifevalStopped"], 3)
for name, val in __import__("re").findall(r"^(curated-1k|bulk-1k-s0) +\| IFEval .*answers that ended with the end token ([0-9.]+)",
                                          "".join(x["data"] for x in json.loads((EXP / "llt-insdata-exp.log").read_text(encoding="utf-8"))), flags=__import__("re").M):
    ended[name + ("-s0" if name == "curated-1k" else "")] = float(val)
print("spearman", rho, ended)
write_run("lltInstructionData-T1-length-check", "T1", "analysis:length",
          {"tokenizer": "HuggingFaceTB/SmolLM2-360M-Instruct chat template", "maxLength": 1024, "rewardSample": "1,500 pool rows, seed 0"},
          None, metrics, evals={"spearmanRewardVsAnswerTokens": rho, "ifevalAnswersEnded": ended})
