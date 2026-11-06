import collections, json, os
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pinned revisions, so every run reads the same rows.
DOLCI = ("allenai/Dolci-Instruct-SFT", "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221", "data/train-00000-of-00015.parquet")
WILD = ("allenai/WildChat-1M", "7d6490e462285cf85d91eabea0f9a954fbddcd1f", "data/train-00000-of-00014.parquet")
N_EACH = 4000

def read_shard(repo, rev, fname, columns):
    path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
    return pq.read_table(path, columns=columns)

def first_turn(msgs):
    # The first user message and the assistant reply right after it.
    for a, b in zip(msgs, msgs[1:]):
        if a["role"] == "user" and b["role"] == "assistant" and a["content"] and b["content"]:
            return a["content"], b["content"]
    return None

pool, skipped = [], collections.Counter()
dolci = read_shard(*DOLCI, columns=["id", "messages", "source_dataset"])
for i in np.random.default_rng(0).choice(dolci.num_rows, size=N_EACH, replace=False):
    row = dolci.slice(int(i), 1).to_pylist()[0]
    turn = first_turn(row["messages"])
    if turn is None:
        skipped["dolci"] += 1
        continue
    pool.append({"id": "dolci-" + row["id"], "source": "dolci", "sub": row["source_dataset"],
                 "prompt": turn[0], "response": turn[1]})

wild = read_shard(*WILD, columns=["conversation_hash", "model", "conversation", "language", "redacted"])
for i in np.random.default_rng(1).choice(wild.num_rows, size=N_EACH, replace=False):
    row = wild.slice(int(i), 1).to_pylist()[0]
    turn = first_turn(row["conversation"])
    if turn is None:
        skipped["wildchat"] += 1
        continue
    pool.append({"id": "wild-" + str(i), "source": "wildchat", "sub": row["model"], "lang": row["language"],
                 "redacted": row["redacted"], "prompt": turn[0], "response": turn[1]})

print(dolci.num_rows, "Dolci rows in shard 0 |", wild.num_rows, "WildChat conversations in shard 0")
print(len(pool), "pool rows | skipped, no user and assistant pair", dict(skipped))
print(collections.Counter(r["sub"] for r in pool if r["source"] == "dolci").most_common(8))
print(collections.Counter(r["lang"] for r in pool if r["source"] == "wildchat").most_common(5))

os.makedirs("data", exist_ok=True)
with open("data/pool.jsonl", "w", encoding="utf-8") as f:
    for r in pool:
        f.write(json.dumps(r, ensure_ascii=False) + chr(10))
