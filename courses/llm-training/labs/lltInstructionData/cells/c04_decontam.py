import json as _json

EVALS = {  # name: (repo, revision, file, text column)
    "IFEval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "ifeval_input_data.jsonl", "prompt"),
    "GSM8K test": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet", "question"),
    "HumanEval": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544", "openai_humaneval/test-00000-of-00001.parquet", "prompt"),
    "MMLU test": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all/test-00000-of-00001.parquet", "question"),
}

def load_eval(repo, rev, fname, col):
    path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
    if fname.endswith(".jsonl"):
        return [_json.loads(line)[col] for line in open(path, encoding="utf-8")]
    return pq.read_table(path, columns=[col]).column(col).to_pylist()

def words(s):
    return "".join(ch if ch.isalnum() else " " for ch in s.lower()).split()

def ngrams(s, n=8):
    w = words(s)
    return {" ".join(w[k:k + n]) for k in range(len(w) - n + 1)}

eval_texts = {name: load_eval(*spec) for name, spec in EVALS.items()}
contaminated = {}
for name, texts in eval_texts.items():
    index = collections.defaultdict(set)          # 8-gram -> eval items that contain it
    sizes = []
    for k, t in enumerate(texts):
        g = ngrams(t)
        sizes.append(len(g))
        for x in g:
            index[x].add(k)
    hits = {}
    for i, r in enumerate(pool):
        counts = collections.Counter(k for x in ngrams(r["prompt"]) for k in index.get(x, ()))
        best = max((c / sizes[k] for k, c in counts.items()), default=0.0)
        if best >= 0.5:                            # at least half of one eval item's 8-grams
            hits[i] = round(best, 3)
    contaminated[name] = hits
    print(name.ljust(11), len(texts), "items |", len(hits), "pool prompts share half or more of one item's 8-grams")

# Paraphrases share few 8-grams, so also compare embeddings of pool prompts and eval prompts.
ifeval_near = []
for name, texts in eval_texts.items():
    Ev = embedder.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    sim = (E @ Ev.T).max(axis=1)
    if name == "IFEval":
        ifeval_near = np.flatnonzero(sim >= 0.95).tolist()     # IFEval is our eval, so drop close paraphrases too
    for t in [0.9, 0.95]:
        near = set(np.flatnonzero(sim >= t).tolist())
        print(name.ljust(11), "cosine", t, "or more to an eval prompt:", len(near), "| not caught by 8-grams:",
              len(near - set(contaminated[name])))
del embedder                                        # free the GPU for the next models
