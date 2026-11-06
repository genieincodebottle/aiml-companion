from sentence_transformers import SentenceTransformer

def norm(s):
    return " ".join(s.lower().split())

# Exact duplicates: the same prompt after lower-casing and collapsing whitespace. Keep the first.
first_seen, exact_dup = {}, set()
for i, r in enumerate(pool):
    key = norm(r["prompt"])
    if key in first_seen:
        exact_dup.add(i)
    else:
        first_seen[key] = i

# Near duplicates: cosine similarity of prompt embeddings from a multilingual embedder (released 2026-04-29).
embedder = SentenceTransformer("ibm-granite/granite-embedding-311m-multilingual-r2", device=device,
                               model_kwargs={"dtype": torch.float32})
embedder.max_seq_length = 512                      # the first 512 tokens of each prompt
E = embedder.encode([r["prompt"] for r in pool], batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
keep_idx = [i for i in range(len(pool)) if i not in exact_dup]
Ek = E[keep_idx]
S = Ek @ Ek.T
np.fill_diagonal(S, -1.0)
nearest = S.max(axis=1)

def greedy_dedup(S, threshold):
    # Walk the rows in order; drop a row if it is too similar to a row already kept.
    kept, dropped = [], []
    for j in range(len(S)):
        if kept and S[j, kept].max() >= threshold:
            dropped.append(j)
        else:
            kept.append(j)
    return kept, dropped

NEAR_T = 0.95
print(len(pool), "rows |", len(exact_dup), "exact duplicates")
for t in [0.80, 0.85, 0.90, 0.95, 0.98]:
    kept, dropped = greedy_dedup(S, t)
    print("threshold", t, "| near duplicates dropped", len(dropped))
np.save("data/prompt_embeddings.npy", E)              # reused by the selection step
