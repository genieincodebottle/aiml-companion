from langdetect import DetectorFactory, detect
from sklearn.cluster import KMeans

DetectorFactory.seed = 0
def is_english(r):
    if r["source"] == "wildchat":
        return r["lang"] == "English"
    try:
        return detect(r["prompt"][:1000]) == "en"
    except Exception:
        return False

def vendi(X):
    # Vendi score: exp of the entropy of the eigenvalues of the similarity matrix divided by n.
    # It reads as the effective number of distinct items, from 1 (all the same) to n (all orthogonal).
    lam = np.linalg.eigvalsh(X @ X.T / len(X))
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))

def mean_cos(X):
    n = len(X)
    return float(((X @ X.T).sum() - n) / (n * (n - 1)))

kept_near, _ = greedy_dedup(S, NEAR_T)
unique_rows = {keep_idx[j] for j in kept_near}
contam = set().union(*[set(h) for h in contaminated.values()]) | set(ifeval_near)
english = {i for i, r in enumerate(pool) if is_english(r)}
floor = float(RM_FLOOR)
stages = [("pool", set(range(len(pool)))), ("no exact or near duplicate", unique_rows), ("not contaminated", unique_rows - contam),
          ("no personal data hit", unique_rows - contam - set(pii)), ("English", (unique_rows - contam - set(pii)) & english)]
stages.append(("answer not cut off", {i for i in stages[-1][1] if pool[i].get("ended", True)}))
stages.append(("reward score " + str(round(floor, 2)) + " or more", {i for i in stages[-1][1] if scores[i] >= floor}))
for label, rows in stages:
    print(label.ljust(28), len(rows), dict(collections.Counter(pool[i]["source"] for i in rows)))

# One cluster per topic region of prompt space; take the best-scored rows from each cluster in turn.
cand = sorted(stages[-1][1])
km = KMeans(n_clusters=min(100, len(cand)), random_state=0, n_init=3).fit(E[cand])
per = collections.defaultdict(list)
for i, c in zip(cand, km.labels_):
    per[c].append(i)
queues = [sorted(v, key=lambda i: -scores[i]) for v in per.values()]
curated, target = [], min(1000, len(cand))
while len(curated) < target:
    for q in queues:
        if q and len(curated) < target:
            curated.append(q.pop(0))

eligible = sorted(set(range(len(pool))) - contam)          # bulk arms skip only decontamination
arms = {"curated-1k": curated}
for seed in [0, 1]:
    pick = np.random.default_rng(seed).permutation(eligible)
    arms["bulk-1k-s" + str(seed)], arms["bulk-4k-s" + str(seed)] = list(pick[:1000]), list(pick[:4000])
for name, rows in arms.items():
    X = E[rows[:1000]]
    print(name.ljust(12), "| mean cosine", round(mean_cos(X), 3), "| Vendi score", round(vendi(X), 1),
          "| median reward", round(float(np.median(scores[rows])), 2), "|", dict(collections.Counter(pool[i]["source"] for i in rows)))

with open("data/arms.json", "w", encoding="utf-8") as f:
    json.dump({k: [int(i) for i in v] for k, v in arms.items()}, f)
with open("data/pool_scored.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(pool):
        f.write(json.dumps(dict(r, reward=float(scores[i])), ensure_ascii=False) + chr(10))
flags = {"exact_dup": sorted(exact_dup), "near_kept_rows": sorted(unique_rows), "contaminated": {k: sorted(v) for k, v in contaminated.items()},
         "ifeval_near": ifeval_near, "pii": {str(i): h for i, h in pii.items()}, "names": {str(i): v for i, v in names.items()},
         "english": sorted(english), "rm_floor": floor, "near_threshold": NEAR_T}
with open("data/flags.json", "w", encoding="utf-8") as f:
    json.dump(flags, f, ensure_ascii=False, default=int)
