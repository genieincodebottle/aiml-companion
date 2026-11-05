import numpy as np
import ptd

texts = [t for _, t in kept]
seen, unique = set(), []
for i, t in enumerate(texts):
    key = ptd.exact_key(t)
    if key not in seen:
        seen.add(key)
        unique.append(i)
print("exact duplicates removed", len(texts) - len(unique))

a, b = ptd.make_hashes(112, seed=1)
sigs = np.stack([ptd.minhash(ptd.shingles(texts[i]), a, b) for i in unique])
pairs = ptd.lsh_candidates(sigs, bands=14, rows=8)
roots = ptd.clusters(len(unique), pairs)
keep = [unique[k] for k in range(len(unique)) if roots[k] == k]
deduped = [kept[i] for i in keep]
print("candidate pairs", len(pairs), "| near duplicates removed", len(unique) - len(keep), "| pages left", len(deduped))
for x, y in sorted(pairs)[:3]:
    j = ptd.jaccard(ptd.shingles(texts[unique[x]]), ptd.shingles(texts[unique[y]]))
    print(f"pair {x}-{y}  exact Jaccard {j:.3f}  signature agreement {float((sigs[x] == sigs[y]).mean()):.3f}")
