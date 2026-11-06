import numpy as np

def greedy_dedup(E, threshold):
    # E: (n, d), each row a unit-length embedding. Walk the rows in order. Keep a row unless its
    # cosine similarity to a row you already kept is threshold or more. Return the kept indices, in order.
    kept = []
    for i in range(len(E)):
        if kept and float((E[kept] @ E[i]).max()) >= threshold:
            continue
        kept.append(i)
    return kept

def vendi(E):
    # Vendi score of unit-length rows: exp of the entropy of the eigenvalues of E E^T / n.
    lam = np.linalg.eigvalsh(E @ E.T / len(E))
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))

# Checks. Run the cell; it prints All checks passed when both functions are right.
rng = np.random.default_rng(0)
topics = rng.normal(size=(6, 32))
E = np.vstack([topics, topics[[1, 4]] + 0.1 * rng.normal(size=(2, 32)), topics[[2]] + 0.6 * rng.normal(size=(1, 32))])
E = E / np.linalg.norm(E, axis=1, keepdims=True)
sims = np.round(E[6:] @ E[[1, 4, 2]].T, 3).diagonal()
print("rows 6, 7, 8 against the rows they were made from:", sims.tolist())
kept = greedy_dedup(E, 0.95)
assert kept == [0, 1, 2, 3, 4, 5, 8], kept
assert greedy_dedup(E, 0.8) == [0, 1, 2, 3, 4, 5], greedy_dedup(E, 0.8)
assert greedy_dedup(E, 0.3) == [0, 1, 2, 4, 5], "at 0.3 even unrelated rows start to count as copies"
assert greedy_dedup(E, 1.01) == list(range(9)), "a threshold above 1 keeps every row"
assert abs(vendi(np.eye(5)) - 5.0) < 1e-9, "5 orthogonal rows count as 5 distinct items"
assert abs(vendi(np.tile(E[:1], (5, 1))) - 1.0) < 1e-6, "5 copies of one row count as 1"
v_all, v_kept = vendi(E), vendi(E[kept])
print("Vendi score, all 9 rows:", round(v_all, 3), "| after dedup, 7 rows:", round(v_kept, 3))
assert round(v_all, 3) == 5.488 and round(v_kept, 3) == 5.562, (round(v_all, 3), round(v_kept, 3))
print("All checks passed")
