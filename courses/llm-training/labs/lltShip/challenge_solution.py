import numpy as np

def trim(d, density):
    # Keep the largest-magnitude share density of the entries of d (at least one entry), set the rest to 0.
    d = np.asarray(d, dtype=float)
    k = max(1, int(density * d.size))
    keep = np.argsort(-np.abs(d).ravel(), kind="stable")[:k]
    out = np.zeros(d.size)
    out[keep] = d.ravel()[keep]
    return out.reshape(d.shape)

def elect(ts):
    # ts: trimmed task vectors stacked on axis 0. Return +1.0 or -1.0 per entry: the sign of their sum, +1.0 on a tie.
    return np.where(np.sum(ts, axis=0) >= 0, 1.0, -1.0)

def ties_merge(base, deltas, density, lam=1.0):
    # TIES: trim each delta, elect a sign per entry, average only the entries that agree with it, scale by lam, add to base.
    ts = np.stack([trim(d, density) for d in deltas])
    agree = np.sign(ts) == elect(ts)
    count = agree.sum(axis=0)
    merged = (ts * agree).sum(axis=0) / np.where(count == 0, 1, count)
    return np.asarray(base, dtype=float) + lam * merged

# Checks. Run the cell; it prints All checks passed when all three functions are right.
b = np.zeros(4)
t1, t2 = np.array([0.5, -0.2, 0.1, 0.0]), np.array([0.3, 0.4, -0.6, 0.05])
assert np.allclose(trim(t1, 0.5), [0.5, -0.2, 0, 0]) and np.allclose(trim(t2, 0.5), [0, 0.4, -0.6, 0]), "keep the 2 largest magnitudes"
assert np.allclose(trim(np.array([1.0, -3.0, 2.0]), 0.1), [0, -3.0, 0]), "a small density still keeps one entry"
assert np.array_equal(elect(np.stack([trim(t1, 0.5), trim(t2, 0.5)])), [1, 1, -1, 1]), "sign of the sum, +1 on a tie"
assert np.allclose(ties_merge(b, [t1, t2], 0.5), [0.5, 0.4, -0.6, 0.0]), "divide by how many agree, not by how many tasks"
assert np.allclose(ties_merge(b, [t1, t2], 1.0), [0.4, 0.4, -0.6, 0.05]), "with no trimming, 0.1 loses the vote at entry 2"
assert np.allclose(ties_merge(b + 1, [t1, t2], 0.5, lam=2.0), [2.0, 1.8, -0.2, 1.0]), "scale the merged delta, not the base"
rng = np.random.default_rng(0)
base, d = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
assert np.allclose(ties_merge(base, [d, d], 1.0), base + d), "two identical fine-tunes merge to that fine-tune"
assert ties_merge(base, [d, -d], 0.5).shape == (3, 8), "keep the shape"
print("TIES, density 0.5:", np.round(ties_merge(b, [t1, t2], 0.5), 3), "| task arithmetic:", np.round(b + t1 + t2, 3))
print("All checks passed")
