import numpy as np

def paired_bootstrap(a, b, n_boot=10000, seed=0):
    # a, b: 0/1 scores of two models on the SAME items, in the same order.
    # Return (mean of b - a, 2.5th percentile, 97.5th percentile) of the bootstrap means.
    # Draw all resamples at once: rng.integers(0, n, size=(n_boot, n)) with rng = np.random.default_rng(seed).
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def unpaired_bootstrap(a, b, n_boot=10000, seed=0):
    # The same difference, resampling each model's items on their own (this one is given).
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boots = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1) - a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(b.mean() - a.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

# Checks. Two models on 500 items: the second fixes 30 items the first got wrong and breaks 12 it got right.
rng = np.random.default_rng(7)
a = (rng.random(500) < 0.6).astype(int)
b = a.copy()
b[rng.choice(np.flatnonzero(a == 0), 30, replace=False)] = 1
b[rng.choice(np.flatnonzero(a == 1), 12, replace=False)] = 0
d, lo, hi = paired_bootstrap(a, b)
_, ulo, uhi = unpaired_bootstrap(a, b)
print("gain", round(d, 3), "| paired 95%", round(lo, 3), "to", round(hi, 3), "| unpaired 95%", round(ulo, 3), "to", round(uhi, 3))
assert abs(d - 0.036) < 1e-9, "the gain is (30 - 12) / 500"
assert lo > 0, "with pairing, the whole interval sits above zero"
assert ulo < 0 < uhi, "without pairing, the same gain looks like noise"
assert (hi - lo) < 0.5 * (uhi - ulo), "pairing removes the item-difficulty noise both models share"
assert paired_bootstrap(a, a) == (0.0, 0.0, 0.0), "a model compared with itself differs by exactly zero"
assert paired_bootstrap(a, b, seed=1) != paired_bootstrap(a, b, seed=0), "the seed must change the resamples"
print("All checks passed")
