def paired_bootstrap(a, b, n_boot=10000, seed=0):
    # Mean of b - a over the same items, with a 95% interval from resampling ITEMS (each item keeps both scores).
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def unpaired_bootstrap(a, b, n_boot=10000, seed=0):
    # The same difference, resampling each model's items independently, as if the items were different.
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boots = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1) - a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(b.mean() - a.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def verdict(lo, hi):
    return "better" if lo > 0 else "worse" if hi < 0 else "no clear change"

rows = [(task, HARNESS[BASE]["tasks"][task]["acc"], HARNESS[TUNED]["tasks"][task]["acc"]) for task in TASKS + ["mmlu"]]
rows += [("ARC-Easy, " + f, FORMATS[BASE][f], FORMATS[TUNED][f]) for f in FORMATS[BASE]]
PAIRED = {}
print("comparison".ljust(34), "gain".rjust(7), "paired 95%".rjust(18), "unpaired 95%".rjust(18), "both right or both wrong".rjust(25), "  verdict (paired)")
for name, a, b in rows:
    d, lo, hi = paired_bootstrap(a, b)
    _, ulo, uhi = unpaired_bootstrap(a, b)
    same = float(np.mean(np.asarray(a) == np.asarray(b)))
    PAIRED[name] = {"gain": d, "paired": [lo, hi], "unpaired": [ulo, uhi], "agree": same}
    fmt = lambda x, y: ("{:+.3f} to {:+.3f}".format(x, y)).rjust(18)
    print(name.ljust(34), "{:+.3f}".format(d).rjust(7), fmt(lo, hi), fmt(ulo, uhi), str(round(same, 3)).rjust(25), " ", verdict(lo, hi))
json.dump(PAIRED, open("results/paired.json", "w"))
