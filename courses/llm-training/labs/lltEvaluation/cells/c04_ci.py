import math

def wilson(k, n, z=1.959964):
    # Wilson score 95% interval for a proportion k/n; stays inside 0 to 1 even near 0 or 1.
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def bootstrap_ci(x, n_boot=10000, seed=0):
    # Resample the items with replacement, recompute the mean, read the 2.5th and 97.5th percentiles.
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

print("task".ljust(14), "n".rjust(5), "acc".rjust(6), "harness stderr".rjust(15), "normal 95%".rjust(16), "Wilson 95%".rjust(16), "bootstrap 95%".rjust(16))
for task in TASKS + ["mmlu"]:
    x = np.array(HARNESS[TUNED]["tasks"][task]["acc"])
    n, p = len(x), x.mean()
    se = x.std(ddof=1) / math.sqrt(n)                  # what lm-eval prints as acc_stderr
    lo_w, hi_w = wilson(x.sum(), n)
    lo_b, hi_b = bootstrap_ci(x)
    fmt = lambda a, b: (str(round(a, 3)) + " to " + str(round(b, 3))).rjust(16)
    print(task.ljust(14), str(n).rjust(5), str(round(p, 3)).rjust(6), str(round(se, 4)).rjust(15),
          fmt(p - 1.96 * se, p + 1.96 * se), fmt(lo_w, hi_w), fmt(lo_b, hi_b))
print("harness stderr reported for arc_easy:", round(HARNESS[TUNED]["tasks"]["arc_easy"]["stderr"], 4))

# How wide is the interval at other benchmark sizes, for a score near 50%?
for n in [100, 250, 500, 1000, 2000, 10000]:
    print("n =", str(n).rjust(5), "| half-width of the 95% interval at 50%:", round(100 * 1.96 * math.sqrt(0.25 / n), 1), "points")
