"""pass@k with confidence intervals.

pass@k for one problem: the chance that at least one of k samples is correct. From n >= k samples with
c correct, the unbiased estimate is 1 - C(n - c, k) / C(n, k) (Chen et al. 2021). The lltRL lab's
pass_at_k computes exactly this; we use it through rl_core and test it in test_capstone.py.

Problems differ far more than samples do, so the interval comes from resampling PROBLEMS (bootstrap),
and a comparison between two models on the same problems uses a PAIRED bootstrap.
"""
import math
import random


def pass_at_k(n, c, k):
    """Same formula as the lltRL lab's pass_at_k (kept here so this file runs without torch)."""
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curve(counts, n, ks):
    """counts: correct samples per problem (each out of n). Returns {k: mean pass@k over problems}."""
    return {k: sum(pass_at_k(n, c, k) for c in counts) / len(counts) for k in ks if k <= n}


def bootstrap_ci(values, reps=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean of per-problem values."""
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(reps))
    return means[int(reps * alpha / 2)], means[int(reps * (1 - alpha / 2)) - 1]


def paired_diff_ci(a, b, reps=2000, seed=0, alpha=0.05):
    """Mean of a - b over the same problems, with a paired bootstrap CI."""
    d = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(d, reps, seed, alpha)
    return sum(d) / len(d), lo, hi


def curve_with_ci(counts, n, ks, reps=2000, seed=0):
    out = {}
    for k in ks:
        if k <= n:
            vals = [pass_at_k(n, c, k) for c in counts]
            lo, hi = bootstrap_ci(vals, reps, seed)
            out[k] = dict(mean=round(sum(vals) / len(vals), 4), lo=round(lo, 4), hi=round(hi, 4))
    return out
