"""Confidence intervals for evaluation scores (lltEvaluation).

All functions take per-item 0/1 scores as NumPy arrays and return plain floats.
"""
import math
import numpy as np


def normal_ci(k, n, z=1.959964):
    """Normal-approximation (Wald) 95% CI for a proportion k/n."""
    p = k / n
    h = z * math.sqrt(p * (1 - p) / n)
    return p - h, p + h


def wilson(k, n, z=1.959964):
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bootstrap_ci(x, n_boot=10_000, seed=0):
    """Percentile bootstrap 95% CI for the mean of per-item scores x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """Mean difference b - a and its percentile bootstrap 95% CI.

    Resamples ITEMS, keeping each item's pair of scores together.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    diffs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def unpaired_bootstrap(a, b, n_boot=10_000, seed=0):
    """Same difference, but resampling the two models independently (ignores pairing)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    return float(b.mean() - a.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fmt(p, lo, hi, digits=1):
    """Percent with CI, e.g. '61.4 (57.1 to 65.6)'."""
    s = f"{{:.{digits}f}}"
    return f"{s.format(100 * p)} ({s.format(100 * lo)} to {s.format(100 * hi)})"
