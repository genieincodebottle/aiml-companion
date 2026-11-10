"""Intervals and the length-controlled win rate. NumPy only.

Paired bootstrap: resample the prompts (with replacement) and recompute the statistic, 2,000 times; the 2.5th and
97.5th percentiles are the 95% interval. Pairing keeps each prompt's two models together, which is what makes a
small difference detectable.

Length-controlled win rate (the idea of AlpacaEval LC, Dubois et al. 2024, reduced to one length term):
fit  P(A wins) = sigmoid(b0 + b1 * tanh(d / s)),  d = len(A) - len(B) in tokens, s = std of d over the set,
then report sigmoid(b0): the predicted win rate when both replies have the same length. AlpacaEval's GLM also has a
per-instruction difficulty term; we leave it out, so this is a simplified version and the page says so.
"""
import math

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def boot(values, stat=np.mean, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bs = np.array([stat(v[i]) for i in idx])
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def paired_diff(a, b, n_boot=2000, seed=0):
    """Mean of a minus mean of b over the same items, with a paired bootstrap interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return {"diff": round(float(d.mean()), 4), "ci": boot(d, n_boot=n_boot, seed=seed)}


def fit_lc(y, d, iters=50):
    """Fractional logistic regression by Newton steps. y in [0, 1] (win = 1, tie = 0.5), d = length difference."""
    y = np.asarray(y, float)
    d = np.asarray(d, float)
    s = d.std() or 1.0
    X = np.stack([np.ones_like(d), np.tanh(d / s)], 1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w, s


def lc_winrate(y, d, n_boot=2000, seed=0):
    y = np.asarray(y, float)
    d = np.asarray(d, float)
    w, s = fit_lc(y, d)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        wb, _ = fit_lc(y[i], d[i])
        bs.append(1 / (1 + math.exp(-wb[0])))
    return {"lc": round(1 / (1 + math.exp(-w[0])), 4), "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "lengthCoef": round(float(w[1]), 4), "lengthScaleTokens": round(float(s), 1)}
