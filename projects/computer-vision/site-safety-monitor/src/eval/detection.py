"""Mean average precision by class and by object size, because small objects
fail first."""
from __future__ import annotations

import numpy as np

SMALL = 32 * 32
MEDIUM = 96 * 96


def size_bucket(area: float, small: float = SMALL, medium: float = MEDIUM) -> str:
    return "small" if area < small else "medium" if area < medium else "large"


def average_precision(scores, matched) -> float:
    """Area under the precision-recall curve, all-point interpolation."""
    if not len(scores):
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(matched, dtype=float)[order]
    n_gt = tp.sum()
    if n_gt == 0:
        return 0.0
    cum_tp = np.cumsum(tp)
    precision = cum_tp / np.arange(1, len(tp) + 1)
    recall = cum_tp / n_gt

    ap, prev_r = 0.0, 0.0
    for p, r in zip(np.maximum.accumulate(precision[::-1])[::-1], recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def map_by_size(preds, gts, small: float = SMALL, medium: float = MEDIUM):
    """Stratify by object area before reporting a single number.

    A helmet on a worker 30 metres from the camera is a small object.
    Aggregate mAP is dominated by near objects, so a model can lose
    badly on distance while the headline number barely moves.
    """
    buckets = {"small": [], "medium": [], "large": []}
    for p, g in zip(preds, gts):
        buckets[size_bucket(g)].append(p)

    out = {}
    for k, rows in buckets.items():
        if not rows:
            out[k] = None
            continue
        out[k] = average_precision([r[0] for r in rows], [r[1] for r in rows])
    return out
