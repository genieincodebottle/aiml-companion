"""Non-maximum suppression, capped, with a note on why it is quadratic."""
from __future__ import annotations

import numpy as np


def _iou(box, others):
    if not len(others):
        return np.zeros(0)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def comparisons(b: int) -> int:
    """b(b-1)/2, the worst case count this function is measured by."""
    return b * (b - 1) // 2


def nms(pred, conf_th, iou_th, max_det):
    """Cost is roughly b(b-1)/2 IoU comparisons in surviving boxes.

    300 boxes is 44,850 comparisons. 1,000 boxes is 499,500, which is
    11.1 times the work for 3.3 times the boxes. The candidate cap is
    a latency control, not a quality one.
    """
    pred = np.asarray(pred, dtype=np.float64)
    if not len(pred):
        return np.zeros((0, 6))
    pred = pred[pred[:, 4] >= conf_th]
    pred = pred[np.argsort(-pred[:, 4])][:max_det]

    keep = []
    while len(pred):
        best, pred = pred[0], pred[1:]
        keep.append(best)
        if not len(pred):
            break
        # Suppress only within the same class, or a helmet box would
        # suppress the person box it sits on.
        same = pred[:, 5] == best[5]
        drop = same & (_iou(best[:4], pred[:, :4]) >= iou_th)
        pred = pred[~drop]
    return np.array(keep)
