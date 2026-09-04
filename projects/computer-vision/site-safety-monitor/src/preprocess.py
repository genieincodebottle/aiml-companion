"""Letterbox to a square, and the inverse that maps boxes back to original
pixels."""
from __future__ import annotations

import numpy as np

from src.config import settings

#: Must match the padding the model saw in training. A mismatch shifts
#: every prediction slightly and reads as a mediocre model.
PAD_VALUE = 114


def _resize(img: np.ndarray, nw: int, nh: int) -> np.ndarray:
    """OpenCV when it is installed, nearest-neighbour numpy when it is not.

    The fallback exists so the pipeline runs offline. It is not
    interpolation-identical to cv2, which is why the export parity test
    pins the letterbox geometry rather than the pixels.
    """
    try:
        import cv2

        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        h, w = img.shape[:2]
        ys = (np.arange(nh) * (h / nh)).astype(np.int64).clip(0, h - 1)
        xs = (np.arange(nw) * (w / nw)).astype(np.int64).clip(0, w - 1)
        return img[ys][:, xs]


def letterbox(img: np.ndarray, imgsz: int | None = None):
    """Resize preserving aspect ratio, pad to square. Returns the inverse too."""
    size = imgsz or settings.imgsz
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    resized = _resize(img, nw, nh)

    top = (size - nh) // 2
    left = (size - nw) // 2
    shape = (size, size, 3) if img.ndim == 3 else (size, size)
    out = np.full(shape, PAD_VALUE, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, (s, left, top)


def apply_letterbox(box, meta):
    """Forward transform, original pixels into padded space."""
    s, left, top = meta
    x1, y1, x2, y2 = box
    return (x1 * s + left, y1 * s + top, x2 * s + left, y2 * s + top)


def unletterbox(box, meta):
    """Undo it IMMEDIATELY after NMS, so nothing downstream sees padded space."""
    s, left, top = meta
    x1, y1, x2, y2 = box
    return ((x1 - left) / s, (y1 - top) / s, (x2 - left) / s, (y2 - top) / s)
