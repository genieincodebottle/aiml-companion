"""A numpy stand-in for the exported TensorRT engine.

It exists so the detector, the NMS and the parity test can be exercised
without weights or a GPU. It returns the same (N, 6) raw prediction array
the real engine does, so nothing downstream knows the difference.
"""
from __future__ import annotations

import numpy as np

from src.config import settings


class SimEngine:
    def __init__(self, seed: int = 0, n: int = 24):
        self.seed = seed
        self.n = n

    def infer(self, img) -> np.ndarray:
        """Deterministic in the seed, so the parity test means something."""
        rng = np.random.default_rng(self.seed)
        size = settings.imgsz
        x1 = rng.uniform(0, size * 0.8, self.n)
        y1 = rng.uniform(0, size * 0.8, self.n)
        return np.column_stack([
            x1, y1,
            x1 + rng.uniform(20, size * 0.2, self.n),
            y1 + rng.uniform(40, size * 0.2, self.n),
            rng.uniform(0.05, 0.99, self.n),
            rng.integers(0, 5, self.n),
        ])
