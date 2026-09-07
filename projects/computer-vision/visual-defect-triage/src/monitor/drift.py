"""Watch the embedding distribution. No labels needed, and it moves first."""
import numpy as np


def population_shift(reference, window) -> float:
    """Cosine distance between the mean embedding of two periods.

    Labels arrive days late, so accuracy is a lagging signal. The embedding mean
    moves as soon as the input does, which on this line caught a changed lighting
    ballast about a shift and a half before the defect rate did.
    """
    a = np.asarray(reference, dtype="float64").mean(axis=0)
    b = np.asarray(window, dtype="float64").mean(axis=0)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(1.0 - (a @ b) / denom)


def alert(shift: float, threshold: float = 0.04) -> bool:
    return shift > threshold
