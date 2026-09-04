"""Expected calibration error, the number that says whether the fit worked."""
import numpy as np


def expected_calibration_error(conf, correct, n_bins: int = 15) -> float:
    """Average gap between claimed confidence and observed accuracy.

    A model claiming 0.95 on a bucket that is 78 percent correct has a gap of
    0.17 in that bin. Weighting the gaps by bin population gives one number you
    can watch over time, and it moves when inputs shift while accuracy holds.
    """
    conf = np.asarray(conf, dtype="float64")
    correct = np.asarray(correct, dtype="float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_table(conf, correct, n_bins: int = 15) -> list[dict]:
    """Per-bin claimed confidence against observed accuracy. Feeds the report."""
    conf = np.asarray(conf, dtype="float64")
    correct = np.asarray(correct, dtype="float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        rows.append({
            "bin_lo": round(float(lo), 4),
            "bin_hi": round(float(hi), 4),
            "n": int(m.sum()),
            "claimed": round(float(conf[m].mean()), 4),
            "observed": round(float(correct[m].mean()), 4),
        })
    return rows
