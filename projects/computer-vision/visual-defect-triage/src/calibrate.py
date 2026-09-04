"""Temperature scaling. One parameter, fitted on validation only.

Dividing every logit by the same positive number cannot reorder them, so the
predicted class never changes and accuracy is untouched. All that changes is how
sure the model claims to be, which is the number the gate reads.

Implemented in numpy so the whole calibration path runs without torch.
"""
import numpy as np


def _nll(logits: np.ndarray, labels: np.ndarray, t: float) -> float:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1))
    return float(-(z[np.arange(len(labels)), labels] - logsumexp).mean())


def fit_temperature(logits, labels, lo: float = 0.05, hi: float = 10.0,
                    iterations: int = 60) -> float:
    """Golden-section search on the negative log likelihood.

    Fit this on VALIDATION. Fitting on test means the number you quote was chosen
    using the data you quote it on, and every threshold from it is then a guess.
    """
    logits = np.asarray(logits, dtype="float64")
    labels = np.asarray(labels, dtype="int64")

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(iterations):
        if _nll(logits, labels, c) < _nll(logits, labels, d):
            b, d = d, c
            c = b - phi * (b - a)
        else:
            a, c = c, d
            d = a + phi * (b - a)
    return float((a + b) / 2.0)


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype="float64") / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
