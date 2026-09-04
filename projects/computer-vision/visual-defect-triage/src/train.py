"""Train the head on cached embeddings, weighting the rare defects.

Pure numpy multinomial logistic regression, so the whole training path runs
without torch. Class weighting is what keeps hairline_crack from being optimised
away: at 3 percent of the data the cheapest way to cut average loss is to almost
never predict it, and the model will take that deal.
"""
import numpy as np

from src.calibrate import softmax
from src.models.head import NumpyHead


def class_weights(labels, num_classes: int, power: float = 0.5) -> np.ndarray:
    """Inverse frequency raised to `power`, so a rare mistake costs more.

    power=1.0 is full inverse frequency and power=0.0 is unweighted. Square root
    is the usual middle, and it matters here: hairline_crack sits next to pass,
    so full weighting buys crack recall by pushing false cracks onto pass, which
    is 62 percent of traffic. That trade is worth measuring rather than assuming.
    """
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype("float64")
    counts[counts == 0] = 1.0
    return (counts.sum() / (num_classes * counts)) ** power


def train(embeddings, labels, num_classes: int, epochs: int = 1500,
          lr: float = 0.5, weight_decay: float = 1e-6, weight_power: float = 0.5,
          seed: int = 0) -> NumpyHead:
    x = np.asarray(embeddings, dtype="float64")
    y = np.asarray(labels, dtype="int64")
    n, d = x.shape

    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=(num_classes, d))
    b = np.zeros(num_classes)

    cw = class_weights(y, num_classes, weight_power)
    sample_w = cw[y]
    sample_w = sample_w / sample_w.sum()
    onehot = np.eye(num_classes)[y]

    for _ in range(epochs):
        probs = softmax(x @ w.T + b, 1.0)
        err = (probs - onehot) * sample_w[:, None]
        w -= lr * (err.T @ x + weight_decay * w)
        b -= lr * err.sum(axis=0)

    return NumpyHead(w, b)
