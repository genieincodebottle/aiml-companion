"""Reviewer agreement puts a ceiling on the accuracy a model can be asked for."""
from collections import Counter


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Agreement corrected for chance.

    Raw agreement flatters you when one class dominates: two reviewers who each
    say 'pass' 90 percent of the time agree 82 percent by luck alone. Asking a
    model to beat the consistency of the people producing its labels is asking it
    to learn noise, and the honest fix is a clearer defect definition.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equal, non-empty label lists")

    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
