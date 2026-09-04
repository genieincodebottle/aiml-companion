"""Sweep the two thresholds and read off what escapes."""
import numpy as np


def simulate(conf, correct, accept_above: float, reject_below: float) -> dict:
    """Answers the only question that matters, what escapes.

    A threshold is not a modelling decision. It is a trade between reviewer cost
    and escaped defects, so simulating it on held-out data turns an argument
    about risk appetite into a table someone can pick a row from.
    """
    conf = np.asarray(conf, dtype="float64")
    correct = np.asarray(correct, dtype=bool)

    accepted = conf >= accept_above
    rejected = conf <= reject_below
    reviewed = ~(accepted | rejected)

    escaped = int((accepted & ~correct).sum())
    return {
        "accept_above": accept_above,
        "reject_below": reject_below,
        "auto_accept_share": float(accepted.mean()),
        "auto_reject_share": float(rejected.mean()),
        "review_share": float(reviewed.mean()),
        "escaped_errors": escaped,
        "escape_rate": escaped / max(int(accepted.sum()), 1),
    }


def sweep(conf, correct, accepts=(0.90, 0.95, 0.98, 0.99), reject_below=0.02) -> list[dict]:
    return [simulate(conf, correct, a, reject_below) for a in accepts]
