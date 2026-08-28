"""Measure how false the independence assumption actually is.

Everyone repeats that Naive Bayes assumes conditional independence and that the
assumption is wrong. Almost nobody measures it. This module does, with one
number per token pair:

    lift(a, b | c)  =  P(a, b | c)  /  ( P(a | c) * P(b | c) )

Under the assumption that ratio is 1. Above 1 the pair co-occurs more than the
model believes possible, so the model counts the same evidence twice and its
posterior goes to an extreme it has not earned. Below 1 they exclude each other.

The point of the ratio rather than a p-value: at 9,000 tickets a chi-square test
rejects independence for pairs that are off by a trivial amount. The effect size
is what tells you whether the violation matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from support_ticket_triage.utils.logging import get_logger

log = get_logger(__name__)


def pair_lift(df: pd.DataFrame, cls: str, a: str, b: str,
              label_col: str = "category") -> dict:
    """Dependence lift for one token pair inside one class."""
    sub = df[df[label_col] == cls]
    n = len(sub)
    if n == 0:
        raise ValueError(f"no rows for class {cls!r}")

    pa = float(sub[a].mean())
    pb = float(sub[b].mean())
    pab = float(((sub[a] == 1) & (sub[b] == 1)).mean())
    expected = pa * pb

    # A chi-square alongside it, to show the p-value saying "dependent!" for
    # pairs whose lift is 1.02 and therefore harmless.
    table = pd.crosstab(sub[a], sub[b])
    if table.shape == (2, 2) and table.values.min() > 0:
        chi2, p_value = chi2_contingency(table.values)[:2]
    else:
        chi2, p_value = float("nan"), float("nan")

    return {
        "class": cls, "token_a": a, "token_b": b, "n": n,
        "p_a": round(pa, 4), "p_b": round(pb, 4),
        "p_ab_observed": round(pab, 4),
        "p_ab_if_independent": round(expected, 4),
        "lift": round(pab / expected, 3) if expected > 0 else float("nan"),
        "chi2": round(float(chi2), 1) if chi2 == chi2 else float("nan"),
        "p_value": float(f"{p_value:.3e}") if p_value == p_value else float("nan"),
    }


def planted_pair_report(df: pd.DataFrame, pairs, label_col: str = "category"
                        ) -> pd.DataFrame:
    """Lift for every pair the generator deliberately wired together."""
    rows = [pair_lift(df, cls, a, b, label_col) for cls, a, b in pairs]
    return (pd.DataFrame(rows)
            .sort_values("lift", ascending=False, ignore_index=True))


def survey_all_pairs(df: pd.DataFrame, tokens: list[str],
                     label_col: str = "category", top: int = 12) -> pd.DataFrame:
    """The worst offenders across every token pair, planted or not.

    Run this on real data, where nobody hands you the list of couplings. It is
    the same computation, and it finds the planted pairs without being told
    about them, which is what makes it trustworthy on data you did not build.
    """
    out = []
    for cls in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cls]
        X = sub[tokens].to_numpy(dtype=float)
        n = len(sub)
        if n < 30:
            log.warning("class %s has only %d rows, skipping the pair survey", cls, n)
            continue
        marg = X.mean(axis=0)
        joint = (X.T @ X) / n
        expected = np.outer(marg, marg)
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(expected > 0, joint / expected, np.nan)
        iu = np.triu_indices(len(tokens), k=1)
        for i, j, lv in zip(iu[0], iu[1], lift[iu]):
            # Ignore pairs too rare for the ratio to mean anything.
            if marg[i] < 0.05 or marg[j] < 0.05 or not np.isfinite(lv):
                continue
            out.append({"class": cls, "token_a": tokens[i], "token_b": tokens[j],
                        "p_a": round(float(marg[i]), 4),
                        "p_b": round(float(marg[j]), 4),
                        "lift": round(float(lv), 3)})
    frame = pd.DataFrame(out)
    if frame.empty:
        return frame
    frame["distance_from_1"] = (frame["lift"] - 1.0).abs()
    return (frame.sort_values("distance_from_1", ascending=False,
                              ignore_index=True)
            .head(top).drop(columns="distance_from_1"))


def recovered_planted_pairs(survey: pd.DataFrame, pairs) -> dict:
    """Did the blind survey rediscover the pairs we planted?

    This is the check that earns the right to trust the survey on real data.
    """
    if survey.empty:
        # Same keys as the populated branch. A caller that has to check which
        # shape it got is a caller that will eventually forget to.
        return {"planted": len(pairs), "surveyed": 0, "found": 0,
                "recall": 0.0, "precision": 0.0, "missed": list(pairs)}
    found = set()
    for _, row in survey.iterrows():
        key = (row["class"], *sorted((row["token_a"], row["token_b"])))
        found.add(key)
    planted = {(c, *sorted((a, b))) for c, a, b in pairs}
    hit = planted & found
    return {
        "planted": len(planted), "surveyed": len(found), "found": len(hit),
        # Recall: how much of the planted structure the sweep recovered.
        "recall": round(len(hit) / len(planted), 3),
        # Precision: how much of what it flagged was genuine. Both matter, and
        # they trade against each other as the cut-off widens: a longer list
        # finds more of the truth and more noise with it. On real data you only
        # ever see this column by chasing the hits down by hand.
        "precision": round(len(hit) / len(found), 3) if found else 0.0,
        "missed": sorted(planted - hit),
    }
