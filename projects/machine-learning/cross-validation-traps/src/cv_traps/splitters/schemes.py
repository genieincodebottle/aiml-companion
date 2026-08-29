"""The splitters, and the one property that separates them.

Every scheme here answers the same question differently: which rows is the
model allowed to have seen before it is asked about this row? Getting that
wrong is not a modelling mistake, it is a bookkeeping mistake, and it is the
most expensive kind because the number it produces looks fine.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold)


def stratified(df: pd.DataFrame, y: np.ndarray, n_folds: int, seed: int
               ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The default almost everyone reaches for first."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from cv.split(df, y)


def plain(df: pd.DataFrame, y: np.ndarray, n_folds: int, seed: int
          ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from cv.split(df)


def grouped(df: pd.DataFrame, y: np.ndarray, n_folds: int, seed: int
            ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """No customer may appear on both sides of a fold.

    Not shuffled, because GroupKFold is deterministic: it packs groups into
    folds by size to keep them balanced. The seed is accepted and ignored so
    every scheme here has one interface.
    """
    cv = GroupKFold(n_splits=n_folds)
    yield from cv.split(df, y, groups=df["customer_id"].to_numpy())


def forward_chaining(df: pd.DataFrame, y: np.ndarray, n_folds: int, seed: int
                     ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Train on the past, validate on the next period. Never the reverse.

    Periods, not rows, are the unit. Slicing a panel by row index would put
    half of one period in train and half in validation, which leaks the very
    drift this scheme exists to respect.
    """
    periods = np.sort(df["period"].unique())
    if len(periods) <= n_folds:
        n_folds = max(1, len(periods) - 1)
    # The last n_folds periods each become a validation fold in turn.
    for p in periods[-n_folds:]:
        train = np.flatnonzero(df["period"].to_numpy() < p)
        test = np.flatnonzero(df["period"].to_numpy() == p)
        if len(train) == 0 or len(test) == 0:
            continue
        yield train, test


def grouped_forward_chaining(df: pd.DataFrame, y: np.ndarray, n_folds: int,
                             seed: int
                             ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Both constraints at once: past only, and unseen customers only.

    This is the scheme that matches how the truth holdout was built. It is the
    honest one, and it is also the one that throws away the most data, which is
    the trade the project is really about.
    """
    rng = np.random.default_rng(seed)
    periods = np.sort(df["period"].unique())
    if len(periods) <= n_folds:
        n_folds = max(1, len(periods) - 1)
    customers = np.sort(df["customer_id"].unique())

    for i, p in enumerate(periods[-n_folds:]):
        held = set(rng.choice(customers, size=max(1, len(customers) // 4),
                              replace=False).tolist())
        in_held = df["customer_id"].isin(held).to_numpy()
        before = df["period"].to_numpy() < p
        at = df["period"].to_numpy() == p
        train = np.flatnonzero(before & ~in_held)
        test = np.flatnonzero(at & in_held)
        if len(train) == 0 or len(test) == 0:
            continue
        yield train, test


def shuffled_grouped(df: pd.DataFrame, y: np.ndarray, n_folds: int, seed: int
                     ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Grouped folds that actually respond to the seed.

    GroupKFold is deterministic: it packs groups into folds by size and ignores
    any random state. That is fine when you want one honest split, and useless
    when the question is how much a CV number moves on reshuffling alone, which
    is what the variance experiment asks. An earlier draft used GroupKFold
    there and every one of 20 repeats returned a standard deviation of exactly
    0.0000, which is not a finding about stability, it is the same split
    twenty times.
    """
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from cv.split(df, y, groups=df["customer_id"].to_numpy())


REGISTRY = {
    "stratified_kfold": stratified,
    "kfold": plain,
    "group_kfold": grouped,
    "shuffled_group_kfold": shuffled_grouped,
    "forward_chaining": forward_chaining,
    "grouped_forward_chaining": grouped_forward_chaining,
}


def build(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown scheme {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]


def leakage_report(df: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]
                   ) -> pd.DataFrame:
    """How much each fold actually leaks, in the two ways that matter here.

    Reported rather than assumed, because a scheme's name is a claim and this
    is the check on it.
    """
    rows = []
    cust = df["customer_id"].to_numpy()
    per = df["period"].to_numpy()
    for i, (tr, te) in enumerate(splits):
        shared = np.intersect1d(cust[tr], cust[te])
        test_min = per[te].min() if len(te) else np.nan
        rows.append({
            "fold": i,
            "n_train": len(tr),
            "n_test": len(te),
            "customers_in_both": int(len(shared)),
            "pct_test_rows_seen_customer": round(
                float(np.isin(cust[te], cust[tr]).mean()), 4),
            "train_rows_from_the_future": int((per[tr] >= test_min).sum())
            if len(te) else 0,
        })
    return pd.DataFrame(rows)
