"""The design matrix, and the models.

There is deliberately no feature engineering. Every number this project reports
is a difference between validation schemes, so anything clever here would be a
confounder rather than a contribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cv_traps.data.generate import (CUSTOMER_FEATURES, ID_COLUMNS,
                                    PERIOD_FEATURES, TARGET)

#: Columns that must never reach a model. `customer_id` is the group key, and
#: handing it to a learner is the group trap in its most literal form.
LEAKY = ID_COLUMNS + (TARGET,)


#: The columns that carry real signal. The noise columns exist for the
#: preprocessing and selection traps, where the point IS that noise gets
#: mistaken for signal. The group and time traps are about splitting, not
#: about feature choice, so they run on the core set: with 150 noise columns
#: in the matrix a boosted tree spends its capacity sifting noise and the
#: group effect it is supposed to expose gets diluted below the fold-to-fold
#: spread. Measured, not assumed: on the full matrix the KFold-to-GroupKFold
#: gap was +0.015 with a fold std of the same size, and on the core set it is
#: several times that.
CORE_FEATURES = tuple(CUSTOMER_FEATURES) + tuple(PERIOD_FEATURES)


def feature_columns(df: pd.DataFrame, core_only: bool = False) -> list[str]:
    if core_only:
        return [c for c in CORE_FEATURES if c in df.columns]
    return [c for c in df.columns if c not in set(LEAKY)]


def design_matrix(df: pd.DataFrame, core_only: bool = False) -> pd.DataFrame:
    cols = feature_columns(df, core_only=core_only)
    if not cols:
        raise ValueError("no feature columns found; is this the right frame?")
    return df[cols].astype(float)


def assert_no_leakage(X: pd.DataFrame) -> None:
    bad = [c for c in X.columns if c in set(LEAKY)]
    if bad:
        raise ValueError(f"leaky columns reached the design matrix: {bad}")


def xy(df: pd.DataFrame, core_only: bool = False
       ) -> tuple[pd.DataFrame, np.ndarray]:
    X = design_matrix(df, core_only=core_only)
    assert_no_leakage(X)
    return X, df[TARGET].to_numpy()


def linear_pipeline(select_k: int | None = None, C: float = 1.0,
                    seed: int = 42) -> Pipeline:
    """Impute, scale, optionally select, then fit.

    Every step before the model is a step that learns from data, which is the
    whole point of the preprocessing trap: `fit` on the wrong rows and the
    score moves even though the model did not change.
    """
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if select_k:
        steps.append(("select", SelectKBest(f_classif, k=select_k)))
    steps.append(("model", LogisticRegression(
        C=C, max_iter=2000, random_state=seed)))
    return Pipeline(steps)


def booster(seed: int = 42, max_leaf_nodes: int = 31,
            learning_rate: float = 0.08) -> HistGradientBoostingClassifier:
    """Flexible enough to memorise a customer from its fingerprint columns.

    That memorisation is not a flaw to be tuned away, it is the mechanism the
    group trap depends on, and a model too rigid to do it would hide the
    effect rather than avoid it. Handles NaN natively, so the group experiment
    is not entangled with imputation choices.
    """
    return HistGradientBoostingClassifier(
        max_leaf_nodes=max_leaf_nodes, learning_rate=learning_rate,
        max_iter=180, early_stopping=False, random_state=seed)
