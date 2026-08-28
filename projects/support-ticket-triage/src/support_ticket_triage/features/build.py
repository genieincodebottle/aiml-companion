"""The feature matrix, and the split.

There is almost nothing here, which is deliberate. The bag of words IS the
feature matrix, so this project has no feature engineering to hide behind. Every
difference between models is a difference in the model.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from support_ticket_triage.config import CFG, Config
from support_ticket_triage.data.generate import VOCAB

#: Columns that must never reach the model. `text` is the tokens themselves, so
#: feeding it back would be a tautology; `category` is the label.
LEAKY = ("category", "text", "ticket_id")

TARGET = "category"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c in set(VOCAB)]


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Token presence flags, in a stable column order."""
    cols = feature_columns(df)
    if not cols:
        raise ValueError("no vocabulary columns found; is this the right frame?")
    return df[cols].astype(float)


def assert_no_leakage(X: pd.DataFrame) -> None:
    """Fail the run rather than train on the answer."""
    bad = [c for c in X.columns if c in LEAKY]
    if bad:
        raise ValueError(f"leaky columns reached the design matrix: {bad}")


def split(df: pd.DataFrame, cfg: Config = CFG):
    """Stratified split.

    Stratified, not random: at a 3% class an unstratified split can hand the
    test set a wildly different rare-class share, and every rare-class metric
    then measures the split rather than the model.
    """
    train, test = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.split_seed,
        stratify=df[TARGET])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = design_matrix(df)
    assert_no_leakage(X)
    return X, df[TARGET]
