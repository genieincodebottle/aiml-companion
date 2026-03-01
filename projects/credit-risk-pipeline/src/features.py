"""
Feature Engineering
===================

Engineers domain-specific features for credit risk modeling:
DTI ratio, credit utilization, loan burden, age buckets, and
log-transformed skewed features.

All transformations use ``df.copy()`` to avoid side effects.

Functions
---------
engineer_features   Apply all feature engineering steps.
add_dti_ratio       Add debt-to-income ratio.
add_utilization     Add credit utilization ratio.
add_loan_burden     Add loan-to-income burden score.
add_age_buckets     Add age group buckets.
add_log_transforms  Apply log1p to skewed numeric features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply all feature engineering steps.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data with original features.
    cfg : dict
        Pipeline configuration with feature engineering settings.

    Returns
    -------
    pd.DataFrame
        Data with engineered features appended.
    """
    df = df.copy()
    n_cols_before = df.shape[1]

    df = add_dti_ratio(df, cfg)
    df = add_utilization(df, cfg)
    df = add_loan_burden(df, cfg)
    df = add_age_buckets(df, cfg)
    df = add_log_transforms(df, cfg)

    n_new = df.shape[1] - n_cols_before
    logger.info(f"Engineered {n_new} new features (total: {df.shape[1]})")
    return df


# ---------------------------------------------------------------------------
# Individual feature transformations
# ---------------------------------------------------------------------------

def add_dti_ratio(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add debt-to-income ratio.

    DTI = (existing_debt + loan_amount) / income.
    Capped at 1.0 to handle edge cases.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cfg : dict
        Config with ``dti_columns`` mapping.

    Returns
    -------
    pd.DataFrame
        Data with ``dti_ratio`` column added.
    """
    df = df.copy()
    col_map = cfg.get("dti_columns", {})
    debt_col = col_map.get("debt", None)
    income_col = col_map.get("income", None)
    loan_col = col_map.get("loan_amount", None)

    # Find columns by common naming patterns
    if debt_col is None:
        debt_col = _find_column(df, ["existing_debt", "debt", "credit_amount"])
    if income_col is None:
        income_col = _find_column(df, ["income", "annual_income", "personal_status"])
    if loan_col is None:
        loan_col = _find_column(df, ["loan_amount", "credit_amount", "duration"])

    if debt_col and income_col:
        income = pd.to_numeric(df[income_col], errors="coerce").replace(0, np.nan)
        debt = pd.to_numeric(df[debt_col], errors="coerce").fillna(0)

        if loan_col and loan_col != debt_col:
            loan = pd.to_numeric(df[loan_col], errors="coerce").fillna(0)
            df["dti_ratio"] = ((debt + loan) / income).clip(upper=1.0)
        else:
            df["dti_ratio"] = (debt / income).clip(upper=1.0)

        logger.info(f"Added dti_ratio (debt={debt_col}, income={income_col})")
    elif loan_col and income_col:
        income = pd.to_numeric(df[income_col], errors="coerce").replace(0, np.nan)
        loan = pd.to_numeric(df[loan_col], errors="coerce").fillna(0)
        df["dti_ratio"] = (loan / income).clip(upper=1.0)
        logger.info(f"Added dti_ratio (loan={loan_col}, income={income_col})")
    else:
        logger.warning("Could not compute DTI - missing debt/income columns")

    return df


def add_utilization(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add credit utilization ratio.

    Utilization = balance / credit_limit.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cfg : dict
        Config with ``utilization_columns`` mapping.

    Returns
    -------
    pd.DataFrame
        Data with ``utilization_ratio`` column added.
    """
    df = df.copy()
    col_map = cfg.get("utilization_columns", {})
    balance_col = col_map.get("balance", None)
    limit_col = col_map.get("limit", None)

    if balance_col is None:
        balance_col = _find_column(df, ["balance", "existing_credits", "credit_amount"])
    if limit_col is None:
        limit_col = _find_column(df, ["credit_limit", "limit", "credit_amount"])

    if balance_col and limit_col and balance_col != limit_col:
        limit = pd.to_numeric(df[limit_col], errors="coerce").replace(0, np.nan)
        balance = pd.to_numeric(df[balance_col], errors="coerce").fillna(0)
        df["utilization_ratio"] = (balance / limit).clip(0, 1.5)
        logger.info(f"Added utilization_ratio (balance={balance_col}, limit={limit_col})")
    else:
        logger.warning("Could not compute utilization - missing balance/limit columns")

    return df


def add_loan_burden(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add loan burden score.

    Burden = loan_amount * duration / income. Captures combined effect
    of loan size and repayment period relative to earning capacity.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cfg : dict
        Config with ``burden_columns`` mapping.

    Returns
    -------
    pd.DataFrame
        Data with ``loan_burden`` column added.
    """
    df = df.copy()
    col_map = cfg.get("burden_columns", {})
    amount_col = col_map.get("amount", _find_column(df, ["credit_amount", "loan_amount", "amount"]))
    duration_col = col_map.get("duration", _find_column(df, ["duration", "term", "installment_rate"]))
    income_col = col_map.get("income", _find_column(df, ["income", "annual_income"]))

    if amount_col and duration_col:
        amount = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
        duration = pd.to_numeric(df[duration_col], errors="coerce").fillna(1)

        if income_col:
            income = pd.to_numeric(df[income_col], errors="coerce").replace(0, np.nan)
            df["loan_burden"] = (amount * duration) / income
        else:
            # Normalize by median amount as fallback
            median_amount = amount.median() or 1
            df["loan_burden"] = (amount * duration) / median_amount

        logger.info(f"Added loan_burden (amount={amount_col}, duration={duration_col})")
    else:
        logger.warning("Could not compute loan_burden - missing amount/duration columns")

    return df


def add_age_buckets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add age group buckets for stratified analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cfg : dict
        Config with optional ``age_column`` and ``age_bins``.

    Returns
    -------
    pd.DataFrame
        Data with ``age_group`` column added.
    """
    df = df.copy()
    age_col = cfg.get("age_column", _find_column(df, ["age", "age_years"]))

    if age_col:
        bins = cfg.get("age_bins", [0, 25, 35, 45, 55, 100])
        labels = cfg.get("age_labels", ["18-25", "26-35", "36-45", "46-55", "55+"])
        df["age_group"] = pd.cut(
            pd.to_numeric(df[age_col], errors="coerce"),
            bins=bins,
            labels=labels,
            right=True,
        )
        logger.info(f"Added age_group from {age_col}")
    else:
        logger.warning("Could not add age_group - no age column found")

    return df


def add_log_transforms(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply log1p transformation to highly skewed numeric features.

    Skewness threshold: |skew| > 2.0 (configurable).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cfg : dict
        Config with optional ``skew_threshold``.

    Returns
    -------
    pd.DataFrame
        Data with ``log_<col>`` columns for skewed features.
    """
    df = df.copy()
    threshold = cfg.get("skew_threshold", 2.0)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude = {"target", "dti_ratio", "utilization_ratio", "loan_burden"}

    for col in numeric_cols:
        if col in exclude:
            continue
        skew = df[col].skew()
        if abs(skew) > threshold:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
            logger.debug(f"Added log_{col} (skew={skew:.2f})")

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for name in candidates:
        if name in df.columns:
            return name
    return None