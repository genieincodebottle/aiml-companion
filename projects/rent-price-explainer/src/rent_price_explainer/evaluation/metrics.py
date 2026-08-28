"""Fit metrics, and an honest account of what each one hides.

R² is the most quoted and least useful number in applied regression. It never
falls when you add a feature -- not even a column of pure noise -- so "our R²
went up" is not evidence of anything. Adjusted R² penalises that, and MAE on a
held-out set is the only one of the three that can actually go the wrong way
when you make the model worse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot


def adjusted_r2(y, yhat, n_features: int) -> float:
    n = len(y)
    if n - n_features - 1 <= 0:
        return float("nan")
    return 1 - (1 - r2(y, yhat)) * (n - 1) / (n - n_features - 1)


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def mape(y, yhat) -> float:
    y = np.asarray(y, float)
    return float(np.mean(np.abs((y - np.asarray(yhat, float)) / np.clip(np.abs(y), 1e-9, None))))


def median_ape(y, yhat) -> float:
    """Median absolute percentage error -- the number to quote on skewed prices,
    because a handful of penthouses will not drag it around."""
    y = np.asarray(y, float)
    return float(np.median(np.abs((y - np.asarray(yhat, float)) / np.clip(np.abs(y), 1e-9, None))))


def all_metrics(y, yhat, n_features: int) -> dict:
    return {
        "r2": round(r2(y, yhat), 4),
        "adj_r2": round(adjusted_r2(y, yhat, n_features), 4),
        "mae": round(mae(y, yhat), 1),
        "rmse": round(rmse(y, yhat), 1),
        "mape": round(mape(y, yhat), 4),
        "median_ape": round(median_ape(y, yhat), 4),
        "n_features": int(n_features),
    }


def r2_inflation_demo(X: pd.DataFrame, y, junk_cols: list[str],
                      base_cols: list[str]) -> pd.DataFrame:
    """Add pure-noise columns one at a time and watch R² climb regardless.

    This is the demonstration that settles the argument: every added junk
    feature raises in-sample R², while adjusted R² and held-out MAE tell the
    truth. If you select features on R², noise wins.
    """
    from sklearn.linear_model import LinearRegression

    rows, cols = [], list(base_cols)
    for i, extra in enumerate([None] + list(junk_cols)):
        if extra is not None:
            cols.append(extra)
        m = LinearRegression().fit(X[cols], y)
        pred = m.predict(X[cols])
        rows.append({"junk_features_added": i,
                     "r2": round(r2(y, pred), 5),
                     "adj_r2": round(adjusted_r2(y, pred, len(cols)), 5)})
    out = pd.DataFrame(rows)
    out["r2_change"] = out["r2"].diff().round(6)
    out["adj_r2_change"] = out["adj_r2"].diff().round(6)
    return out
