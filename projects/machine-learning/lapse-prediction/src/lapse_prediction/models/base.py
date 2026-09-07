"""The interface every model in this package implements.

    m.fit(train, valid=None) -> self
    m.predict_proba(df)      -> (n_rows, n_classes) over CFG.class_names

Multiclass, ordinal chain, hurdle, AFT, Cox, discrete hazard and neural models
all reduce to the same bucket distribution, which is what makes the bake-off
in pipelines/benchmark.py an apples-to-apples comparison and what lets serving
code swap one for another without changing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lapse_prediction.config import CFG


class Base:
    """Subclasses set `name` and implement fit/predict_proba.

    Calibration lives here rather than on individual models: ops reads
    `p_lapse` as a probability, so the LEVEL has to be right for whichever
    model is deployed, not only for the ones that happened to implement it.
    """
    name: str = "base"
    _iso = None
    _calibration_method: str | None = None

    # ------------------------------------------------------------ calibration
    def calibrate(self, holdout: pd.DataFrame, method: str | None = None
                  ) -> "Base":
        """Fit a calibration map on the lapse probability, then rescale the
        remaining bucket mass so rows still sum to 1.

        `holdout` must be a cohort used for NEITHER fitting NOR early stopping.
        Early stopping chooses the tree count by watching a cohort, which makes
        that cohort partly in-sample: scores on it are better behaved than the
        model deserves, and a calibrator fitted there is learning to correct a
        distortion that is not present on fresh data.

        On `method`, this project defaults to Platt (a one-parameter logistic
        on the log-odds) rather than isotonic, and the reason is measured
        rather than stylistic. Isotonic is a step function. With ~3,500
        calibration rows at a 10% lapse rate it has very few effective steps,
        and it collapsed 3,356 distinct scores into 32 levels here. That is
        catastrophic for the thing ops actually consumes: the retention queue
        is a RANKING, and mass ties destroy the order inside each step.
        Measured on the untouched validation cohort:

            raw       PR-AUC 0.3381   Brier 0.07124   ECE 0.0072
            isotonic  PR-AUC 0.3119   Brier 0.07198   ECE 0.0118
            platt     PR-AUC 0.3381   Brier 0.07114   ECE 0.0048

        Isotonic made every column worse. Platt left the ranking untouched to
        four decimal places and cut ECE by a third.

        The general lesson is the one worth carrying: "calibration preserves
        the ranking" is true of a STRICTLY monotone map. Isotonic is only
        weakly monotone, and on small calibration sets the ties are where your
        ranking goes. Calibration is a change to the model like any other, so
        measure it instead of assuming it helped -- which is what
        `calibration_audit` below exists to do.
        """
        method = method or CFG.calibration_method
        raw = self._raw_proba(holdout)[:, CFG.lapse_index]
        y = holdout["lapsed"].to_numpy()
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self._iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
        elif method == "platt":
            self._iso = _PlattMap().fit(raw, y)
        else:
            raise ValueError(
                f"unknown calibration method {method!r}; use 'platt' or "
                "'isotonic'")
        self._calibration_method = method
        return self

    def _raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated probabilities. Subclasses implement THIS; predict_proba
        applies calibration on top."""
        raise NotImplementedError

    def _apply_calibration(self, p: np.ndarray) -> np.ndarray:
        if self._iso is None:
            return p
        li = CFG.lapse_index
        lapse = np.clip(self._iso.predict(p[:, li]), 1e-6, 1 - 1e-6)
        rest = p[:, :li]
        rest = rest / np.clip(rest.sum(axis=1, keepdims=True), 1e-9, None)
        out = np.empty_like(p)
        out[:, :li] = rest * (1 - lapse)[:, None]
        out[:, li] = lapse
        return out

    @property
    def is_calibrated(self) -> bool:
        return self._iso is not None

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "Base":
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def from_cdf(F: np.ndarray, cfg=CFG) -> np.ndarray:
    """CDF at the bucket edges -> bucket probabilities plus the lapse tail.

    Monotonicity is enforced here rather than assumed: AFT, Cox and the ordinal
    chain can each emit a non-monotone curve, and a model that claims
    P(paid by day 15) < P(paid by day 7) cannot be shown to an ops user.
    """
    li = cfg.lapse_index
    F = np.clip(np.maximum.accumulate(np.clip(F, 0, 1), axis=1), 1e-6, 1 - 1e-6)
    p = np.empty((F.shape[0], cfg.n_classes))
    p[:, 0] = F[:, 0]
    p[:, 1:li] = np.diff(F, axis=1)
    p[:, li] = 1 - F[:, -1]
    return p / p.sum(axis=1, keepdims=True)


class _PlattMap:
    """Platt scaling: a logistic regression on the log-odds of the raw score.

    Strictly monotone by construction, so it cannot reorder the queue -- it
    only rescales how extreme the scores are. Two parameters, which is why it
    survives a small calibration cohort where isotonic's step function does
    not.
    """

    def __init__(self):
        self._lr = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_PlattMap":
        from sklearn.linear_model import LogisticRegression
        self._lr = LogisticRegression().fit(self._logit(raw), y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(self._logit(raw))[:, 1]


def calibration_audit(model: "Base", frame: pd.DataFrame,
                      cfg=CFG) -> pd.DataFrame:
    """Did calibration actually help? Answered on an untouched cohort.

    Shipping a calibrator because calibration is a good idea in general is the
    same species of mistake as shipping a model because boosting usually wins.
    This compares the raw and calibrated scores on rows used for neither
    fitting, early stopping nor calibration, on both axes that matter: the
    RANKING ops consumes, and the LEVEL ops reads as a probability. It also
    reports how many distinct score values survive, because that is the
    mechanism behind a ranking that quietly collapses.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss

    li = cfg.lapse_index
    y = frame["lapsed"].to_numpy()
    raw = model._raw_proba(frame)[:, li]
    rows = [("raw", raw)]
    if model.is_calibrated:
        rows.append((model._calibration_method or "calibrated",
                     np.clip(model._iso.predict(raw), 1e-6, 1 - 1e-6)))
    out = []
    for name, p in rows:
        out.append({
            "scores": name,
            # ranking: what the retention queue is actually built from
            "lapse_pr_auc": round(float(average_precision_score(y, p)), 4),
            # level: what ops reads as "a 12% chance"
            "brier": round(float(brier_score_loss(y, p)), 5),
            "ece": round(_ece(y, p), 5),
            "distinct_scores": int(len(np.unique(np.round(p, 6)))),
        })
    return pd.DataFrame(out)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 12) -> float:
    """Expected calibration error: mean gap between claimed and actual risk."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)
