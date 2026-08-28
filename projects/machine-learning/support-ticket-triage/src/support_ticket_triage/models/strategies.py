"""The four ways to turn binary classifiers into a six-class decision.

  native       one model that is multiclass internally (MultinomialNB, trees)
  ovr          one-vs-rest: K binary models, argmax of their scores
  ovo          one-vs-one: K(K-1)/2 binary models, vote
  softmax      one model, multinomial logistic, a joint probability by design

They are not interchangeable. The differences that actually bite:

  * cost. OvO fits 15 models here against OvR's 6. That gap grows as K squared.
  * calibration. OvR normalises K independent scores that were never fitted to
    sum to 1, so the result is a ranking wearing a probability costume.
  * imbalance. OvR trains each binary model on a 3%-positive problem for the
    rare class; OvO compares it against one class at a time, which is a much
    more balanced sub-problem, and it usually shows in the rare-class recall.
  * ties. OvO can produce a vote tie and needs a documented tiebreak.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from support_ticket_triage.config import CFG, Config
from support_ticket_triage.utils.logging import get_logger

log = get_logger(__name__)


class Strategy:
    """One interface so the comparison is apples to apples."""

    name = "base"
    #: how many binary sub-models this strategy fits, for the cost column
    n_submodels = 1

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.classes_: np.ndarray | None = None
        self.fit_seconds: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Strategy":
        t0 = time.perf_counter()
        self._fit(X, y)
        self.fit_seconds = round(time.perf_counter() - t0, 3)
        return self

    def _fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class NativeNB(Strategy):
    """MultinomialNB, which is multiclass without any wrapper at all.

    One pass over the counts, no per-class model, no meta-strategy. It is the
    cheapest thing in the comparison by a wide margin, which is exactly why it
    survived being wrong about independence for forty years.
    """

    name = "native_nb"

    def _fit(self, X, y):
        self.model = MultinomialNB(alpha=self.cfg.alpha).fit(X, y)
        self.classes_ = self.model.classes_

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class OvRNB(Strategy):
    """One binary Naive Bayes per class, scores normalised afterwards.

    The normalisation is the part to look at. Each sub-model was fitted without
    any knowledge of the others, so dividing by the sum manufactures a
    distribution that no model ever estimated.
    """

    name = "ovr_nb"

    def _fit(self, X, y):
        self.model = OneVsRestClassifier(MultinomialNB(alpha=self.cfg.alpha)).fit(X, y)
        self.classes_ = self.model.classes_
        self.n_submodels = len(self.classes_)

    def predict_proba(self, X):
        p = self.model.predict_proba(X)
        s = p.sum(axis=1, keepdims=True)
        # A row can be all-zeros if every sub-model rejects it. Falling back to
        # the uniform is honest; dividing by zero is not.
        return np.where(s > 0, p / np.where(s == 0, 1, s), 1.0 / p.shape[1])


class OvONB(Strategy):
    """Every pair of classes gets its own model, then they vote.

    K(K-1)/2 = 15 models for six classes. Each is trained on only the two
    classes it arbitrates, so the rare class is never a 3% needle: against any
    single opponent it is a much fairer fight.
    """

    name = "ovo_nb"

    def _fit(self, X, y):
        self.model = OneVsOneClassifier(MultinomialNB(alpha=self.cfg.alpha)).fit(X, y)
        self.classes_ = self.model.classes_
        k = len(self.classes_)
        self.n_submodels = k * (k - 1) // 2

    def predict_proba(self, X):
        # OvO has no probabilities, only vote counts via the decision function.
        # Turning votes into something that sums to 1 is a convenience for the
        # comparison, not a probability, and the calibration numbers say so.
        d = self.model.decision_function(X)
        d = d - d.min(axis=1, keepdims=True)
        s = d.sum(axis=1, keepdims=True)
        return np.where(s > 0, d / np.where(s == 0, 1, s), 1.0 / d.shape[1])


class SoftmaxLR(Strategy):
    """Multinomial logistic regression: one model, a joint fit over all classes.

    The only member of the four whose probabilities are estimated jointly rather
    than assembled after the fact. It is the reference point for what calibrated
    output looks like before anyone applies a calibrator.
    """

    name = "softmax_lr"

    def _fit(self, X, y):
        self.model = LogisticRegression(
            max_iter=2000, C=1.0, random_state=self.cfg.seed).fit(X, y)
        self.classes_ = self.model.classes_

    def predict_proba(self, X):
        return self.model.predict_proba(X)


REGISTRY: dict[str, type[Strategy]] = {
    c.name: c for c in (NativeNB, OvRNB, OvONB, SoftmaxLR)
}


def build(name: str, cfg: Config = CFG) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](cfg)
