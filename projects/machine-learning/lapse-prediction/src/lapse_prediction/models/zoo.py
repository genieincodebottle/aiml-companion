"""Every candidate algorithm behind ONE interface, so the comparison is honest.

    m.fit(train, valid) -> self
    m.predict_proba(df) -> (n_rows, n_classes) over CFG.class_names

Whatever the internal machinery -- multiclass, ordinal chain, hurdle, AFT,
Cox, discrete hazard, neural -- the output is the same bucket distribution,
scored with the same metrics on the same out-of-time cohort.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lapse_prediction.config import CFG
from lapse_prediction.features.build import CATEGORICAL, assert_no_leakage, feature_columns
from lapse_prediction.features.matrix import Matrix
from lapse_prediction.models.base import Base, from_cdf

EDGES = np.array(CFG.edges, dtype=float)
LI = CFG.lapse_index


# ---------------------------------------------------------------- baselines
class Prior(Base):
    """Book-average bucket mix. Any model that cannot beat this is noise."""
    name = "prior"

    def fit(self, train, valid=None):
        self.p = np.bincount(train["bucket"], minlength=CFG.n_classes) / len(train)
        return self

    def predict_proba(self, df):
        return np.tile(self.p, (len(df), 1))


class Logit(Base):
    """Multinomial logistic regression. Slow to lose, easy to explain to an
    actuary or a regulator -- the reference point, not the answer."""
    name = "logit"

    def fit(self, train, valid=None):
        self.mx = Matrix().fit(train)
        self.m = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        self.m.fit(self.mx.transform(train), train["bucket"])
        return self

    def predict_proba(self, df):
        return self.m.predict_proba(self.mx.transform(df))


class RF(Base):
    name = "random_forest"

    def fit(self, train, valid=None):
        self.mx = Matrix(standardize=False).fit(train)
        self.m = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=25, max_features="sqrt",
            n_jobs=-1, random_state=CFG.seed)
        self.m.fit(self.mx.transform(train), train["bucket"])
        return self

    def predict_proba(self, df):
        return self.m.predict_proba(self.mx.transform(df))


# ------------------------------------------------------------ boosted trees
class LGBMulti(Base):
    """The model shipped in model_bucket.py, uncalibrated, for a fair fight."""
    name = "lgbm_multiclass"

    def fit(self, train, valid=None):
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        self.cols = feature_columns(train)
        assert_no_leakage(self.cols)
        self.m = LGBMClassifier(objective="multiclass", num_class=CFG.n_classes,
                                n_estimators=1500, learning_rate=0.04, num_leaves=63,
                                min_child_samples=80, subsample=0.85, subsample_freq=1,
                                colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1,
                                random_state=CFG.seed, verbosity=-1)
        kw = {}
        if valid is not None and len(valid):
            kw = dict(eval_set=[(valid[self.cols], valid["bucket"])],
                      callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        self.m.fit(train[self.cols], train["bucket"],
                   categorical_feature=CATEGORICAL, **kw)
        return self

    def predict_proba(self, df):
        return self.m.predict_proba(df[self.cols])


class XGBMulti(Base):
    name = "xgb_multiclass"

    def fit(self, train, valid=None):
        from xgboost import XGBClassifier
        self.cols = feature_columns(train)
        assert_no_leakage(self.cols)
        self.m = XGBClassifier(objective="multi:softprob", num_class=CFG.n_classes,
                               n_estimators=900, learning_rate=0.05, max_depth=7,
                               min_child_weight=10, subsample=0.85,
                               colsample_bytree=0.8, reg_lambda=5.0,
                               tree_method="hist", enable_categorical=True,
                               n_jobs=-1, random_state=CFG.seed)
        self.m.fit(train[self.cols], train["bucket"], verbose=False)
        return self

    def predict_proba(self, df):
        return self.m.predict_proba(df[self.cols])


from lapse_prediction.models.ordinal import OrdinalChain  # noqa: F401


# ------------------------------------------------------------------- hurdle
class Hurdle(Base):
    """The originally proposed two-stage design."""
    name = "hurdle_2stage"

    def fit(self, train, valid=None):
        from lapse_prediction.models.hurdle import HurdleModel
        self.m = HurdleModel().fit(train)
        return self

    def predict_proba(self, df):
        return self.m.predict_proba(df)


# ---------------------------------------------------------------- survival
class XGBAFT(Base):
    """Accelerated failure time on boosted trees -- the family the question
    asked about. Lapses enter as RIGHT-CENSORED at the grace boundary
    (label_upper = inf), which is the honest encoding: we know the premium had
    not arrived by day 45, not that it never will. Bucket probabilities come
    from the fitted log-normal CDF."""
    name = "xgb_aft"

    def fit(self, train, valid=None):
        import xgboost as xgb
        self.cols = feature_columns(train)
        assert_no_leakage(self.cols)
        d = train["days_to_pay"].to_numpy(float)
        paid = ~np.isnan(d) & (d <= CFG.grace_days)
        lo = np.where(paid, np.maximum(d, 0.5), float(CFG.grace_days))
        hi = np.where(paid, np.maximum(d, 0.5), np.inf)
        dm = xgb.DMatrix(train[self.cols], enable_categorical=True)
        dm.set_float_info("label_lower_bound", lo)
        dm.set_float_info("label_upper_bound", hi)
        self.sigma = 1.0
        self.m = xgb.train(
            {"objective": "survival:aft", "eval_metric": "aft-nloglik",
             "aft_loss_distribution": "normal",
             "aft_loss_distribution_scale": self.sigma, "tree_method": "hist",
             "learning_rate": 0.05, "max_depth": 6, "subsample": 0.85,
             "colsample_bytree": 0.8, "lambda": 5.0, "seed": CFG.seed},
            dm, num_boost_round=600)
        return self

    def predict_proba(self, df):
        import xgboost as xgb
        mu = np.log(np.clip(self.m.predict(
            xgb.DMatrix(df[self.cols], enable_categorical=True)), 1e-6, None))
        z = (np.log(EDGES)[None, :] - mu[:, None]) / self.sigma
        return from_cdf(norm.cdf(z))


class CoxPH(Base):
    """Classical semi-parametric Cox, via statsmodels PHReg. Included because
    it is what everyone asks about; subsampled because it does not scale like
    the trees do. Proportional hazards is a strong assumption here -- payment
    behaviour has a spike at day 0 and another at the grace deadline."""
    name = "cox_ph"

    def __init__(self, n_sub: int = 25_000):
        self.n_sub = n_sub

    def fit(self, train, valid=None):
        from statsmodels.duration.hazard_regression import PHReg
        tr = train.sample(min(self.n_sub, len(train)), random_state=CFG.seed)
        self.mx = Matrix(drop_first=True).fit(tr)
        X = self.mx.transform(tr)
        d = tr["days_to_pay"].to_numpy(float)
        paid = (~np.isnan(d)) & (d <= CFG.grace_days)
        t = np.where(paid, np.maximum(d, 0.5), float(CFG.grace_days))
        mod = PHReg(t, X, status=paid.astype(int))
        try:
            self.res = mod.fit(method="lbfgs", maxiter=300)
        except (ValueError, np.linalg.LinAlgError):
            # Near-collinear one-hot columns make the Hessian singular on small
            # or thin books. A ridge penalty recovers usable coefficients; it
            # costs the standard errors, which this model does not use.
            from statsmodels.duration.hazard_regression import PHRegResults
            ridge = mod.fit_regularized(alpha=0.05, L1_wt=0.0)
            # RegularizedResults cannot compute the baseline hazard; rewrap the
            # penalised coefficients in a full results object that can.
            k = len(np.asarray(ridge.params))
            self.res = PHRegResults(mod, np.asarray(ridge.params), np.eye(k))
        # baseline hazard is defined at x=0; features are standardised so x=0 is
        # the training mean, but re-centre explicitly to keep levels honest
        self.lp0 = float(np.mean(X @ self.res.params))
        # PHReg returns [times, cumulative_hazard, survival] -- index 1, not 2
        times, cumhaz, _ = self.res.baseline_cumulative_hazard[0]
        order = np.argsort(times)
        times, cumhaz = np.asarray(times)[order], np.asarray(cumhaz)[order]
        self.h0 = np.interp(EDGES, times, cumhaz, left=0.0, right=cumhaz[-1])
        return self

    def predict_proba(self, df):
        lp = self.mx.transform(df) @ self.res.params - self.lp0
        S = np.exp(-self.h0[None, :] * np.exp(lp)[:, None])
        return from_cdf(1 - S)


class DiscreteHazard(Base):
    """Discrete-time hazard restricted to the grace window, so it competes on
    the same footing as everything else here. Its real advantage shows up on
    the revival horizon, which this table cannot measure."""
    name = "discrete_hazard"

    def fit(self, train, valid=None):
        from lapse_prediction.models.hazard import HazardModel, expand
        self.m = HazardModel(n_estimators=500).fit(expand(train, sample=0.5))
        return self

    def predict_proba(self, df):
        S, days = self.m.survival(
            df, horizon_days=CFG.grace_days + CFG.hazard_period_days)
        F = 1 - S
        idx = [int(np.clip(np.searchsorted(days, e, side="right") - 1, 0, F.shape[1] - 1))
               for e in EDGES]
        return from_cdf(F[:, idx])


# ------------------------------------------------------------------- neural
class DeepHit(Base):
    """DeepHit-style: an MLP with a softmax over discrete time bins plus a
    'never within grace' bin. Same target as the multiclass tree model, a very
    different inductive bias. On tabular data with this much categorical
    structure it is the underdog, and it should be shown losing rather than
    assumed to lose."""
    name = "deephit_mlp"

    def __init__(self, epochs: int = 20):
        self.epochs = epochs

    def fit(self, train, valid=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(CFG.seed)
        self.mx = Matrix().fit(train)
        X = torch.tensor(self.mx.transform(train), dtype=torch.float32)
        y = torch.tensor(train["bucket"].to_numpy(), dtype=torch.long)
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, CFG.n_classes))
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), batch_size=1024, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                lossf(self.net(xb), yb).backward()
                opt.step()
        return self

    def predict_proba(self, df):
        import torch
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.mx.transform(df), dtype=torch.float32)
            return torch.softmax(self.net(x), dim=1).numpy()


class Blend(Base):
    """Probability average of already-fitted models. Cheap insurance against
    any single family's idiosyncratic failure mode."""

    def __init__(self, models, name="blend"):
        self.models, self.name = models, name

    def fit(self, train, valid=None):
        return self

    def predict_proba(self, df):
        return np.mean([m.predict_proba(df) for m in self.models], axis=0)


REGISTRY = [Prior, Logit, RF, LGBMulti, XGBMulti, OrdinalChain, Hurdle,
            XGBAFT, CoxPH, DiscreteHazard, DeepHit]
