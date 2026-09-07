"""The whole argument, as a runnable pipeline.

  1. fit the naive OLS everyone actually writes
  2. run the diagnostics -- watch it fail four of seven checks
  3. fix exactly what the diagnostics named, refit
  4. fit a GBM with none of that specification work
  5. compare on ACCURACY (where the GBM wins a little)
     and on RECOVERY  (where the naive OLS is simply wrong and the GBM
                       cannot answer the question at all)
  6. reconcile the two with SHAP
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rent_price_explainer.config import CFG, Config
from rent_price_explainer.data.io import load_listings
from rent_price_explainer.diagnostics import assumptions
from rent_price_explainer.evaluation import metrics, recovery
from rent_price_explainer.explain import shap_report
from rent_price_explainer.features.build import TARGET, design_matrix, junk_columns
from rent_price_explainer.models.gbm import GBM
from rent_price_explainer.models.linear import (InteractionOLS, NaiveOLS,
                                                SpecifiedOLS)
from rent_price_explainer.utils.logging import get_logger

log = get_logger(__name__)


def split(df: pd.DataFrame, cfg: Config = CFG):
    return train_test_split(df, test_size=cfg.test_size,
                            random_state=cfg.split_seed)


def run(cfg: Config = CFG, write: bool = True) -> dict:
    df = load_listings(cfg=cfg)
    train, test = split(df, cfg)
    log.info("train=%d  test=%d  median rent=%.0f", len(train), len(test),
             df[TARGET].median())

    out: dict = {}

    # ---------------------------------------------------------- 1. naive OLS
    naive = NaiveOLS().fit(train)
    naive_metrics = metrics.all_metrics(test[TARGET], naive.predict(test),
                                        len(naive.cols))
    log.info("naive OLS      %s", naive_metrics)

    # ------------------------------------------------------- 2. diagnostics
    X_naive = naive._prepare(train)
    diag = assumptions.run_all(naive.res, X_naive)
    log.info("diagnostics on the naive model: %d of %d checks FAILED",
             diag["n_failed"], len(diag["checks"]))
    for c in diag["checks"]:
        log.info("  %s", c)
    out["diagnostics"] = assumptions.summary_frame(diag["checks"])
    out["vif_table"] = diag["vif_table"]

    # ------------------------------------------------- 3. specified OLS refit
    spec = SpecifiedOLS().fit(train)
    spec_metrics = metrics.all_metrics(test[TARGET], spec.predict(test),
                                       len(spec.cols))
    log.info("specified OLS  %s", spec_metrics)

    spec_diag = assumptions.run_all(spec.res, spec._prepare(train))
    log.info("diagnostics after the fixes: %d of %d checks FAILED",
             spec_diag["n_failed"], len(spec_diag["checks"]))
    out["diagnostics_after"] = assumptions.summary_frame(spec_diag["checks"])

    # --------------------------- 3b. the interaction the GBM will point at
    inter = InteractionOLS().fit(train)
    inter_metrics = metrics.all_metrics(test[TARGET], inter.predict(test),
                                        len(inter.cols))
    log.info("interaction OLS %s", inter_metrics)

    # ------------------------------------------------------------- 4. the GBM
    gbm = GBM(log_target=cfg.gbm_log_target, seed=cfg.seed).fit(train)
    gbm_metrics = metrics.all_metrics(test[TARGET], gbm.predict(test),
                                      len(gbm.cols))
    log.info("GBM            %s", gbm_metrics)

    out["accuracy"] = pd.DataFrame([
        {"model": "naive_ols", **naive_metrics},
        {"model": "specified_ols", **spec_metrics},
        {"model": "interaction_ols", **inter_metrics},
        {"model": "gbm", **gbm_metrics},
    ])

    # -------------------------------------------------------- 5. recovery
    rec_spec = recovery.score_recovery(spec, "specified_ols")
    rec_inter = recovery.score_recovery(inter, "interaction_ols")
    out["recovery"] = pd.concat([rec_spec, rec_inter], ignore_index=True)
    out["recovery_summary"] = pd.DataFrame([
        {"model": "specified_ols", **recovery.recovery_summary(rec_spec)},
        {"model": "interaction_ols", **recovery.recovery_summary(rec_inter)},
        {"model": "naive_ols", "comparable": False,
         "note": "fitted on levels; its coefficients are not the truth's units"},
        {"model": "gbm", "comparable": False,
         "note": "has no coefficients to compare -- the trade, stated plainly"},
    ])
    out["collinearity_damage"] = recovery.collinearity_damage(train)

    # ------------------------------------------------- 6. R2 inflation demo
    Xj = design_matrix(train, log_area=True, drop_collinear=True,
                       add_age_curve=True, include_junk=True)
    junk = [c for c in Xj.columns if c.startswith("junk_")]
    base = [c for c in Xj.columns if not c.startswith("junk_")]
    out["r2_inflation"] = metrics.r2_inflation_demo(
        Xj, np.log(train[TARGET]), junk, base)

    # ----------------------------------------------------- 7. SHAP bridge
    # shap is the one heavy, optional dependency here. Losing it should cost
    # you the attribution table, not the six sections of work above it.
    out["ledger"] = shap_report.interpretability_ledger(inter, gbm)
    try:
        out["attribution"] = shap_report.compare_attributions(inter, gbm, test)
        log.info("attribution: %s",
                 shap_report.split_attribution_note(out["attribution"]))
    except ImportError:
        log.warning("shap is not installed -- skipping the attribution table. "
                    "Everything else above is complete. `pip install shap` "
                    "to enable it.")

    out["models"] = {"naive_ols": naive, "specified_ols": spec,
                     "interaction_ols": inter, "gbm": gbm}

    if write:
        d = Path(cfg.artifacts)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            if isinstance(v, pd.DataFrame):
                v.to_csv(d / f"{k}.csv", index=False)
        inter.coefficients().to_csv(d / "interaction_ols_coefficients.csv")
        log.info("wrote %d tables to %s",
                 sum(isinstance(v, pd.DataFrame) for v in out.values()), d)
    return out


if __name__ == "__main__":
    run()
