"""Bake-off across every algorithm family on ONE identical out-of-time split.

Scored on what the business consumes:
  lapse_pr_auc  - ranking for the retention queue under heavy imbalance
  lapse_brier   - is the risk score a real probability (uncalibrated)
  mlogloss      - quality of the WHOLE time distribution, not just the tail
  days_mae      - expected days-to-payment, among dues actually paid
  capture@20    - share of lapses caught in the top 20% of the queue
  mono_viol     - share of rows whose implied CDF is non-monotone
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from lapse_prediction.config import CFG, Config
from lapse_prediction.evaluation import report
from lapse_prediction.features.labels import time_split
from lapse_prediction.models.zoo import REGISTRY, Blend
from lapse_prediction.pipelines.train import prepare
from lapse_prediction.utils.logging import get_logger

log = get_logger(__name__)


def _row(name, proba, valid, fit_s, pred_s, cfg) -> dict:
    h = report.full_report(valid, proba, cfg)["headline"]
    return {"model": name, "lapse_pr_auc": h["lapse_pr_auc"],
            "lapse_auc": h["lapse_roc_auc"], "lapse_brier": h["lapse_brier"],
            "mlogloss": h["bucket_mlogloss"], "days_mae": h["days_mae"],
            "capture@20": h["capture_at_20pct"],
            "mono_viol": h["monotonicity_violation"],
            "fit_s": round(fit_s, 1),
            "pred_us_per_row": round(1e6 * pred_s / max(len(valid), 1), 1)}


def run(cfg: Config = CFG, n_policies: int = 12_000, only: set[str] | None = None,
        refresh: bool = False, write: bool = True) -> pd.DataFrame:
    df = prepare(cfg, n_policies=n_policies, refresh=refresh)
    train, test, valid = time_split(df, cfg)
    log.info("train=%d  test(OOT)=%d  valid=%d  lapse_rate(valid)=%.3f",
             len(train), len(test), len(valid), valid["lapsed"].mean())

    rows, fitted = [], {}
    for cls in REGISTRY:
        if only and cls.name not in only:
            continue
        try:
            t0 = time.perf_counter()
            m = cls().fit(train, valid=test)   # early stopping uses TEST, never VALID
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            proba = m.predict_proba(valid)
            rows.append(_row(cls.name, proba, valid, fit_s,
                             time.perf_counter() - t0, cfg))
            fitted[cls.name] = m
            log.info("ok   %-18s %6.1fs  pr_auc=%.4f", cls.name, fit_s,
                     rows[-1]["lapse_pr_auc"])
        except Exception as e:     # a family that cannot run is itself a result
            log.error("FAIL %-18s %s: %s", cls.name, type(e).__name__, e)
            rows.append({"model": cls.name, "lapse_pr_auc": np.nan, "fit_s": np.nan})

    res = pd.DataFrame(rows).sort_values("lapse_pr_auc", ascending=False)

    top = [n for n in res["model"].head(3) if n in fitted][:2]
    if len(top) == 2:
        b = Blend([fitted[n] for n in top], name=f"blend({'+'.join(top)})")
        t0 = time.perf_counter()
        res = pd.concat([res, pd.DataFrame([_row(
            b.name, b.predict_proba(valid), valid, 0.0,
            time.perf_counter() - t0, cfg)])], ignore_index=True)

    res = res.sort_values("lapse_pr_auc", ascending=False, ignore_index=True)
    log.info("BAKE-OFF (out-of-time, identical features, no calibration)\n%s",
             res.to_string(index=False))
    if write:
        path = Path(cfg.artifacts) / "benchmark.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(path, index=False)
        log.info("results -> %s", path)
    return res


if __name__ == "__main__":
    run()
