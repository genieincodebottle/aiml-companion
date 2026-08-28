"""Training pipeline: ledger -> features -> OOT split -> fit -> calibrate ->
evaluate -> persist a versioned model with its card.

Run via `lapse train` (see cli.py) or `python -m lapse_prediction.pipelines.train`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lapse_prediction.config import CFG, Config
from lapse_prediction.data.io import load_ledger, save
from lapse_prediction.evaluation import metrics, report
from lapse_prediction.features.build import build
from lapse_prediction.features.labels import add_labels, mature, time_split
from lapse_prediction.models import registry
from lapse_prediction.models.bucket import BucketModel
from lapse_prediction.models.ordinal import OrdinalChain
from lapse_prediction.utils.logging import get_logger

log = get_logger(__name__)

MODELS = {"ordinal_chain": OrdinalChain, "bucket": BucketModel}


def prepare(cfg: Config = CFG, n_policies: int = 20_000, refresh: bool = False,
            data_path: str | None = None) -> pd.DataFrame:
    """Ledger -> modelling table, cached to disk."""
    raw = load_ledger(data_path or cfg.raw_data, refresh=refresh, cfg=cfg,
                      n_policies=n_policies)
    df = add_labels(build(raw), cfg)
    df = mature(df, cfg=cfg)
    save(df, cfg.modelling_table)
    log.info("modelling table: %d rows x %d cols -> %s",
             len(df), df.shape[1], cfg.modelling_table)
    return df


def run(model_name: str = "ordinal_chain", cfg: Config = CFG,
        n_policies: int = 20_000, refresh: bool = False,
        calibrate: bool = True, persist: bool = True) -> dict:
    df = prepare(cfg, n_policies=n_policies, refresh=refresh)
    train, test, valid = time_split(df, cfg)
    log.info("split  train=%d  test(OOT)=%d  valid=%d  lapse_rate(valid)=%.4f",
             len(train), len(test), len(valid), valid["lapsed"].mean())

    if model_name not in MODELS:
        raise KeyError(f"unknown model {model_name!r}; choose from {list(MODELS)}")
    model = MODELS[model_name]().fit(train, valid=test)

    if calibrate:
        if not hasattr(model, "calibrate"):
            # Silently shipping an uncalibrated model is exactly the failure
            # this pipeline exists to prevent -- ops reads p_lapse as a number.
            raise TypeError(
                f"{model_name} cannot be calibrated but calibrate=True. "
                f"Implement _raw_proba/calibrate, or pass --no-calibrate "
                f"and accept that the scores are rankings, not probabilities.")
        model.calibrate(test)          # isotonic on a cohort never trained on
        log.info("calibrated %s on the out-of-time cohort", model_name)
    else:
        log.warning("%s is NOT calibrated -- treat its scores as rankings only",
                    model_name)

    proba = model.predict_proba(valid)
    scores = report.full_report(valid, proba, cfg)
    report.log_report(scores, log)

    if persist:
        card = registry.build_card(
            model_name, model, train, metrics=scores["headline"], cfg=cfg,
            notes=("in-grace bucket distribution; "
                   f"calibrated={getattr(model, 'is_calibrated', False)}; "
                   "see README for model selection"))
        registry.save(model, card, cfg.model_store)

    return {"model": model, "metrics": scores, "splits":
            {"train": len(train), "test": len(test), "valid": len(valid)}}


if __name__ == "__main__":
    run()
