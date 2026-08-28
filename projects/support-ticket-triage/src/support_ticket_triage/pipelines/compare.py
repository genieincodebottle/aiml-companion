"""The experiments, end to end.

Three of them, in the order the argument needs:

  1. `run_strategies`  the four multiclass strategies on identical data
  2. `run_calibration` what calibration repairs, and what it leaves alone
  3. `run_sweep`       the headline: accuracy and ECE as the assumption breaks
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from support_ticket_triage.config import CFG, Config
from support_ticket_triage.data.generate import DEPENDENT_PAIRS, generate
from support_ticket_triage.data.io import load_tickets
from support_ticket_triage.evaluation import independence, metrics
from support_ticket_triage.features.build import feature_columns, split, xy
from support_ticket_triage.models.calibrate import (CalibratedNB,
                                                    confidence_shift,
                                                    ranking_preserved)
from support_ticket_triage.models.strategies import REGISTRY, build
from support_ticket_triage.routing import decide
from support_ticket_triage.utils.logging import get_logger

log = get_logger(__name__)


def _write(tables: dict[str, pd.DataFrame], cfg: Config) -> None:
    out = Path(cfg.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    log.info("wrote %d tables to %s", len(tables), out)


def run_independence(cfg: Config = CFG, write: bool = True) -> dict:
    """Prove the assumption is false, then prove the proof works blind."""
    df = load_tickets(cfg=cfg)
    tokens = feature_columns(df)

    planted = independence.planted_pair_report(df, DEPENDENT_PAIRS)
    survey = independence.survey_all_pairs(df, tokens, top=30)
    recovery = independence.recovered_planted_pairs(survey, DEPENDENT_PAIRS)

    log.info("planted pairs: median lift %.2f (1.0 would mean the assumption holds)",
             float(planted["lift"].median()))
    log.info("blind survey: %d of the top %d flags were genuine (precision "
             "%.0f%%), covering %.0f%% of the planted structure",
             recovery["found"], recovery["surveyed"],
             recovery["precision"] * 100, recovery["recall"] * 100)

    if write:
        _write({"independence_planted": planted,
                "independence_survey": survey}, cfg)
    return {"planted": planted, "survey": survey, "recovery": recovery}


def run_strategies(cfg: Config = CFG, write: bool = True) -> dict:
    """All four multiclass strategies, one identical split."""
    df = load_tickets(cfg=cfg)
    train, test = split(df, cfg)
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)

    rows, per_class = [], {}
    for name in REGISTRY:
        model = build(name, cfg).fit(Xtr, ytr)
        proba = model.predict_proba(Xte)
        row = metrics.score(yte.to_numpy(), proba, model.classes_, name, cfg=cfg)
        row["n_submodels"] = model.n_submodels
        row["fit_seconds"] = model.fit_seconds
        rows.append(row)
        per_class[name] = metrics.per_class_table(yte.to_numpy(), proba,
                                                  model.classes_)
        log.info("%-12s acc=%.4f macro_f1=%.4f ece=%.4f submodels=%d %.2fs",
                 name, row["accuracy"], row["macro_f1"], row["ece"],
                 model.n_submodels, model.fit_seconds)

    table = pd.DataFrame(rows)
    if write:
        _write({"strategies": table,
                "per_class_native_nb": per_class["native_nb"]}, cfg)
    return {"strategies": table, "per_class": per_class}


def run_calibration(cfg: Config = CFG, write: bool = True) -> dict:
    """Does calibration repair the probabilities without moving the decisions?"""
    df = load_tickets(cfg=cfg)
    train, test = split(df, cfg)
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)

    raw = build("native_nb", cfg).fit(Xtr, ytr)
    cal = CalibratedNB(cfg).fit(Xtr, ytr)
    p_raw, p_cal = raw.predict_proba(Xte), cal.predict_proba(Xte)

    table = pd.DataFrame([
        metrics.score(yte.to_numpy(), p_raw, raw.classes_, "native_nb", cfg=cfg),
        metrics.score(yte.to_numpy(), p_cal, cal.classes_, "calibrated_nb", cfg=cfg),
    ])
    preserved = ranking_preserved(p_raw, p_cal, raw.classes_)
    shift = confidence_shift(p_raw, p_cal)
    reliability_raw = metrics.reliability_table(
        yte.to_numpy(), p_raw, raw.classes_, cfg.n_calibration_bins)
    reliability_cal = metrics.reliability_table(
        yte.to_numpy(), p_cal, cal.classes_, cfg.n_calibration_bins)

    routed_raw = decide.route(p_raw, raw.classes_, yte.to_numpy(), cfg)
    routed_cal = decide.route(p_cal, cal.classes_, yte.to_numpy(), cfg)
    routing = pd.DataFrame([
        {"model": "native_nb", **decide.routing_summary(routed_raw, cfg)},
        {"model": "calibrated_nb", **decide.routing_summary(routed_cal, cfg)},
    ])
    sweep = decide.threshold_sweep(p_cal, cal.classes_, yte.to_numpy(), cfg=cfg)

    log.info("calibration kept %.1f%% of top-1 decisions identical; "
             "ECE %.4f -> %.4f", preserved["top1_agreement"] * 100,
             float(table.loc[0, "ece"]), float(table.loc[1, "ece"]))

    if write:
        _write({"calibration": table, "confidence_shift": shift,
                "reliability_raw": reliability_raw,
                "reliability_calibrated": reliability_cal,
                "routing": routing, "threshold_sweep": sweep}, cfg)
    return {"table": table, "preserved": preserved, "shift": shift,
            "reliability_raw": reliability_raw,
            "reliability_calibrated": reliability_cal,
            "routing": routing, "sweep": sweep}


def run_sweep(cfg: Config = CFG, write: bool = True) -> pd.DataFrame:
    """The headline experiment, done as a controlled comparison.

    The naive version of this experiment just watches Naive Bayes get worse as
    dependence rises, and concludes the assumption matters. That conclusion does
    not follow. Redundant tokens carry LESS TOTAL INFORMATION than independent
    ones: five words that always co-occur are one signal wearing five hats. Any
    model degrades on that data, including models that make no independence
    assumption at all.

    So the honest measurement is a difference. Fit Naive Bayes, which assumes
    independence, alongside multinomial logistic regression, which does not, on
    the identical data. The GAP between them is the cost of the assumption; the
    shared decline is the data getting harder and is nobody's fault.
    """
    rows = []
    for strength in cfg.sweep_points:
        df = generate(n_tickets=cfg.n_tickets, seed=cfg.seed,
                      dependency_strength=strength,
                      boilerplate_rate=cfg.boilerplate_rate)
        train, test = split(df, cfg)
        Xtr, ytr = xy(train)
        Xte, yte = xy(test)
        y = yte.to_numpy()

        nb = build("native_nb", cfg).fit(Xtr, ytr)
        lr = build("softmax_lr", cfg).fit(Xtr, ytr)
        s_nb = metrics.score(y, nb.predict_proba(Xte), nb.classes_, cfg=cfg)
        s_lr = metrics.score(y, lr.predict_proba(Xte), lr.classes_, cfg=cfg)

        planted = independence.planted_pair_report(df, DEPENDENT_PAIRS)
        rows.append({
            "dependency_strength": strength,
            "median_planted_lift": round(float(planted["lift"].median()), 3),
            # the control: a model with no independence assumption
            "accuracy_lr": s_lr["accuracy"],
            "accuracy_nb": s_nb["accuracy"],
            # what the assumption actually costs, once the data is controlled for
            "accuracy_gap": round(s_lr["accuracy"] - s_nb["accuracy"], 4),
            "ece_lr": s_lr["ece"],
            "ece_nb": s_nb["ece"],
            "ece_gap": round(s_nb["ece"] - s_lr["ece"], 4),
            "nb_pct_over_99pct_sure": s_nb["pct_over_99pct_sure"],
        })
        log.info("strength=%.2f lift=%.2f | acc lr=%.4f nb=%.4f gap=%.4f | "
                 "ece gap=%.4f", strength, rows[-1]["median_planted_lift"],
                 s_lr["accuracy"], s_nb["accuracy"], rows[-1]["accuracy_gap"],
                 rows[-1]["ece_gap"])

    table = pd.DataFrame(rows)
    if write:
        _write({"dependency_sweep": table}, cfg)
    return table


def headline(sweep: pd.DataFrame) -> dict:
    """The one-sentence version, COMPUTED from the table rather than asserted.

    Whatever the sweep says is what this returns, including if it says the
    opposite of what the project expected. A hardcoded conclusion that the data
    contradicts is worse than no conclusion.
    """
    first, last = sweep.iloc[0], sweep.iloc[-1]
    shared_decline = float(first["accuracy_lr"] - last["accuracy_lr"])
    gap_growth = float(last["accuracy_gap"] - first["accuracy_gap"])
    worst_gap = float(sweep["accuracy_gap"].max())
    ece_gap_growth = float(last["ece_gap"] - first["ece_gap"])

    if worst_gap < 0.02:
        verdict = ("the assumption is provably false and costs under 2 accuracy "
                   "points against a model that does not make it: it works anyway")
    elif worst_gap < 0.05:
        verdict = ("the assumption costs a few accuracy points; usable, but no "
                   "longer free")
    else:
        verdict = ("the assumption costs more than 5 accuracy points here, so "
                   "the it-works-anyway folklore does not hold on this data")

    return {
        "lift_at_worst": float(last["median_planted_lift"]),
        "shared_accuracy_decline": round(shared_decline, 4),
        "assumption_cost_worst": round(worst_gap, 4),
        "assumption_cost_growth": round(gap_growth, 4),
        "ece_gap_growth": round(ece_gap_growth, 4),
        "verdict": verdict,
    }
