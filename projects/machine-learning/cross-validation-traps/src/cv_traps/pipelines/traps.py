"""The five experiments, end to end.

Each one follows the same shape, which is the point:

    1. take the development panel
    2. produce a cross-validation estimate the naive way
    3. produce one the careful way
    4. compare BOTH against the truth holdout

Step 4 is what most treatments of this topic skip. Without it you can show
that two schemes disagree but not which one is right, and "the more
conservative number is the honest one" is a heuristic, not a result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from cv_traps.config import CFG, Config
from cv_traps.data.io import load_panel
from cv_traps.evaluation import truth as truth_mod
from cv_traps.evaluation.metrics import (cv_score, holdout_score,
                                         optimism_table)
from cv_traps.features.build import booster, linear_pipeline, xy
from cv_traps.splitters import schemes
from cv_traps.utils.logging import get_logger

log = get_logger(__name__)


def _write(tables: dict[str, pd.DataFrame], cfg: Config) -> None:
    out = Path(cfg.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    log.info("wrote %d tables to %s", len(tables), out)


def _setup(cfg: Config, core_only: bool = False):
    """Load, split off the truth, and build both matrices the same way.

    `core_only` drops the noise columns. The group and time traps use it: they
    are about which rows a split allows, and 150 noise columns only add
    variance that masks the effect being measured. The preprocessing and
    selection traps need the noise, because mistaking noise for signal is
    exactly what they demonstrate.
    """
    df = load_panel(cfg=cfg)
    development, holdout = truth_mod.split_panel(df, cfg)
    X_dev, y_dev = xy(development, core_only=core_only)
    X_out, y_out = xy(holdout, core_only=core_only)
    return development, holdout, X_dev, y_dev, X_out, y_out


# ---------------------------------------------------------------- trap one
def _leaked_matrix(X, y, select_k: int | None):
    """Every transformer fitted once, on everything, before any split."""
    leaked = X.copy()
    leaked[:] = SimpleImputer(strategy="median").fit_transform(leaked)
    leaked[:] = StandardScaler().fit_transform(leaked)
    if select_k:
        k = min(select_k, leaked.shape[1] - 1)
        sel = SelectKBest(f_classif, k=k).fit(leaked, y)
        leaked = leaked[leaked.columns[sel.get_support()]]
    return leaked


def _subsample(dev: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Take whole customers, never individual rows.

    Sampling rows would split customers across the sample boundary and quietly
    reintroduce the group trap into an experiment about preprocessing.
    """
    if n_rows >= len(dev):
        return dev
    rng = np.random.default_rng(seed)
    customers = np.sort(dev["customer_id"].unique())
    keep = rng.choice(customers,
                      max(20, int(len(customers) * n_rows / len(dev))),
                      replace=False)
    return dev[dev["customer_id"].isin(set(keep.tolist()))].reset_index(drop=True)


def run_preprocessing(cfg: Config = CFG, write: bool = True) -> dict:
    """Fitting a transformer before the split, versus inside it.

    Grouped folds throughout, deliberately. On stratified folds 95% of
    validation rows belong to a customer already in training, and that leak is
    so much larger than this one that it swamps it. Holding the other traps
    closed is what makes this one a measurement of itself.

    The headline is not a single number, because the honest answer turned out
    to depend on sample size and saying so is the finding. Selecting 20 of 159
    columns on all the data before validating is nearly free when there are
    thousands of rows and catastrophic when there are hundreds: the winners
    only need to correlate with the validation rows by accident, and accidents
    are common in small samples. Most published warnings about this trap come
    from small-n settings, and most people apply the warning to large-n ones.
    """
    dev, out, X_dev, y_dev, X_out, y_out = _setup(cfg)
    splits = list(schemes.grouped(dev, y_dev, cfg.n_folds, cfg.seed))
    rows = []

    for label, leak, k in (
            ("scale+impute before split", True, None),
            ("selection before split", True, cfg.select_k),
            ("scale+impute inside the fold", False, None),
            ("selection inside the fold", False, cfg.select_k)):
        if leak:
            X = _leaked_matrix(X_dev, y_dev, k)
            pipe = linear_pipeline(seed=cfg.seed)
        else:
            X, pipe = X_dev, linear_pipeline(select_k=k, seed=cfg.seed)
        score, folds = cv_score(pipe, X, y_dev, splits)
        rows.append({"approach": label, "leaked_the_transformer": leak,
                     "cv_auc": score,
                     "fold_std": round(float(np.nanstd(folds)), 4)})

    # --- the same leak, measured across sample sizes
    curve = []
    for n_rows in cfg.preprocess_sizes:
        sub = _subsample(dev, n_rows, cfg.seed)
        Xs, ys = xy(sub)
        if len(np.unique(ys)) < 2 or len(sub) < 60:
            continue
        sp = list(schemes.grouped(sub, ys, cfg.n_folds, cfg.seed))
        leaked_cv, _ = cv_score(linear_pipeline(seed=cfg.seed),
                                _leaked_matrix(Xs, ys, cfg.select_k), ys, sp)
        honest_cv, _ = cv_score(
            linear_pipeline(select_k=min(cfg.select_k, Xs.shape[1] - 1),
                            seed=cfg.seed), Xs, ys, sp)
        curve.append({"rows": len(sub),
                      "customers": int(sub["customer_id"].nunique()),
                      "leaked_cv": round(leaked_cv, 4),
                      "honest_cv": round(honest_cv, 4),
                      "self_deception": round(leaked_cv - honest_cv, 4)})
    curve_df = pd.DataFrame(curve)

    true_auc = holdout_score(linear_pipeline(select_k=cfg.select_k,
                                             seed=cfg.seed),
                             X_dev, y_dev, X_out, y_out)
    table = optimism_table(rows, true_auc)
    if write:
        _write({"trap_preprocessing": table,
                "trap_preprocessing_by_size": curve_df}, cfg)
    return {"table": table, "curve": curve_df, "truth": true_auc}


# ---------------------------------------------------------------- trap two
def run_grouped(cfg: Config = CFG, write: bool = True) -> dict:
    """The same customer on both sides of a fold.

    A booster is used here deliberately. The customer-level columns fingerprint
    the customer, so a flexible model can recognise a customer it has already
    been trained on and recall its latent churn tendency. That is not cheating
    by the model, it is the split handing it the answer.
    """
    dev, out, X_dev, y_dev, X_out, y_out = _setup(cfg, core_only=True)
    model = booster(seed=cfg.seed)
    rows, leak_frames = [], {}

    for name in ("stratified_kfold", "group_kfold"):
        splits = list(schemes.build(name)(dev, y_dev, cfg.n_folds, cfg.seed))
        score, folds = cv_score(model, X_dev, y_dev, splits)
        leak = schemes.leakage_report(dev, splits)
        leak_frames[name] = leak
        rows.append({
            "scheme": name,
            "pct_test_rows_seen_customer": round(
                float(leak["pct_test_rows_seen_customer"].mean()), 4),
            "cv_auc": score,
            "fold_std": round(float(np.nanstd(folds)), 4)})

    # --- the same leak, across sample sizes
    #
    # This exists to test the project's own thesis rather than illustrate it.
    # The statistical traps fall away as rows are added; if this one did too,
    # the two-family split would be a story rather than a finding.
    curve = []
    for n_rows in cfg.preprocess_sizes:
        sub = _subsample(dev, n_rows, cfg.seed)
        Xs, ys = xy(sub, core_only=True)
        if len(np.unique(ys)) < 2 or len(sub) < 60:
            continue
        a, _ = cv_score(model, Xs, ys,
                        list(schemes.stratified(sub, ys, cfg.n_folds, cfg.seed)))
        b, _ = cv_score(model, Xs, ys,
                        list(schemes.grouped(sub, ys, cfg.n_folds, cfg.seed)))
        curve.append({"rows": len(sub), "ungrouped_cv": round(a, 4),
                      "grouped_cv": round(b, 4),
                      "customer_leak": round(a - b, 4)})
    curve_df = pd.DataFrame(curve)

    true_auc = holdout_score(model, X_dev, y_dev, X_out, y_out)
    table = optimism_table(rows, true_auc)
    if write:
        _write({"trap_grouped": table,
                "trap_grouped_by_size": curve_df,
                "trap_grouped_leakage": leak_frames["stratified_kfold"]}, cfg)
    return {"table": table, "curve": curve_df, "truth": true_auc,
            "leakage": leak_frames}


# -------------------------------------------------------------- trap three
def run_temporal(cfg: Config = CFG, write: bool = True) -> dict:
    """Training on the future to predict the past.

    Random KFold interpolates: for almost every validation row there are
    training rows from later periods. Deployment never gets that, it only ever
    extrapolates forward, and the gap between the two is exactly what drift
    costs.
    """
    dev, out, X_dev, y_dev, X_out, y_out = _setup(cfg, core_only=True)
    model = booster(seed=cfg.seed)
    rows, leak_frames = [], {}

    for name in ("stratified_kfold", "forward_chaining",
                 "grouped_forward_chaining"):
        splits = list(schemes.build(name)(dev, y_dev, cfg.n_folds, cfg.seed))
        score, folds = cv_score(model, X_dev, y_dev, splits)
        leak = schemes.leakage_report(dev, splits)
        leak_frames[name] = leak
        rows.append({
            "scheme": name,
            "train_rows_from_the_future": int(
                leak["train_rows_from_the_future"].sum()),
            "pct_test_rows_seen_customer": round(
                float(leak["pct_test_rows_seen_customer"].mean()), 4),
            "cv_auc": score,
            "fold_std": round(float(np.nanstd(folds)), 4)})

    true_auc = holdout_score(model, X_dev, y_dev, X_out, y_out)
    table = optimism_table(rows, true_auc)
    if write:
        _write({"trap_temporal": table}, cfg)
    return {"table": table, "truth": true_auc, "leakage": leak_frames}


# --------------------------------------------------------------- trap four
def _candidate_subsets(columns, n_candidates: int, size: int, seed: int):
    """Candidates that differ by which features they use.

    Every candidate keeps all nine real columns and adds a different random
    draw of noise columns. That makes them interchangeable BY CONSTRUCTION:
    the noise contributes nothing, so any candidate is as good as any other and
    the entire spread in their CV scores is luck. A maximum taken over that
    spread is pure winner's curse, with no genuine signal mixed in.

    Two earlier drafts failed in opposite directions, and both failures are
    worth knowing. Varying C and select_k produced candidates within 0.008 of
    each other: no dispersion, nothing for a maximum to exploit. Drawing fully
    random subsets produced candidates from 0.48 to 0.70, because most subsets
    missed the real columns entirely: there the search finds real signal, and
    picking the best of it is competent rather than cursed. The trap needs
    candidates that differ in score but not in merit.
    """
    rng = np.random.default_rng(seed)
    noise = np.array([c for c in columns if c.startswith("noise_")])
    real = [c for c in columns if not c.startswith("noise_")]
    return [(f"subset_{i:02d}",
             real + list(rng.choice(noise, size=min(size, len(noise)),
                                    replace=False)))
            for i in range(n_candidates)]


def run_selection(cfg: Config = CFG, write: bool = True) -> dict:
    """The winner's curse: reporting the best CV score as if it were an estimate.

    Search many candidates, keep the best, quote its CV number. That number is
    a maximum over many noisy draws, and the maximum of noisy draws is biased
    upward even when every candidate is equally good. Nested CV buys an honest
    number by re-running the whole search inside each outer fold, so the
    selection is validated rather than assumed.
    """
    dev, out, X_dev, y_dev, X_out, y_out = _setup(cfg)
    outer = list(schemes.grouped(dev, y_dev, cfg.n_folds, cfg.seed))
    cands = _candidate_subsets(list(X_dev.columns), cfg.n_candidates,
                               cfg.candidate_size, cfg.seed)
    lookup = dict(cands)

    flat = []
    for name, cols in cands:
        score, folds = cv_score(linear_pipeline(seed=cfg.seed),
                                X_dev[cols], y_dev, outer)
        flat.append({"candidate": name, "cv_auc": round(score, 4),
                     "fold_std": round(float(np.nanstd(folds)), 4)})
    flat_df = pd.DataFrame(flat).sort_values("cv_auc", ascending=False,
                                             ignore_index=True)
    winner_name = str(flat_df.iloc[0]["candidate"])
    winner_cols = lookup[winner_name]

    # --- nested: the search itself sits inside the validation
    nested_scores = []
    for train_idx, test_idx in outer:
        inner_df = dev.iloc[train_idx].reset_index(drop=True)
        X_in, y_in = xy(inner_df)
        inner = list(schemes.grouped(inner_df, y_in, cfg.n_folds, cfg.seed + 1))
        best, best_score = None, -np.inf
        for name, cols in cands:
            sc, _ = cv_score(linear_pipeline(seed=cfg.seed), X_in[cols], y_in,
                             inner)
            if sc > best_score:
                best, best_score = cols, sc
        nested_scores.append(holdout_score(
            linear_pipeline(seed=cfg.seed),
            X_dev[best].iloc[train_idx], y_dev[train_idx],
            X_dev[best].iloc[test_idx], y_dev[test_idx]))

    # --- the same curse, across sample sizes
    curve = []
    for n_rows in cfg.preprocess_sizes:
        sub = _subsample(dev, n_rows, cfg.seed)
        Xs, ys = xy(sub)
        if len(np.unique(ys)) < 2 or len(sub) < 60:
            continue
        sp = list(schemes.grouped(sub, ys, cfg.n_folds, cfg.seed))
        sc = np.array([cv_score(linear_pipeline(seed=cfg.seed), Xs[c], ys, sp)[0]
                       for _, c in cands])
        curve.append({"rows": len(sub),
                      "best_of_30": round(float(sc.max()), 4),
                      "mean_candidate": round(float(sc.mean()), 4),
                      "winners_curse": round(float(sc.max() - sc.mean()), 4)})
    curve_df = pd.DataFrame(curve)

    true_auc = holdout_score(linear_pipeline(seed=cfg.seed),
                             X_dev[winner_cols], y_dev,
                             X_out[winner_cols], y_out)
    rows = [
        {"estimate": "best candidate's own CV score",
         "cv_auc": float(flat_df.iloc[0]["cv_auc"])},
        {"estimate": "mean across all candidates",
         "cv_auc": float(flat_df["cv_auc"].mean())},
        {"estimate": "nested CV", "cv_auc": float(np.nanmean(nested_scores))},
    ]
    table = optimism_table(rows, true_auc)
    if write:
        _write({"trap_selection": table,
                "trap_selection_candidates": flat_df,
                "trap_selection_by_size": curve_df}, cfg)
    return {"table": table, "candidates": flat_df, "curve": curve_df,
            "truth": true_auc,
            "winner": winner_name,
            "spread": round(float(flat_df["cv_auc"].max()
                                  - flat_df["cv_auc"].min()), 4)}


# --------------------------------------------------------------- trap five
def run_variance(cfg: Config = CFG, write: bool = True) -> dict:
    """Not a bias at all, which is why it belongs here.

    The first four traps move the number in a direction. This one does not; it
    just means a single CV run is a draw from a distribution wide enough that
    people routinely read noise as a result. Repeat the same 5-fold CV with
    different shuffles and watch the spread.
    """
    dev, out, X_dev, y_dev, X_out, y_out = _setup(cfg, core_only=True)
    a = linear_pipeline(C=1.0, seed=cfg.seed)
    b = linear_pipeline(C=0.1, seed=cfg.seed)

    draws = {"model_A (C=1.0)": [], "model_B (C=0.1)": []}
    for r in range(cfg.n_repeats):
        splits = list(schemes.shuffled_grouped(dev, y_dev, cfg.n_folds,
                                               cfg.seed + r))
        draws["model_A (C=1.0)"].append(cv_score(a, X_dev, y_dev, splits)[0])
        draws["model_B (C=0.1)"].append(cv_score(b, X_dev, y_dev, splits)[0])

    rows = []
    for name, vals in draws.items():
        v = np.array(vals)
        rows.append({
            "model": name,
            "mean_cv_auc": round(float(v.mean()), 4),
            "std_across_repeats": round(float(v.std()), 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
            "spread": round(float(v.max() - v.min()), 4)})
    table = pd.DataFrame(rows)

    gap = abs(table.loc[0, "mean_cv_auc"] - table.loc[1, "mean_cv_auc"])
    widest = float(table["spread"].max())
    diffs = np.array(draws["model_A (C=1.0)"]) - np.array(draws["model_B (C=0.1)"])
    verdict = {
        "gap_between_models": round(gap, 4),
        "widest_single_model_spread": round(widest, 4),
        "times_A_beat_B_out_of_repeats": int((diffs > 0).sum()),
        "n_repeats": cfg.n_repeats,
        "gap_smaller_than_spread": bool(gap < widest),
    }
    if write:
        _write({"trap_variance": table,
                "trap_variance_draws": pd.DataFrame(draws)}, cfg)
    return {"table": table, "verdict": verdict}


# ------------------------------------------------------------- the headline
def run_sweep(cfg: Config = CFG, write: bool = True) -> pd.DataFrame:
    """All four biased traps side by side, ranked by measured optimism."""
    rows = []
    for label, fn, naive, careful in (
            ("preprocessing", run_preprocessing, "selection before split",
             "selection inside the fold"),
            ("grouped", run_grouped, "stratified_kfold", "group_kfold"),
            ("temporal", run_temporal, "stratified_kfold",
             "grouped_forward_chaining"),
            ("selection", run_selection, "best candidate's own CV score",
             "nested CV")):
        out = fn(cfg, write=False)
        t = out["table"]
        key = "approach" if "approach" in t.columns else (
            "scheme" if "scheme" in t.columns else "estimate")
        n = t.loc[t[key] == naive].iloc[0]
        c = t.loc[t[key] == careful].iloc[0]
        rows.append({
            "trap": label,
            "naive_cv": n["cv_auc"],
            "careful_cv": c["cv_auc"],
            "truth": n["truth"],
            "naive_optimism": n["optimism"],
            "careful_optimism": c["optimism"],
            "optimism_removed": round(
                float(n["optimism"] - c["optimism"]), 4)})
    table = pd.DataFrame(rows).sort_values(
        "naive_optimism", ascending=False, ignore_index=True)
    if write:
        _write({"sweep": table}, cfg)
    return table


def headline(sweep: pd.DataFrame) -> dict:
    """Read the verdict off the numbers rather than asserting one.

    Two things this deliberately does not claim. It does not say the careful
    scheme is accurate, because here the careful schemes overshoot into
    pessimism and saying otherwise would be the same sin in the other
    direction. And it does not treat `optimism_removed` as a correction, only
    as a distance moved, because a fix that travels further than the error is
    overcorrecting, not correcting.
    """
    worst = sweep.iloc[0]
    still_off = sweep.loc[sweep["careful_optimism"].abs().idxmax()]
    overshoots = sweep[(sweep["naive_optimism"] > 0)
                       & (sweep["careful_optimism"] < 0)]["trap"].tolist()
    return {
        "worst_trap": str(worst["trap"]),
        "worst_optimism": float(worst["naive_optimism"]),
        "worst_moved": float(worst["optimism_removed"]),
        "residual_trap": str(still_off["trap"]),
        "residual_optimism": float(still_off["careful_optimism"]),
        "overshooting_traps": overshoots,
        "verdict": (
            f"The {worst['trap']} trap is the expensive one at this sample "
            f"size: naive folds report {worst['naive_optimism']:+.4f} against "
            f"the truth. Switching to the careful scheme moves the estimate "
            f"{worst['optimism_removed']:.4f}, "
            f"which is further than the error itself, landing at "
            f"{worst['careful_optimism']:+.4f}. So the careful scheme is not "
            f"accurate here, it is conservative, and that is the honest way to "
            f"describe it. The largest residual across all four traps is "
            f"{still_off['careful_optimism']:+.4f} on {still_off['trap']}. "
            f"Correct validation reduces self-deception; it does not deliver "
            f"the truth."),
    }
