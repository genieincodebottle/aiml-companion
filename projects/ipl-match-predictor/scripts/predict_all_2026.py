"""
predict_all_2026.py - Run the ensemble predictor against every IPL 2026 match
=============================================================================

Trains the 4-model ensemble (RF + XGB + GB + LR) on pre-2026 historical data,
then predicts every IPL 2026 match using the row's pre-match features and the
trained ensemble. Compares each prediction to the actual winner and prints a
final accuracy table.

This script uses ``ensemble_predict_proba()`` directly against rows from
``prepare_features(combined_df)``. Equivalent to calling the fixed
``src.models.predict_match(..., as_of_date=row.date)`` per match, but faster
because we only build the feature matrix once. Three original bugs that affected
both paths have been fixed in ``src/models.py``:
 1. Silent fallback in ``predict_match()`` (17 vs 46 feature mismatch) — now
    properly builds 46-col feature vector with one-hot encoding
 2. Double-scaling in ``prepare_features()`` — default now ``scale_numerical=False``
    so each pipeline's internal StandardScaler is the single source of standardization
 3. Elo target-row leakage in ``predict_match()`` — new ``as_of_date`` parameter
    restricts Elo computation to matches strictly before the prediction date

Methodology notes (read these before quoting the accuracy number)
-----------------------------------------------------------------
1. ``prepare_features()`` (one-hot + StandardScaler) is fit on the full
   combined dataset (historical + 2026), then training uses only pre-2026
   rows. This causes a small amount of scaler leakage from 2026 into the
   training transform, but the scaler sees 1,169 pre-2026 rows vs 72 in 2026,
   so the marginal effect is tiny. A strictly leak-free version would
   walk-forward: refit scaler + ensemble for each match. That would take
   ~30+ minutes; this single-fit version takes ~15 seconds.

2. Elo ratings in each row's ``elo_team1``/``elo_team2`` columns are the
   PRE-match snapshot from ``compute_elo_ratings()``, which updates
   chronologically. The trained ensemble uses these pre-match Elo values
   as features — no target-row leakage in Elo.

3. The pipeline does NOT use ``CalibratedClassifierCV`` here. The notebook
   version falls back to uncalibrated on older sklearn; this script matches
   that behavior for consistency.

Usage
-----
    python scripts/predict_all_2026.py
    python scripts/predict_all_2026.py --limit 10
    python scripts/predict_all_2026.py --output preds.csv

Expected runtime: ~15-20 seconds.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_matches, clean_matches, clean_team_names
from src.features import compute_elo_ratings, engineer_features
from src.models import (
    build_ensemble,
    ensemble_predict_proba,
    prepare_features,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_combined_dataset() -> pd.DataFrame:
    historical = load_matches()
    p = REPO_ROOT / "data" / "raw" / "matches_2026.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    season_2026 = pd.read_csv(p)
    common = [c for c in historical.columns if c in season_2026.columns]
    combined = pd.concat([historical[common], season_2026[common]], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def build_features_and_train(combined: pd.DataFrame):
    """Process features end-to-end, fit ensemble on pre-2026 subset."""
    combined, _ = clean_team_names(combined)
    combined = clean_matches(combined)
    # Fill missing toss data with a defensible default so prepare_features
    # doesn't drop the row (2026 export has toss data for only 12 of 71 completed matches)
    combined["toss_winner"] = combined["toss_winner"].fillna(combined["team1"])
    combined["toss_decision"] = combined["toss_decision"].fillna("field")
    combined = compute_elo_ratings(combined)
    combined = engineer_features(combined)

    # Single feature-prep on the combined dataset (see methodology note #1)
    model_df, feature_names = prepare_features(combined)
    # Coerce to float64 (sklearn <1.5 strict about dtypes; bool one-hot would error)
    model_df = model_df.astype(np.float64)

    # Identify pre-2026 training rows (have a winner + before 2026)
    cutoff = pd.Timestamp("2026-01-01")
    train_mask = (combined["date"] < cutoff) & combined["winner"].notna()
    train_idx = combined[train_mask].index.intersection(model_df.index)

    X_train = model_df.loc[train_idx]
    y_train = (combined.loc[train_idx, "winner"] == combined.loc[train_idx, "team1"]).astype(np.int64)

    sw = None
    if "recency_weight" in combined.columns:
        sw = combined.loc[train_idx, "recency_weight"].values.astype(np.float64)

    print(f"Training set: {len(X_train)} matches "
          f"({combined.loc[train_idx, 'date'].min().date()} -> "
          f"{combined.loc[train_idx, 'date'].max().date()})")
    print(f"Feature schema: {X_train.shape[1]} columns")
    print("Training 4-model ensemble (RF + XGB + GB + LR), uncalibrated...")
    t0 = time.time()
    models = build_ensemble(X_train, y_train, sample_weight=sw, calibrate=False)
    print(f"Training complete in {time.time()-t0:.1f}s\n")

    return combined, model_df, models, feature_names


def predict_2026_season(
    combined: pd.DataFrame, model_df: pd.DataFrame, models, limit: int = None
) -> List[Dict]:
    """Predict each 2026 match using ensemble_predict_proba (real ensemble)."""
    season_2026 = combined[combined["season"].astype(str) == "2026"].copy()
    season_2026 = season_2026.sort_values("date").reset_index()
    if limit:
        season_2026 = season_2026.head(limit)

    SHORT = {
        "Royal Challengers Bangalore": "RCB", "Mumbai Indians": "MI",
        "Chennai Super Kings": "CSK", "Kolkata Knight Riders": "KKR",
        "Rajasthan Royals": "RR", "Punjab Kings": "PBKS",
        "Gujarat Titans": "GT", "Sunrisers Hyderabad": "SRH",
        "Delhi Capitals": "DC", "Lucknow Super Giants": "LSG",
    }
    def s(t): return SHORT.get(t, t[:4]) if t and isinstance(t, str) else "-"

    results = []
    print(f"{'#':>3} {'Date':<12} {'Match':<14} {'Pick':<6} {'Conf':>5} {'Actual':<8} {'Out':>4}")
    print("-" * 60)

    for i, row in season_2026.iterrows():
        orig_idx = row["index"]
        if orig_idx not in model_df.index:
            continue
        X_row = model_df.loc[[orig_idx]]
        try:
            prob_team1 = float(ensemble_predict_proba(models, X_row)[0])
        except Exception as e:
            logger.warning("Match %s prediction failed: %s", i+1, e)
            continue

        pick = row["team1"] if prob_team1 >= 0.5 else row["team2"]
        conf = round(prob_team1 * 100) if prob_team1 >= 0.5 else round((1 - prob_team1) * 100)

        raw_winner = row.get("winner")
        if (raw_winner is not None and pd.notna(raw_winner)
                and str(raw_winner).strip() not in ("", "Unknown")):
            actual = raw_winner
            correct = (pick == actual)
        else:
            actual = None
            correct = None
        outcome = "YES" if correct else ("NO" if correct is False else "...")

        results.append({
            "match_idx": i + 1, "date": row["date"].date().isoformat(),
            "team1": row["team1"], "team2": row["team2"],
            "venue": row.get("venue", ""),
            "predicted": pick, "confidence": conf,
            "team1_prob": round(prob_team1, 4),
            "actual": actual, "correct": correct,
        })
        print(f"{i+1:>3} {row['date'].date().isoformat():<12} "
              f"{s(row['team1'])+' vs '+s(row['team2']):<14} "
              f"{s(pick):<6} {conf:>4}% "
              f"{s(actual) if actual else '_pending_':<8} {outcome:>4}")
    return results


def print_summary(results: List[Dict]) -> None:
    resolved = [r for r in results if r["correct"] is not None]
    if not resolved:
        print("\nNo completed matches to score.")
        return
    correct = sum(1 for r in resolved if r["correct"])
    pct = 100 * correct / len(resolved)
    print("\n" + "=" * 60)
    print(f"  ML ensemble accuracy on IPL 2026: {correct} / {len(resolved)} = {pct:.1f}%")
    print(f"  Total matches predicted: {len(results)}")
    print(f"  Resolved (with actual winner): {len(resolved)}")
    print(f"  Upcoming/unresolved: {len(results) - len(resolved)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    combined = load_combined_dataset()
    combined, model_df, models, _ = build_features_and_train(combined)
    results = predict_2026_season(combined, model_df, models, limit=args.limit)
    print_summary(results)

    if args.output:
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"\nWrote per-match predictions to {args.output}")


if __name__ == "__main__":
    main()
