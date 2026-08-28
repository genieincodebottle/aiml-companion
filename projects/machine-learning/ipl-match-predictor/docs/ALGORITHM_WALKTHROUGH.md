# IPL Match Predictor — Algorithm Walkthrough

This is a step-by-step walkthrough of how the prediction algorithm works, using **two real IPL 2026 matches** as case studies. The numbers come from running [`scripts/predict_all_2026.py`](../scripts/predict_all_2026.py) on the production data export — they are NOT hand-curated.

Pair this doc with [`notebooks/algorithm_walkthrough.ipynb`](../notebooks/algorithm_walkthrough.ipynb) for the runnable version.

## Bugs fixed in this version

Three bugs in the original pipeline are now fixed in `src/models.py`:

1. **Silent fallback in `predict_match()`** — built a 17-feature dict but trained models expected 46; bare `except Exception:` caught the resulting `ValueError` and silently fell back to pure Elo expectation, producing inflated accuracy numbers (~65%) that weren't from the trained ensemble at all. **Fix:** added one-hot `team1_X`, `team2_X`, `toss_decision_field` columns + reindex to training schema. Real failures now surface as exceptions.

2. **Double-scaling skew** — `prepare_features()` applied `StandardScaler` externally, then each pipeline applied `StandardScaler` again internally. Models learned from doubly-transformed features, then broke at inference when handed raw input (LR could saturate to 0/1). **Fix:** `prepare_features(scale_numerical=False)` is now the default. Each pipeline's internal scaler is the single source of standardization.

3. **Elo target-row leakage in `predict_match()`** — recomputed Elo by iterating the entire `matches_df`. If the target match was in that df with its winner filled, Elo reflected that outcome (using M71's result to predict M71). **Fix:** added `as_of_date` parameter; when provided, filters `matches_df` to matches strictly before that date.

**True ensemble accuracy on IPL 2026 (post-fix): 34/70 = 48.6%** — still barely better than a coin flip. The model is genuinely weak on this dataset; that number isn't a bug, it's the real performance.

## Why two case studies?

| Case | Match | Result |
|------|-------|--------|
| **1** (wrong, low confidence) | M71 Qualifier 1: RCB vs GT, 26 May 2026 | Ensemble said GT 64% — RCB won by 92 runs [FAIL] |
| **2** (wrong, moderate confidence) | M10: SRH vs LSG, 5 Apr 2026 | Ensemble said SRH 61% — LSG won by 5 wickets [FAIL] |

Both case studies are predictions the model got wrong. This is intentional — the model is wrong on 36 of 70 matches, so two wrong cases are representative. The post-mortems show **why** the trained ensemble is barely better than random, and where a learner can intervene to improve it.

---

## The 7 stages

```
1. Load        ← src/data_loader.py    pandas reads matches.csv (2008-2025) + matches_2026.csv
2. Clean       ← src/data_loader.py    standardize team names, impute nulls
3. Elo         ← src/features.py       chronological K=32 rating updates across every match
4. Features    ← src/features.py       momentum, H2H, home, toss, venue chase bias
5. Train       ← src/models.py         4-model ensemble (RF + XGB + GB + LR), uncalibrated
6. Predict     ← src/models.py         ensemble_predict_proba() on the 46-feature row
7. Output      ←                       weighted average → pick + confidence
```

Note: this script does NOT use `CalibratedClassifierCV` (sklearn <1.5 has a known dtype bug in `_sigmoid_calibration`). Uncalibrated probabilities work fine for the walkthrough.

---

# Case 1 — RCB vs GT (Qualifier 1, 26 May 2026)

**Setup:** Top-of-table RCB (20 pts) vs runners-up GT (18 pts) at the neutral playoff venue HPCA Stadium, Dharamshala. GT won the toss and chose to field. The model has zero direct knowledge of this match; it predicts using only data through 25 May 2026.

## Stage 1 — Load

```python
historical = load_matches()          # 1,169 matches, 2008-04-18 → 2025-06-03
season_2026 = pd.read_csv("data/raw/matches_2026.csv")  # 72 matches
combined = pd.concat([historical, season_2026]).sort_values("date")
```

After loading, `combined` has **1,241 matches** across 18 seasons.

## Stage 2 — Clean

Standardizes the 4 franchise renames (e.g., `Royal Challengers Bengaluru` → `Royal Challengers Bangalore`), imputes nulls, fills missing toss data with `team1` + `field` defaults (the 2026 export has toss data for only 12 of 71 completed matches).

## Stage 3 — Elo ratings

[Elo](https://en.wikipedia.org/wiki/Elo_rating_system) is a self-correcting rating system. The algorithm walks through **all 1,240 matches chronologically** before M71 and updates Elo after each:

```python
exp_team1 = 1 / (1 + 10 ** ((elo_team2 - elo_team1) / 400))
elo_team1_new = elo_team1 + 32 * (actual_team1_won - exp_team1)
```

By 26 May 2026 (just before M71), the **pre-match** snapshot is:

| Team | Elo | Interpretation |
|------|-----|----------------|
| Royal Challengers Bangalore | **1610.5** | Strong, top of season |
| Gujarat Titans | **1597.8** | Strong, runners-up |
| Elo difference | +12.7 | Tiny edge to RCB |
| Elo-expected P(RCB win) | **51.8%** | Practically a coin flip |

## Stage 4 — Feature engineering

For this specific match, the algorithm builds the 17-numerical-feature core (plus ~29 one-hot team/toss columns):

| Feature | Value for M71 | What it means |
|---------|--------------|---------------|
| `elo_team1` | 1610.5 | Pre-match RCB rating |
| `elo_team2` | 1597.8 | Pre-match GT rating |
| `elo_diff` | +12.7 | RCB slight edge |
| `elo_expected` | 0.5182 | Elo-implied P(RCB win) |
| `momentum_team1` | 0.60 | RCB won 3 of last 5 |
| `momentum_team2` | **0.80** | GT won 4 of last 5 — hotter |
| `momentum_diff` | -0.20 | GT has momentum edge |
| `h2h_team1_winrate` | 0.50 | Even split historically |
| `h2h_matches` | 8 | Small sample |
| `home_team1` | 0 | RCB not at home |
| `home_team2` | 0 | GT not at home (neutral venue) |
| `toss_winner_is_team1` | 0 | GT won the toss |
| `toss_chose_field` | 1 | GT chose to bowl |
| `venue_chase_bias` | 0.50 | Dharamshala neutral for bat/chase |

## Stage 5 — Train

The 4-model ensemble was trained **once** on 1,169 pre-2026 matches with `recency_weight` exponential decay. Training takes ~2 seconds. Uncalibrated.

## Stage 6 — Predict

Each model scores the M71 feature vector (46 columns including one-hot encodings):

| Model | P(RCB wins) | Weight | Contribution |
|-------|-------------|--------|--------------|
| Random Forest | **43.1%** | 0.30 | 12.9 |
| XGBoost | **28.7%** | 0.35 | 10.1 |
| Gradient Boosting | **46.3%** | 0.20 | 9.3 |
| Logistic Regression | **23.4%** | 0.15 | 3.5 |
| **Weighted ensemble** | **35.7%** | — | — |

The per-model numbers are now genuinely different (RF and GB are close to coin flip; XGB and LR strongly lean GT).

## Final output for M71

```
Pick: Gujarat Titans
Confidence: 64% (= 100 - 35.7)
Actual: Royal Challengers Bangalore (won by 92 runs)
Outcome: [FAIL] WRONG
```

**Reading the result:** The model used Elo (slight RCB edge) + GT's momentum (+20%) and ended up confident in GT. In reality, RCB posted 254 and dismantled GT for 162. The model's signals — Elo nearly equal, GT momentum hot, neutral venue — were all defensible. They were just wrong about this specific match.

---

# Case 2 — SRH vs LSG (Match 10, 5 Apr 2026)

**Setup:** SRH at home at Rajiv Gandhi Intl Stadium. LSG won toss + chose to field.

## Feature vector for M10

| Feature | Value | Signal |
|---------|-------|--------|
| `elo_team1` (SRH) | 1519.3 | |
| `elo_team2` (LSG) | 1485.0 | |
| `elo_diff` | +34.3 | Strong SRH edge |
| `elo_expected` | 0.5492 | Elo: SRH 55% |
| `momentum_team1` (SRH) | **0.80** | SRH won 4 of last 5 |
| `momentum_team2` (LSG) | 0.20 | LSG lost 4 of last 5 |
| `momentum_diff` | **+0.60** | Huge SRH momentum edge |
| `h2h_team1_winrate` | **0.33** | LSG won 4 of 6 historically |
| `h2h_matches` | 6 | Small but consistent |
| `home_team1` (SRH) | **1** | SRH home advantage |
| `home_team2` (LSG) | 0 | |
| `toss_winner_is_team1` | 0 | LSG won toss |
| `toss_chose_field` | 1 | LSG bowling first |
| `venue_chase_bias` | 0.50 | Neutral |

## Ensemble output

| Model | P(SRH wins) | Weight |
|-------|-------------|--------|
| Random Forest | **59.0%** | 0.30 |
| XGBoost | **71.7%** | 0.35 |
| Gradient Boosting | **47.7%** | 0.20 |
| Logistic Regression | **58.8%** | 0.15 |
| **Weighted ensemble** | **61.1%** | — |

GB was the lone voice saying "actually LSG" (47.7%) but got outvoted. XGB was particularly confident in SRH (71.7%) — likely because it overweighted the +60% momentum gap.

## Final output for M10

```
Pick: Sunrisers Hyderabad
Confidence: 61%
Actual: Lucknow Super Giants (won by 5 wickets, chase)
Outcome: [FAIL] WRONG
```

## What went wrong?

| Signal | What the model saw | What actually mattered |
|--------|--------------------|------------------------|
| **Momentum (+60%)** | "SRH on a heater" | Last-5 form too small a window to override... |
| **H2H (LSG 67%)** | "...but only 6 matches, mostly noise" | ...LSG's structural matchup advantage |
| **Home (+SRH)** | "SRH at home, +3-5% bump" | LSG bowled first under dew — chase advantage flipped the home factor |
| **Toss/decision** | "Bowling first = +small chase edge" | Was the strongest signal in retrospect |

---

# The bigger picture — model accuracy

When run end-to-end on all 70 resolved IPL 2026 matches:

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Random coin flip | 50.0% | Baseline |
| Always pick higher-Elo team | ~58-60% | Single-feature heuristic |
| **Trained 4-model ensemble (this pipeline)** | **34 / 70 = 48.6%** | Real reproducible number |
| Hand-curated archive (expert overlay) | 43 / 70 = 61.4% | What was actually posted to live site |

**The trained ensemble does NOT beat naive Elo.** It performs *at random*. The hand-curated archive — which uses contextual reasoning on top of a similar feature set — substantially beats the model.

Why might the trained ensemble underperform?

1. **Feature set is too thin** — 17 features can't capture playing-XI changes, weather, dew, pitch reports, key-player matchups, captaincy, injury rotations
2. **Sample size is small** — 1,169 training matches across 18 seasons means each team appears ~120 times; in T20 cricket, team identity drifts a lot in that window
3. **Recency vs depth tension** — `recency_weight` upweights recent seasons, but 2025 has only ~70 matches per team, not enough for the model to lock onto current strengths
4. **No interaction encoding** — the 4-model ensemble can't see "this RCB lineup has Kohli + Patidar + Hazlewood" because team identity is just a one-hot column

The 12-percentage-point gap (61.4% vs 48.6%) is roughly the value of human contextual reasoning. That's also the target a learner should try to close.

---

# Try it yourself

```bash
cd ipl-match-predictor
pip install -r requirements.txt
python scripts/predict_all_2026.py
```

Expected: `ML ensemble accuracy on IPL 2026: 34 / 70 = 48.6%`. Two matches unresolved (M12 washout, M72 still upcoming).

## Where to read the actual code

| To understand... | Read this file |
|------------------|----------------|
| How Elo updates work | [`src/features.py`](../src/features.py) → `compute_elo_ratings()` |
| How momentum / H2H / venue features are built | [`src/features.py`](../src/features.py) → `engineer_features()` |
| How the 4 models combine | [`src/models.py`](../src/models.py) → `build_ensemble()`, `ensemble_predict_proba()` |
| Single-match predict helper (CURRENTLY BROKEN) | [`src/models.py`](../src/models.py) → `predict_match()` — see honesty note at top |
| Full season runner (correct ensemble usage) | [`scripts/predict_all_2026.py`](../scripts/predict_all_2026.py) |

## Next steps for learners

1. **Run the pipeline**, prove the 48.6% number (verify the model really is bad)
2. **Pick 3 high-confidence wrong picks** (e.g., M11 RCB-vs-CSK 78%, M22 CSK-vs-KKR 72%, M23 RCB-vs-LSG 76%) — hypothesize why each failed
3. **Add a feature** to `src/features.py` that you think captures something the current model misses (toss × venue interaction, win-streak-against-this-opponent, captaincy, day-of-week)
4. **Re-run** — did accuracy go up?
5. **Fix `predict_match()`** — the one-hot encoding mismatch is a real bug; building the proper 46-feature vector at predict time would let the live single-match API work too
6. **Try to beat 61.4%** — that's the hand-curated archive's score (see [`predictions-2026/INDEX.md`](../predictions-2026/INDEX.md))
