# IPL Match Predictor

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

End-to-end ML pipeline that predicts IPL match outcomes using 18 seasons of historical data (2008-2025). Combines Elo rating system, statistical hypothesis testing, and a 4-model weighted ensemble (Random Forest, XGBoost, Gradient Boosting, Logistic Regression) validated with 10,000-match Monte Carlo simulation.

**What this project demonstrates:** The complete ML lifecycle - from raw data to production-ready predictions - using 1,169 matches (1,161 with a decided winner) and 260,920 ball-by-ball records (the ball-by-ball file covers 2008-2024 only).

## Architecture

```
matches.csv + deliveries.csv
        |
   [Data Ingestion]       Multi-source: Kaggle / local / GitHub URL
        |
   [Data Cleaning]        4 franchise renames, null imputation, type casting
        |
   [EDA]                  12 interactive Plotly charts, 7 analysis categories
        |
   [Feature Engineering]   Elo (K=32), momentum, H2H, home, toss, venue chase bias
        |
   [Hypothesis Testing]   Binomial test (toss), chi-squared (decision impact)
        |
   [Model Training]       4-model calibrated ensemble (RF + XGB + GB + LR)
        |                  + CalibratedClassifierCV + season recency weighting
        |
   [Monte Carlo]          10,000-match simulation to validate prediction stability
        |
   [Evaluation]           Accuracy, F1, ensemble weights, feature importances
        |
   [Predict Match]        predict_match() for unseen future matches
```

## Key Results

> **Read the first three rows together.** Accuracy on a near-balanced binary
> problem means nothing on its own, and no model here reliably beats "always
> pick the same team". That is the honest result, and the sections below explain
> how it was arrived at rather than explaining it away.

Every number below is measured on the shipped `data/raw/matches.csv`
(2008-2025, 1,169 matches, 1,161 decided, 8 no-result) by `python main.py
--stage evaluate`, with the training seasons 2008-2023 (1,019 matches) and the
test seasons 2024-25 (142 matches).

| Metric | Value | Method |
|--------|-------|--------|
| Match winner accuracy | **0.500** | Calibrated, recency-weighted 4-model ensemble (RF + XGB + GB + LR), **out-of-time** split, test = 2024-25 |
| Majority-class baseline | **0.479** | Always predict team 1 (the more common training outcome) |
| Lift over baseline | **+0.021** | About 0.5 standard errors: within sampling noise, not evidence of skill |
| Single models, same test seasons | RF 0.451, XGB 0.486, Neural Net 0.500 (opt-in, `--neural-net`) | All within one standard error (about 0.042 for 142 matches) of the baseline |
| Model-selection CV (training seasons 2008-2023 only) | RF 0.5290, XGB 0.5172, GB 0.5089, LR 0.5373, Neural Net 0.5089 | 5 forward-in-time folds (expanding window), see "How the cross-validation is done" below. `main.py` picks among RF / XGB (/ Neural Net with `--neural-net`), so the pick is the Random Forest |
| Run-margin regression | CV MAE 23.39, R2 -0.467 | Gradient Boosting, same 5 forward folds on the 463 runs-wins of 2008-2023; predicting the mean margin gives MAE 20.35. On the 68 runs-wins of 2024-25: MAE 21.99, R2 -0.186, mean-margin baseline 20.81 |
| Monte Carlo validation | 10,000 sims | Gaussian noise (std=0.05) stability check |
| Toss advantage | Not significant | Two-sided binomial test: toss winner won 598 of 1,161 (51.5%), p=0.318 |
| Top predictive features | Elo features | `elo_team2`, `elo_team1`, `elo_diff`, `elo_expected` are ranks 1-4 in the Random Forest importances |
| Engineered features | 20+ | Elo, momentum, H2H, home, toss, venue chase bias, interactions |
| Probability calibration | CalibratedClassifierCV | Sigmoid method, cv=3 |
| Training weighting | Season recency | Exponential decay (0.5 to 1.0) |

`main.py` reports the single classifier with the best forward-CV accuracy on the
training seasons as "the CV pick": the Random Forest (CV 0.5290), which scores
0.4507 on the 2024-25 test seasons. `torch` is optional (commented out in
`requirements.txt`) and the Neural Net is opt-in with `python main.py
--neural-net`, so the default result does not depend on what is installed. With
`--neural-net` the pick is still the Random Forest (Neural Net CV 0.5089, test
0.5000). All of these are inside the noise of the 0.479 baseline, which is why
the headline row above is the ensemble, not the CV pick.

### Why the split is by season, not at random

The deliverable here is predictions for matches that have not been played, so
the test set has to sit in the *future* of the training set. A random split
trains on 2025 to predict 2024 -- something deployment never gets to do -- and
between seasons the squads, the rules and the team strengths all move, so
interpolating across them is an easier problem than the real one.

Test accuracy by holdout window (train = every season before the window):

| holdout | test matches | majority baseline | Random Forest | XGBoost | 4-model ensemble |
|---|---|---|---|---|---|
| 2025 only | 71 | 0.4648 | 0.4789 | 0.4789 | 0.4648 |
| **2024-25 (what `main.py` uses)** | 142 | **0.4789** | 0.4507 | 0.4859 | **0.5000** |
| 2023-25 | 215 | 0.5023 | 0.4744 | 0.5070 | 0.4558 |
| random 25% split (not used) | 291 | -- | 0.5258 | -- | 0.5120 |

With one standard error of 0.04 to 0.06 on windows this small, no model in any
window is distinguishable from "always predict the same team". The ensemble is
level with the baseline on 2025 alone, 2.1 points above it on 2024-25 and 4.7
points below it on 2023-25, which is what noise around zero looks like. The
random split flatters the Random Forest by about 7 points (0.5258 against 0.4507
on 2024-25) and the ensemble by about 1 point (0.5120 against 0.5000), so the gap
between the two evaluations is itself noisy. The point is not the size of the gap; it is that a random split answers a
question deployment never asks, and no baseline was printed next to it, which is
how 0.53 comes to sound like skill.

This also independently corroborates the 48.6% (34 of 70) this project
measures on the 2026 season below. Two different honest evaluations agree that
pre-match features do not predict IPL results, and that agreement is worth
more than either number alone.

> **Data vintages.** The shipped `matches.csv` is the 2008-2025 export (1,169
> rows). Earlier write-ups, the Kaggle notebook text and the blog diagrams used
> the 2008-2024 file (1,095 matches, 1,090 decided), where the toss test gave
> p=0.6066 (554 of 1,090, 50.8%). On the shipped 2008-2025 data it is p=0.3184.
> `deliveries.csv` still covers only the 1,095 matches of 2008-2024.

### How the cross-validation is done

Model selection uses **only the training seasons (2008-2023, 1,019 matches)** and
**forward-in-time folds**: `forward_cv(5)` in `src/models.py` is scikit-learn's
`TimeSeriesSplit(n_splits=5)`, an expanding window over the date-sorted rows.
`cross_validate_model()` runs it with `cross_val_score(pipeline, X_train, y_train,
cv=forward_cv(5))`, and `main.py` asserts the rows are date-sorted. The test
seasons (2024-25) are never seen by selection.

| fold | train rows (seasons) | validate rows (seasons) |
|---|---|---|
| 1 | 174 (2008-2010) | 169 (2010-2013) |
| 2 | 343 (2008-2013) | 169 (2013-2015) |
| 3 | 512 (2008-2015) | 169 (2015-2018) |
| 4 | 681 (2008-2018) | 169 (2018-2021) |
| 5 | 850 (2008-2021) | 169 (2021-2023) |

Folds are cut by row count, so a boundary can fall inside a season. The regressor
uses the same 5 forward folds on the 463 runs-wins of 2008-2023. Test accuracy
comes from a separate fit on all of 2008-2023, scored once on 2024-25.

Earlier versions ran `cross_val_score(cv=5)` (unshuffled `StratifiedKFold`) on all
1,161 matches, which trained on later seasons to predict earlier ones and let the
CV pick see the test seasons. Those numbers (RF 0.5125, XGB 0.4798, GB 0.5091,
LR 0.5168, regression MAE 21.80 / R2 -0.198) are retired; forward CV is lower
for the regressor and different by a few points for each classifier.

## Quick Start

### Option 1: Run locally (recommended)

```bash
# Clone and navigate
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/machine-learning/ipl-match-predictor

# Set up environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install and run
pip install -r requirements.txt
python main.py                    # Full pipeline
```

### Option 2: Kaggle Notebook

1. Open: [IPL Dataset Analysis on Kaggle](https://www.kaggle.com/code/genieincodebottle/ipl-dataset-analysis-2008-2025)
2. Click **"Copy & Edit"** (top-right)
3. Click **"Run All"** (`Shift + Ctrl + Enter`)

> After the first cell installs Plotly, restart the kernel for interactive charts.

### Option 3: Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Upload: `File > Upload notebook > notebooks/ipl-dataset-analysis.ipynb`
3. Run: `Runtime > Run all`

> Dataset auto-loads from GitHub. No manual download needed.

## CLI Usage

```bash
python main.py                      # Run full pipeline (all 7 stages)
python main.py --stage clean        # Data cleaning only
python main.py --stage eda          # EDA with chart export
python main.py --stage features     # Feature engineering
python main.py --stage hypothesis   # Statistical hypothesis tests
python main.py --stage train        # Model training + cross-validation
python main.py --stage evaluate     # Evaluation + report generation
python main.py --neural-net         # Also compare the optional PyTorch Neural Net (needs torch)
python main.py --verbose            # Debug-level logging
```

## Predict a Match

```python
from src.predict import run_prediction

result = run_prediction(
    team1="Mumbai Indians",
    team2="Chennai Super Kings",
    venue="Wankhede Stadium, Mumbai",
    city="Mumbai",
    toss_winner="Mumbai Indians",
    toss_decision="field",
    contextual_adjustment=0.0,   # Expert overlay (+/- probability)
    calibrate=True,              # CalibratedClassifierCV wrapping
    use_recency_weight=True,     # Season recency weighting
)

print(f"{result['winner']} wins with {result['confidence']}% confidence")
# Per-model probabilities: result['model_scores']
# Monte Carlo validation:  result['monte_carlo']
# Feature vector used:     result['features']
```

`run_prediction()` trains the ensemble on **every decided match** in the dataset
(no held-out split): the match being predicted is in the future, so there is
nothing later to test on. The out-of-time evaluation lives in `main.py`.

## Reproduce IPL 2026 Predictions

The repo ships with all 72 IPL 2026 matches as a CSV ([`data/raw/matches_2026.csv`](data/raw/matches_2026.csv)) plus a script that predicts the entire season end-to-end:

```bash
python scripts/predict_all_2026.py              # all 72 matches (~15s)
python scripts/predict_all_2026.py --limit 10   # first 10 only
python scripts/predict_all_2026.py --output preds.csv   # save per-match output
```

What it does:
1. Loads historical IPL matches (2008-2025) + the 2026 season export
2. Trains the 4-model ensemble (RF + XGB + GB + LR) on pre-2026 data
3. For each 2026 match, looks up the pre-match feature row and calls `ensemble_predict_proba()` directly (bypassing `src.models.predict_match()`, which has a known bug - see `docs/ALGORITHM_WALKTHROUGH.md`)
4. Compares each prediction to the actual winner and prints a final accuracy table

**Honest expected output:** `ML ensemble accuracy on IPL 2026: 34 / 70 = 48.6%` - barely better than a coin flip. The trained 4-model ensemble does NOT beat naive baselines on this season. The hand-curated archive (with expert contextual overlay) at 61.4% substantially outperforms the trained model. See [`predictions-2026/README.md`](predictions-2026/README.md) for the side-by-side and [`docs/ALGORITHM_WALKTHROUGH.md`](docs/ALGORITHM_WALKTHROUGH.md) for the step-by-step.

## Understand the Algorithm

For a learner-focused step-by-step walkthrough of how the prediction algorithm works - using two real IPL 2026 matches as case studies (both of which the model got wrong, intentionally - see why below):

- **Read:** [`docs/ALGORITHM_WALKTHROUGH.md`](docs/ALGORITHM_WALKTHROUGH.md) - static markdown walkthrough with real numbers from the pipeline
- **Run:** [`notebooks/algorithm_walkthrough.ipynb`](notebooks/algorithm_walkthrough.ipynb) - same content as runnable cells you can modify and re-execute

Both cover all 7 pipeline stages (load → clean → Elo → features → train → predict → output), the 14-feature vector for each match, the per-model ensemble scores (RF/XGB/GB/LR), and a post-mortem on why the model gets these matches wrong. Both case studies are wrong because the trained ensemble's overall accuracy is only **48.6%** - barely better than random. The walkthrough docs are upfront about this and explain where the headroom is for learners to add better features.

## Make Targets

```bash
make install    # Install dependencies
make run        # Full pipeline via CLI
make test       # Run 77 unit tests
make notebook   # Execute Jupyter notebook end-to-end
make clean      # Remove generated artifacts
make help       # Show all targets
```

## Project Structure

```
ipl-match-predictor/
├── main.py                                # CLI entry point - 7-stage pipeline
├── configs/
│   └── base.yaml                          # Dataset URLs, model hyperparams, feature flags
├── data/
│   └── raw/                               # CSVs auto-loaded from GitHub at runtime
├── notebooks/
│   └── ipl-dataset-analysis.ipynb         # Interactive analysis notebook (full pipeline)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Data loading, cleaning, team standardization
│   ├── eda.py                             # 8 EDA functions with Plotly charts
│   ├── features.py                        # Elo, momentum, H2H, home, toss, venue chase bias, recency
│   ├── hypothesis.py                      # Binomial + chi-squared hypothesis tests
│   ├── models.py                          # Ensemble (RF+XGB+GB+LR), calibration, predict_match, Monte Carlo
│   ├── predict.py                         # End-to-end prediction pipeline (run_prediction)
│   └── evaluate.py                        # Markdown report generation, console summary
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py                # Data loading/cleaning (19 tests)
│   ├── test_features.py                   # Elo + feature engineering (20 tests)
│   ├── test_hypothesis.py                 # Statistical tests (9 tests)
│   └── test_models.py                     # ML model pipelines, split, baseline, report (29 tests)
├── artifacts/
│   ├── results/key_findings.md            # Pre-documented key findings
│   └── figures/                           # Exported chart PNGs
├── docs/
│   └── analysis_summary.md               # Detailed analysis documentation
├── scripts/
│   └── run_analysis.sh                    # Notebook execution script
├── Makefile
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

| File | Records | Description |
|------|---------|-------------|
| `matches.csv` | 1,169 | Match-level data: teams, toss, result, venue, player of match (2008-2025); 1,161 have a decided winner |
| `deliveries.csv` | 260,920 | Ball-by-ball: batsman, bowler, runs, extras, wickets (2008-2024, 1,095 matches; none for 2025) |

Source: [Kaggle IPL Complete Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) + [IPL 2025 Dataset](https://www.kaggle.com/datasets/maratheabhishek/ipl-dataset-2008-to-2025)

> Dataset is auto-loaded from [GitHub](https://github.com/genieincodebottle/generative-ai/tree/main/docs/ipl-dataset). No manual download needed.

## ML Pipeline Details

### Data Cleaning

- **Team name standardization:** 4 franchise renames mapped to canonical names (e.g., Royal Challengers Bengaluru to Bangalore)
- **Missing values:** Winners for no-result matches, umpire names, cities, result margins - all systematically imputed
- **Derived columns:** `win_by_runs`, `win_by_wickets`, `toss_winner_won_match`

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `elo_team1/2` | Elo rating before the match (K=32, initial=1500) |
| `elo_diff` | Elo rating difference (team1 - team2) |
| `elo_expected` | Expected win probability from Elo |
| `momentum_team1/2` | Rolling win rate over last 5 matches |
| `momentum_diff` | Momentum difference between teams |
| `h2h_team1_winrate` | Historical head-to-head win rate |
| `h2h_matches` | Number of prior head-to-head encounters |
| `home_team1/2` | Binary home advantage indicator |
| `toss_winner_is_team1` | Whether team1 won the toss (known pre-match) |
| `toss_chose_field` | Whether toss winner chose to field |
| `venue_chase_bias` | Historical chase win% at the venue (no future leakage) |
| `elo_x_momentum_t1/t2` | Interaction: Elo * momentum per team |
| `elo_x_home_t1` | Interaction: Elo * home advantage |
| `recency_weight` | Exponential decay weight for training (0.5-1.0) |

### Models

| Task | Algorithm | Key Params | Ensemble Weight |
|------|-----------|------------|-----------------|
| Classification | Random Forest | 100 trees, max_depth=10, balanced weights | 30% |
| Classification | XGBoost | 200 estimators, L1/L2 regularization | 35% |
| Classification | Gradient Boosting | 150 stages, lr=0.1 | 20% |
| Classification | Logistic Regression | L2 regularization, baseline classifier | 15% |
| Calibration | CalibratedClassifierCV | Sigmoid method, cv=3 (wraps all 4 models) | - |
| Weighting | Season recency | Exponential decay: `0.5 + 0.5 * exp(-days/1500)` | - |
| Validation | Monte Carlo Simulation | 10,000 matches, Gaussian noise (std=0.05) | - |

All 4 classifiers use sklearn Pipelines with StandardScaler preprocessing and are compared with 5-fold forward-in-time cross-validation on the training seasons only (see "How the cross-validation is done"). Each model is optionally wrapped with `CalibratedClassifierCV` for well-calibrated probability estimates. Training uses season recency weighting so recent matches have more influence. Predictions are combined via weighted average, then validated with Monte Carlo simulation. The `predict_match()` function in `src/models.py` constructs feature vectors for unseen future matches.

### Hypothesis Testing

| Test | Method | Result |
|------|--------|--------|
| Toss advantage | Two-sided binomial (H0: p=0.5) | 598 of 1,161 (51.5%), p=0.318, NOT significant |
| Home advantage | One-sided binomial | Venue-dependent |
| Decision impact (bat vs field) | Chi-squared contingency | Analyzed per era |

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas, numpy |
| Visualization | Plotly (interactive), matplotlib, seaborn |
| ML | scikit-learn (RandomForest, GradientBoosting, LogisticRegression, Pipeline), xgboost |
| Statistics | scipy (binomial test, chi-squared) |
| Config | PyYAML |
| Testing | pytest |

## Configuration

All parameters are configurable via `configs/base.yaml`:

```yaml
features:
  elo:
    initial_rating: 1500
    k_factor: 32

models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    class_weight: "balanced"
  xgboost:
    n_estimators: 200
    learning_rate: 0.1
    reg_alpha: 0.1
    reg_lambda: 1.0
  gradient_boosting_classifier:
    n_estimators: 200
    learning_rate: 0.1
  logistic_regression:
    C: 1.0
    penalty: "l2"
  ensemble_weights:
    rf: 0.30
    xgb: 0.35
    gb: 0.20
    lr: 0.15
  monte_carlo:
    n_simulations: 10000
    noise_std: 0.05

feature_flags:
  use_elo_ratings: true
  use_momentum_features: true
  use_home_advantage: true
  use_toss_features: true
  use_venue_chase_bias: true
  use_recency_weight: true
  calibrate_models: true
  run_hypothesis_tests: true
```

## Testing

```bash
# Run all 77 tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Analysis Highlights

### Toss Has No Significant Advantage

Binomial test (two-sided) on the 1,161 decided matches of 2008-2025: the toss winner won 598 of them (51.5%), which does not deviate meaningfully from 50%. p-value = 0.318 (alpha = 0.05). On the older 2008-2024 file (1,090 decided matches) the same test gives 50.8% and p = 0.607. This is one of the most commonly asked IPL trivia questions, and the data provides a definitive answer.

### Elo Ratings Are the Strongest Predictor

The Elo features (`elo_team2`, `elo_team1`, `elo_diff`, `elo_expected`) take the top four Random Forest importances, although the importances are spread thinly (0.064 to 0.089 across the top ten), so this is a ranking, not a strong signal. It is consistent with historical team strength (captured via incremental rating updates) carrying more of what little signal there is than raw team identity or venue.

### Run Margin Prediction Needs Richer Features

The regression model achieves a forward-CV MAE of 23.4 runs with R2 of -0.47 (463 training-season matches won by runs); simply predicting the mean margin gives MAE 20.35, so the model is worse than the baseline. On the 68 runs-wins of 2024-25 it scores MAE 21.99 and R2 -0.19 against a 20.81 baseline. Cricket run margins depend on in-match factors (weather, pitch conditions, individual player form) that richer data sources could capture. The negative R2 with current features is consistent with published sports prediction literature.

---

**Author:** [Rajesh Srivastava](https://github.com/genieincodebottle)

**Part of:** [AI/ML Companion](https://github.com/genieincodebottle/aiml-companion) - A hands-on learning platform for ML, Deep Learning, LLMs, and AI Engineering.