# IPL Match Predictor

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

End-to-end ML pipeline that predicts IPL match outcomes using 18 seasons of historical data (2008-2025). Combines Elo rating system, statistical hypothesis testing, and a 4-model weighted ensemble (Random Forest, XGBoost, Gradient Boosting, Logistic Regression) validated with 10,000-match Monte Carlo simulation.

**What this project demonstrates:** The complete ML lifecycle - from raw data to production-ready predictions - using 1,169 matches and 260,920 ball-by-ball records.

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

| Metric | Value | Method |
|--------|-------|--------|
| Match winner accuracy | ~50% | 4-model weighted ensemble, 5-fold CV |
| Monte Carlo validation | 10,000 sims | Gaussian noise (std=0.05) stability check |
| Toss advantage | Not significant | Binomial test (two-sided), p=0.61 |
| Top predictive feature | Elo rating difference | Feature importance ranking |
| Engineered features | 20+ | Elo, momentum, H2H, home, toss, venue chase bias, interactions |
| Probability calibration | CalibratedClassifierCV | Sigmoid method, cv=3 |
| Training weighting | Season recency | Exponential decay (0.5 to 1.0) |

## Quick Start

### Option 1: Run locally (recommended)

```bash
# Clone and navigate
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/ipl-match-predictor

# Set up environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install and run
pip install -r requirements.txt
python main.py                    # Full pipeline
```

### Option 2: Kaggle Notebook

1. Open: [IPL Dataset Analysis on Kaggle](https://www.kaggle.com/code/genieincodebottle/ipl-dataset-analysis)
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

## Make Targets

```bash
make install    # Install dependencies
make run        # Full pipeline via CLI
make test       # Run 59 unit tests
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
│   └── test_models.py                     # ML model pipelines (11 tests)
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
| `matches.csv` | 1,169 | Match-level data: teams, toss, result, venue, player of match (2008-2025) |
| `deliveries.csv` | 260,920 | Ball-by-ball: batsman, bowler, runs, extras, wickets (2008-2024) |

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

All 4 classifiers use sklearn Pipelines with StandardScaler preprocessing and 5-fold stratified cross-validation. Each model is optionally wrapped with `CalibratedClassifierCV` for well-calibrated probability estimates. Training uses season recency weighting so recent matches have more influence. Predictions are combined via weighted average, then validated with Monte Carlo simulation. The `predict_match()` function in `src/models.py` constructs feature vectors for unseen future matches.

### Hypothesis Testing

| Test | Method | Result |
|------|--------|--------|
| Toss advantage | Two-sided binomial (H0: p=0.5) | p=0.61, NOT significant |
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
# Run all 59 tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Analysis Highlights

### Toss Has No Significant Advantage

Binomial test (two-sided) on 1,160+ decided matches: the toss winner's match win rate (~50.8%) does not deviate meaningfully from 50%. p-value = 0.61 (alpha = 0.05). This is one of the most commonly asked IPL trivia questions, and the data provides a definitive answer.

### Elo Ratings Are the Strongest Predictor

The Elo rating difference between teams consistently ranks as the top feature in importance analysis. This confirms that historical team strength (captured via incremental rating updates) provides more signal than raw team identity or venue.

### Run Margin Prediction Needs Richer Features

The regression model achieves MAE ~20 runs with R2 near zero using pre-match aggregate features. Cricket run margins depend on in-match factors (weather, pitch conditions, individual player form) that richer data sources could capture. The near-zero R2 with current features is consistent with published sports prediction literature.

---

**Author:** [Rajesh Srivastava](https://github.com/genieincodebottle)

**Part of:** [AI/ML Companion](https://github.com/genieincodebottle/aiml-companion) - A hands-on learning platform for ML, Deep Learning, LLMs, and AI Engineering.