# Analysis Summary - IPL Match Predictor (2008-2025)

## Objective

Build a production-grade ML pipeline that predicts IPL match outcomes (winner classification and run margin regression) using 18 seasons of historical data (2008-2025). The project demonstrates the complete data science lifecycle: data ingestion, cleaning, EDA, feature engineering with Elo ratings, statistical hypothesis testing, and predictive modeling with cross-validation.

## Data Sources

| Dataset | Records | Columns | Source |
|---------|---------|---------|--------|
| `matches.csv` | 1,169 (1,161 decided) | 20 | [Kaggle / GitHub](https://github.com/genieincodebottle/generative-ai/tree/main/docs/ipl-dataset) |
| `deliveries.csv` | 260,920 | 17 | Same as above (covers the 1,095 matches of 2008-2024 only) |

Data is auto-loaded at runtime from GitHub raw URLs. No manual download is required.

## Pipeline Architecture

```
matches.csv + deliveries.csv
        |
        v
[1. Data Ingestion]     Multi-source loader (Kaggle / local / GitHub URL)
        |
        v
[2. Data Cleaning]      Team name standardization (4 renames),
                        missing value imputation, type casting
        |
        v
[3. EDA]                Season trends, team performance, toss analysis,
                        venue stats, batsman deep-dive, win margins
        |
        v
[4. Feature Engineering] Elo ratings (K=32), momentum (5-match window),
                        head-to-head stats, home advantage, interactions
        |
        v
[5. Hypothesis Testing]  Toss advantage (binomial, p=0.318),
                        home advantage, decision impact (chi-squared)
        |
        v
[6. Model Training]      Random Forest (classification, balanced),
                        Gradient Boosting (regression, squared error),
                        forward-in-time 5-fold cross-validation (training seasons)
        |
        v
[7. Evaluation]          Accuracy, F1, confusion matrix, MAE, RMSE, R2,
                        feature importances, Markdown report
```

## Key Results

| Metric | Value |
|--------|-------|
| Classification accuracy | 0.500 calibrated, recency-weighted 4-model ensemble on the 2024-25 out-of-time test (142 matches), against a 0.479 majority baseline (within noise); forward CV on 2008-2023: RF 0.529, XGB 0.517, GB 0.509, LR 0.537 |
| Regression MAE | 23.39 runs (forward CV on 2008-2023), against 20.35 for predicting the mean margin; 21.99 vs 20.81 on 2024-25 |
| Regression R2 | -0.467 (forward CV); -0.186 on 2024-25 |
| Toss advantage p-value | 0.318 (598 of 1,161, 51.5%; not significant, two-sided). On the older 2008-2024 file: p = 0.607 |
| Top features | Elo features (`elo_team2`, `elo_team1`, `elo_diff`, `elo_expected`) |
| Engineered features | 20+ |

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | pandas, numpy |
| Visualization | Plotly (interactive), matplotlib, seaborn |
| Machine learning | scikit-learn (RandomForest, GradientBoosting, Pipeline) |
| Statistics | scipy (binomial test, chi-squared) |
| Configuration | PyYAML |
| Testing | pytest (77 unit tests across 4 test files) |

## Project Structure

```
ipl-match-predictor/
├── main.py                                # CLI entry point (7-stage pipeline)
├── configs/base.yaml                      # Dataset URLs, hyperparams, feature flags
├── data/raw/.gitkeep                      # Local CSV storage (git-ignored)
├── notebooks/ipl-dataset-analysis.ipynb   # Interactive analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Data loading, cleaning, team standardization
│   ├── eda.py                             # EDA plots and aggregations (Plotly)
│   ├── features.py                        # Elo ratings, momentum, H2H, home advantage
│   ├── hypothesis.py                      # Statistical hypothesis tests
│   ├── models.py                          # ML pipeline builders and evaluation
│   └── evaluate.py                        # Report generation and summary printing
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py                # Data loading and cleaning tests (19 tests)
│   ├── test_features.py                   # Feature engineering tests (20 tests)
│   ├── test_hypothesis.py                 # Hypothesis testing tests (9 tests)
│   └── test_models.py                     # ML model tests (29 tests)
├── artifacts/
│   ├── results/key_findings.md            # Documented key findings
│   └── figures/.gitkeep                   # Exported chart images
├── docs/analysis_summary.md               # This document
├── scripts/run_analysis.sh                # Shell script to run the notebook
├── Makefile                               # Make targets: run, test, notebook, clean
├── requirements.txt                       # Python dependencies
├── .gitignore
└── README.md
```

## Running the Pipeline

### Option 1: CLI (Recommended)
```bash
python main.py                    # Full pipeline
python main.py --stage eda        # Just EDA
python main.py --stage train      # Just model training
python main.py --verbose          # Debug logging
```

### Option 2: Make
```bash
make install    # Install dependencies
make run        # Full pipeline
make test       # Run 77 unit tests
make notebook   # Execute the Jupyter notebook
make clean      # Remove generated artifacts
```

### Option 3: Notebook
```bash
jupyter notebook notebooks/ipl-dataset-analysis.ipynb
```