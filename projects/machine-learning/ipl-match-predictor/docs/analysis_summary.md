# Analysis Summary - IPL Match Predictor (2008-2024)

## Objective

Build a production-grade ML pipeline that predicts IPL match outcomes (winner classification and run margin regression) using 17 seasons of historical data (2008-2024). The project demonstrates the complete data science lifecycle: data ingestion, cleaning, EDA, feature engineering with Elo ratings, statistical hypothesis testing, and predictive modeling with cross-validation.

## Data Sources

| Dataset | Records | Columns | Source |
|---------|---------|---------|--------|
| `matches.csv` | 1,095 | 20 | [Kaggle / GitHub](https://github.com/genieincodebottle/generative-ai/tree/main/docs/ipl-dataset) |
| `deliveries.csv` | 260,920 | 17 | Same as above |

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
[5. Hypothesis Testing]  Toss advantage (binomial, p=0.61),
                        home advantage, decision impact (chi-squared)
        |
        v
[6. Model Training]      Random Forest (classification, balanced),
                        Gradient Boosting (regression, squared error),
                        5-fold stratified cross-validation
        |
        v
[7. Evaluation]          Accuracy, F1, confusion matrix, MAE, RMSE, R2,
                        feature importances, Markdown report
```

## Key Results

| Metric | Value |
|--------|-------|
| Classification accuracy | ~50% (5-fold CV) |
| Regression MAE | ~20 runs (baseline) |
| Regression R2 | Near zero |
| Toss advantage p-value | 0.61 (not significant, two-sided) |
| Top feature | Elo rating difference |
| Engineered features | 15+ |

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | pandas, numpy |
| Visualization | Plotly (interactive), matplotlib, seaborn |
| Machine learning | scikit-learn (RandomForest, GradientBoosting, Pipeline) |
| Statistics | scipy (binomial test, chi-squared) |
| Configuration | PyYAML |
| Testing | pytest (48 unit tests across 4 test files) |

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
│   ├── test_data_loader.py                # Data loading and cleaning tests (11 tests)
│   ├── test_features.py                   # Feature engineering tests (15 tests)
│   ├── test_hypothesis.py                 # Hypothesis testing tests (10 tests)
│   └── test_models.py                     # ML model tests (12 tests)
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
make test       # Run 48 unit tests
make notebook   # Execute the Jupyter notebook
make clean      # Remove generated artifacts
```

### Option 3: Notebook
```bash
jupyter notebook notebooks/ipl-dataset-analysis.ipynb
```