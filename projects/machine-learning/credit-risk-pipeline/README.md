# Credit Risk Pipeline

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

End-to-end ML pipeline for credit default prediction. Demonstrates data cleaning, EDA, feature engineering, model training with cross-validation, cost-sensitive evaluation, FastAPI serving, and PSI drift monitoring.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/machine-learning/credit-risk-pipeline
python -m venv .venv
source .venv/Scripts/activate    # Windows
# source .venv/bin/activate      # Mac/Linux
pip install -r requirements.txt

# Run the full pipeline
python main.py --verbose
```

## Pipeline Stages

```bash
python main.py                     # Run all stages
python main.py --stage clean       # Data loading + cleaning
python main.py --stage eda         # Exploratory data analysis
python main.py --stage features    # Feature engineering
python main.py --stage train       # Model training (LR + GBC)
python main.py --stage evaluate    # Cost-sensitive evaluation
```

## Project Structure

```
credit-risk-pipeline/
├── main.py                  # CLI entry point
├── configs/
│   └── base.yaml            # Pipeline configuration
├── src/
│   ├── data_loader.py       # Data loading (OpenML/CSV/Kaggle) + cleaning
│   ├── eda.py               # EDA charts and summary stats
│   ├── features.py          # Feature engineering (DTI, utilization, burden)
│   ├── models.py            # sklearn Pipeline + ColumnTransformer training
│   ├── evaluate.py          # Cost-sensitive threshold tuning, SHAP, reports
│   ├── serve.py             # FastAPI prediction endpoint
│   └── monitor.py           # PSI drift detection + data quality checks
├── tests/
│   ├── test_data_loader.py  # Data loading/cleaning tests
│   ├── test_features.py     # Feature engineering tests
│   ├── test_models.py       # Model pipeline tests
│   └── test_monitor.py      # Monitoring tests
├── notebooks/
│   └── credit-risk-pipeline.ipynb
├── data/raw/                # Raw data files
├── artifacts/
│   ├── figures/             # EDA and evaluation charts
│   └── results/             # Trained models and reports
├── Dockerfile
├── Makefile
├── requirements.txt
└── model_card.md
```

## Key Features

- **KNN Imputation**: handles missing patterns in the employment/savings fields
- **Domain Features**: loan burden score and age bucketing. DTI and credit
  utilization are **deliberately not built** -- see "What this dataset cannot
  support" below
- **Fair lending**: sex and marital status are dropped before modelling, not
  left to a caveat -- see "The column that had to go"
- **sklearn Pipeline**: ColumnTransformer, fitted inside each CV fold
- **Dual Models**: Logistic Regression (interpretable, ECOA-compliant) + Gradient Boosting (performance)
- **Cost-Sensitive Evaluation**: 10:1 FN/FP cost ratio, tuned on **out-of-fold**
  predictions -- see "Why the cost used to be $1"
- **SHAP Explanations**: Feature importance for adverse action reasons
- **FastAPI Serving**: REST API with health check, single and batch prediction
- **PSI Monitoring**: Population Stability Index for production drift detection

## Three things this pipeline gets right, because it used to get them wrong

### Why the cost used to be $1

Evaluation called `pipeline.predict_proba(X)` on a model fitted to all of `X`.
A gradient booster memorises 1,000 rows without effort, so it scored its own
training data almost perfectly and the pipeline reported a **total
misclassification cost of $1**.

The flattering number was not the expensive part. The threshold was chosen from
those memorised scores, and *that* threshold is what got written to
`threshold.json` and loaded by the serving API:

| | threshold | reported cost | real cost |
|---|---|---|---|
| in-sample (old) | 0.30 | **$1** | **$1,372** |
| out-of-fold (now) | 0.05 | $685 | $685 |

Because false negatives cost 10x false positives, the honest threshold is
*aggressive* -- 0.05. The overfitted model looked so confident that 0.30 seemed
safe, and shipping it roughly **doubled** real-world cost against a correctly
tuned model. Everything downstream now runs on `cross_val_predict` output:
every row scored by a copy of the model that never trained on it.

### The column that had to go

`personal_status` reads "male single", "female div/dep/mar". It encodes **sex**,
and it was going into the model as an ordinary categorical. ECOA / Regulation B
prohibits that outright.

The usual objection is that accuracy requires it. It does not:

| | cross-validated AUC |
|---|---|
| with `personal_status` | 0.7869 |
| without it | 0.7818 |

Half a point of AUC. What it buys on the other side of the ledger: the female
default rate in this data is **35.2%** against 27.7% for men, so a model holding
that column learns to charge women more for being women -- 31% of applicants.
`drop_protected` is on by default and dropping the column is the *floor*, not
the finish line: proxies survive, which is why a real deployment needs a
disparate-impact audit rather than a deleted column.

### What this dataset cannot support

The README used to advertise DTI ratio and credit utilization as domain
features. Neither can exist here:

- **German Credit has no income column.** DTI resolved `income` to
  `personal_status` -- a categorical -- `to_numeric` coerced every value to
  `NaN`, and the result was a **100% null column** that logged "Added
  dti_ratio" and got quietly imputed downstream.
- **There is no credit limit either.** Utilization resolved to
  `existing_credits` (a count of 1-4) divided by `credit_amount` (this loan's
  size), giving a column ranging 5e-05 to 0.01 and named after a ratio it did
  not compute.

Dropping both *raised* cross-validated AUC (0.7869 -> 0.7873), which is what a
feature contributing nothing looks like. They are now skipped with a warning
rather than fabricated. **A feature that cannot be computed must be absent, so
its absence is visible -- a trustworthy name on an empty column is worse than
no column.**

## Serve the Model

```bash
# Train first (also writes the tuned decision threshold to
# artifacts/results/threshold.json, which the API loads on startup)
python main.py

# Start API server
uvicorn src.serve:app --reload --port 8000

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"duration": 24, "credit_amount": 5000, "age": 35, "income": 45000}'
```

## Run Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Tech Stack

Python 3.10+, pandas, scikit-learn, SHAP, FastAPI, Docker, pytest

## License

MIT