# Credit Risk Pipeline

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

End-to-end ML pipeline for credit default prediction. Demonstrates data cleaning, EDA, feature engineering, model training with cross-validation, cost-sensitive evaluation, FastAPI serving, and PSI drift monitoring.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/genieincodebottle/aiml-companion.git
cd projects/credit-risk-pipeline
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
│   └── Credit_Risk_Pipeline.ipynb
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

- **KNN Imputation**: Handles MNAR (Missing Not At Random) patterns in income/employment data
- **Domain Features**: DTI ratio, credit utilization, loan burden score, age bucketing
- **sklearn Pipeline**: ColumnTransformer prevents data leakage between train/test splits
- **Dual Models**: Logistic Regression (interpretable, ECOA-compliant) + Gradient Boosting (performance)
- **Cost-Sensitive Evaluation**: 10:1 FN/FP cost ratio for threshold tuning
- **SHAP Explanations**: Feature importance for adverse action reasons
- **FastAPI Serving**: REST API with health check, single and batch prediction
- **PSI Monitoring**: Population Stability Index for production drift detection

## Serve the Model

```bash
# Train first
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