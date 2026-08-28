# Algorithm Showdown: Medical Diagnostic Classifier

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-brightgreen)

Build a medical diagnostic classifier that balances catching cancer (99%+ recall) against false alarms, with full explainability for regulatory review.

## Architecture

```
UCI Breast Cancer Data (569 samples, 30 features)
    |
    v
+-----------------------------------+
|   Data Pipeline                   |
|   StandardScaler + StratifiedKFold(5) |
+-----------------------------------+
    |
    v
+-----------------------------------+
|   6 Competing Algorithms          |
|   LogReg | SVM | KNN | DT | RF | XGB |
+-----------------------------------+
    |
    v
+-----------------------------------+
|   Cost-Sensitive Evaluation       |
|   Threshold tuned for 99% recall  |
+-----------------------------------+
    |
    v
+-----------------------------------+
|   SHAP Explainability Report      |
|   Global importance + top-3 local |
+-----------------------------------+
```

## Problem Statement

Cancer screening demands high recall -- **missing a malignant tumor is far worse than a false alarm**. This project systematically compares 6 ML algorithms on real clinical data, tunes decision thresholds for a 99% recall target, and generates SHAP explainability reports suitable for regulatory review.

## Approach

1. **Baseline**: Logistic Regression with StandardScaler -- establishes the accuracy/recall floor
2. **Competition**: 5 additional algorithms (SVM, KNN, Decision Tree, Random Forest, XGBoost) evaluated with identical StratifiedKFold(5) cross-validation
3. **Threshold Tuning**: Best model's decision threshold adjusted from 0.5 to achieve 99% recall
4. **Explainability**: SHAP TreeExplainer generates per-patient and global feature importance reports

## Results

| Algorithm | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.968 | 0.978 | 0.973 | 0.993 |
| SVM (RBF) | 0.968 | 0.971 | 0.980 | 0.976 | 0.994 |
| KNN (k=5) | 0.958 | 0.965 | 0.969 | 0.967 | 0.985 |
| Decision Tree | 0.940 | 0.951 | 0.955 | 0.953 | 0.960 |
| Random Forest | 0.963 | 0.966 | 0.975 | 0.971 | 0.994 |
| **XGBoost** | **0.970** | **0.968** | **0.983** | **0.975** | **0.995** |

> Sample results from UCI Breast Cancer Wisconsin dataset. Your results may vary slightly due to random seed differences.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/genieincodebottle/ml-algorithms-capstone.git
cd ml-algorithms-capstone
pip install -r requirements.txt

# Run the full pipeline (train -> evaluate -> explain)
make all

# Or run individual stages
make train       # Train 6 algorithms and compare
make evaluate    # Evaluate saved model with threshold analysis
make explain     # Generate SHAP explainability report

# Run tests
make test

# Or use the pipeline script directly
bash scripts/run_pipeline.sh          # Full pipeline
bash scripts/run_pipeline.sh train    # Train only
```

## Project Structure

```
ml-algorithms-capstone/
|-- configs/
|   +-- base.yaml               # Hyperparameters, CV settings, threshold target
|-- src/
|   |-- __init__.py              # Package marker
|   |-- train.py                 # 6-algorithm comparison pipeline
|   |-- evaluate.py              # Confusion matrices, PR curves, threshold analysis
|   +-- explain.py               # SHAP waterfall + global importance
|-- tests/
|   |-- __init__.py
|   +-- test_pipeline.py         # pytest: data loading, fit, predict, threshold
|-- notebooks/
|   +-- Algorithm_Showdown.ipynb # Interactive walkthrough
|-- artifacts/
|   |-- models/                  # Saved model artifacts (.gitkeep)
|   +-- results/
|       |-- comparison_table.md  # Pre-generated metrics table
|       +-- shap_summary.md      # Pre-generated SHAP report
|-- docs/
|   |-- model_card.md            # Google Model Card for the classifier
|   +-- experiment_log.csv       # Experiment progression tracking
|-- scripts/
|   +-- run_pipeline.sh          # Shell script: train -> evaluate -> explain
|-- .gitignore
|-- Makefile                     # make train / evaluate / explain / test / all
|-- requirements.txt             # Pinned dependencies
+-- README.md
```

## Experiment Log

| Experiment | What Changed | Accuracy | AUC | Notes |
|---|---|---|---|---|
| baseline_logreg | Initial LogReg baseline | 0.965 | 0.993 | Strong baseline |
| svm_rbf | Added SVM RBF | 0.968 | 0.994 | Slightly better |
| knn_k5 | KNN k=5 | 0.958 | 0.985 | Lowest performer |
| decision_tree | DTree depth=5 | 0.940 | 0.960 | Most interpretable |
| random_forest | RF n=200 | 0.963 | 0.994 | High variance |
| xgboost | XGB lr=0.1 | 0.970 | 0.995 | Best AUC |
| xgboost_tuned | Threshold for 99% recall | 0.956 | 0.995 | Production target met |

## Interview Guide: How to Talk About This Project

### "Walk me through this project."

"I built a cancer diagnostic classifier that compares 6 ML algorithms on real clinical data. The key insight is that accuracy isn't enough -- for cancer screening, we need 99%+ recall because missing a malignant tumor is the worst outcome. I tuned the decision threshold to meet that clinical constraint and generated SHAP reports for explainability."

### "What was the hardest part?"

"Threshold tuning. Lowering the decision threshold from 0.5 to 0.14 achieved 99% recall, but precision dropped from 0.968 to 0.942 -- meaning more false alarms. I had to justify this tradeoff quantitatively: in cancer screening, a false alarm means an extra biopsy, but a missed diagnosis can be fatal."

### "What would you do differently?"

"Three things: (1) Use a larger, more imbalanced dataset -- breast cancer has a 63/37 split, which is relatively balanced. Real-world medical data is often 95/5. (2) Add calibration plots to verify the predicted probabilities are meaningful, not just the rank ordering. (3) Implement a voting ensemble of the top 3 models instead of relying on a single model."

### "How does this scale to production?"

"The sklearn Pipeline handles preprocessing and scaling consistently. For deployment, I'd wrap the threshold-tuned XGBoost in a FastAPI endpoint, add input validation with Pydantic, and containerize with Docker. The SHAP explainability component would be critical for any regulated healthcare deployment."

### "Explain the threshold tuning to a non-technical person."

"Imagine a smoke detector. A sensitive detector catches every fire but sometimes triggers on burnt toast. A less sensitive one never false alarms but might miss a real fire. For cancer screening, we set the sensitivity to maximum -- we'd rather investigate 100 false alarms than miss 1 real case. That's what threshold tuning does."