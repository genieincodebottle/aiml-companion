# SHAP Explainability Report

> **Note:** Re-measured 2026-11-13 with `python -m src.explain` (shap 0.54.1, XGBoost 3.4.1, seed 42). This explains a **separately trained XGBoost** (80/20 split, test accuracy 0.9474), not the logistic-regression pipeline that `src/train.py` selects and saves. Your results may vary slightly across library versions.

## Global Top 10 Features (mean |SHAP value|, XGBoost)

| Rank | Feature | Mean SHAP |
|---|---|---|
| 1 | worst perimeter | 1.1892 |
| 2 | worst texture | 0.8682 |
| 3 | worst concave points | 0.8618 |
| 4 | area error | 0.8603 |
| 5 | worst radius | 0.7741 |
| 6 | worst area | 0.5484 |
| 7 | worst concavity | 0.5253 |
| 8 | mean smoothness | 0.4815 |
| 9 | mean concave points | 0.4681 |
| 10 | worst smoothness | 0.3735 |

## Data Leakage Check

No suspicious ID-like features detected. Top features (perimeter, area error, texture, concave points) are morphological measurements of cell nuclei.
