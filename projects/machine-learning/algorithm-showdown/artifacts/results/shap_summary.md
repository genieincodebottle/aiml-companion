# SHAP Explainability Report

> **Note:** These are sample outputs from running on the UCI Breast Cancer Wisconsin dataset. Your results may vary slightly due to random seed differences.

## Global Top 10 Features (mean |SHAP value|)

| Rank | Feature | Mean SHAP |
|---|---|---|
| 1 | worst concave points | 0.8234 |
| 2 | worst perimeter | 0.6891 |
| 3 | worst radius | 0.6745 |
| 4 | mean concave points | 0.5123 |
| 5 | worst area | 0.4567 |
| 6 | mean perimeter | 0.3891 |
| 7 | mean radius | 0.3456 |
| 8 | worst texture | 0.2890 |
| 9 | worst smoothness | 0.2345 |
| 10 | mean texture | 0.1987 |

## Data Leakage Check

No suspicious ID-like features detected. Top features (concave points, perimeter, radius) are clinically meaningful morphological measurements.
