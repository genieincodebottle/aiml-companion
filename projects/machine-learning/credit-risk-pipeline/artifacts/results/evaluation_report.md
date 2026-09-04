# Credit Risk Pipeline - Evaluation Report

## Model Comparison

| Model | ROC-AUC | Recall | F1 |
|-------|---------|--------|----|
| LogisticRegression | 0.783 +/- 0.025 | 0.700 | 0.595 |
| GradientBoosting ** | 0.784 +/- 0.025 | 0.493 | 0.557 |

**Best Model**: GradientBoosting

## Cost-Sensitive Threshold Tuning

- **Optimal Threshold**: 0.01
- **Minimum Cost**: $662
- **Cost Ratio**: FN=10x, FP=1x

## Classification Report (at optimal threshold)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.946 | 0.126 | 0.222 | 700 |
| 1 | 0.325 | 0.983 | 0.489 | 300 |

## Top Features (SHAP)

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | cat__checking_status_no checking | 0.7445 |
| 2 | num__credit_amount | 0.4816 |
| 3 | num__loan_burden | 0.3798 |
| 4 | num__duration | 0.3459 |
| 5 | cat__checking_status_<0 | 0.2747 |
| 6 | num__age | 0.2691 |
| 7 | cat__credit_history_critical/other existing credit | 0.2546 |
| 8 | cat__purpose_used car | 0.2329 |
| 9 | cat__savings_status_<100 | 0.2079 |
| 10 | cat__employment_4<=X<7 | 0.2013 |

## Artifacts

- `artifacts/figures/precision_recall_curve.png`
- `artifacts/figures/confusion_matrix.png`
- `artifacts/figures/shap_importance.png`
- `artifacts/results/best_model.joblib`