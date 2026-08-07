# Credit Risk Pipeline - Evaluation Report

## Model Comparison

| Model | ROC-AUC | Recall | F1 |
|-------|---------|--------|----|
| LogisticRegression ** | 0.788 +/- 0.022 | 0.703 | 0.599 |
| GradientBoosting | 0.776 +/- 0.021 | 0.460 | 0.524 |

**Best Model**: LogisticRegression

## Cost-Sensitive Threshold Tuning

- **Optimal Threshold**: 0.21
- **Minimum Cost**: $528
- **Cost Ratio**: FN=10x, FP=1x

## Classification Report (at optimal threshold)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.962 | 0.403 | 0.568 | 700 |
| 1 | 0.409 | 0.963 | 0.574 | 300 |

## Top Features (SHAP)

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | num__credit_amount | 0.5075 |
| 2 | cat__checking_status_no checking | 0.4885 |
| 3 | num__duration | 0.4109 |
| 4 | cat__credit_history_critical/other existing credit | 0.3688 |
| 5 | num__installment_commitment | 0.3221 |
| 6 | num__loan_burden | 0.3154 |
| 7 | cat__checking_status_<0 | 0.2605 |
| 8 | cat__purpose_new car | 0.2425 |
| 9 | cat__savings_status_<100 | 0.2239 |
| 10 | cat__personal_status_male single | 0.2225 |

## Artifacts

- `artifacts/figures/precision_recall_curve.png`
- `artifacts/figures/confusion_matrix.png`
- `artifacts/figures/shap_importance.png`
- `artifacts/results/best_model.joblib`