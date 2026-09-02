# Credit Risk Pipeline - Evaluation Report

## Model Comparison

| Model | ROC-AUC | Recall | F1 |
|-------|---------|--------|----|
| LogisticRegression | 0.783 +/- 0.025 | 0.700 | 0.595 |
| GradientBoosting ** | 0.784 +/- 0.025 | 0.493 | 0.557 |

**Best Model**: GradientBoosting

## Cost-Sensitive Threshold Tuning

- **Optimal Threshold**: 0.05
- **Minimum Cost**: $685
- **Cost Ratio**: FN=10x, FP=1x

## Classification Report (at optimal threshold)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.913 | 0.421 | 0.577 | 700 |
| 1 | 0.402 | 0.907 | 0.557 | 300 |

## Artifacts

- `artifacts/figures/precision_recall_curve.png`
- `artifacts/figures/confusion_matrix.png`
- `artifacts/figures/shap_importance.png`
- `artifacts/results/best_model.joblib`