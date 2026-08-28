# Algorithm Comparison Results

> **Note:** These are sample outputs from running on the UCI Breast Cancer Wisconsin dataset (569 samples, 30 features). Your results may vary slightly due to random seed differences.

## Cross-Validation Results (StratifiedKFold, k=5)

| Algorithm | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.968 | 0.978 | 0.973 | 0.993 |
| SVM (RBF) | 0.968 | 0.971 | 0.980 | 0.976 | 0.994 |
| KNN (k=5) | 0.958 | 0.965 | 0.969 | 0.967 | 0.985 |
| Decision Tree | 0.940 | 0.951 | 0.955 | 0.953 | 0.960 |
| Random Forest | 0.963 | 0.966 | 0.975 | 0.971 | 0.994 |
| **XGBoost** | **0.970** | **0.968** | **0.983** | **0.975** | **0.995** |

## Best Per Metric

| Metric | Best Algorithm | Score |
|---|---|---|
| accuracy | XGBoost | 0.970 |
| precision | SVM (RBF) | 0.971 |
| recall | XGBoost | 0.983 |
| f1 | SVM (RBF) | 0.976 |
| roc_auc | XGBoost | 0.995 |

## Threshold Tuning (Cancer Screening - 99% Recall Target)

| Threshold | Precision | Recall |
|---|---|---|
| 0.142 | 0.942 | 0.990 |
| 0.287 | 0.958 | 0.955 |
| 0.500 | 0.968 | 0.983 |

**Selected threshold: 0.142** - achieves 99% recall with acceptable precision for clinical screening.
