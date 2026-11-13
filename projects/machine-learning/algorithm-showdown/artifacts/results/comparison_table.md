# Algorithm Comparison Results

> **Note:** Re-measured 2026-11-13 with `python -m src.train` on the UCI Breast Cancer Wisconsin dataset (569 samples, 30 features; scikit-learn 1.9.1, XGBoost 3.4.1, seed 42). Results can move in the third decimal across library versions.

## Cross-Validation Results (StratifiedKFold, k=5, shuffle=True, random_state=42)

| Algorithm | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.974 | 0.968 | **0.992** | 0.979 | **0.995** |
| SVM (RBF) | **0.977** | **0.976** | 0.989 | **0.982** | 0.995 |
| KNN (k=5) | 0.963 | 0.957 | 0.986 | 0.971 | 0.985 |
| Decision Tree | 0.928 | 0.932 | 0.958 | 0.944 | 0.907 |
| Random Forest | 0.954 | 0.962 | 0.966 | 0.964 | 0.990 |
| XGBoost | 0.963 | 0.963 | 0.980 | 0.971 | 0.994 |

## Best Per Metric

| Metric | Best Algorithm | Score |
|---|---|---|
| accuracy | SVM (RBF) | 0.9772 |
| precision | SVM (RBF) | 0.9756 |
| recall | Logistic Regression | 0.9916 |
| f1 | SVM (RBF) | 0.9820 |
| roc_auc | Logistic Regression | 0.9953 |

**Selected for threshold tuning: Logistic Regression** (highest roc_auc, chosen by `select_best()`). By accuracy XGBoost ties KNN for third of six.

## Threshold Tuning (Cancer Screening, 95% Malignant Recall Target)

Model fit on train (341), threshold chosen on validation (114), reported on test (114; 43 malignant, 71 benign).

| Target recall | Threshold (P(malignant)) | Validation recall | Validation precision | Test recall | Test precision |
|---|---|---|---|---|---|
| 100% | 0.000 | 1.000 | 0.368 | 1.000 | 0.381 |
| 95% | 0.759 | 0.952 | 1.000 | 0.860 | 1.000 |
| 90% | 0.922 | 0.905 | 1.000 | 0.814 | 1.000 |
| 85% | 0.971 | 0.857 | 1.000 | 0.767 | 1.000 |
| 80% | 0.994 | 0.810 | 1.000 | 0.674 | 1.000 |

**Selected threshold: 0.759** (95% target). Held-out recall is **0.860** (37 of 43 malignant caught, 6 missed), so the 95% target is not met on the test split. For comparison, the untuned 0.5 threshold catches 42 of 43 (recall 0.977, precision 0.977) on the same split.
