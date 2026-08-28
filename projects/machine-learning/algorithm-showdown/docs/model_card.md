# Model Card: Breast Cancer Diagnostic Classifier

## Model Details

### Overview
- **Model name:** Breast Cancer Diagnostic Classifier (XGBoost, threshold-tuned)
- **Model version:** 1.0
- **Model type:** Gradient-boosted decision tree (XGBClassifier) wrapped in a scikit-learn Pipeline with StandardScaler preprocessing
- **Framework:** scikit-learn 1.4+, XGBoost 2.0+
- **License:** MIT
- **Contact:** Project maintainer via repository issues

### Training Algorithm
XGBoost with the following hyperparameters:
- `n_estimators`: 200
- `learning_rate`: 0.1
- `max_depth`: 4
- `eval_metric`: logloss
- `random_state`: 42

The decision threshold was tuned from the default 0.5 to achieve ~95% recall on the malignant class, prioritizing sensitivity for cancer screening. The exact threshold is computed dynamically from the precision-recall curve and varies by library version (typically 0.1-0.5).

> **Note on reproducibility:** Exact metric values depend on scikit-learn and XGBoost versions. The numbers below were produced with scikit-learn 1.4.0 and XGBoost 2.1.4. Your results may differ slightly, but the relative ranking and patterns should be consistent.

## Intended Use

### Primary Use Cases
- **Educational demonstration** of ML algorithm comparison, threshold tuning, and SHAP explainability on clinical data.
- **Portfolio project** showcasing cost-sensitive evaluation for medical diagnostics.
- **Reference implementation** for comparing multiple classifiers on tabular biomedical data.

### Out-of-Scope Uses
- **Not for clinical deployment.** This model is trained on a small, well-curated benchmark dataset and has not been validated on real-world clinical populations.
- **Not for automated medical decision-making.** Any clinical application would require regulatory approval (e.g., FDA 510(k)), prospective validation, and integration with clinical workflows.
- **Not for populations outside the training distribution.** The UCI dataset represents a specific patient population and may not generalize.

## Training Data

### Dataset
- **Name:** UCI Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository / `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569 (212 malignant, 357 benign)
- **Features:** 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses. Features describe characteristics of cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension (mean, standard error, and worst for each).
- **Class distribution:** 62.7% benign, 37.3% malignant (relatively balanced)

### Preprocessing
- StandardScaler normalization (zero mean, unit variance)
- No feature selection or engineering applied
- No missing values in the dataset
- Stratified train/test split (80/20) preserving class proportions

## Evaluation Data
- **Test set:** 20% stratified holdout (114 samples)
- **Cross-validation:** StratifiedKFold with k=5, shuffle=True, random_state=42

## Metrics

### Cross-Validation Performance (6 algorithms compared)

| Algorithm           | Accuracy | Precision | Recall | F1    | AUC   |
|---------------------|----------|-----------|--------|-------|-------|
| Logistic Regression | 0.974    | 0.968     | 0.992  | 0.979 | 0.995 |
| **SVM (RBF)**       | **0.977**| **0.976** | 0.989  | **0.982** | 0.995 |
| KNN (k=5)           | 0.963    | 0.957     | 0.986  | 0.971 | 0.985 |
| Decision Tree       | 0.928    | 0.932     | 0.958  | 0.944 | 0.907 |
| Random Forest       | 0.954    | 0.962     | 0.966  | 0.964 | 0.990 |
| XGBoost             | 0.961    | 0.968     | 0.972  | 0.969 | 0.994 |

All algorithms achieve >92% accuracy. SVM and Logistic Regression typically lead on most metrics, with XGBoost competitive on AUC. Decision Tree is consistently the weakest.

### Threshold-Tuned Performance (XGBoost)

The threshold is computed dynamically from the precision-recall curve to achieve the target recall:

| Target Malignant Recall | Threshold (on P(malignant)) | Precision |
|-------------------------|----------------------------|-----------|
| 100%                    | ~0.00                      | ~0.37     |
| 95%                     | ~0.18                      | ~0.85     |
| 90%                     | ~0.28                      | ~0.90     |
| 85%                     | ~0.97                      | ~1.00     |

### Selection Rationale
XGBoost was selected as the production model for its balance of high AUC (~0.994), strong recall (~0.97), and compatibility with SHAP TreeExplainer for interpretability. For cancer screening, the threshold is lowered to achieve ~95% malignant recall, accepting a modest decrease in precision.

## Explainability

### Method
SHAP (SHapley Additive exPlanations) using TreeExplainer for the XGBoost model.

### Top 5 Global Features (by mean |SHAP value|)
1. **worst perimeter** (1.29) - boundary length of the worst-case nucleus
2. **area error** (0.89) - variability in nucleus area measurements
3. **worst texture** (0.87) - texture of the worst-case nucleus
4. **worst concave points** (0.77) - morphological irregularity of cell nuclei
5. **worst area** (0.64) - area of the worst-case nucleus

All top features are clinically meaningful morphological measurements. No ID-like or data-leakage features detected. Feature rankings may shift slightly across library versions, but the dominant features (`worst perimeter`, `worst concave points`, `worst area`) remain consistent.

## Ethical Considerations

### Fairness
- The UCI Breast Cancer dataset does not include demographic attributes (race, ethnicity, age, socioeconomic status). As a result, fairness across subgroups cannot be assessed.
- Real-world deployment would require fairness evaluation across protected classes.

### Privacy
- The dataset is fully de-identified and publicly available.
- No personally identifiable information (PII) is present.

### Clinical Risk
- **False negatives** (missed malignant tumors) carry severe consequences. The threshold was tuned to minimize this risk at the expense of more false positives.
- **False positives** result in unnecessary follow-up procedures (e.g., additional imaging, biopsies), causing patient anxiety and healthcare costs but are clinically preferable to missed diagnoses.

## Limitations

- **Small dataset:** 569 samples is insufficient for production clinical models. Results may not generalize to larger, more diverse populations.
- **Balanced classes:** Real-world cancer screening datasets are typically highly imbalanced (e.g., 95/5), making this benchmark optimistic.
- **No temporal validation:** All data comes from a single source and time period. Model performance may degrade with distributional shift.
- **Feature extraction assumed:** The model operates on pre-computed cell nucleus features, not raw medical images. A production system would need an upstream image processing pipeline.
- **No calibration analysis:** Predicted probabilities have not been validated for calibration. Rank ordering (AUC) is reliable, but absolute probability values may be miscalibrated.
- **Version sensitivity:** Exact metrics vary by scikit-learn/XGBoost version. The ranking of algorithms and general patterns are stable, but absolute numbers should not be treated as fixed benchmarks.

## Recommendations

- Use this model strictly for educational and portfolio purposes.
- For clinical applications, retrain on larger, more representative datasets with proper regulatory oversight.
- Add probability calibration (Platt scaling or isotonic regression) before using predicted probabilities for clinical decision-making.
- Conduct prospective validation before any deployment scenario.
