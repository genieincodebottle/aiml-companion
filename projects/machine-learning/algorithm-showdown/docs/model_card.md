# Model Card: Breast Cancer Diagnostic Classifier

## Model Details

### Overview
- **Model name:** Breast Cancer Diagnostic Classifier (Logistic Regression, threshold-tuned)
- **Model version:** 1.1
- **Model type:** Logistic regression (`LogisticRegression`) wrapped in a scikit-learn Pipeline with StandardScaler preprocessing. It is the model `src/train.py` carries forward because it has the highest cross-validated AUC of the six candidates (0.9953).
- **Framework:** scikit-learn 1.9+ (XGBoost 3.4+ is needed only for the comparison and the SHAP step)
- **License:** MIT
- **Contact:** Project maintainer via repository issues

### Training Algorithm
Logistic regression with the following hyperparameters:
- `max_iter`: 1000
- `random_state`: 42
- default L2 penalty, `C=1.0`, on StandardScaler-scaled features

The model is not hardcoded. `select_best()` in `src/train.py` picks it from the six-algorithm comparison by cross-validated ROC AUC; if the comparison changes, so does the selected model.

The decision threshold was moved from the default 0.5 to the value that reaches ~95% recall on the malignant class **on a validation split**, prioritizing sensitivity for cancer screening. On the default seed (42) the tuned threshold on P(malignant) is **0.759**. That threshold gives validation recall 0.952 but **test recall 0.860** (37 of 43 malignant test cases caught, 6 missed), so the 95% target is **not** met on held-out data.

> **Note on reproducibility:** Exact metric values depend on scikit-learn and XGBoost versions. The numbers below were re-measured with scikit-learn 1.9.1, XGBoost 3.4.1 and SHAP 0.54.1 (seed 42). Your results may differ slightly, but the relative ranking and patterns should be consistent.

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
- Stratified three-way split (60/20/20) preserving class proportions, defined once in `make_splits()` and shared by `train.py` and `evaluate.py`: train 341, validation 114, test 114

## Evaluation Data
- **Test set:** 20% stratified holdout (114 samples: 43 malignant, 71 benign), used by neither model fitting nor threshold tuning
- **Validation set:** 20% stratified (114 samples), used only to choose the threshold
- **Cross-validation (algorithm comparison):** StratifiedKFold with k=5, shuffle=True, random_state=42, on all 569 samples

## Metrics

### Cross-Validation Performance (6 algorithms compared)

| Algorithm           | Accuracy | Precision | Recall | F1    | AUC   |
|---------------------|----------|-----------|--------|-------|-------|
| Logistic Regression | 0.974    | 0.968     | 0.992  | 0.979 | 0.995 |
| **SVM (RBF)**       | **0.977**| **0.976** | 0.989  | **0.982** | 0.995 |
| KNN (k=5)           | 0.963    | 0.957     | 0.986  | 0.971 | 0.985 |
| Decision Tree       | 0.928    | 0.932     | 0.958  | 0.944 | 0.907 |
| Random Forest       | 0.954    | 0.962     | 0.966  | 0.964 | 0.990 |
| XGBoost             | 0.963    | 0.963     | 0.980  | 0.971 | 0.994 |

All algorithms achieve >92% accuracy. SVM takes accuracy, precision and F1; Logistic Regression takes recall and AUC. By accuracy XGBoost ties KNN for third of six (SVM 0.977, LR 0.974, KNN 0.963, XGBoost 0.963, RF 0.954, DT 0.928). Decision Tree is consistently the weakest.

### Threshold-Tuned Performance (Logistic Regression)

The threshold is chosen on the validation split from the precision-recall curve (the point closest to the target recall) and then reported on the untouched test split (43 malignant cases):

| Target malignant recall | Threshold (on P(malignant)) | Validation recall / precision | **Test recall** / precision | Missed on test |
|---|---|---|---|---|
| 100% | 0.000 | 1.000 / 0.368 | 1.000 / 0.381 | 0 of 43 |
| 95% (shipped) | 0.759 | 0.952 / 1.000 | **0.860** / 1.000 | 6 of 43 |
| 90% | 0.922 | 0.905 / 1.000 | 0.814 / 1.000 | 8 of 43 |
| 85% | 0.971 | 0.857 / 1.000 | 0.767 / 1.000 | 10 of 43 |
| 80% | 0.994 | 0.810 / 1.000 | 0.674 / 1.000 | 14 of 43 |

For comparison, the untuned 0.5 threshold on the same test split catches 42 of 43 malignant cases (recall 0.977, precision 0.977, accuracy 0.991). The tuned 0.759 threshold has test accuracy 0.947 and test AUC 0.997 (AUC does not depend on the threshold). On 43 malignant cases one case is about 2.3 recall points, so these test-recall figures carry a spread of several points.

### Selection Rationale
Logistic Regression was selected by `select_best()` because it has the highest cross-validated ROC AUC (0.9953, with SVM at 0.995 and XGBoost at 0.994 within noise) and the highest recall (0.992). AUC is the selection metric because no threshold has been chosen yet at that stage. For cancer screening, the threshold is then raised on P(malignant) to the value that reaches ~95% malignant recall on the validation split. That target is **not** reproduced on the test split (recall 0.860), which is the finding to carry forward: the screening threshold as shipped does not meet its stated recall.

## Explainability

### Method
SHAP (SHapley Additive exPlanations) using TreeExplainer. `src/explain.py` trains its **own XGBoost model** on a separate 80/20 split (test accuracy 0.9474) and explains that model. It does **not** explain the shipped logistic-regression pipeline, so these rankings describe how a gradient-boosted model uses the features on this dataset, not the selected model's coefficients.

### Top 5 Global Features (by mean |SHAP value|, XGBoost)
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
- **False negatives** (missed malignant tumors) carry severe consequences. The threshold was tuned to minimize this risk, but held-out recall is 0.860 against the 0.95 target (6 of 43 malignant test cases missed), so the risk is reduced but not controlled to the stated level.
- **False positives** result in unnecessary follow-up procedures (e.g., additional imaging, biopsies), causing patient anxiety and healthcare costs but are clinically preferable to missed diagnoses.

## Limitations

- **Small dataset:** 569 samples is insufficient for production clinical models. Results may not generalize to larger, more diverse populations.
- **Balanced classes:** Real-world cancer screening datasets are typically highly imbalanced (e.g., 95/5), making this benchmark optimistic.
- **No temporal validation:** All data comes from a single source and time period. Model performance may degrade with distributional shift.
- **Feature extraction assumed:** The model operates on pre-computed cell nucleus features, not raw medical images. A production system would need an upstream image processing pipeline.
- **No calibration analysis:** Predicted probabilities have not been validated for calibration. Rank ordering (AUC) is reliable, but absolute probability values may be miscalibrated.
- **Recall target not met on held-out data:** the validation-tuned threshold reaches 0.952 on validation but 0.860 on test. With about 43 malignant cases in a test split, the estimate moves by about 2.3 points per case, and no tuning method on this dataset size can pin it tighter.
- **Explanations are for a different model:** the SHAP step explains a separately trained XGBoost, not the selected logistic regression.
- **Version sensitivity:** Exact metrics vary by scikit-learn/XGBoost version. The ranking of algorithms and general patterns are stable, but absolute numbers should not be treated as fixed benchmarks.

## Recommendations

- Use this model strictly for educational and portfolio purposes.
- For clinical applications, retrain on larger, more representative datasets with proper regulatory oversight.
- Add probability calibration (Platt scaling or isotonic regression) before using predicted probabilities for clinical decision-making.
- Conduct prospective validation before any deployment scenario.
