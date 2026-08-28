"""
SHAP Explanation for Clinical Decision.

Generate per-prediction SHAP explanations for the best model.
Includes global feature importance and data leakage detection.

Usage:
    python explain.py
"""
import numpy as np
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_model():
    """Train XGBoost on breast cancer data and return model + data."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    model.fit(X_train_s, y_train)
    print(f"Test accuracy: {model.score(X_test_s, y_test):.4f}")

    return model, X_test, X_test_s, y_test, feature_names


def explain_single_prediction(model, X_test, X_test_s, y_test, feature_names,
                               shap_values, patient_idx=0):
    """Explain a single patient's prediction with SHAP."""
    patient_pred = model.predict_proba(X_test_s[patient_idx:patient_idx + 1])[0]
    print(f"\nPatient #{patient_idx} prediction:")
    print(f"  P(malignant) = {patient_pred[0]:.3f}")
    print(f"  P(benign)    = {patient_pred[1]:.3f}")
    print(f"  True label:    {'benign' if y_test[patient_idx] else 'malignant'}")

    patient_shap = shap_values[patient_idx]
    top_indices = np.argsort(np.abs(patient_shap))[::-1][:5]

    print(f"\nTop 5 features driving prediction:")
    print(f"  {'Feature':<25} {'Value':>8} {'SHAP':>8} {'Direction'}")
    print("  " + "-" * 60)
    for i in top_indices:
        direction = "-> benign" if patient_shap[i] > 0 else "-> malignant"
        print(f"  {feature_names[i]:<25} {X_test[patient_idx, i]:>8.3f}"
              f" {patient_shap[i]:>+8.4f} {direction}")


def global_feature_importance(shap_values, feature_names):
    """Show global top 10 features by mean |SHAP value|."""
    mean_shap = np.abs(shap_values).mean(axis=0)
    global_top = np.argsort(mean_shap)[::-1][:10]

    print(f"\nGlobal Top 10 Features (mean |SHAP value|):")
    for rank, i in enumerate(global_top, 1):
        print(f"  {rank:>2}. {feature_names[i]:<25} {mean_shap[i]:.4f}")

    return mean_shap


def check_data_leakage(feature_names, mean_shap):
    """Check for suspiciously high-importance ID-like features."""
    print("\nData Leakage Check:")
    suspicious = [f for i, f in enumerate(feature_names)
                  if mean_shap[i] > 0.5 and 'id' in f.lower()]
    if suspicious:
        print(f"  WARNING: Suspicious features with high SHAP: {suspicious}")
    else:
        print("  No suspicious ID-like features detected.")
    print("  Always verify top features are clinically meaningful!")


if __name__ == "__main__":
    model, X_test, X_test_s, y_test, feature_names = train_model()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)

    explain_single_prediction(model, X_test, X_test_s, y_test,
                               feature_names, shap_values, patient_idx=0)

    mean_shap = global_feature_importance(shap_values, feature_names)

    check_data_leakage(feature_names, mean_shap)
