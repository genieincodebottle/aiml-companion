"""
Standalone evaluation - confusion matrices, PR curves, threshold tuning.

Loads the saved model from train.py and runs full evaluation on the test set.

Usage:
    python train.py   # first, to save the model
    python evaluate.py
"""
import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score
)
import joblib


def load_model():
    """Load the saved model and config."""
    model_path = 'artifacts/models/best_model.joblib'
    config_path = 'artifacts/models/model_config.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Run 'python train.py' first to train and save the model."
        )
    model = joblib.load(model_path)
    config = joblib.load(config_path)
    return model, config


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Run full evaluation with confusion matrix and classification report."""
    # P(malignant) for threshold-based decisions
    y_proba_malign = model.predict_proba(X_test)[:, 0]  # P(malignant)
    # Flag as malignant if P(malignant) >= threshold
    y_pred = np.where(y_proba_malign >= threshold, 0, 1)  # 0=malignant, 1=benign

    print("=" * 60)
    print(f"EVALUATION REPORT (threshold={threshold:.3f} on P(malignant))")
    print("=" * 60)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  {'':>15} Predicted:0  Predicted:1")
    print(f"  {'Actual:0':>15} {cm[0, 0]:>10}  {cm[0, 1]:>10}")
    print(f"  {'Actual:1':>15} {cm[1, 0]:>10}  {cm[1, 1]:>10}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['malignant', 'benign']))

    # AUC: use P(benign) since label 1 = benign
    auc = roc_auc_score(y_test, 1 - y_proba_malign)
    print(f"ROC AUC: {auc:.4f}")

    return y_proba_malign, y_pred


def threshold_analysis(y_test, y_proba_malign):
    """Analyze precision-recall tradeoff at different malignant recall targets."""
    precisions, recalls, thresholds = precision_recall_curve(
        (y_test == 0).astype(int),  # 1 = malignant (positive class)
        y_proba_malign               # P(malignant)
    )

    print("\nThreshold Analysis (malignant recall targets):")
    print(f"  {'Target':>8} {'Threshold':>10} {'Precision':>10} {'Recall':>10}")
    print("  " + "-" * 40)
    for target_recall in [0.95, 0.90, 0.85, 0.80]:
        idx = np.argmin(np.abs(recalls - target_recall))
        if idx < len(thresholds):
            print(f"  {target_recall:>7.0%} {thresholds[idx]:>10.3f} {precisions[idx]:>10.3f} {recalls[idx]:>10.3f}")


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model, config = load_model()
    threshold = config.get('threshold', 0.5)

    y_proba_malign, y_pred = evaluate_model(model, X_test, y_test, threshold)

    print("\n--- Default threshold (0.5) for comparison ---")
    evaluate_model(model, X_test, y_test, threshold=0.5)

    threshold_analysis(y_test, y_proba_malign)
