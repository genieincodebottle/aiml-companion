"""
Algorithm Showdown - Train 6 competing algorithms on clinical data.

Capstone: ML Algorithms Track
Dataset: UCI Breast Cancer Wisconsin (569 samples, 30 features)
Goal: Compare algorithms with cost-sensitive evaluation and threshold tuning.

Usage:
    python train.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load UCI Breast Cancer dataset."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.bincount(y)} (0=malignant, 1=benign)")
    print(f"Class balance: {y.mean():.1%} benign")
    return X, y, feature_names


def build_algorithms():
    """Build dictionary of 6 competing algorithms, each in a Pipeline with scaling."""
    algorithms = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        'KNN (k=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ]),
        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(max_depth=5, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=200, max_features='sqrt', random_state=42, n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=4,
                random_state=42, eval_metric='logloss', verbosity=0
            ))
        ]),
    }
    return algorithms


def cross_validate_all(X, y, algorithms):
    """Run StratifiedKFold cross-validation on all algorithms."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    print("\n" + "=" * 75)
    print(f"{'Algorithm':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("=" * 75)

    results = {}
    for name, pipeline in algorithms.items():
        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        results[name] = {
            metric: cv_results[f'test_{metric}'].mean()
            for metric in scoring
        }
        r = results[name]
        print(f"{name:<22} {r['accuracy']:>8.3f}  {r['precision']:>9.3f}"
              f"  {r['recall']:>7.3f}  {r['f1']:>7.3f}  {r['roc_auc']:>7.3f}")

    print("=" * 75)
    return results


def find_best_per_metric(results):
    """Identify best algorithm per metric."""
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    print("\nBest per metric:")
    for metric in scoring:
        best_algo = max(results, key=lambda a: results[a][metric])
        print(f"  {metric:<12}: {best_algo} ({results[best_algo][metric]:.4f})")


def tune_threshold(X, y, algorithms, target_recall=0.95):
    """Tune decision threshold for cancer screening (~95% malignant recall target)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best_pipe = algorithms['XGBoost']
    best_pipe.fit(X_train, y_train)

    # For cancer screening: optimize MALIGNANT recall (catch maximum cancers)
    y_proba_malign = best_pipe.predict_proba(X_test)[:, 0]  # P(malignant)

    # Compute PR curve for malignant class (flip labels so malignant=1)
    precisions, recalls, thresholds = precision_recall_curve(
        (y_test == 0).astype(int),  # 1 = malignant (positive class)
        y_proba_malign               # P(malignant)
    )

    idx = np.argmin(np.abs(recalls - target_recall))
    opt_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

    print(f"\nThreshold for {target_recall:.0%} malignant recall: {opt_threshold:.3f}")
    print(f"  Precision at this threshold: {precisions[idx]:.3f}")

    return best_pipe, opt_threshold


if __name__ == "__main__":
    # Load data
    X, y, feature_names = load_data()

    # Build and evaluate all algorithms
    algorithms = build_algorithms()
    results = cross_validate_all(X, y, algorithms)

    # Best per metric
    find_best_per_metric(results)

    # Threshold tuning for cancer screening
    best_pipe, threshold = tune_threshold(X, y, algorithms)

    # Save best model
    os.makedirs('artifacts/models', exist_ok=True)
    joblib.dump(best_pipe, 'artifacts/models/best_model.joblib')
    joblib.dump({'threshold': threshold, 'features': list(feature_names)},
                'artifacts/models/model_config.joblib')
    print(f"\nModel saved to artifacts/models/best_model.joblib")

    print("\nKey insight: No single algorithm wins all metrics.")
    print("Choose based on your specific cost tradeoff.")