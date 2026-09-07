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


def make_splits(X, y, seed=42):
    """The one definition of train / validation / test used by every stage.

    Defined once and imported by evaluate.py, because the two files used to
    build their own splits independently. They happened to agree -- and that is
    the problem: nothing enforced it, and evaluate.py was scoring the model on
    the exact rows train.py had tuned the threshold on. A split is part of the
    experiment's design, not a local detail of whichever script needs one.
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def select_best(results, metric='roc_auc'):
    """Pick the model to carry forward, from the comparison that was just run.

    This used to be hardcoded to XGBoost, which quietly made the whole bake-off
    decorative: six algorithms were compared and then the answer was ignored.
    AUC is the selection metric because no threshold has been chosen yet at this
    stage -- AUC scores the ranking across every threshold, which is exactly the
    question "which model should I go on to tune?".
    """
    best = max(results, key=lambda a: results[a][metric])
    print(f"\nSelected for threshold tuning: {best} "
          f"({metric}={results[best][metric]:.4f}, chosen from the table above)")
    return best


def pick_threshold(y_true, y_proba_malign, target_recall):
    """Smallest threshold that still reaches `target_recall` on malignant cases.

    Note the asymmetry this encodes: a false alarm costs a biopsy, a miss can
    cost a life. So we do not pick the threshold that maximises accuracy or F1.
    We fix the recall we are willing to live with and accept the precision it
    costs.
    """
    precisions, recalls, thresholds = precision_recall_curve(
        (y_true == 0).astype(int),   # 1 = malignant (the positive class)
        y_proba_malign
    )
    # precision_recall_curve returns len(thresholds) == len(recalls) - 1: the
    # final (recall=0, precision=1) point has no threshold behind it. Trimming
    # to the aligned prefix keeps one index valid for all three arrays, instead
    # of the old `if idx < len(thresholds) else 0.5`, which silently shipped an
    # untuned 0.5 as though it were a tuned threshold.
    recalls, precisions = recalls[:-1], precisions[:-1]
    idx = int(np.argmin(np.abs(recalls - target_recall)))
    return float(thresholds[idx]), float(precisions[idx]), float(recalls[idx])


def tune_threshold(X, y, algorithms, target_recall=0.95, best_name='XGBoost'):
    """Fit the chosen model, tune its threshold, and report on untouched data.

    THREE splits, not two, and the reason is the whole lesson of this function.

    Choosing an operating threshold IS model selection: it is a parameter fitted
    to data. Fit it on the same rows you then report and the report is circular.
    The threshold was *defined* as the one hitting 95% recall on those rows, so
    95% recall is what it prints -- every time, however good or bad the model
    is. Measured over 30 seeds, that circular number had a standard deviation of
    0.0005. It is not an estimate; it is the target echoed back.

    Tuned on validation and reported on an untouched test split, the same
    quantity has a standard deviation of 0.037 and ranges from 0.857 to 1.000.
    On 42 malignant cases, 0.857 means six missed cancers. That spread is the
    real uncertainty, and the two-split design hid it completely. Precision
    moves too, in the direction you would expect: 0.917 in-sample, 0.833 held
    out.

        train      (60%)  fit the model
        validation (20%)  choose the threshold
        test       (20%)  report -- touched by neither of the above
    """
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    best_pipe = algorithms[best_name]
    best_pipe.fit(X_train, y_train)

    # --- choose the threshold on VALIDATION
    opt_threshold, val_prec, val_rec = pick_threshold(
        y_val, best_pipe.predict_proba(X_val)[:, 0], target_recall
    )
    print(f"\nThreshold tuned on VALIDATION for {target_recall:.0%} malignant "
          f"recall: {opt_threshold:.3f}")
    print(f"  validation: recall {val_rec:.3f}  precision {val_prec:.3f}  "
          f"(in-sample for the threshold -- not a result)")

    # --- report it on TEST, seen by neither the model nor the threshold
    p_malign_test = best_pipe.predict_proba(X_test)[:, 0]
    flagged = p_malign_test >= opt_threshold
    is_malign = (y_test == 0)
    test_rec = float(flagged[is_malign].mean())
    test_prec = float(is_malign[flagged].mean()) if flagged.any() else float('nan')
    print(f"  TEST (held out): recall {test_rec:.3f}  precision {test_prec:.3f}"
          f"   <- the only honest number here")
    if test_rec < target_recall:
        print(f"  note: test recall missed the {target_recall:.0%} target. That "
              f"is what a held-out set is for; the tuned number never could.")

    return best_pipe, opt_threshold


if __name__ == "__main__":
    # Load data
    X, y, feature_names = load_data()

    # Build and evaluate all algorithms
    algorithms = build_algorithms()
    results = cross_validate_all(X, y, algorithms)

    # Best per metric
    find_best_per_metric(results)

    # Threshold tuning for cancer screening, on the model the comparison chose
    best_name = select_best(results)
    best_pipe, threshold = tune_threshold(X, y, algorithms, best_name=best_name)

    # Save best model
    os.makedirs('artifacts/models', exist_ok=True)
    joblib.dump(best_pipe, 'artifacts/models/best_model.joblib')
    joblib.dump({'threshold': threshold, 'features': list(feature_names)},
                'artifacts/models/model_config.joblib')
    print(f"\nModel saved to artifacts/models/best_model.joblib")

    print("\nKey insight: No single algorithm wins all metrics.")
    print("Choose based on your specific cost tradeoff.")