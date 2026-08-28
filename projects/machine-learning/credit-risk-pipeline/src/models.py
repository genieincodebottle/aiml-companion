"""
Model Training & Selection
===========================

Trains two models using sklearn Pipeline + ColumnTransformer:
- Logistic Regression (interpretable baseline, ECOA/FCRA compliant)
- Gradient Boosting Classifier (performance challenger)

Both pipelines use:
- KNN imputation for MNAR-pattern missing values
- StandardScaler for numeric features
- OneHotEncoder for categorical features
- StratifiedKFold cross-validation

Functions
---------
train_and_evaluate     Train models with cross-validation.
build_preprocessor     Build ColumnTransformer for mixed types.
build_lr_pipeline      Build Logistic Regression pipeline.
build_gbc_pipeline     Build Gradient Boosting pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, cfg: dict) -> dict:
    """Train models with stratified cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered data with ``target`` column.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Model name -> metrics dict with keys:
        ``roc_auc_mean``, ``roc_auc_std``, ``recall_mean``,
        ``f1_mean``, ``pipeline`` (fitted on full data).
    """
    target_col = cfg.get("target_column", "target")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    # Build pipelines
    pipelines = {
        "LogisticRegression": build_lr_pipeline(numeric_cols, categorical_cols, cfg),
        "GradientBoosting": build_gbc_pipeline(numeric_cols, categorical_cols, cfg),
    }

    # Cross-validation
    n_folds = cfg.get("cv_folds", 5)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scoring = ["roc_auc", "recall", "f1"]

    results = {}
    for name, pipeline in pipelines.items():
        logger.info(f"Training {name} ({n_folds}-fold CV)...")

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        results[name] = {
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "recall_mean": scores["test_recall"].mean(),
            "recall_std": scores["test_recall"].std(),
            "f1_mean": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
        }

        logger.info(
            f"  {name}: ROC-AUC={results[name]['roc_auc_mean']:.3f} "
            f"(+/-{results[name]['roc_auc_std']:.3f})"
        )

        # Fit on full data for deployment
        pipeline.fit(X, y)
        results[name]["pipeline"] = pipeline

    # Save best model
    best_name = max(results, key=lambda k: results[k]["roc_auc_mean"])
    model_path = PROJECT_ROOT / "artifacts" / "results" / "best_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results[best_name]["pipeline"], model_path)
    logger.info(f"Best model ({best_name}) saved to {model_path}")

    return results


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: dict,
) -> ColumnTransformer:
    """Build a ColumnTransformer for mixed-type preprocessing.

    Numeric: KNN imputation -> StandardScaler.
    Categorical: OneHotEncoder (handle_unknown='ignore').

    Parameters
    ----------
    numeric_cols : list[str]
        Numeric column names.
    categorical_cols : list[str]
        Categorical column names.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    ColumnTransformer
        Fitted preprocessor.
    """
    knn_neighbors = cfg.get("knn_imputer_neighbors", 5)

    numeric_pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=knn_neighbors)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def build_lr_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: dict,
) -> Pipeline:
    """Build Logistic Regression pipeline.

    LR is the primary model for regulatory compliance (ECOA/FCRA).
    Coefficients provide interpretable adverse action reasons.

    Parameters
    ----------
    numeric_cols : list[str]
        Numeric column names.
    categorical_cols : list[str]
        Categorical column names.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    Pipeline
        sklearn Pipeline ready for fit/predict.
    """
    lr_params = cfg.get("logistic_regression", {})

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, cfg)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=lr_params.get("C", 1.0),
            max_iter=lr_params.get("max_iter", 1000),
            class_weight=lr_params.get("class_weight", "balanced"),
            random_state=42,
            solver="lbfgs",
        )),
    ])

    return pipeline


def build_gbc_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: dict,
) -> Pipeline:
    """Build Gradient Boosting Classifier pipeline.

    GBC is the performance challenger model, typically achieving
    higher ROC-AUC than LR but with less interpretability.

    Parameters
    ----------
    numeric_cols : list[str]
        Numeric column names.
    categorical_cols : list[str]
        Categorical column names.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    Pipeline
        sklearn Pipeline ready for fit/predict.
    """
    gbc_params = cfg.get("gradient_boosting", {})

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, cfg)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=gbc_params.get("n_estimators", 200),
            learning_rate=gbc_params.get("learning_rate", 0.1),
            max_depth=gbc_params.get("max_depth", 4),
            subsample=gbc_params.get("subsample", 0.8),
            random_state=42,
        )),
    ])

    return pipeline