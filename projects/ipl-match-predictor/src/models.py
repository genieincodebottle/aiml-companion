"""
models.py - Machine Learning Pipeline Utilities for IPL Prediction
===================================================================

Provides functions to build, train, and evaluate classification
(match-winner prediction) and regression (run-margin prediction)
models on engineered IPL features.

Usage
-----
>>> from src.models import build_classifier, build_regressor, evaluate_model
>>> pipeline = build_classifier()
>>> pipeline.fit(X_train, y_train)
>>> metrics = evaluate_model(pipeline, X_test, y_test, task="classification")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


# ===================================================================
# 1. Feature preparation
# ===================================================================
def prepare_features(
    matches_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    cat_columns: Optional[List[str]] = None,
    scale_numerical: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode categorical columns and optionally scale numerics.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.
    feature_cols : list of str, optional
        Columns to retain for modeling.  When *None*, a sensible default
        set is used.
    cat_columns : list of str, optional
        Categorical columns to one-hot encode.  Defaults to
        ``['team1', 'team2', 'toss_winner', 'venue', 'city',
        'toss_decision']``.
    scale_numerical : bool
        Whether to apply ``StandardScaler`` to numerical columns.

    Returns
    -------
    tuple of (pd.DataFrame, list of str)
        ``(encoded_df, feature_names)``
    """
    if feature_cols is None:
        feature_cols = [
            "season_year",
            # Engineered features (from compute_elo_ratings + engineer_features)
            "elo_team1",
            "elo_team2",
            "elo_diff",
            "elo_expected",
            "momentum_team1",
            "momentum_team2",
            "momentum_diff",
            "h2h_team1_winrate",
            "home_team1",
            "home_team2",
            # Interaction features
            "elo_x_momentum_t1",
            "elo_x_momentum_t2",
            # Team identity (moderate-cardinality categorical)
            "team1",
            "team2",
            # Toss decision (low-cardinality categorical)
            "toss_decision",
        ]

    available = [c for c in feature_cols if c in matches_df.columns]
    model_df = matches_df[available].copy()

    # Drop rows with missing key categoricals
    if cat_columns is None:
        cat_columns = ["team1", "team2", "toss_winner", "venue", "city", "toss_decision"]
    cat_present = [c for c in cat_columns if c in model_df.columns]
    model_df.dropna(subset=cat_present, inplace=True)

    # One-hot encode
    model_df = pd.get_dummies(model_df, columns=cat_present, drop_first=True, dummy_na=False)

    # Scale numerical columns
    if scale_numerical:
        num_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            scaler = StandardScaler()
            model_df[num_cols] = scaler.fit_transform(model_df[num_cols])

    feature_names = model_df.columns.tolist()
    logger.info(
        "Prepared feature matrix — shape: %s, features: %d",
        model_df.shape,
        len(feature_names),
    )
    return model_df, feature_names


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature matrix and target into train / test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float
        Fraction of data for testing.
    random_state : int
        Reproducibility seed.
    stratify : bool
        Whether to stratify on ``y`` (classification tasks).

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``
    """
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )


# ===================================================================
# 2. Model builders
# ===================================================================
def build_classifier(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> Pipeline:
    """Build a Random Forest classification pipeline.

    The pipeline consists of a ``StandardScaler`` followed by a
    ``RandomForestClassifier``.  This mirrors the approach used in the
    notebook for predicting whether Team 1 wins.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum tree depth.
    min_samples_split : int
        Minimum samples to split an internal node.
    min_samples_leaf : int
        Minimum samples at a leaf node.
    random_state : int
        Reproducibility seed.
    class_weight : str
        Strategy for handling class imbalance.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted classification pipeline.

    Examples
    --------
    >>> clf = build_classifier(n_estimators=200)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    class_weight=class_weight,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    logger.info("Built RF classifier pipeline — %d estimators", n_estimators)
    return pipeline


def build_regressor(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    loss: str = "huber",
) -> Pipeline:
    """Build a Gradient Boosting regression pipeline.

    Used for predicting continuous targets such as run margins.
    Defaults to Huber loss, which is robust to outlier blowout wins
    (e.g., 146-run margins) by combining MSE for small errors with
    MAE for large errors.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.
    max_depth : int
        Maximum depth of individual trees.
    learning_rate : float
        Shrinkage parameter.
    min_samples_split : int
        Minimum samples to split an internal node.
    min_samples_leaf : int
        Minimum samples at a leaf node.
    random_state : int
        Reproducibility seed.
    loss : str
        Loss function (``'huber'``, ``'squared_error'``, ``'absolute_error'``, etc.).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted regression pipeline.

    Examples
    --------
    >>> reg = build_regressor(n_estimators=150, learning_rate=0.05)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    loss=loss,
                ),
            ),
        ]
    )
    logger.info(
        "Built GBR pipeline — %d estimators, lr=%.3f", n_estimators, learning_rate
    )
    return pipeline


def build_xgb_classifier(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Build an XGBoost classification pipeline.

    XGBoost adds L1/L2 regularization, column subsampling, and
    native missing-value handling on top of gradient boosting.
    Typically 1-3% more accurate than Random Forest on structured
    tabular data like IPL match features.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage (eta).
    subsample : float
        Fraction of rows sampled per tree (row subsampling).
    colsample_bytree : float
        Fraction of features sampled per tree (column subsampling).
    reg_alpha : float
        L1 regularization on leaf weights.
    reg_lambda : float
        L2 regularization on leaf weights.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted XGBoost classification pipeline.

    Raises
    ------
    ImportError
        If ``xgboost`` is not installed.

    Examples
    --------
    >>> clf = build_xgb_classifier(n_estimators=300)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """
    if not _HAS_XGBOOST:
        raise ImportError(
            "xgboost is required for build_xgb_classifier. "
            "Install it with: pip install xgboost"
        )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    logger.info(
        "Built XGBoost classifier pipeline — %d estimators, lr=%.3f, depth=%d",
        n_estimators,
        learning_rate,
        max_depth,
    )
    return pipeline


# ===================================================================
# 2b. Neural Network classifier (PyTorch)
# ===================================================================
if _HAS_TORCH:

    class _IPLNet(nn.Module):
        """Simple feedforward network for binary match-winner prediction.

        Architecture: Input -> 128 (ReLU, Dropout) -> 64 (ReLU, Dropout)
                      -> 32 (ReLU) -> 1 (Sigmoid)
        """

        def __init__(self, input_dim: int, dropout: float = 0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.67),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)


    class NeuralClassifier(BaseEstimator, ClassifierMixin):
        """Scikit-learn compatible wrapper around a PyTorch neural network.

        Implements ``fit`` / ``predict`` / ``predict_proba`` so it drops
        into a ``sklearn.pipeline.Pipeline`` seamlessly and works with
        ``cross_val_score``, ``classification_report``, etc.

        Parameters
        ----------
        hidden_sizes : tuple of int
            Neuron counts for hidden layers (architecture is fixed to
            128-64-32 inside ``_IPLNet``, this param is reserved for
            future flexibility).
        dropout : float
            Dropout probability for regularization.
        lr : float
            Adam optimizer learning rate.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size for training.
        random_state : int
            Reproducibility seed.
        """

        def __init__(
            self,
            dropout: float = 0.3,
            lr: float = 0.001,
            epochs: int = 100,
            batch_size: int = 32,
            random_state: int = 42,
        ):
            self.dropout = dropout
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            X_arr = np.asarray(X, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

            self.model_ = _IPLNet(X_arr.shape[1], dropout=self.dropout)
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
            criterion = nn.BCELoss()

            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_arr), torch.from_numpy(y_arr)
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )

            self.model_.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    preds = self.model_(X_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            logger.debug(
                "Neural net training complete — %d epochs, final batch loss: %.4f",
                self.epochs,
                epoch_loss / max(len(loader), 1),
            )
            return self

        def predict_proba(self, X):
            self.model_.eval()
            X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
            with torch.no_grad():
                probs = self.model_(X_t).numpy().flatten()
            return np.column_stack([1 - probs, probs])

        def predict(self, X):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int)


def build_neural_classifier(
    dropout: float = 0.3,
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 42,
) -> Pipeline:
    """Build a PyTorch neural network classification pipeline.

    Uses a 3-layer MLP (128-64-32) with batch normalization and
    dropout. Wrapped in a scikit-learn compatible estimator so it
    works with ``cross_val_score``, ``Pipeline``, etc.

    Parameters
    ----------
    dropout : float
        Dropout probability (applied after first two hidden layers).
    lr : float
        Adam optimizer learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted neural network classification pipeline.

    Raises
    ------
    ImportError
        If ``torch`` is not installed.

    Examples
    --------
    >>> clf = build_neural_classifier(epochs=150, lr=0.0005)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for build_neural_classifier. "
            "Install it with: pip install torch"
        )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                NeuralClassifier(
                    dropout=dropout,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    random_state=random_state,
                ),
            ),
        ]
    )
    logger.info(
        "Built Neural Net classifier pipeline — epochs=%d, lr=%.4f, dropout=%.2f",
        epochs,
        lr,
        dropout,
    )
    return pipeline


# ===================================================================
# 3. Evaluation
# ===================================================================
def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "classification",
) -> Dict[str, Any]:
    """Evaluate a fitted pipeline on a test set.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted model pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True labels / values.
    task : str
        ``'classification'`` or ``'regression'``.

    Returns
    -------
    dict
        Dictionary of metric names to values.  For classification:
        ``accuracy``, ``f1``, ``confusion_matrix``, ``report``.
        For regression: ``mae``, ``rmse``, ``r2``.

    Examples
    --------
    >>> metrics = evaluate_model(clf, X_test, y_test, task="classification")
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    y_pred = pipeline.predict(X_test)
    metrics: Dict[str, Any] = {}

    if task == "classification":
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["f1"] = f1_score(y_test, y_pred, average="weighted")
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        metrics["report"] = classification_report(
            y_test,
            y_pred,
            target_names=["Team 2 Wins/Other", "Team 1 Wins"],
            output_dict=True,
        )
        logger.info(
            "Classification — Accuracy: %.4f  F1: %.4f",
            metrics["accuracy"],
            metrics["f1"],
        )
    elif task == "regression":
        metrics["mae"] = mean_absolute_error(y_test, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["r2"] = r2_score(y_test, y_pred)
        logger.info(
            "Regression — MAE: %.4f  RMSE: %.4f  R2: %.4f",
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )
    else:
        raise ValueError(f"Unknown task: {task!r}. Use 'classification' or 'regression'.")

    return metrics


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, float]:
    """Run k-fold cross-validation and return summary statistics.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Unfitted or fitted pipeline (will be re-fitted per fold).
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full target vector.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric (e.g. ``'accuracy'``, ``'neg_mean_squared_error'``).

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``scores`` (list of per-fold values).
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
    result = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores.tolist(),
    }
    logger.info(
        "Cross-validation (%d folds, %s) — mean: %.4f +/- %.4f",
        cv,
        scoring,
        result["mean"],
        result["std"],
    )
    return result


def get_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract feature importances from a fitted tree-based pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline whose last step exposes ``feature_importances_``.
    feature_names : list of str
        Column names corresponding to the feature matrix.
    top_n : int
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``Feature``, ``Importance``, sorted descending.
    """
    # Access the last step of the pipeline
    estimator = pipeline.steps[-1][1]

    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError(
            f"{type(estimator).__name__} does not expose feature_importances_."
        )

    importances = estimator.feature_importances_
    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    return fi_df.head(top_n).reset_index(drop=True)
