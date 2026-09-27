from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


@dataclass(frozen=True)
class ClassificationMetrics:
    roc_auc: float
    log_loss: float
    brier: float


def make_scoring_baseline(
    categorical_features: list[str], numeric_features: list[str]
) -> Pipeline:
    """Logistic-regression baseline for P(score >= 1 run before inning end)."""
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", categorical, categorical_features),
            ("numeric", numeric, numeric_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )


def make_xgboost_scoring_model(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Gradient-boosted tree model for contextual scoring probability."""
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", categorical, categorical_features),
            ("numeric", numeric, numeric_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                XGBClassifier(
                    n_estimators=350,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=5,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_probabilities(y: pd.Series, probabilities: pd.Series | list[float]) -> ClassificationMetrics:
    """Evaluate probability predictions for a binary scoring target."""
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y, probabilities)),
        log_loss=float(log_loss(y, probabilities, labels=[0, 1])),
        brier=float(brier_score_loss(y, probabilities)),
    )


def evaluate_binary_classifier(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> ClassificationMetrics:
    probabilities = model.predict_proba(x)[:, 1]
    return evaluate_probabilities(y, probabilities)


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    pearson_r: float
    spearman_rho: float


def _make_regression_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, categorical_features),
            ("numeric", numeric, numeric_features),
        ]
    )


def make_ridge_run_value_model(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Regularized linear baseline for expected terminal-event RE24."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                _make_regression_preprocessor(
                    categorical_features, numeric_features, scale_numeric=True
                ),
            ),
            ("model", Ridge(alpha=2.0)),
        ]
    )


def make_xgboost_run_value_model(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Gradient-boosted regression model for expected terminal-event RE24."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                _make_regression_preprocessor(
                    categorical_features, numeric_features, scale_numeric=False
                ),
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=450,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=6,
                    reg_lambda=1.5,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_regression_predictions(
    y: pd.Series,
    predictions: pd.Series | list[float],
) -> RegressionMetrics:
    """Evaluate continuous run-value predictions."""
    y_true = pd.Series(y, dtype="float64").reset_index(drop=True)
    y_pred = pd.Series(predictions, dtype="float64").reset_index(drop=True)
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=rmse,
        r2=float(r2_score(y_true, y_pred)),
        pearson_r=float(y_true.corr(y_pred, method="pearson")),
        spearman_rho=float(y_true.corr(y_pred, method="spearman")),
    )


def evaluate_regressor(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> RegressionMetrics:
    return evaluate_regression_predictions(y, model.predict(x))


def make_multiclass_logistic_model(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Multinomial logistic model for terminal PA outcome probabilities."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                _make_regression_preprocessor(
                    categorical_features,
                    numeric_features,
                    scale_numeric=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def make_xgboost_outcome_model(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Gradient-boosted multiclass model for terminal PA outcomes."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                _make_regression_preprocessor(
                    categorical_features,
                    numeric_features,
                    scale_numeric=False,
                ),
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=6,
                    reg_lambda=1.5,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _make_low_memory_pitch_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Sparse float32 preprocessing for full-season pitch-level models."""
    categorical = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )

    if scale_numeric:
        numeric = Pipeline(
            steps=[
                ("scale", StandardScaler()),
            ]
        )
    else:
        numeric = "passthrough"

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, categorical_features),
            ("numeric", numeric, numeric_features),
        ],
        sparse_threshold=1.0,
    )


def make_pitch_logistic_model(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Memory-efficient sparse logistic model for swing/take decisions."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                _make_low_memory_pitch_preprocessor(
                    categorical_features,
                    numeric_features,
                    scale_numeric=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    solver="saga",
                    random_state=42,
                ),
            ),
        ]
    )


def make_pitch_xgboost_model(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Memory-efficient sparse XGBoost model for swing/take decisions."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                _make_low_memory_pitch_preprocessor(
                    categorical_features,
                    numeric_features,
                    scale_numeric=False,
                ),
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=5,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    max_bin=128,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
