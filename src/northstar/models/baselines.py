from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


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
