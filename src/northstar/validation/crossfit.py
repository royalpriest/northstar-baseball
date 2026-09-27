from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

from northstar.models.baselines import (
    ClassificationMetrics,
    evaluate_probabilities,
    make_pitch_xgboost_model,
)


@dataclass(frozen=True)
class CrossFitPropensityResult:
    oof_raw: np.ndarray
    fold_id: np.ndarray
    fold_metrics: pd.DataFrame


@dataclass(frozen=True)
class WeightDiagnostics:
    cap: float
    rows: int
    clipped_rows: int
    clipped_pct: float
    raw_max: float
    clipped_max: float
    mean_weight: float
    effective_sample_size: float
    effective_sample_pct: float
    swing_effective_sample_size: float
    take_effective_sample_size: float


def crossfit_pitch_propensity(
    frame: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
    target: str,
    group_column: str = "game_pk",
    folds: int = 3,
    seed: int = 42,
) -> CrossFitPropensityResult:
    """Build game-grouped out-of-fold swing propensities.

    Every pitch from a game stays in one fold, so the model predicting that
    pitch was not trained on another pitch from the same game.
    """
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if group_column not in frame.columns:
        raise ValueError(f"Missing group column: {group_column}")
    if target not in frame.columns:
        raise ValueError(f"Missing target column: {target}")

    features = categorical_features + numeric_features
    y = frame[target].to_numpy(dtype="int8")
    groups = frame[group_column].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    oof = np.full(len(frame), np.nan, dtype="float64")
    fold_id = np.full(len(frame), -1, dtype="int16")
    metric_rows: list[dict[str, object]] = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(frame[features], y, groups=groups),
        start=1,
    ):
        model = make_pitch_xgboost_model(
            categorical_features,
            numeric_features,
        )
        model.fit(
            frame.iloc[train_idx][features],
            frame.iloc[train_idx][target],
        )
        probability = model.predict_proba(
            frame.iloc[valid_idx][features]
        )[:, 1]

        oof[valid_idx] = probability
        fold_id[valid_idx] = fold

        metrics = evaluate_probabilities(
            frame.iloc[valid_idx][target],
            probability,
        )
        metric_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "validation_rows": len(valid_idx),
                "validation_games": int(
                    frame.iloc[valid_idx][group_column].nunique()
                ),
                "roc_auc": metrics.roc_auc,
                "log_loss": metrics.log_loss,
                "brier": metrics.brier,
                "observed_swing_rate": float(
                    frame.iloc[valid_idx][target].mean()
                ),
                "mean_predicted_swing": float(probability.mean()),
            }
        )

    if np.isnan(oof).any() or (fold_id < 0).any():
        raise RuntimeError("Cross-fitting did not produce every OOF prediction.")

    return CrossFitPropensityResult(
        oof_raw=oof,
        fold_id=fold_id,
        fold_metrics=pd.DataFrame(metric_rows),
    )


def fit_isotonic_calibrator(
    probabilities: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
) -> IsotonicRegression:
    """Fit a monotonic propensity calibrator on out-of-fold predictions."""
    probability = np.asarray(probabilities, dtype="float64")
    y = np.asarray(target, dtype="float64")

    if len(probability) != len(y):
        raise ValueError("probabilities and target must have equal length")

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    calibrator.fit(probability, y)
    return calibrator


def apply_propensity_calibrator(
    calibrator: IsotonicRegression,
    probabilities: np.ndarray | pd.Series,
    epsilon: float = 1e-4,
) -> np.ndarray:
    values = np.asarray(
        calibrator.predict(
            np.asarray(probabilities, dtype="float64")
        ),
        dtype="float64",
    )
    return np.clip(values, epsilon, 1.0 - epsilon)


def propensity_metric_row(
    sample: str,
    y: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
) -> dict[str, object]:
    metrics: ClassificationMetrics = evaluate_probabilities(y, probabilities)
    probability = np.asarray(probabilities, dtype="float64")
    target = np.asarray(y, dtype="float64")
    return {
        "sample": sample,
        "rows": len(target),
        "roc_auc": metrics.roc_auc,
        "log_loss": metrics.log_loss,
        "brier": metrics.brier,
        "mean_predicted_swing": float(probability.mean()),
        "observed_swing_rate": float(target.mean()),
    }


def stabilized_raw_weights(
    actual_swing: pd.Series | np.ndarray,
    swing_probability: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return uncapped stabilized inverse-propensity weights."""
    action = np.asarray(actual_swing, dtype="float64")
    propensity = np.clip(
        np.asarray(swing_probability, dtype="float64"),
        1e-6,
        1.0 - 1e-6,
    )
    swing_rate = float(action.mean())
    return np.where(
        action == 1.0,
        swing_rate / propensity,
        (1.0 - swing_rate) / (1.0 - propensity),
    )


def effective_sample_size(weights: np.ndarray | pd.Series) -> float:
    values = np.asarray(weights, dtype="float64")
    denominator = float(np.square(values).sum())
    if denominator <= 0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def cap_weights(
    raw_weights: np.ndarray | pd.Series,
    cap: float,
) -> np.ndarray:
    if cap <= 0:
        raise ValueError("cap must be > 0")
    return np.minimum(
        np.asarray(raw_weights, dtype="float64"),
        float(cap),
    )


def weight_diagnostics(
    actual_swing: pd.Series | np.ndarray,
    raw_weights: np.ndarray | pd.Series,
    cap: float,
) -> WeightDiagnostics:
    action = np.asarray(actual_swing, dtype="int8")
    raw = np.asarray(raw_weights, dtype="float64")
    clipped = cap_weights(raw, cap)

    swing_weights = clipped[action == 1]
    take_weights = clipped[action == 0]
    clipped_rows = int((raw > cap).sum())

    return WeightDiagnostics(
        cap=float(cap),
        rows=len(raw),
        clipped_rows=clipped_rows,
        clipped_pct=float(clipped_rows / len(raw) * 100.0),
        raw_max=float(raw.max()),
        clipped_max=float(clipped.max()),
        mean_weight=float(clipped.mean()),
        effective_sample_size=effective_sample_size(clipped),
        effective_sample_pct=float(
            effective_sample_size(clipped) / len(clipped) * 100.0
        ),
        swing_effective_sample_size=effective_sample_size(swing_weights),
        take_effective_sample_size=effective_sample_size(take_weights),
    )
