from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_corr(
    actual: pd.Series,
    predicted: np.ndarray,
    method: str,
) -> float:
    left = pd.Series(actual.to_numpy(dtype="float64"))
    right = pd.Series(
        np.asarray(predicted, dtype="float64"),
        index=left.index,
    )
    if len(left) < 2:
        return float("nan")
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.corr(right, method=method))


def _regression_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    y = actual.to_numpy(dtype="float64")
    p = np.asarray(predicted, dtype="float64")
    return {
        "r2": float(r2_score(y, p)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(np.mean(np.square(y - p)))),
        "pearson_r": _safe_corr(actual, p, method="pearson"),
        "spearman_rho": _safe_corr(actual, p, method="spearman"),
    }


def compare_future_incremental_prediction(
    half_frame: pd.DataFrame,
    *,
    future_target: str,
    northstar_feature: str,
    feature_sets: Mapping[str, Sequence[str]],
    component: str,
    folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether an early NorthStar feature improves late-skill prediction.

    Each comparison uses the same players and the same cross-validation folds
    for:
      A) early conventional constructs only
      B) the same constructs plus the early NorthStar adjustment

    The target is a *late-half observed conventional skill*, not a NorthStar
    residual. This avoids the algebraic trap of residualizing
    (observed - expected) against observed and then interpreting a stable
    negative expected component as new player skill.
    """
    required = {
        "batter",
        "season_half",
        future_target,
        northstar_feature,
    }
    missing = required.difference(half_frame.columns)
    if missing:
        raise ValueError(
            f"Future-prediction data is missing: {sorted(missing)}"
        )
    if folds < 3:
        raise ValueError("folds must be >= 3")
    if not component:
        raise ValueError("component must be non-empty")

    early = half_frame.loc[
        half_frame["season_half"].eq("early")
    ].copy()
    late = half_frame.loc[
        half_frame["season_half"].eq("late"),
        ["batter", future_target],
    ].copy()
    late = late.rename(
        columns={future_target: "future_target_value"}
    )

    prediction_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for label, requested_features in feature_sets.items():
        conventional_features = [
            feature
            for feature in requested_features
            if feature in early.columns
        ]
        if not conventional_features:
            continue

        columns = (
            ["batter", northstar_feature]
            + conventional_features
        )
        current = early.loc[:, columns].merge(
            late,
            on="batter",
            how="inner",
            validate="one_to_one",
        )

        numeric_columns = (
            [northstar_feature, "future_target_value"]
            + conventional_features
        )
        for column in numeric_columns:
            current[column] = pd.to_numeric(
                current[column],
                errors="coerce",
            )
        current = current.dropna(
            subset=numeric_columns
        ).reset_index(drop=True)

        if len(current) < max(30, folds * 3):
            continue

        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
        splits = list(splitter.split(current))

        def _model() -> Pipeline:
            return Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=1.0)),
                ]
            )

        conventional_prediction = cross_val_predict(
            _model(),
            current[conventional_features],
            current["future_target_value"],
            cv=splits,
        )
        augmented_features = (
            conventional_features + [northstar_feature]
        )
        augmented_prediction = cross_val_predict(
            _model(),
            current[augmented_features],
            current["future_target_value"],
            cv=splits,
        )

        conventional_metrics = _regression_metrics(
            current["future_target_value"],
            conventional_prediction,
        )
        augmented_metrics = _regression_metrics(
            current["future_target_value"],
            augmented_prediction,
        )

        summary_rows.append(
            {
                "component": component,
                "future_target": future_target,
                "construct_set": label,
                "players": len(current),
                "conventional_features": ";".join(
                    conventional_features
                ),
                "northstar_feature": northstar_feature,
                "base_oof_r2": conventional_metrics["r2"],
                "augmented_oof_r2": augmented_metrics["r2"],
                "delta_oof_r2": (
                    augmented_metrics["r2"]
                    - conventional_metrics["r2"]
                ),
                "base_mae": conventional_metrics["mae"],
                "augmented_mae": augmented_metrics["mae"],
                "delta_mae": (
                    augmented_metrics["mae"]
                    - conventional_metrics["mae"]
                ),
                "base_rmse": conventional_metrics["rmse"],
                "augmented_rmse": augmented_metrics["rmse"],
                "delta_rmse": (
                    augmented_metrics["rmse"]
                    - conventional_metrics["rmse"]
                ),
                "base_pearson_r": conventional_metrics["pearson_r"],
                "augmented_pearson_r": augmented_metrics["pearson_r"],
                "delta_pearson_r": (
                    augmented_metrics["pearson_r"]
                    - conventional_metrics["pearson_r"]
                ),
                "base_spearman_rho": conventional_metrics[
                    "spearman_rho"
                ],
                "augmented_spearman_rho": augmented_metrics[
                    "spearman_rho"
                ],
                "delta_spearman_rho": (
                    augmented_metrics["spearman_rho"]
                    - conventional_metrics["spearman_rho"]
                ),
            }
        )

        part = current[
            [
                "batter",
                "future_target_value",
                northstar_feature,
            ]
            + conventional_features
        ].copy()
        part.insert(1, "component", component)
        part.insert(2, "construct_set", label)
        part["base_oof_prediction"] = conventional_prediction
        part["augmented_oof_prediction"] = augmented_prediction
        part["base_error"] = (
            part["future_target_value"]
            - part["base_oof_prediction"]
        )
        part["augmented_error"] = (
            part["future_target_value"]
            - part["augmented_oof_prediction"]
        )
        prediction_parts.append(part)

    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    return predictions, pd.DataFrame(summary_rows)
