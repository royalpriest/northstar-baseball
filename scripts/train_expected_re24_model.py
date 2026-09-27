from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from northstar.features.context import add_backward_looking_terminal_context
from northstar.models.baselines import (
    evaluate_regression_predictions,
    evaluate_regressor,
    make_ridge_run_value_model,
    make_xgboost_run_value_model,
)
from northstar.models.expected_run_value import (
    EXPECTED_RE24_CATEGORICAL_FEATURES,
    EXPECTED_RE24_NUMERIC_FEATURES,
    EXPECTED_RE24_TARGET,
    chronological_run_value_split,
    fit_state_run_value_baseline,
    predict_state_run_value,
    prepare_expected_re24_frame,
)


DEFAULT_INPUT = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_METRICS = Path("reports/tables/expected_re24_model_metrics_2025.csv")
DEFAULT_RIDGE = Path("reports/tables/expected_re24_ridge_coefficients_2025.csv")
DEFAULT_XGB = Path("reports/tables/expected_re24_xgboost_feature_importance_2025.csv")
DEFAULT_FIGURE = Path("reports/figures/expected_re24_holdout_calibration_2025.png")
DEFAULT_PREDICTIONS = Path("data/processed/expected_re24_test_predictions_2025.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NorthStar expected terminal-event RE24 models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--validation-start", default="2025-08-01")
    parser.add_argument("--test-start", default="2025-09-01")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--include-shifted-pa",
        action="store_true",
        help="Include PAs with within-PA state changes in the expected-value model.",
    )
    return parser.parse_args()


def _metrics_row(model_name: str, split_name: str, metrics) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split_name,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "pearson_r": metrics.pearson_r,
        "spearman_rho": metrics.spearman_rho,
    }


def _feature_names(model) -> list[str]:
    return list(model.named_steps["preprocess"].get_feature_names_out())


def _ridge_coefficients(model) -> pd.DataFrame:
    names = _feature_names(model)
    coef = model.named_steps["model"].coef_
    return (
        pd.DataFrame(
            {
                "feature": names,
                "coefficient": coef,
                "absolute_importance": abs(coef),
            }
        )
        .sort_values("absolute_importance", ascending=False)
        .reset_index(drop=True)
    )


def _xgb_importance(model) -> pd.DataFrame:
    names = _feature_names(model)
    importance = model.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _binned_calibration(
    y_true: pd.Series,
    predictions,
    bins: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "actual": pd.to_numeric(y_true, errors="coerce"),
            "predicted": pd.to_numeric(pd.Series(predictions), errors="coerce"),
        }
    ).dropna()
    unique_predictions = frame["predicted"].nunique()
    q = min(bins, unique_predictions)
    if q < 2:
        return pd.DataFrame(
            {
                "mean_predicted": [frame["predicted"].mean()],
                "mean_actual": [frame["actual"].mean()],
                "samples": [len(frame)],
            }
        )
    frame["bin"] = pd.qcut(frame["predicted"], q=q, duplicates="drop")
    return (
        frame.groupby("bin", observed=True, as_index=False)
        .agg(
            mean_predicted=("predicted", "mean"),
            mean_actual=("actual", "mean"),
            samples=("actual", "size"),
        )
    )


def _save_holdout_chart(
    y_true: pd.Series,
    prediction_sets: dict[str, object],
    output: Path,
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    all_values: list[float] = []
    for label, predictions in prediction_sets.items():
        calibration = _binned_calibration(y_true, predictions, bins)
        ax.plot(
            calibration["mean_predicted"],
            calibration["mean_actual"],
            marker="o",
            label=label,
        )
        all_values.extend(calibration["mean_predicted"].tolist())
        all_values.extend(calibration["mean_actual"].tolist())

    if all_values:
        low = min(all_values)
        high = max(all_values)
        padding = max((high - low) * 0.08, 0.01)
        ax.plot(
            [low - padding, high + padding],
            [low - padding, high + padding],
            linestyle="--",
            label="Perfect mean prediction",
        )

    ax.set_xlabel("Mean predicted terminal-event RE24")
    ax.set_ylabel("Mean observed terminal-event RE24")
    ax.set_title("2025 expected RE24 — September holdout")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bins < 3:
        raise ValueError("--bins must be >= 3")
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Rebuild the 2025 baseline first."
        )

    plate_appearances = pd.read_parquet(args.input)
    contextual = add_backward_looking_terminal_context(plate_appearances)
    frame = prepare_expected_re24_frame(
        contextual,
        clean_attribution_only=not args.include_shifted_pa,
    )
    split = chronological_run_value_split(
        frame,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )

    features = EXPECTED_RE24_CATEGORICAL_FEATURES + EXPECTED_RE24_NUMERIC_FEATURES
    ridge = make_ridge_run_value_model(
        EXPECTED_RE24_CATEGORICAL_FEATURES,
        EXPECTED_RE24_NUMERIC_FEATURES,
    )
    xgb = make_xgboost_run_value_model(
        EXPECTED_RE24_CATEGORICAL_FEATURES,
        EXPECTED_RE24_NUMERIC_FEATURES,
    )

    ridge.fit(split.train[features], split.train[EXPECTED_RE24_TARGET])
    xgb.fit(split.train[features], split.train[EXPECTED_RE24_TARGET])
    state_mapping, state_fallback = fit_state_run_value_baseline(split.train)

    rows: list[dict[str, object]] = []
    for split_name, partition in [
        ("validation", split.validation),
        ("test", split.test),
    ]:
        state_pred = predict_state_run_value(partition, state_mapping, state_fallback)
        state_metrics = evaluate_regression_predictions(
            partition[EXPECTED_RE24_TARGET], state_pred
        )
        ridge_metrics = evaluate_regressor(
            ridge, partition[features], partition[EXPECTED_RE24_TARGET]
        )
        xgb_metrics = evaluate_regressor(
            xgb, partition[features], partition[EXPECTED_RE24_TARGET]
        )

        rows.append(_metrics_row("state_mean", split_name, state_metrics))
        rows.append(_metrics_row("ridge_context", split_name, ridge_metrics))
        rows.append(_metrics_row("xgboost_context", split_name, xgb_metrics))

    metrics = pd.DataFrame(rows)

    state_test = predict_state_run_value(split.test, state_mapping, state_fallback)
    ridge_test = ridge.predict(split.test[features])
    xgb_test = xgb.predict(split.test[features])

    predictions = split.test[
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "batter",
            "pitcher",
            "events",
            "base_out_state_before",
            "action_base_out_state_before",
            EXPECTED_RE24_TARGET,
            "re24",
            "within_pa_state_re24",
        ]
    ].copy()
    predictions["state_expected_re24"] = state_test.values
    predictions["ridge_expected_re24"] = ridge_test
    predictions["xgboost_expected_re24"] = xgb_test
    predictions["ridge_value_above_expected"] = (
        predictions[EXPECTED_RE24_TARGET] - predictions["ridge_expected_re24"]
    )
    predictions["xgboost_value_above_expected"] = (
        predictions[EXPECTED_RE24_TARGET] - predictions["xgboost_expected_re24"]
    )

    ridge_importance = _ridge_coefficients(ridge)
    xgb_importance = _xgb_importance(xgb)

    DEFAULT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DEFAULT_METRICS, index=False)
    ridge_importance.to_csv(DEFAULT_RIDGE, index=False)
    xgb_importance.to_csv(DEFAULT_XGB, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(DEFAULT_PREDICTIONS, index=False)

    _save_holdout_chart(
        split.test[EXPECTED_RE24_TARGET],
        {
            "State mean": state_test,
            "Ridge context": ridge_test,
            "XGBoost context": xgb_test,
        },
        DEFAULT_FIGURE,
        bins=args.bins,
    )

    print()
    print("NorthStar expected terminal-event RE24 model")
    print(
        "Attribution sample: "
        + ("all terminal-event PAs" if args.include_shifted_pa else "clean PAs only")
    )
    print(
        f"Train:      {len(split.train):,} PA "
        f"({split.train['game_date'].min().date()} to "
        f"{split.train['game_date'].max().date()})"
    )
    print(
        f"Validation: {len(split.validation):,} PA "
        f"({split.validation['game_date'].min().date()} to "
        f"{split.validation['game_date'].max().date()})"
    )
    print(
        f"Test:       {len(split.test):,} PA "
        f"({split.test['game_date'].min().date()} to "
        f"{split.test['game_date'].max().date()})"
    )
    print()
    print(
        metrics.round(
            {
                "mae": 4,
                "rmse": 4,
                "r2": 4,
                "pearson_r": 3,
                "spearman_rho": 3,
            }
        ).to_string(index=False)
    )
    print()
    print("Top XGBoost features")
    print(xgb_importance.head(15).round({"importance": 4}).to_string(index=False))
    print()
    print(f"Saved metrics:     {DEFAULT_METRICS}")
    print(f"Saved ridge coef:  {DEFAULT_RIDGE}")
    print(f"Saved XGB imp:     {DEFAULT_XGB}")
    print(f"Saved holdout plot:{DEFAULT_FIGURE}")
    print(f"Saved test preds:  {DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
