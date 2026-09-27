from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from northstar.features.outcomes import CODE_TO_OUTCOME, OUTCOME_CLASSES
from northstar.features.outcomes import add_backward_looking_outcome_context
from northstar.models.baselines import (
    evaluate_regression_predictions,
    make_multiclass_logistic_model,
    make_xgboost_outcome_model,
)
from northstar.models.pa_outcome import (
    OUTCOME_CATEGORICAL_FEATURES,
    OUTCOME_NUMERIC_FEATURES,
    OUTCOME_TARGET,
    align_model_probabilities,
    chronological_outcome_split,
    evaluate_multiclass_probabilities,
    expected_re24_from_outcome_probabilities,
    fit_class_prior,
    fit_outcome_run_value_table,
    prepare_outcome_frame,
    repeat_class_prior,
)


DEFAULT_INPUT = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_CLASS_DISTRIBUTION = Path(
    "reports/tables/pa_outcome_class_distribution_2025.csv"
)
DEFAULT_CLASS_METRICS = Path("reports/tables/pa_outcome_model_metrics_2025.csv")
DEFAULT_VALUE_METRICS = Path(
    "reports/tables/pa_outcome_expected_re24_metrics_2025.csv"
)
DEFAULT_XGB_IMPORTANCE = Path(
    "reports/tables/pa_outcome_xgboost_feature_importance_2025.csv"
)
DEFAULT_CONFUSION = Path(
    "reports/tables/pa_outcome_xgboost_confusion_2025.csv"
)
DEFAULT_FIGURE = Path(
    "reports/figures/pa_outcome_xgboost_confusion_2025.png"
)
DEFAULT_PREDICTIONS = Path(
    "data/processed/pa_outcome_test_predictions_2025.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NorthStar multiclass plate-appearance outcome models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--validation-start", default="2025-08-01")
    parser.add_argument("--test-start", default="2025-09-01")
    parser.add_argument(
        "--include-shifted-pa",
        action="store_true",
        help="Include PAs with within-PA state changes.",
    )
    return parser.parse_args()


def _classification_row(model_name: str, split_name: str, metrics) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split_name,
        "log_loss": metrics.log_loss,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "top2_accuracy": metrics.top2_accuracy,
    }


def _value_row(model_name: str, split_name: str, metrics) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split_name,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "pearson_r": metrics.pearson_r,
        "spearman_rho": metrics.spearman_rho,
    }


def _xgb_importance(model) -> pd.DataFrame:
    names = model.named_steps["preprocess"].get_feature_names_out()
    importance = model.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame["outcome_class"]
        .value_counts()
        .reindex(OUTCOME_CLASSES, fill_value=0)
        .rename_axis("outcome_class")
        .reset_index(name="plate_appearances")
    )
    counts["share"] = counts["plate_appearances"] / counts["plate_appearances"].sum()
    return counts


def _save_confusion(
    y_true: pd.Series,
    probabilities: np.ndarray,
    csv_output: Path,
    figure_output: Path,
) -> None:
    labels = list(range(len(OUTCOME_CLASSES)))
    predicted = probabilities.argmax(axis=1)

    raw = confusion_matrix(y_true, predicted, labels=labels)
    raw_frame = pd.DataFrame(
        raw,
        index=OUTCOME_CLASSES,
        columns=OUTCOME_CLASSES,
    )
    raw_frame.index.name = "actual"
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(csv_output)

    normalized = confusion_matrix(
        y_true,
        predicted,
        labels=labels,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(normalized, aspect="auto")
    fig.colorbar(image, ax=ax, label="Share of actual class")
    ax.set_xticks(range(len(OUTCOME_CLASSES)), OUTCOME_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(OUTCOME_CLASSES)), OUTCOME_CLASSES)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title("2025 PA outcome model — September holdout")

    for row in range(len(OUTCOME_CLASSES)):
        for col in range(len(OUTCOME_CLASSES)):
            value = normalized[row, col]
            if value >= 0.05:
                ax.text(col, row, f"{value:.2f}", ha="center", va="center")

    fig.tight_layout()
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Build the 2025 baseline first."
        )

    plate_appearances = pd.read_parquet(args.input)
    contextual = add_backward_looking_outcome_context(plate_appearances)
    frame = prepare_outcome_frame(
        contextual,
        clean_attribution_only=not args.include_shifted_pa,
    )
    split = chronological_outcome_split(
        frame,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )

    features = OUTCOME_CATEGORICAL_FEATURES + OUTCOME_NUMERIC_FEATURES

    logistic = make_multiclass_logistic_model(
        OUTCOME_CATEGORICAL_FEATURES,
        OUTCOME_NUMERIC_FEATURES,
    )
    xgboost = make_xgboost_outcome_model(
        OUTCOME_CATEGORICAL_FEATURES,
        OUTCOME_NUMERIC_FEATURES,
    )

    logistic.fit(split.train[features], split.train[OUTCOME_TARGET])
    xgboost.fit(split.train[features], split.train[OUTCOME_TARGET])

    prior = fit_class_prior(split.train)
    state_outcome_values, outcome_fallback, overall_fallback = (
        fit_outcome_run_value_table(split.train)
    )

    classification_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []

    for split_name, partition in [
        ("validation", split.validation),
        ("test", split.test),
    ]:
        probability_sets = {
            "class_prior": repeat_class_prior(prior, len(partition)),
            "logistic_outcome": align_model_probabilities(
                logistic, partition[features]
            ),
            "xgboost_outcome": align_model_probabilities(
                xgboost, partition[features]
            ),
        }

        for model_name, probabilities in probability_sets.items():
            classification_metrics = evaluate_multiclass_probabilities(
                partition[OUTCOME_TARGET],
                probabilities,
            )
            classification_rows.append(
                _classification_row(
                    model_name,
                    split_name,
                    classification_metrics,
                )
            )

            expected_re24 = expected_re24_from_outcome_probabilities(
                partition,
                probabilities,
                state_outcome_values,
                outcome_fallback,
                overall_fallback,
            )
            run_value_metrics = evaluate_regression_predictions(
                partition["terminal_event_re24"],
                expected_re24,
            )
            value_rows.append(
                _value_row(model_name, split_name, run_value_metrics)
            )

    class_metrics = pd.DataFrame(classification_rows)
    value_metrics = pd.DataFrame(value_rows)

    test_logistic = align_model_probabilities(logistic, split.test[features])
    test_xgb = align_model_probabilities(xgboost, split.test[features])
    test_xgb_expected = expected_re24_from_outcome_probabilities(
        split.test,
        test_xgb,
        state_outcome_values,
        outcome_fallback,
        overall_fallback,
    )

    predictions = split.test[
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "batter",
            "pitcher",
            "events",
            "outcome_class",
            OUTCOME_TARGET,
            "base_out_state_before",
            "terminal_event_re24",
        ]
    ].copy()

    predictions["logistic_predicted_code"] = test_logistic.argmax(axis=1)
    predictions["xgboost_predicted_code"] = test_xgb.argmax(axis=1)
    predictions["logistic_predicted_outcome"] = predictions[
        "logistic_predicted_code"
    ].map(CODE_TO_OUTCOME)
    predictions["xgboost_predicted_outcome"] = predictions[
        "xgboost_predicted_code"
    ].map(CODE_TO_OUTCOME)
    predictions["xgboost_expected_re24"] = test_xgb_expected
    predictions["xgboost_value_above_expected"] = (
        predictions["terminal_event_re24"]
        - predictions["xgboost_expected_re24"]
    )

    for code, outcome in enumerate(OUTCOME_CLASSES):
        predictions[f"logistic_p_{outcome}"] = test_logistic[:, code]
        predictions[f"xgboost_p_{outcome}"] = test_xgb[:, code]

    distribution = _class_distribution(frame)
    xgb_importance = _xgb_importance(xgboost)

    DEFAULT_CLASS_METRICS.parent.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(DEFAULT_CLASS_DISTRIBUTION, index=False)
    class_metrics.to_csv(DEFAULT_CLASS_METRICS, index=False)
    value_metrics.to_csv(DEFAULT_VALUE_METRICS, index=False)
    xgb_importance.to_csv(DEFAULT_XGB_IMPORTANCE, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(DEFAULT_PREDICTIONS, index=False)

    _save_confusion(
        split.test[OUTCOME_TARGET],
        test_xgb,
        DEFAULT_CONFUSION,
        DEFAULT_FIGURE,
    )

    print()
    print("NorthStar PA outcome model V1")
    print(
        "Attribution sample: "
        + ("all PAs" if args.include_shifted_pa else "clean PAs only")
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
    print("Outcome distribution")
    print(
        distribution.assign(
            share=distribution["share"] * 100.0
        ).round({"share": 2}).to_string(index=False)
    )

    print()
    print("Multiclass holdout metrics")
    print(
        class_metrics.round(
            {
                "log_loss": 4,
                "accuracy": 4,
                "macro_f1": 4,
                "top2_accuracy": 4,
            }
        ).to_string(index=False)
    )

    print()
    print("Expected RE24 implied by outcome probabilities")
    print(
        value_metrics.round(
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
    print(f"Saved distribution: {DEFAULT_CLASS_DISTRIBUTION}")
    print(f"Saved class metrics:{DEFAULT_CLASS_METRICS}")
    print(f"Saved value metrics:{DEFAULT_VALUE_METRICS}")
    print(f"Saved XGB imp:      {DEFAULT_XGB_IMPORTANCE}")
    print(f"Saved confusion:    {DEFAULT_CONFUSION}")
    print(f"Saved figure:       {DEFAULT_FIGURE}")
    print(f"Saved test preds:   {DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
