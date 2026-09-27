from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

from northstar.models.baselines import (
    evaluate_binary_classifier,
    evaluate_probabilities,
    make_scoring_baseline,
)
from northstar.models.scoring_probability import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    chronological_split,
    fit_state_probability_baseline,
    predict_state_probability,
    prepare_scoring_frame,
)


DEFAULT_INPUT = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_METRICS = Path("reports/tables/scoring_probability_model_metrics_2025.csv")
DEFAULT_COEFFICIENTS = Path("reports/tables/scoring_probability_model_coefficients_2025.csv")
DEFAULT_FIGURE = Path("reports/figures/scoring_probability_calibration_2025.png")
DEFAULT_PREDICTIONS = Path("data/processed/scoring_probability_test_predictions_2025.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NorthStar's first leakage-safe scoring-probability model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--validation-start", default="2025-08-01")
    parser.add_argument("--test-start", default="2025-09-01")
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def metrics_row(model_name: str, split_name: str, metrics) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split_name,
        "roc_auc": metrics.roc_auc,
        "log_loss": metrics.log_loss,
        "brier": metrics.brier,
    }


def extract_coefficients(model) -> pd.DataFrame:
    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    coefficients = classifier.coef_[0]

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "absolute_coefficient": abs(coefficients),
            }
        )
        .sort_values("absolute_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def save_calibration_chart(
    y_true: pd.Series,
    logistic_probabilities,
    state_probabilities,
    output: Path,
    bins: int,
) -> None:
    logistic_true, logistic_pred = calibration_curve(
        y_true,
        logistic_probabilities,
        n_bins=bins,
        strategy="quantile",
    )
    state_true, state_pred = calibration_curve(
        y_true,
        state_probabilities,
        n_bins=bins,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(logistic_pred, logistic_true, marker="o", label="Logistic model")
    ax.plot(state_pred, state_true, marker="o", label="24-state baseline")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed scoring frequency")
    ax.set_title("2025 MLB scoring-probability calibration — September holdout")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.calibration_bins < 3:
        raise ValueError("--calibration-bins must be >= 3")
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Run scripts/build_baseline.py first."
        )

    plate_appearances = pd.read_parquet(args.input)
    frame = prepare_scoring_frame(plate_appearances)
    split = chronological_split(
        frame,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )

    feature_columns = MODEL_CATEGORICAL_FEATURES + MODEL_NUMERIC_FEATURES
    target = "scored_before_inning_end"

    model = make_scoring_baseline(
        categorical_features=MODEL_CATEGORICAL_FEATURES,
        numeric_features=MODEL_NUMERIC_FEATURES,
    )
    model.fit(split.train[feature_columns], split.train[target])

    state_mapping, state_fallback = fit_state_probability_baseline(split.train)

    rows = []
    for split_name, partition in [
        ("validation", split.validation),
        ("test", split.test),
    ]:
        model_metrics = evaluate_binary_classifier(
            model,
            partition[feature_columns],
            partition[target],
        )
        state_probabilities = predict_state_probability(
            partition,
            state_mapping,
            state_fallback,
        )
        state_metrics = evaluate_probabilities(
            partition[target],
            state_probabilities,
        )

        rows.append(metrics_row("logistic_context", split_name, model_metrics))
        rows.append(metrics_row("state_24", split_name, state_metrics))

    metrics = pd.DataFrame(rows)

    test_logistic_probability = model.predict_proba(split.test[feature_columns])[:, 1]
    test_state_probability = predict_state_probability(
        split.test,
        state_mapping,
        state_fallback,
    )

    predictions = split.test[
        [
            "game_date",
            "base_out_state_before",
            "base_state_before",
            "outs_before",
            "inning_topbot",
            "inning",
            target,
        ]
    ].copy()
    for optional_column in ("batter", "pitcher", "game_pk", "at_bat_number"):
        if optional_column in split.test.columns:
            predictions[optional_column] = split.test[optional_column].values
    predictions["logistic_probability"] = test_logistic_probability
    predictions["state_probability"] = test_state_probability.values

    coefficients = extract_coefficients(model)

    DEFAULT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DEFAULT_METRICS, index=False)
    coefficients.to_csv(DEFAULT_COEFFICIENTS, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(DEFAULT_PREDICTIONS, index=False)

    save_calibration_chart(
        split.test[target],
        test_logistic_probability,
        test_state_probability,
        DEFAULT_FIGURE,
        bins=args.calibration_bins,
    )

    print()
    print("NorthStar scoring-probability model")
    print(
        f"Train:      {len(split.train):,} PA "
        f"({split.train['game_date'].min().date()} to {split.train['game_date'].max().date()})"
    )
    print(
        f"Validation: {len(split.validation):,} PA "
        f"({split.validation['game_date'].min().date()} to "
        f"{split.validation['game_date'].max().date()})"
    )
    print(
        f"Test:       {len(split.test):,} PA "
        f"({split.test['game_date'].min().date()} to {split.test['game_date'].max().date()})"
    )

    print()
    print(metrics.round({"roc_auc": 3, "log_loss": 4, "brier": 4}).to_string(index=False))

    print()
    print(f"Saved metrics:      {DEFAULT_METRICS}")
    print(f"Saved coefficients: {DEFAULT_COEFFICIENTS}")
    print(f"Saved calibration:  {DEFAULT_FIGURE}")
    print(f"Saved test preds:   {DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
