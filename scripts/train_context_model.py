from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

from northstar.features.context import add_backward_looking_player_context
from northstar.models.baselines import (
    evaluate_binary_classifier,
    evaluate_probabilities,
    make_scoring_baseline,
    make_xgboost_scoring_model,
)
from northstar.models.scoring_probability import (
    CONTEXT_CATEGORICAL_FEATURES,
    CONTEXT_NUMERIC_FEATURES,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    chronological_split,
    fit_state_probability_baseline,
    predict_state_probability,
    prepare_scoring_frame,
    validate_context_columns,
)


DEFAULT_INPUT = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_METRICS = Path("reports/tables/context_model_metrics_2025.csv")
DEFAULT_LOGISTIC_IMPORTANCE = Path(
    "reports/tables/context_logistic_coefficients_2025.csv"
)
DEFAULT_XGB_IMPORTANCE = Path(
    "reports/tables/context_xgboost_feature_importance_2025.csv"
)
DEFAULT_FIGURE = Path("reports/figures/context_model_calibration_2025.png")
DEFAULT_PREDICTIONS = Path(
    "data/processed/context_model_test_predictions_2025.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NorthStar V2 contextual scoring-probability models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--validation-start", default="2025-08-01")
    parser.add_argument("--test-start", default="2025-09-01")
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def _metrics_row(model_name: str, split_name: str, metrics) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split_name,
        "roc_auc": metrics.roc_auc,
        "log_loss": metrics.log_loss,
        "brier": metrics.brier,
    }


def _feature_names(model) -> list[str]:
    return list(model.named_steps["preprocess"].get_feature_names_out())


def _logistic_coefficients(model) -> pd.DataFrame:
    feature_names = _feature_names(model)
    coefficients = model.named_steps["model"].coef_[0]
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "absolute_importance": abs(coefficients),
            }
        )
        .sort_values("absolute_importance", ascending=False)
        .reset_index(drop=True)
    )


def _xgb_importance(model) -> pd.DataFrame:
    feature_names = _feature_names(model)
    importance = model.named_steps["model"].feature_importances_
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _save_calibration_chart(
    y_true: pd.Series,
    predictions: dict[str, pd.Series | list[float]],
    output: Path,
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for label, probabilities in predictions.items():
        observed, predicted = calibration_curve(
            y_true,
            probabilities,
            n_bins=bins,
            strategy="quantile",
        )
        ax.plot(predicted, observed, marker="o", label=label)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed scoring frequency")
    ax.set_title("2025 MLB scoring calibration — September holdout")
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
            f"{args.input} does not exist. Rebuild the baseline first."
        )

    plate_appearances = pd.read_parquet(args.input)
    contextual = add_backward_looking_player_context(plate_appearances)
    frame = prepare_scoring_frame(contextual)
    validate_context_columns(frame)

    split = chronological_split(
        frame,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )

    target = "scored_before_inning_end"
    v1_features = MODEL_CATEGORICAL_FEATURES + MODEL_NUMERIC_FEATURES
    context_features = CONTEXT_CATEGORICAL_FEATURES + CONTEXT_NUMERIC_FEATURES

    logistic_v1 = make_scoring_baseline(
        MODEL_CATEGORICAL_FEATURES,
        MODEL_NUMERIC_FEATURES,
    )
    logistic_v2 = make_scoring_baseline(
        CONTEXT_CATEGORICAL_FEATURES,
        CONTEXT_NUMERIC_FEATURES,
    )
    xgboost_v2 = make_xgboost_scoring_model(
        CONTEXT_CATEGORICAL_FEATURES,
        CONTEXT_NUMERIC_FEATURES,
    )

    logistic_v1.fit(split.train[v1_features], split.train[target])
    logistic_v2.fit(split.train[context_features], split.train[target])
    xgboost_v2.fit(split.train[context_features], split.train[target])

    state_mapping, state_fallback = fit_state_probability_baseline(split.train)

    models = {
        "logistic_v1": (logistic_v1, v1_features),
        "logistic_context_v2": (logistic_v2, context_features),
        "xgboost_context_v2": (xgboost_v2, context_features),
    }

    rows: list[dict[str, object]] = []
    for split_name, partition in [
        ("validation", split.validation),
        ("test", split.test),
    ]:
        state_probabilities = predict_state_probability(
            partition,
            state_mapping,
            state_fallback,
        )
        state_metrics = evaluate_probabilities(
            partition[target],
            state_probabilities,
        )
        rows.append(_metrics_row("state_24", split_name, state_metrics))

        for model_name, (model, features) in models.items():
            metrics = evaluate_binary_classifier(
                model,
                partition[features],
                partition[target],
            )
            rows.append(_metrics_row(model_name, split_name, metrics))

    metrics = pd.DataFrame(rows)

    test_probabilities = {
        "24-state baseline": predict_state_probability(
            split.test,
            state_mapping,
            state_fallback,
        ),
        "Logistic V1": logistic_v1.predict_proba(
            split.test[v1_features]
        )[:, 1],
        "Logistic context V2": logistic_v2.predict_proba(
            split.test[context_features]
        )[:, 1],
        "XGBoost context V2": xgboost_v2.predict_proba(
            split.test[context_features]
        )[:, 1],
    }

    predictions = split.test[
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "batter",
            "pitcher",
            "base_out_state_before",
            target,
        ]
    ].copy()
    predictions["state_probability"] = test_probabilities["24-state baseline"].values
    predictions["logistic_v1_probability"] = test_probabilities["Logistic V1"]
    predictions["logistic_v2_probability"] = test_probabilities[
        "Logistic context V2"
    ]
    predictions["xgboost_v2_probability"] = test_probabilities[
        "XGBoost context V2"
    ]

    logistic_importance = _logistic_coefficients(logistic_v2)
    xgb_importance = _xgb_importance(xgboost_v2)

    DEFAULT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DEFAULT_METRICS, index=False)
    logistic_importance.to_csv(DEFAULT_LOGISTIC_IMPORTANCE, index=False)
    xgb_importance.to_csv(DEFAULT_XGB_IMPORTANCE, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(DEFAULT_PREDICTIONS, index=False)

    _save_calibration_chart(
        split.test[target],
        test_probabilities,
        DEFAULT_FIGURE,
        bins=args.calibration_bins,
    )

    print()
    print("NorthStar context model V2")
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
            {"roc_auc": 3, "log_loss": 4, "brier": 4}
        ).to_string(index=False)
    )

    print()
    print("Top XGBoost features")
    print(xgb_importance.head(15).round({"importance": 4}).to_string(index=False))

    print()
    print(f"Saved metrics:       {DEFAULT_METRICS}")
    print(f"Saved logistic coef: {DEFAULT_LOGISTIC_IMPORTANCE}")
    print(f"Saved XGB importance:{DEFAULT_XGB_IMPORTANCE}")
    print(f"Saved calibration:   {DEFAULT_FIGURE}")
    print(f"Saved test preds:    {DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
