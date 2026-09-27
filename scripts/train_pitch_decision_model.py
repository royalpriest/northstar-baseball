from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

from northstar.data.statcast import read_statcast
from northstar.features.pitches import build_pitch_decision_frame
from northstar.models.baselines import (
    evaluate_binary_classifier,
    evaluate_probabilities,
    make_scoring_baseline,
    make_xgboost_scoring_model,
)
from northstar.models.pitch_decision import (
    PITCH_DECISION_CATEGORICAL_FEATURES,
    PITCH_DECISION_NUMERIC_FEATURES,
    PITCH_DECISION_TARGET,
    chronological_pitch_split,
    decision_surprise,
    fit_count_zone_swing_baseline,
    predict_count_zone_swing_baseline,
    prepare_pitch_decision_model_frame,
)


DEFAULT_RAW_DIR = Path("data/raw/statcast")
DEFAULT_METRICS = Path("reports/tables/pitch_decision_model_metrics_2025.csv")
DEFAULT_IMPORTANCE = Path(
    "reports/tables/pitch_decision_xgboost_feature_importance_2025.csv"
)
DEFAULT_BATTER_SUMMARY = Path(
    "reports/tables/pitch_decision_batter_holdout_2025.csv"
)
DEFAULT_FIGURE = Path("reports/figures/pitch_decision_calibration_2025.png")
DEFAULT_PREDICTIONS = Path(
    "data/processed/pitch_decision_test_predictions_2025.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train NorthStar's first pitch-level swing/take decision models "
            "from cached raw Statcast data."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
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


def _xgb_importance(model) -> pd.DataFrame:
    names = model.named_steps["preprocess"].get_feature_names_out()
    importance = model.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _save_calibration(
    y_true: pd.Series,
    probabilities: dict[str, object],
    output: Path,
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for label, values in probabilities.items():
        observed, predicted = calibration_curve(
            y_true,
            values,
            n_bins=bins,
            strategy="quantile",
        )
        ax.plot(predicted, observed, marker="o", label=label)

    ax.set_xlabel("Predicted swing probability")
    ax.set_ylabel("Observed swing rate")
    ax.set_title("2025 MLB pitch-decision calibration — September holdout")
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

    raw_paths = sorted(args.raw_dir.glob("statcast_*.parquet"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No cached Statcast Parquet files found under {args.raw_dir}."
        )

    print(f"Reading {len(raw_paths)} cached Statcast chunks...")
    pitches = read_statcast(raw_paths)
    decision_frame = build_pitch_decision_frame(pitches)
    frame = prepare_pitch_decision_model_frame(decision_frame)
    split = chronological_pitch_split(
        frame,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )

    features = PITCH_DECISION_CATEGORICAL_FEATURES + PITCH_DECISION_NUMERIC_FEATURES

    logistic = make_scoring_baseline(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )
    xgboost = make_xgboost_scoring_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )

    logistic.fit(split.train[features], split.train[PITCH_DECISION_TARGET])
    xgboost.fit(split.train[features], split.train[PITCH_DECISION_TARGET])

    count_zone_map, count_zone_fallback = fit_count_zone_swing_baseline(
        split.train
    )

    rows: list[dict[str, object]] = []
    for split_name, partition in [
        ("validation", split.validation),
        ("test", split.test),
    ]:
        count_zone_probability = predict_count_zone_swing_baseline(
            partition,
            count_zone_map,
            count_zone_fallback,
        )
        count_zone_metrics = evaluate_probabilities(
            partition[PITCH_DECISION_TARGET],
            count_zone_probability,
        )
        logistic_metrics = evaluate_binary_classifier(
            logistic,
            partition[features],
            partition[PITCH_DECISION_TARGET],
        )
        xgboost_metrics = evaluate_binary_classifier(
            xgboost,
            partition[features],
            partition[PITCH_DECISION_TARGET],
        )

        rows.append(
            _metrics_row("count_zone_baseline", split_name, count_zone_metrics)
        )
        rows.append(_metrics_row("logistic_pitch", split_name, logistic_metrics))
        rows.append(_metrics_row("xgboost_pitch", split_name, xgboost_metrics))

    metrics = pd.DataFrame(rows)

    test_count_zone = predict_count_zone_swing_baseline(
        split.test,
        count_zone_map,
        count_zone_fallback,
    )
    test_logistic = logistic.predict_proba(split.test[features])[:, 1]
    test_xgboost = xgboost.predict_proba(split.test[features])[:, 1]

    predictions = split.test[
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "batter",
            "pitcher",
            "description",
            "decision",
            PITCH_DECISION_TARGET,
            "balls",
            "strikes",
            "pitch_type",
            "zone",
            "plate_x",
            "plate_z",
            "release_speed",
        ]
    ].copy()
    predictions["count_zone_swing_probability"] = test_count_zone.values
    predictions["logistic_swing_probability"] = test_logistic
    predictions["xgboost_swing_probability"] = test_xgboost
    predictions["decision_surprise"] = decision_surprise(
        predictions[PITCH_DECISION_TARGET],
        test_xgboost,
    ).values

    batter_summary = (
        predictions.groupby("batter", as_index=False)
        .agg(
            pitches=(PITCH_DECISION_TARGET, "size"),
            swing_rate=(PITCH_DECISION_TARGET, "mean"),
            expected_swing_rate=("xgboost_swing_probability", "mean"),
            mean_decision_surprise=("decision_surprise", "mean"),
        )
        .sort_values("pitches", ascending=False)
        .reset_index(drop=True)
    )

    xgb_importance = _xgb_importance(xgboost)

    DEFAULT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DEFAULT_METRICS, index=False)
    xgb_importance.to_csv(DEFAULT_IMPORTANCE, index=False)
    batter_summary.to_csv(DEFAULT_BATTER_SUMMARY, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(DEFAULT_PREDICTIONS, index=False)

    _save_calibration(
        split.test[PITCH_DECISION_TARGET],
        {
            "Count + zone baseline": test_count_zone,
            "Logistic pitch model": test_logistic,
            "XGBoost pitch model": test_xgboost,
        },
        DEFAULT_FIGURE,
        args.calibration_bins,
    )

    recognized_share = len(frame) / len(pitches) if len(pitches) else 0.0

    print()
    print("NorthStar pitch-decision model V1")
    print(f"Raw pitches read:       {len(pitches):,}")
    print(f"Physical decisions kept:{len(frame):,} ({recognized_share:.2%})")
    print(
        f"Train:      {len(split.train):,} pitches "
        f"({split.train['game_date'].min().date()} to "
        f"{split.train['game_date'].max().date()})"
    )
    print(
        f"Validation: {len(split.validation):,} pitches "
        f"({split.validation['game_date'].min().date()} to "
        f"{split.validation['game_date'].max().date()})"
    )
    print(
        f"Test:       {len(split.test):,} pitches "
        f"({split.test['game_date'].min().date()} to "
        f"{split.test['game_date'].max().date()})"
    )
    print(f"Training swing rate:    {split.train[PITCH_DECISION_TARGET].mean():.3f}")

    print()
    print(
        metrics.round(
            {"roc_auc": 3, "log_loss": 4, "brier": 4}
        ).to_string(index=False)
    )

    print()
    print("Top XGBoost pitch features")
    print(xgb_importance.head(20).round({"importance": 4}).to_string(index=False))

    print()
    print(f"Saved metrics:       {DEFAULT_METRICS}")
    print(f"Saved XGB importance:{DEFAULT_IMPORTANCE}")
    print(f"Saved batter summary:{DEFAULT_BATTER_SUMMARY}")
    print(f"Saved calibration:   {DEFAULT_FIGURE}")
    print(f"Saved test preds:    {DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
