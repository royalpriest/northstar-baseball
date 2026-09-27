from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from northstar.features.decision_value import (
    PITCH_VALUE_TARGET,
    attach_pitch_value_target,
    overlap_mask,
    stabilized_action_weights,
)
from northstar.features.pitches import (
    PITCH_DECISION_REQUIRED_COLUMNS,
    build_pitch_decision_frame,
)
from northstar.metrics.run_expectancy import build_run_expectancy_table
from northstar.models.baselines import (
    evaluate_regression_predictions,
    make_pitch_value_xgboost_model,
    make_pitch_xgboost_model,
)
from northstar.models.pitch_decision import (
    PITCH_DECISION_CATEGORICAL_FEATURES,
    PITCH_DECISION_NUMERIC_FEATURES,
    PITCH_DECISION_TARGET,
    chronological_pitch_split,
    prepare_pitch_decision_model_frame,
)
from northstar.models.pitch_value import (
    doubly_robust_policy_value,
    factual_action_prediction,
    fit_count_zone_action_value_baseline,
    modeled_action_advantage,
    optimal_action,
    predict_count_zone_action_values,
)


DEFAULT_RAW_DIR = Path("data/raw/statcast")
DEFAULT_PA_INPUT = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_FACTUAL_METRICS = Path(
    "reports/tables/pitch_decision_value_factual_metrics_2025.csv"
)
DEFAULT_POLICY_VALUES = Path(
    "reports/tables/pitch_decision_value_policy_estimates_2025.csv"
)
DEFAULT_BATTER_SUMMARY = Path(
    "reports/tables/pitch_decision_value_batter_holdout_2025.csv"
)
DEFAULT_SWING_IMPORTANCE = Path(
    "reports/tables/pitch_value_swing_xgboost_importance_2025.csv"
)
DEFAULT_TAKE_IMPORTANCE = Path(
    "reports/tables/pitch_value_take_xgboost_importance_2025.csv"
)
DEFAULT_FIGURE = Path(
    "reports/figures/pitch_decision_value_holdout_2025.png"
)
DEFAULT_PREDICTIONS = Path(
    "data/processed/pitch_decision_value_test_predictions_2025.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate observational swing-vs-take action values from pitch "
            "opportunities using propensity-adjusted XGBoost outcome models."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--pa-input", type=Path, default=DEFAULT_PA_INPUT)
    parser.add_argument("--validation-start", default="2025-08-01")
    parser.add_argument("--test-start", default="2025-09-01")
    parser.add_argument("--overlap-lower", type=float, default=0.05)
    parser.add_argument("--overlap-upper", type=float, default=0.95)
    return parser.parse_args()


def _load_pitch_chunks(
    raw_paths: list[Path],
    projected_columns: list[str],
) -> tuple[pd.DataFrame, int]:
    decision_frames: list[pd.DataFrame] = []
    raw_pitch_count = 0

    for index, path in enumerate(raw_paths, start=1):
        raw_chunk = pd.read_parquet(path, columns=projected_columns)
        raw_pitch_count += len(raw_chunk)
        decision_frames.append(build_pitch_decision_frame(raw_chunk))

        if index == 1 or index % 5 == 0 or index == len(raw_paths):
            print(f"  prepared chunk {index}/{len(raw_paths)}")

    return (
        pd.concat(decision_frames, ignore_index=True, copy=False),
        raw_pitch_count,
    )


def _feature_importance(model) -> pd.DataFrame:
    names = model.named_steps["preprocess"].get_feature_names_out()
    importance = model.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _factual_row(model_name: str, split_name: str, metrics) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split_name,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "pearson_r": metrics.pearson_r,
        "spearman_rho": metrics.spearman_rho,
    }


def _policy_rows(
    partition: pd.DataFrame,
    propensity: np.ndarray,
    q_take: np.ndarray,
    q_swing: np.ndarray,
    split_name: str,
) -> list[dict[str, object]]:
    observed_action = partition[PITCH_DECISION_TARGET].to_numpy(dtype="int8")
    outcome = partition[PITCH_VALUE_TARGET].to_numpy(dtype="float64")
    policies = {
        "always_take": np.zeros(len(partition), dtype="int8"),
        "always_swing": np.ones(len(partition), dtype="int8"),
        "model_policy": optimal_action(q_take, q_swing),
    }

    rows = []
    for name, policy in policies.items():
        estimate = doubly_robust_policy_value(
            outcome,
            observed_action,
            propensity,
            q_take,
            q_swing,
            policy,
            policy_name=name,
        )
        rows.append(
            {
                "split": split_name,
                "policy": estimate.policy,
                "value": estimate.value,
                "standard_error": estimate.standard_error,
                "swing_share": estimate.swing_share,
                "pitches": estimate.pitches,
            }
        )

    rows.append(
        {
            "split": split_name,
            "policy": "observed_actions",
            "value": float(outcome.mean()),
            "standard_error": float(outcome.std(ddof=1) / np.sqrt(len(outcome))),
            "swing_share": float(observed_action.mean()),
            "pitches": int(len(outcome)),
        }
    )
    return rows


def _save_decision_value_chart(
    predictions: pd.DataFrame,
    output: Path,
) -> None:
    values = predictions["modeled_action_advantage"].dropna()
    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values.clip(-0.5, 0.5), bins=70)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Modeled chosen-action advantage (runs)")
    ax.set_ylabel("September pitches")
    ax.set_title("NorthStar pitch decision-value V1 — September holdout")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not 0.0 < args.overlap_lower < args.overlap_upper < 1.0:
        raise ValueError("Overlap bounds must satisfy 0 < lower < upper < 1.")
    if not args.pa_input.exists():
        raise FileNotFoundError(
            f"{args.pa_input} does not exist. Build the 2025 baseline first."
        )

    raw_paths = sorted(args.raw_dir.glob("statcast_*.parquet"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No cached Statcast Parquet files found under {args.raw_dir}."
        )

    plate_appearances = pd.read_parquet(args.pa_input)
    plate_appearances["game_date"] = pd.to_datetime(
        plate_appearances["game_date"], errors="coerce"
    )
    training_pa = plate_appearances.loc[
        plate_appearances["game_date"] < pd.Timestamp(args.validation_start)
    ].copy()
    run_expectancy = build_run_expectancy_table(training_pa)

    projected_columns = sorted(PITCH_DECISION_REQUIRED_COLUMNS)
    print(
        f"Reading {len(raw_paths)} cached Statcast chunks "
        f"with {len(projected_columns)} projected columns..."
    )
    pitch_frame, raw_pitch_count = _load_pitch_chunks(
        raw_paths,
        projected_columns,
    )
    pitch_frame = attach_pitch_value_target(
        pitch_frame,
        plate_appearances,
        run_expectancy,
    )
    frame = prepare_pitch_decision_model_frame(pitch_frame)
    frame[PITCH_VALUE_TARGET] = pd.to_numeric(
        frame[PITCH_VALUE_TARGET], errors="coerce"
    )
    frame = frame.dropna(subset=[PITCH_VALUE_TARGET]).reset_index(drop=True)

    split = chronological_pitch_split(
        frame,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )
    features = PITCH_DECISION_CATEGORICAL_FEATURES + PITCH_DECISION_NUMERIC_FEATURES

    propensity_model = make_pitch_xgboost_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )
    propensity_model.fit(
        split.train[features],
        split.train[PITCH_DECISION_TARGET],
    )

    train_propensity = propensity_model.predict_proba(split.train[features])[:, 1]
    validation_propensity = propensity_model.predict_proba(
        split.validation[features]
    )[:, 1]
    test_propensity = propensity_model.predict_proba(split.test[features])[:, 1]

    train_overlap = overlap_mask(
        train_propensity,
        args.overlap_lower,
        args.overlap_upper,
    )
    train_for_value = split.train.loc[train_overlap].reset_index(drop=True)
    train_propensity_overlap = train_propensity[train_overlap]

    weights = stabilized_action_weights(
        train_for_value[PITCH_DECISION_TARGET],
        train_propensity_overlap,
    )

    swing_mask = train_for_value[PITCH_DECISION_TARGET].eq(1).to_numpy()
    take_mask = ~swing_mask

    swing_model = make_pitch_value_xgboost_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )
    take_model = make_pitch_value_xgboost_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )

    swing_model.fit(
        train_for_value.loc[swing_mask, features],
        train_for_value.loc[swing_mask, PITCH_VALUE_TARGET],
        model__sample_weight=weights[swing_mask],
    )
    take_model.fit(
        train_for_value.loc[take_mask, features],
        train_for_value.loc[take_mask, PITCH_VALUE_TARGET],
        model__sample_weight=weights[take_mask],
    )

    count_zone_map, action_fallback, overall_fallback = (
        fit_count_zone_action_value_baseline(train_for_value)
    )

    factual_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []

    split_payload = [
        ("validation", split.validation, validation_propensity),
        ("test", split.test, test_propensity),
    ]

    test_predictions: pd.DataFrame | None = None

    for split_name, partition, propensity in split_payload:
        overlap = overlap_mask(
            propensity,
            args.overlap_lower,
            args.overlap_upper,
        )
        partition_overlap = partition.loc[overlap].reset_index(drop=True)
        propensity_overlap = propensity[overlap]

        baseline_take, baseline_swing = predict_count_zone_action_values(
            partition_overlap,
            count_zone_map,
            action_fallback,
            overall_fallback,
        )
        xgb_take = take_model.predict(partition_overlap[features])
        xgb_swing = swing_model.predict(partition_overlap[features])

        baseline_factual = factual_action_prediction(
            partition_overlap[PITCH_DECISION_TARGET],
            baseline_take,
            baseline_swing,
        )
        xgb_factual = factual_action_prediction(
            partition_overlap[PITCH_DECISION_TARGET],
            xgb_take,
            xgb_swing,
        )

        baseline_metrics = evaluate_regression_predictions(
            partition_overlap[PITCH_VALUE_TARGET],
            baseline_factual,
        )
        xgb_metrics = evaluate_regression_predictions(
            partition_overlap[PITCH_VALUE_TARGET],
            xgb_factual,
        )

        factual_rows.append(
            _factual_row("count_zone_action_value", split_name, baseline_metrics)
        )
        factual_rows.append(
            _factual_row("xgboost_action_value", split_name, xgb_metrics)
        )

        policy_rows.extend(
            _policy_rows(
                partition_overlap,
                propensity_overlap,
                xgb_take,
                xgb_swing,
                split_name,
            )
        )

        if split_name == "test":
            test_predictions = partition_overlap[
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
                    PITCH_VALUE_TARGET,
                ]
            ].copy()
            test_predictions["swing_propensity"] = propensity_overlap
            test_predictions["q_take"] = xgb_take
            test_predictions["q_swing"] = xgb_swing
            test_predictions["optimal_action"] = np.where(
                xgb_swing >= xgb_take,
                "swing",
                "take",
            )
            test_predictions["optimal_choice"] = (
                test_predictions["decision"]
                == test_predictions["optimal_action"]
            )
            test_predictions["modeled_action_advantage"] = modeled_action_advantage(
                test_predictions[PITCH_DECISION_TARGET],
                xgb_take,
                xgb_swing,
            )

    if test_predictions is None:
        raise RuntimeError("No test predictions were generated.")

    factual_metrics = pd.DataFrame(factual_rows)
    policy_values = pd.DataFrame(policy_rows)

    batter_summary = (
        test_predictions.groupby("batter", as_index=False)
        .agg(
            pitches=(PITCH_DECISION_TARGET, "size"),
            swing_rate=(PITCH_DECISION_TARGET, "mean"),
            optimal_choice_rate=("optimal_choice", "mean"),
            mean_modeled_action_advantage=(
                "modeled_action_advantage",
                "mean",
            ),
            mean_abs_decision_gap=(
                "modeled_action_advantage",
                lambda s: s.abs().mean(),
            ),
        )
        .sort_values("pitches", ascending=False)
        .reset_index(drop=True)
    )

    swing_importance = _feature_importance(swing_model)
    take_importance = _feature_importance(take_model)

    DEFAULT_FACTUAL_METRICS.parent.mkdir(parents=True, exist_ok=True)
    factual_metrics.to_csv(DEFAULT_FACTUAL_METRICS, index=False)
    policy_values.to_csv(DEFAULT_POLICY_VALUES, index=False)
    batter_summary.to_csv(DEFAULT_BATTER_SUMMARY, index=False)
    swing_importance.to_csv(DEFAULT_SWING_IMPORTANCE, index=False)
    take_importance.to_csv(DEFAULT_TAKE_IMPORTANCE, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_parquet(DEFAULT_PREDICTIONS, index=False)
    _save_decision_value_chart(test_predictions, DEFAULT_FIGURE)

    test_overlap_share = overlap_mask(
        test_propensity,
        args.overlap_lower,
        args.overlap_upper,
    ).mean()

    print()
    print("NorthStar pitch decision-value V1")
    print(f"Raw pitches read:        {raw_pitch_count:,}")
    print(f"Pitch-value rows:        {len(frame):,}")
    print(
        f"Training overlap rows:   {len(train_for_value):,} "
        f"({len(train_for_value) / len(split.train):.2%})"
    )
    print(f"September overlap share: {test_overlap_share:.2%}")

    print()
    print("Factual action-value prediction")
    print(
        factual_metrics.round(
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
    print("Doubly robust policy estimates")
    print(
        policy_values.round(
            {
                "value": 5,
                "standard_error": 5,
                "swing_share": 3,
            }
        ).to_string(index=False)
    )

    print()
    print("Top swing-value features")
    print(swing_importance.head(15).round({"importance": 4}).to_string(index=False))

    print()
    print("Top take-value features")
    print(take_importance.head(15).round({"importance": 4}).to_string(index=False))

    print()
    print(f"Saved factual metrics: {DEFAULT_FACTUAL_METRICS}")
    print(f"Saved policy values:   {DEFAULT_POLICY_VALUES}")
    print(f"Saved batter summary:  {DEFAULT_BATTER_SUMMARY}")
    print(f"Saved swing importance:{DEFAULT_SWING_IMPORTANCE}")
    print(f"Saved take importance: {DEFAULT_TAKE_IMPORTANCE}")
    print(f"Saved figure:          {DEFAULT_FIGURE}")
    print(f"Saved test preds:      {DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
