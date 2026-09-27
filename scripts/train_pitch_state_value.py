from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from northstar.features.decision_value import (
    overlap_mask,
    stabilized_action_weights,
)
from northstar.features.pitches import (
    PITCH_DECISION_REQUIRED_COLUMNS,
    build_pitch_decision_frame,
)
from northstar.features.state_value import (
    ABSOLUTE_STATE_RETURN,
    DECISION_TRANSITION_TARGET,
    attach_absolute_future_value,
    attach_decision_transition_values,
    build_telescoping_audit,
    fit_decision_state_values,
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
    modeled_action_advantage,
    optimal_action,
)
from northstar.models.state_value import (
    fit_count_zone_transition_baseline,
    predict_count_zone_transition_values,
)


DEFAULT_RAW_DIR = Path("data/raw/statcast")
DEFAULT_PA_INPUT = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_STATE_TABLE = Path(
    "reports/tables/pitch_decision_state_values_2025.csv"
)
DEFAULT_TELESCOPE = Path(
    "reports/tables/pitch_state_telescoping_audit_2025.csv"
)
DEFAULT_FACTUAL = Path(
    "reports/tables/pitch_state_value_factual_metrics_2025.csv"
)
DEFAULT_POLICY = Path(
    "reports/tables/pitch_state_value_one_step_policy_2025.csv"
)
DEFAULT_BATTER = Path(
    "reports/tables/pitch_state_value_batter_holdout_2025.csv"
)
DEFAULT_SWING_IMPORTANCE = Path(
    "reports/tables/pitch_state_value_swing_importance_2025.csv"
)
DEFAULT_TAKE_IMPORTANCE = Path(
    "reports/tables/pitch_state_value_take_importance_2025.csv"
)
DEFAULT_FIGURE = Path(
    "reports/figures/pitch_state_value_advantage_2025.png"
)
DEFAULT_PREDICTIONS = Path(
    "data/processed/pitch_state_value_test_predictions_2025.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train NorthStar V2 one-step pitch-state action-value models."
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
    frames: list[pd.DataFrame] = []
    raw_count = 0

    for index, path in enumerate(raw_paths, start=1):
        chunk = pd.read_parquet(path, columns=projected_columns)
        raw_count += len(chunk)
        frames.append(build_pitch_decision_frame(chunk))

        if index == 1 or index % 5 == 0 or index == len(raw_paths):
            print(f"  prepared chunk {index}/{len(raw_paths)}")

    return pd.concat(frames, ignore_index=True, copy=False), raw_count


def _feature_importance(model) -> pd.DataFrame:
    names = model.named_steps["preprocess"].get_feature_names_out()
    importance = model.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _metric_row(model_name: str, split_name: str, metrics) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split_name,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "pearson_r": metrics.pearson_r,
        "spearman_rho": metrics.spearman_rho,
    }


def _one_step_policy_rows(
    partition: pd.DataFrame,
    propensity: np.ndarray,
    q_take: np.ndarray,
    q_swing: np.ndarray,
    split_name: str,
) -> list[dict[str, object]]:
    outcome = partition[DECISION_TRANSITION_TARGET].to_numpy(dtype="float64")
    observed = partition[PITCH_DECISION_TARGET].to_numpy(dtype="int8")
    policies = {
        "always_take": np.zeros(len(partition), dtype="int8"),
        "always_swing": np.ones(len(partition), dtype="int8"),
        "model_action": optimal_action(q_take, q_swing),
    }

    rows: list[dict[str, object]] = []
    for name, action in policies.items():
        estimate = doubly_robust_policy_value(
            outcome,
            observed,
            propensity,
            q_take,
            q_swing,
            action,
            policy_name=name,
        )
        rows.append(
            {
                "split": split_name,
                "policy": estimate.policy,
                "one_step_value": estimate.value,
                "standard_error": estimate.standard_error,
                "swing_share": estimate.swing_share,
                "pitches": estimate.pitches,
            }
        )

    rows.append(
        {
            "split": split_name,
            "policy": "observed_actions",
            "one_step_value": float(outcome.mean()),
            "standard_error": float(outcome.std(ddof=1) / np.sqrt(len(outcome))),
            "swing_share": float(observed.mean()),
            "pitches": int(len(outcome)),
        }
    )
    return rows


def _telescoping_summary(
    split_name: str,
    audit: pd.DataFrame,
) -> dict[str, object]:
    absolute_error = audit["telescoping_error"].abs()
    return {
        "split": split_name,
        "plate_appearances": len(audit),
        "mean_abs_error": float(absolute_error.mean()),
        "p99_abs_error": float(absolute_error.quantile(0.99)),
        "max_abs_error": float(absolute_error.max()),
        "mean_decisions_per_pa": float(
            audit["decision_opportunities"].mean()
        ),
    }


def _save_advantage_chart(
    predictions: pd.DataFrame,
    output: Path,
) -> None:
    values = predictions["modeled_action_advantage"].dropna()
    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values.clip(-0.25, 0.25), bins=70)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Modeled chosen-action one-step advantage (runs)")
    ax.set_ylabel("September decision opportunities")
    ax.set_title("NorthStar pitch-state value V2 — September holdout")
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
    decision_frame, raw_pitch_count = _load_pitch_chunks(
        raw_paths,
        projected_columns,
    )
    decision_frame = attach_absolute_future_value(
        decision_frame,
        plate_appearances,
        run_expectancy,
    )
    model_frame = prepare_pitch_decision_model_frame(decision_frame)
    split = chronological_pitch_split(
        model_frame,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )

    (
        state_mapping,
        base_out_fallback,
        overall_state_value,
        state_table,
    ) = fit_decision_state_values(split.train)

    train = attach_decision_transition_values(
        split.train,
        state_mapping,
        base_out_fallback,
        overall_state_value,
    )
    validation = attach_decision_transition_values(
        split.validation,
        state_mapping,
        base_out_fallback,
        overall_state_value,
    )
    test = attach_decision_transition_values(
        split.test,
        state_mapping,
        base_out_fallback,
        overall_state_value,
    )

    telescope_rows = [
        _telescoping_summary(
            "train",
            build_telescoping_audit(train),
        ),
        _telescoping_summary(
            "validation",
            build_telescoping_audit(validation),
        ),
        _telescoping_summary(
            "test",
            build_telescoping_audit(test),
        ),
    ]
    telescope_summary = pd.DataFrame(telescope_rows)

    # The action-value model uses only direct next-decision transitions or
    # terminal decisions. Gaps remain in the telescoping audit but are excluded
    # from counterfactual training/evaluation.
    train = train.loc[train["clean_one_step_transition"]].reset_index(drop=True)
    validation = validation.loc[
        validation["clean_one_step_transition"]
    ].reset_index(drop=True)
    test = test.loc[test["clean_one_step_transition"]].reset_index(drop=True)

    features = (
        PITCH_DECISION_CATEGORICAL_FEATURES
        + PITCH_DECISION_NUMERIC_FEATURES
    )

    propensity_model = make_pitch_xgboost_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )
    propensity_model.fit(
        train[features],
        train[PITCH_DECISION_TARGET],
    )

    train_propensity = propensity_model.predict_proba(train[features])[:, 1]
    validation_propensity = propensity_model.predict_proba(
        validation[features]
    )[:, 1]
    test_propensity = propensity_model.predict_proba(test[features])[:, 1]

    train_overlap = overlap_mask(
        train_propensity,
        args.overlap_lower,
        args.overlap_upper,
    )
    train_value = train.loc[train_overlap].reset_index(drop=True)
    train_propensity_overlap = train_propensity[train_overlap]

    weights = stabilized_action_weights(
        train_value[PITCH_DECISION_TARGET],
        train_propensity_overlap,
    )
    swing_mask = train_value[PITCH_DECISION_TARGET].eq(1).to_numpy()
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
        train_value.loc[swing_mask, features],
        train_value.loc[swing_mask, DECISION_TRANSITION_TARGET],
        model__sample_weight=weights[swing_mask],
    )
    take_model.fit(
        train_value.loc[take_mask, features],
        train_value.loc[take_mask, DECISION_TRANSITION_TARGET],
        model__sample_weight=weights[take_mask],
    )

    baseline_map, action_fallback, overall_fallback = (
        fit_count_zone_transition_baseline(train_value)
    )

    metric_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    test_predictions: pd.DataFrame | None = None

    for split_name, partition, propensity in [
        ("validation", validation, validation_propensity),
        ("test", test, test_propensity),
    ]:
        overlap = overlap_mask(
            propensity,
            args.overlap_lower,
            args.overlap_upper,
        )
        current = partition.loc[overlap].reset_index(drop=True)
        current_propensity = propensity[overlap]

        baseline_take, baseline_swing = (
            predict_count_zone_transition_values(
                current,
                baseline_map,
                action_fallback,
                overall_fallback,
            )
        )
        q_take = take_model.predict(current[features])
        q_swing = swing_model.predict(current[features])

        baseline_factual = factual_action_prediction(
            current[PITCH_DECISION_TARGET],
            baseline_take,
            baseline_swing,
        )
        xgb_factual = factual_action_prediction(
            current[PITCH_DECISION_TARGET],
            q_take,
            q_swing,
        )

        metric_rows.append(
            _metric_row(
                "count_zone_transition",
                split_name,
                evaluate_regression_predictions(
                    current[DECISION_TRANSITION_TARGET],
                    baseline_factual,
                ),
            )
        )
        metric_rows.append(
            _metric_row(
                "xgboost_transition",
                split_name,
                evaluate_regression_predictions(
                    current[DECISION_TRANSITION_TARGET],
                    xgb_factual,
                ),
            )
        )

        policy_rows.extend(
            _one_step_policy_rows(
                current,
                current_propensity,
                q_take,
                q_swing,
                split_name,
            )
        )

        if split_name == "test":
            test_predictions = current[
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
                    "base_out_state_before",
                    "current_state_value",
                    "next_state_value",
                    "interval_runs",
                    DECISION_TRANSITION_TARGET,
                ]
            ].copy()
            test_predictions["swing_propensity"] = current_propensity
            test_predictions["q_take"] = q_take
            test_predictions["q_swing"] = q_swing
            test_predictions["optimal_action"] = np.where(
                q_swing >= q_take,
                "swing",
                "take",
            )
            test_predictions["optimal_choice"] = (
                test_predictions["decision"]
                == test_predictions["optimal_action"]
            )
            test_predictions["modeled_action_advantage"] = (
                modeled_action_advantage(
                    test_predictions[PITCH_DECISION_TARGET],
                    q_take,
                    q_swing,
                )
            )

    if test_predictions is None:
        raise RuntimeError("No September predictions were produced.")

    factual_metrics = pd.DataFrame(metric_rows)
    one_step_policy = pd.DataFrame(policy_rows)

    batter_summary = (
        test_predictions.groupby("batter", as_index=False)
        .agg(
            decisions=(PITCH_DECISION_TARGET, "size"),
            swing_rate=(PITCH_DECISION_TARGET, "mean"),
            optimal_choice_rate=("optimal_choice", "mean"),
            observed_transition_value=(
                DECISION_TRANSITION_TARGET,
                "mean",
            ),
            mean_modeled_action_advantage=(
                "modeled_action_advantage",
                "mean",
            ),
            mean_abs_modeled_gap=(
                "modeled_action_advantage",
                lambda s: s.abs().mean(),
            ),
        )
        .sort_values("decisions", ascending=False)
        .reset_index(drop=True)
    )

    swing_importance = _feature_importance(swing_model)
    take_importance = _feature_importance(take_model)

    DEFAULT_STATE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    state_table.to_csv(DEFAULT_STATE_TABLE, index=False)
    telescope_summary.to_csv(DEFAULT_TELESCOPE, index=False)
    factual_metrics.to_csv(DEFAULT_FACTUAL, index=False)
    one_step_policy.to_csv(DEFAULT_POLICY, index=False)
    batter_summary.to_csv(DEFAULT_BATTER, index=False)
    swing_importance.to_csv(DEFAULT_SWING_IMPORTANCE, index=False)
    take_importance.to_csv(DEFAULT_TAKE_IMPORTANCE, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_parquet(DEFAULT_PREDICTIONS, index=False)
    _save_advantage_chart(test_predictions, DEFAULT_FIGURE)

    test_overlap = overlap_mask(
        test_propensity,
        args.overlap_lower,
        args.overlap_upper,
    )

    print()
    print("NorthStar pitch-state value V2")
    print(f"Raw pitches read:          {raw_pitch_count:,}")
    print(f"Decision rows modeled:     {len(model_frame):,}")
    print(
        f"Clean train transitions:   {len(train):,} "
        f"({len(train) / len(split.train):.2%})"
    )
    print(
        f"Training overlap rows:     {len(train_value):,} "
        f"({len(train_value) / len(train):.2%})"
    )
    print(f"September overlap share:   {test_overlap.mean():.2%}")

    print()
    print("Telescoping audit")
    print(
        telescope_summary.round(
            {
                "mean_abs_error": 10,
                "p99_abs_error": 10,
                "max_abs_error": 10,
                "mean_decisions_per_pa": 3,
            }
        ).to_string(index=False)
    )

    print()
    print("Factual one-step transition prediction")
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
    print("One-step doubly robust action estimates")
    print(
        one_step_policy.round(
            {
                "one_step_value": 5,
                "standard_error": 5,
                "swing_share": 3,
            }
        ).to_string(index=False)
    )

    print()
    print("Top swing transition-value features")
    print(
        swing_importance.head(15)
        .round({"importance": 4})
        .to_string(index=False)
    )

    print()
    print("Top take transition-value features")
    print(
        take_importance.head(15)
        .round({"importance": 4})
        .to_string(index=False)
    )

    print()
    print(f"Saved state table:     {DEFAULT_STATE_TABLE}")
    print(f"Saved telescope audit: {DEFAULT_TELESCOPE}")
    print(f"Saved factual metrics: {DEFAULT_FACTUAL}")
    print(f"Saved one-step policy: {DEFAULT_POLICY}")
    print(f"Saved batter summary:  {DEFAULT_BATTER}")
    print(f"Saved swing importance:{DEFAULT_SWING_IMPORTANCE}")
    print(f"Saved take importance: {DEFAULT_TAKE_IMPORTANCE}")
    print(f"Saved figure:          {DEFAULT_FIGURE}")
    print(f"Saved test predictions:{DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
