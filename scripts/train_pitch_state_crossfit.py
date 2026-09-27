from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from northstar.features.decision_value import overlap_mask
from northstar.features.pitches import (
    PITCH_DECISION_REQUIRED_COLUMNS,
    build_pitch_decision_frame,
)
from northstar.features.state_value import (
    DECISION_TRANSITION_TARGET,
    attach_absolute_future_value,
    attach_decision_transition_values,
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
    factual_action_prediction,
    optimal_action,
)
from northstar.validation.crossfit import (
    apply_propensity_calibrator,
    cap_weights,
    crossfit_pitch_propensity,
    fit_isotonic_calibrator,
    propensity_metric_row,
    stabilized_raw_weights,
    weight_diagnostics,
)
from northstar.validation.pitch_state import (
    cluster_bootstrap_mean,
    doubly_robust_policy_score,
    propensity_calibration_table,
)


DEFAULT_RAW_DIR = Path("data/raw/statcast")
DEFAULT_PA_INPUT = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_FOLD_METRICS = Path(
    "reports/tables/pitch_state_crossfit_fold_metrics_2025.csv"
)
DEFAULT_PROPENSITY_METRICS = Path(
    "reports/tables/pitch_state_crossfit_propensity_metrics_2025.csv"
)
DEFAULT_PROPENSITY_BINS = Path(
    "reports/tables/pitch_state_crossfit_propensity_bins_2025.csv"
)
DEFAULT_WEIGHT_DIAGNOSTICS = Path(
    "reports/tables/pitch_state_crossfit_weight_diagnostics_2025.csv"
)
DEFAULT_FACTUAL_METRICS = Path(
    "reports/tables/pitch_state_crossfit_factual_metrics_2025.csv"
)
DEFAULT_POLICY_SENSITIVITY = Path(
    "reports/tables/pitch_state_crossfit_policy_sensitivity_2025.csv"
)
DEFAULT_FIGURE = Path(
    "reports/figures/pitch_state_crossfit_weight_sensitivity_2025.png"
)
DEFAULT_PREDICTIONS = Path(
    "data/processed/pitch_state_crossfit_test_predictions_2025.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit NorthStar pitch-state nuisance models, calibrate swing "
            "propensity, and refit action-value models across weight caps."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--pa-input", type=Path, default=DEFAULT_PA_INPUT)
    parser.add_argument("--validation-start", default="2025-08-01")
    parser.add_argument("--test-start", default="2025-09-01")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--weight-caps",
        type=float,
        nargs="+",
        default=[3.0, 5.0, 10.0, 20.0],
    )
    parser.add_argument("--canonical-cap", type=float, default=10.0)
    parser.add_argument("--overlap-lower", type=float, default=0.05)
    parser.add_argument("--overlap-upper", type=float, default=0.95)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
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

    if not frames:
        raise FileNotFoundError("No readable Statcast chunks were found.")

    return pd.concat(frames, ignore_index=True, copy=False), raw_count


def _prepare_state_splits(
    decision_frame: pd.DataFrame,
    plate_appearances: pd.DataFrame,
    validation_start: str,
    test_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plate_appearances = plate_appearances.copy()
    plate_appearances["game_date"] = pd.to_datetime(
        plate_appearances["game_date"],
        errors="coerce",
    )

    training_pa = plate_appearances.loc[
        plate_appearances["game_date"] < pd.Timestamp(validation_start)
    ].copy()
    run_expectancy = build_run_expectancy_table(training_pa)

    absolute = attach_absolute_future_value(
        decision_frame,
        plate_appearances,
        run_expectancy,
    )
    model_frame = prepare_pitch_decision_model_frame(absolute)
    split = chronological_pitch_split(
        model_frame,
        validation_start=validation_start,
        test_start=test_start,
    )

    (
        state_mapping,
        base_out_fallback,
        overall_state_value,
        _,
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

    train = train.loc[
        train["clean_one_step_transition"]
    ].reset_index(drop=True)
    validation = validation.loc[
        validation["clean_one_step_transition"]
    ].reset_index(drop=True)
    test = test.loc[
        test["clean_one_step_transition"]
    ].reset_index(drop=True)

    return train, validation, test


def _weight_row(diagnostics) -> dict[str, object]:
    return {
        "weight_cap": diagnostics.cap,
        "rows": diagnostics.rows,
        "clipped_rows": diagnostics.clipped_rows,
        "clipped_pct": diagnostics.clipped_pct,
        "raw_max": diagnostics.raw_max,
        "clipped_max": diagnostics.clipped_max,
        "mean_weight": diagnostics.mean_weight,
        "effective_sample_size": diagnostics.effective_sample_size,
        "effective_sample_pct": diagnostics.effective_sample_pct,
        "swing_effective_sample_size": (
            diagnostics.swing_effective_sample_size
        ),
        "take_effective_sample_size": diagnostics.take_effective_sample_size,
    }


def _factual_row(
    cap: float,
    split_name: str,
    metrics,
) -> dict[str, object]:
    return {
        "weight_cap": cap,
        "split": split_name,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "pearson_r": metrics.pearson_r,
        "spearman_rho": metrics.spearman_rho,
    }


def _policy_row(
    cap: float,
    split_name: str,
    frame: pd.DataFrame,
    propensity: np.ndarray,
    q_take: np.ndarray,
    q_swing: np.ndarray,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, object]:
    outcome = frame[DECISION_TRANSITION_TARGET].to_numpy(dtype="float64")
    observed_action = frame[PITCH_DECISION_TARGET].to_numpy(dtype="int8")
    model_action = optimal_action(q_take, q_swing)

    model_score = doubly_robust_policy_score(
        outcome,
        observed_action,
        propensity,
        q_take,
        q_swing,
        model_action,
    )
    advantage = model_score - outcome

    bootstrap = cluster_bootstrap_mean(
        advantage,
        frame["game_pk"],
        reps=bootstrap_reps,
        seed=seed,
    )

    return {
        "weight_cap": cap,
        "split": split_name,
        "pitches": len(frame),
        "games": int(frame["game_pk"].nunique()),
        "observed_swing_share": float(observed_action.mean()),
        "model_swing_share": float(model_action.mean()),
        "observed_value": float(outcome.mean()),
        "model_dr_value": float(model_score.mean()),
        "model_advantage": float(advantage.mean()),
        "cluster_bootstrap_se": bootstrap.standard_error,
        "ci_low": bootstrap.ci_low,
        "ci_high": bootstrap.ci_high,
        "optimal_choice_rate": float(
            (observed_action == model_action).mean()
        ),
        "mean_abs_q_gap": float(np.abs(q_swing - q_take).mean()),
    }


def _save_sensitivity_figure(
    policy: pd.DataFrame,
    output: Path,
) -> None:
    current = policy.loc[policy["split"].eq("test")].copy()
    if current.empty:
        return

    current = current.sort_values("weight_cap")
    estimate = current["model_advantage"].to_numpy()
    lower_error = estimate - current["ci_low"].to_numpy()
    upper_error = current["ci_high"].to_numpy() - estimate

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        current["weight_cap"].astype(str),
        estimate,
        yerr=[lower_error, upper_error],
        marker="o",
        capsize=4,
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Stabilized IPW cap")
    ax.set_ylabel("Model-action advantage vs observed (runs / decision)")
    ax.set_title("Pitch-state cross-fit weight sensitivity — September 2025")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    if args.bootstrap_reps < 100:
        raise ValueError("--bootstrap-reps must be >= 100")
    if not 0.0 < args.overlap_lower < args.overlap_upper < 1.0:
        raise ValueError("Overlap bounds must satisfy 0 < lower < upper < 1.")
    if any(cap <= 0 for cap in args.weight_caps):
        raise ValueError("All --weight-caps must be > 0")
    if args.canonical_cap not in args.weight_caps:
        raise ValueError("--canonical-cap must be included in --weight-caps")
    if not args.pa_input.exists():
        raise FileNotFoundError(
            f"{args.pa_input} does not exist. Build the 2025 baseline first."
        )

    raw_paths = sorted(args.raw_dir.glob("statcast_*.parquet"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No cached Statcast Parquet files found under {args.raw_dir}."
        )

    projected_columns = sorted(PITCH_DECISION_REQUIRED_COLUMNS)
    print(
        f"Reading {len(raw_paths)} cached Statcast chunks "
        f"with {len(projected_columns)} projected columns..."
    )
    decision_frame, raw_pitch_count = _load_pitch_chunks(
        raw_paths,
        projected_columns,
    )
    plate_appearances = pd.read_parquet(args.pa_input)

    train, validation, test = _prepare_state_splits(
        decision_frame,
        plate_appearances,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )

    features = (
        PITCH_DECISION_CATEGORICAL_FEATURES
        + PITCH_DECISION_NUMERIC_FEATURES
    )

    print()
    print(
        f"Cross-fitting swing propensity across {args.folds} game-grouped folds..."
    )
    crossfit = crossfit_pitch_propensity(
        train,
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
        target=PITCH_DECISION_TARGET,
        group_column="game_pk",
        folds=args.folds,
        seed=42,
        progress=True,
    )

    calibrator = fit_isotonic_calibrator(
        crossfit.oof_raw,
        train[PITCH_DECISION_TARGET],
    )
    train_oof_calibrated = apply_propensity_calibrator(
        calibrator,
        crossfit.oof_raw,
    )

    print("Fitting final propensity model on the full pre-August training sample...")
    final_propensity_model = make_pitch_xgboost_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )
    final_propensity_model.fit(
        train[features],
        train[PITCH_DECISION_TARGET],
    )

    validation_raw = final_propensity_model.predict_proba(
        validation[features]
    )[:, 1]
    test_raw = final_propensity_model.predict_proba(
        test[features]
    )[:, 1]
    validation_propensity = apply_propensity_calibrator(
        calibrator,
        validation_raw,
    )
    test_propensity = apply_propensity_calibrator(
        calibrator,
        test_raw,
    )

    propensity_metrics = pd.DataFrame(
        [
            propensity_metric_row(
                "train_oof_raw",
                train[PITCH_DECISION_TARGET],
                crossfit.oof_raw,
            ),
            propensity_metric_row(
                "train_oof_calibrated",
                train[PITCH_DECISION_TARGET],
                train_oof_calibrated,
            ),
            propensity_metric_row(
                "validation_raw",
                validation[PITCH_DECISION_TARGET],
                validation_raw,
            ),
            propensity_metric_row(
                "validation_calibrated",
                validation[PITCH_DECISION_TARGET],
                validation_propensity,
            ),
            propensity_metric_row(
                "test_raw",
                test[PITCH_DECISION_TARGET],
                test_raw,
            ),
            propensity_metric_row(
                "test_calibrated",
                test[PITCH_DECISION_TARGET],
                test_propensity,
            ),
        ]
    )

    test_bins = propensity_calibration_table(
        test[PITCH_DECISION_TARGET],
        test_propensity,
        bins=10,
    )

    train_overlap = overlap_mask(
        train_oof_calibrated,
        args.overlap_lower,
        args.overlap_upper,
    )
    train_value = train.loc[train_overlap].reset_index(drop=True)
    train_propensity_overlap = train_oof_calibrated[train_overlap]

    raw_weights = stabilized_raw_weights(
        train_value[PITCH_DECISION_TARGET],
        train_propensity_overlap,
    )

    validation_overlap = overlap_mask(
        validation_propensity,
        args.overlap_lower,
        args.overlap_upper,
    )
    test_overlap = overlap_mask(
        test_propensity,
        args.overlap_lower,
        args.overlap_upper,
    )
    validation_value = validation.loc[
        validation_overlap
    ].reset_index(drop=True)
    test_value = test.loc[test_overlap].reset_index(drop=True)
    validation_propensity_overlap = validation_propensity[validation_overlap]
    test_propensity_overlap = test_propensity[test_overlap]

    weight_rows: list[dict[str, object]] = []
    factual_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    canonical_predictions: pd.DataFrame | None = None

    for index, cap in enumerate(sorted(set(args.weight_caps))):
        diagnostics = weight_diagnostics(
            train_value[PITCH_DECISION_TARGET],
            raw_weights,
            cap,
        )
        weight_rows.append(_weight_row(diagnostics))

        weights = cap_weights(raw_weights, cap)
        swing_mask = train_value[PITCH_DECISION_TARGET].eq(1).to_numpy()
        take_mask = ~swing_mask

        print()
        print(
            f"Weight cap {cap:g}: fitting swing/take value models "
            f"(clipped {diagnostics.clipped_pct:.2f}% of rows; "
            f"ESS {diagnostics.effective_sample_pct:.1f}%)..."
        )

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
            train_value.loc[
                swing_mask,
                DECISION_TRANSITION_TARGET,
            ],
            model__sample_weight=weights[swing_mask],
        )
        take_model.fit(
            train_value.loc[take_mask, features],
            train_value.loc[
                take_mask,
                DECISION_TRANSITION_TARGET,
            ],
            model__sample_weight=weights[take_mask],
        )

        for split_name, current, current_propensity in [
            (
                "validation",
                validation_value,
                validation_propensity_overlap,
            ),
            ("test", test_value, test_propensity_overlap),
        ]:
            q_take = take_model.predict(current[features])
            q_swing = swing_model.predict(current[features])

            factual = factual_action_prediction(
                current[PITCH_DECISION_TARGET],
                q_take,
                q_swing,
            )
            metrics = evaluate_regression_predictions(
                current[DECISION_TRANSITION_TARGET],
                factual,
            )
            factual_rows.append(
                _factual_row(cap, split_name, metrics)
            )

            policy_rows.append(
                _policy_row(
                    cap,
                    split_name,
                    current,
                    current_propensity,
                    q_take,
                    q_swing,
                    bootstrap_reps=args.bootstrap_reps,
                    seed=42 + index * 10 + (1 if split_name == "test" else 0),
                )
            )

            if (
                split_name == "test"
                and float(cap) == float(args.canonical_cap)
            ):
                canonical_predictions = current[
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
                        DECISION_TRANSITION_TARGET,
                    ]
                ].copy()
                canonical_predictions["swing_propensity_raw"] = (
                    test_raw[test_overlap]
                )
                canonical_predictions["swing_propensity_calibrated"] = (
                    current_propensity
                )
                canonical_predictions["q_take"] = q_take
                canonical_predictions["q_swing"] = q_swing
                canonical_predictions["model_action"] = np.where(
                    q_swing >= q_take,
                    "swing",
                    "take",
                )

    weight_table = pd.DataFrame(weight_rows)
    factual_metrics = pd.DataFrame(factual_rows)
    policy_sensitivity = pd.DataFrame(policy_rows)

    if canonical_predictions is None:
        raise RuntimeError("Canonical test predictions were not generated.")

    DEFAULT_FOLD_METRICS.parent.mkdir(parents=True, exist_ok=True)
    crossfit.fold_metrics.to_csv(DEFAULT_FOLD_METRICS, index=False)
    propensity_metrics.to_csv(DEFAULT_PROPENSITY_METRICS, index=False)
    test_bins.to_csv(DEFAULT_PROPENSITY_BINS, index=False)
    weight_table.to_csv(DEFAULT_WEIGHT_DIAGNOSTICS, index=False)
    factual_metrics.to_csv(DEFAULT_FACTUAL_METRICS, index=False)
    policy_sensitivity.to_csv(DEFAULT_POLICY_SENSITIVITY, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    canonical_predictions.to_parquet(DEFAULT_PREDICTIONS, index=False)
    _save_sensitivity_figure(
        policy_sensitivity,
        DEFAULT_FIGURE,
    )

    print()
    print("NorthStar pitch-state cross-fit V1")
    print(f"Raw pitches read:            {raw_pitch_count:,}")
    print(f"Clean train transitions:     {len(train):,}")
    print(
        f"OOF training overlap rows:   {len(train_value):,} "
        f"({len(train_value) / len(train):.2%})"
    )
    print(
        f"Calibrated validation overlap:{len(validation_value):,} "
        f"({len(validation_value) / len(validation):.2%})"
    )
    print(
        f"Calibrated September overlap: {len(test_value):,} "
        f"({len(test_value) / len(test):.2%})"
    )

    print()
    print("Cross-fit fold propensity metrics")
    print(
        crossfit.fold_metrics.round(
            {
                "roc_auc": 3,
                "log_loss": 4,
                "brier": 4,
                "observed_swing_rate": 3,
                "mean_predicted_swing": 3,
            }
        ).to_string(index=False)
    )

    print()
    print("Propensity calibration metrics")
    print(
        propensity_metrics.round(
            {
                "roc_auc": 3,
                "log_loss": 4,
                "brier": 4,
                "mean_predicted_swing": 3,
                "observed_swing_rate": 3,
            }
        ).to_string(index=False)
    )

    print()
    print("Weight diagnostics")
    print(
        weight_table.round(
            {
                "clipped_pct": 3,
                "raw_max": 3,
                "clipped_max": 3,
                "mean_weight": 3,
                "effective_sample_size": 0,
                "effective_sample_pct": 2,
                "swing_effective_sample_size": 0,
                "take_effective_sample_size": 0,
            }
        ).to_string(index=False)
    )

    print()
    print("Factual transition-value metrics by weight cap")
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
    print("Cross-fit / calibrated policy sensitivity")
    print(
        policy_sensitivity.round(
            {
                "observed_swing_share": 3,
                "model_swing_share": 3,
                "observed_value": 5,
                "model_dr_value": 5,
                "model_advantage": 5,
                "cluster_bootstrap_se": 5,
                "ci_low": 5,
                "ci_high": 5,
                "optimal_choice_rate": 3,
                "mean_abs_q_gap": 5,
            }
        ).to_string(index=False)
    )

    print()
    print(f"Saved fold metrics:       {DEFAULT_FOLD_METRICS}")
    print(f"Saved propensity metrics: {DEFAULT_PROPENSITY_METRICS}")
    print(f"Saved calibration bins:   {DEFAULT_PROPENSITY_BINS}")
    print(f"Saved weight diagnostics: {DEFAULT_WEIGHT_DIAGNOSTICS}")
    print(f"Saved factual metrics:    {DEFAULT_FACTUAL_METRICS}")
    print(f"Saved policy sensitivity: {DEFAULT_POLICY_SENSITIVITY}")
    print(f"Saved figure:             {DEFAULT_FIGURE}")
    print(f"Saved canonical preds:    {DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
