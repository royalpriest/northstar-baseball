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
from northstar.validation.external_season import (
    season_sample_summary,
    state_transport_coverage,
)
from northstar.validation.pitch_state import (
    cluster_bootstrap_mean,
    doubly_robust_policy_score,
    propensity_calibration_table,
)


DEFAULT_FOLD_METRICS = Path(
    "reports/tables/external_season_propensity_folds_2024_2025.csv"
)
DEFAULT_SAMPLE_SUMMARY = Path(
    "reports/tables/external_season_sample_summary_2024_2025.csv"
)
DEFAULT_STATE_COVERAGE = Path(
    "reports/tables/external_season_state_coverage_2024_2025.csv"
)
DEFAULT_PROPENSITY_METRICS = Path(
    "reports/tables/external_season_propensity_metrics_2024_2025.csv"
)
DEFAULT_PROPENSITY_BINS = Path(
    "reports/tables/external_season_propensity_bins_2024_2025.csv"
)
DEFAULT_WEIGHT_DIAGNOSTICS = Path(
    "reports/tables/external_season_weight_diagnostics_2024_2025.csv"
)
DEFAULT_FACTUAL_METRICS = Path(
    "reports/tables/external_season_factual_metrics_2024_2025.csv"
)
DEFAULT_POLICY_SENSITIVITY = Path(
    "reports/tables/external_season_policy_sensitivity_2024_2025.csv"
)
DEFAULT_FIGURE = Path(
    "reports/figures/external_season_weight_sensitivity_2024_2025.png"
)
DEFAULT_PREDICTIONS = Path(
    "data/processed/external_season_test_predictions_2024_2025.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train NorthStar pitch-state models on one MLB season and evaluate "
            "them without refitting on a later external season."
        )
    )
    parser.add_argument("--train-season", type=int, default=2024)
    parser.add_argument("--external-season", type=int, default=2025)
    parser.add_argument("--train-raw-dir", type=Path, default=None)
    parser.add_argument("--external-raw-dir", type=Path, default=None)
    parser.add_argument("--train-pa", type=Path, default=None)
    parser.add_argument("--external-pa", type=Path, default=None)
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


def _default_raw_dir(season: int) -> Path:
    if season == 2025:
        return Path("data/raw/statcast")
    return Path(f"data/raw/statcast_{season}")


def _default_pa_path(season: int) -> Path:
    return Path(f"data/processed/plate_appearances_{season}.parquet")


def _load_pitch_chunks(
    raw_dir: Path,
    projected_columns: list[str],
    label: str,
) -> tuple[pd.DataFrame, int, int]:
    raw_paths = sorted(raw_dir.glob("statcast_*.parquet"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No cached Statcast Parquet files found under {raw_dir}."
        )

    frames: list[pd.DataFrame] = []
    raw_count = 0

    print(
        f"Reading {label}: {len(raw_paths)} cached Statcast chunks "
        f"with {len(projected_columns)} projected columns..."
    )
    for index, path in enumerate(raw_paths, start=1):
        chunk = pd.read_parquet(path, columns=projected_columns)
        raw_count += len(chunk)
        frames.append(build_pitch_decision_frame(chunk))

        if index == 1 or index % 5 == 0 or index == len(raw_paths):
            print(
                f"  {label} prepared chunk {index}/{len(raw_paths)}"
            )

    return (
        pd.concat(frames, ignore_index=True, copy=False),
        raw_count,
        len(raw_paths),
    )


def _prepare_transport_frames(
    train_decisions: pd.DataFrame,
    external_decisions: pd.DataFrame,
    train_pa: pd.DataFrame,
    external_pa: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
    pd.DataFrame,
]:
    """Build all state/transition quantities from the training season only."""
    run_expectancy = build_run_expectancy_table(train_pa)

    train_absolute = attach_absolute_future_value(
        train_decisions,
        train_pa,
        run_expectancy,
    )
    external_absolute = attach_absolute_future_value(
        external_decisions,
        external_pa,
        run_expectancy,
    )

    train_model = prepare_pitch_decision_model_frame(train_absolute)
    external_model = prepare_pitch_decision_model_frame(external_absolute)

    (
        state_mapping,
        base_out_fallback,
        overall_state_value,
        state_table,
    ) = fit_decision_state_values(train_model)

    train_transition = attach_decision_transition_values(
        train_model,
        state_mapping,
        base_out_fallback,
        overall_state_value,
    )
    external_transition = attach_decision_transition_values(
        external_model,
        state_mapping,
        base_out_fallback,
        overall_state_value,
    )

    train_transition = train_transition.loc[
        train_transition["clean_one_step_transition"]
    ].reset_index(drop=True)
    external_transition = external_transition.loc[
        external_transition["clean_one_step_transition"]
    ].reset_index(drop=True)

    return (
        train_transition,
        external_transition,
        state_mapping,
        state_table,
    )


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


def _factual_row(cap: float, metrics) -> dict[str, object]:
    return {
        "weight_cap": cap,
        "split": "external",
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "pearson_r": metrics.pearson_r,
        "spearman_rho": metrics.spearman_rho,
    }


def _policy_row(
    cap: float,
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
        "split": "external",
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


def _save_weight_figure(
    policy: pd.DataFrame,
    output: Path,
    train_season: int,
    external_season: int,
) -> None:
    if policy.empty:
        return

    current = policy.sort_values("weight_cap")
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
    ax.set_title(
        f"External-season validation: {train_season} → {external_season}"
    )
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.train_season >= args.external_season:
        raise ValueError("--train-season must be earlier than --external-season")
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

    train_raw_dir = (
        args.train_raw_dir
        if args.train_raw_dir is not None
        else _default_raw_dir(args.train_season)
    )
    external_raw_dir = (
        args.external_raw_dir
        if args.external_raw_dir is not None
        else _default_raw_dir(args.external_season)
    )
    train_pa_path = (
        args.train_pa
        if args.train_pa is not None
        else _default_pa_path(args.train_season)
    )
    external_pa_path = (
        args.external_pa
        if args.external_pa is not None
        else _default_pa_path(args.external_season)
    )

    for label, path in [
        ("training plate appearances", train_pa_path),
        ("external plate appearances", external_pa_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    projected_columns = sorted(PITCH_DECISION_REQUIRED_COLUMNS)
    train_decisions, train_raw_count, _ = _load_pitch_chunks(
        train_raw_dir,
        projected_columns,
        f"{args.train_season} training season",
    )
    external_decisions, external_raw_count, _ = _load_pitch_chunks(
        external_raw_dir,
        projected_columns,
        f"{args.external_season} external season",
    )

    train_pa = pd.read_parquet(train_pa_path)
    external_pa = pd.read_parquet(external_pa_path)

    (
        train,
        external,
        state_mapping,
        _,
    ) = _prepare_transport_frames(
        train_decisions,
        external_decisions,
        train_pa,
        external_pa,
    )

    sample_summary = season_sample_summary(
        train,
        external,
        train_season=args.train_season,
        external_season=args.external_season,
    )
    state_coverage = state_transport_coverage(
        state_mapping,
        external,
    )
    state_coverage.insert(0, "train_season", args.train_season)
    state_coverage.insert(1, "external_season", args.external_season)

    features = (
        PITCH_DECISION_CATEGORICAL_FEATURES
        + PITCH_DECISION_NUMERIC_FEATURES
    )

    print()
    print(
        f"Cross-fitting {args.train_season} swing propensity across "
        f"{args.folds} game-grouped folds..."
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

    print(
        f"Fitting final propensity model on all {args.train_season} "
        "training pitches..."
    )
    final_propensity_model = make_pitch_xgboost_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )
    final_propensity_model.fit(
        train[features],
        train[PITCH_DECISION_TARGET],
    )

    external_raw_propensity = final_propensity_model.predict_proba(
        external[features]
    )[:, 1]
    external_propensity = apply_propensity_calibrator(
        calibrator,
        external_raw_propensity,
    )

    propensity_metrics = pd.DataFrame(
        [
            propensity_metric_row(
                f"{args.train_season}_oof_raw",
                train[PITCH_DECISION_TARGET],
                crossfit.oof_raw,
            ),
            propensity_metric_row(
                f"{args.train_season}_oof_calibrated",
                train[PITCH_DECISION_TARGET],
                train_oof_calibrated,
            ),
            propensity_metric_row(
                f"{args.external_season}_external_raw",
                external[PITCH_DECISION_TARGET],
                external_raw_propensity,
            ),
            propensity_metric_row(
                f"{args.external_season}_external_calibrated",
                external[PITCH_DECISION_TARGET],
                external_propensity,
            ),
        ]
    )
    external_bins = propensity_calibration_table(
        external[PITCH_DECISION_TARGET],
        external_propensity,
        bins=10,
    )

    train_overlap = overlap_mask(
        train_oof_calibrated,
        args.overlap_lower,
        args.overlap_upper,
    )
    train_value = train.loc[train_overlap].reset_index(drop=True)
    train_propensity_overlap = train_oof_calibrated[train_overlap]

    external_overlap = overlap_mask(
        external_propensity,
        args.overlap_lower,
        args.overlap_upper,
    )
    external_value = external.loc[
        external_overlap
    ].reset_index(drop=True)
    external_propensity_overlap = external_propensity[external_overlap]

    raw_weights = stabilized_raw_weights(
        train_value[PITCH_DECISION_TARGET],
        train_propensity_overlap,
    )

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
            f"Weight cap {cap:g}: fitting {args.train_season} swing/take "
            f"value models (clipped {diagnostics.clipped_pct:.2f}%; "
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

        q_take = take_model.predict(external_value[features])
        q_swing = swing_model.predict(external_value[features])

        factual = factual_action_prediction(
            external_value[PITCH_DECISION_TARGET],
            q_take,
            q_swing,
        )
        factual_metrics = evaluate_regression_predictions(
            external_value[DECISION_TRANSITION_TARGET],
            factual,
        )
        factual_rows.append(_factual_row(cap, factual_metrics))

        policy_rows.append(
            _policy_row(
                cap,
                external_value,
                external_propensity_overlap,
                q_take,
                q_swing,
                bootstrap_reps=args.bootstrap_reps,
                seed=100 + index,
            )
        )

        if float(cap) == float(args.canonical_cap):
            canonical_predictions = external_value[
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
                    "decision_state",
                    DECISION_TRANSITION_TARGET,
                ]
            ].copy()
            canonical_predictions["swing_propensity_raw"] = (
                external_raw_propensity[external_overlap]
            )
            canonical_predictions["swing_propensity_calibrated"] = (
                external_propensity_overlap
            )
            canonical_predictions["q_take"] = q_take
            canonical_predictions["q_swing"] = q_swing
            canonical_predictions["model_action"] = np.where(
                q_swing >= q_take,
                "swing",
                "take",
            )

    weight_table = pd.DataFrame(weight_rows)
    factual_table = pd.DataFrame(factual_rows)
    policy_table = pd.DataFrame(policy_rows)

    if canonical_predictions is None:
        raise RuntimeError("Canonical external predictions were not generated.")

    DEFAULT_SAMPLE_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    sample_summary.to_csv(DEFAULT_SAMPLE_SUMMARY, index=False)
    state_coverage.to_csv(DEFAULT_STATE_COVERAGE, index=False)
    crossfit.fold_metrics.to_csv(DEFAULT_FOLD_METRICS, index=False)
    propensity_metrics.to_csv(DEFAULT_PROPENSITY_METRICS, index=False)
    external_bins.to_csv(DEFAULT_PROPENSITY_BINS, index=False)
    weight_table.to_csv(DEFAULT_WEIGHT_DIAGNOSTICS, index=False)
    factual_table.to_csv(DEFAULT_FACTUAL_METRICS, index=False)
    policy_table.to_csv(DEFAULT_POLICY_SENSITIVITY, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    canonical_predictions.to_parquet(DEFAULT_PREDICTIONS, index=False)
    _save_weight_figure(
        policy_table,
        DEFAULT_FIGURE,
        args.train_season,
        args.external_season,
    )

    print()
    print(
        f"NorthStar external-season validation "
        f"{args.train_season} -> {args.external_season}"
    )
    print(f"{args.train_season} raw pitches:       {train_raw_count:,}")
    print(f"{args.external_season} raw pitches:       {external_raw_count:,}")
    print(f"{args.train_season} clean transitions:  {len(train):,}")
    print(f"{args.external_season} clean transitions:  {len(external):,}")
    print(
        f"OOF training overlap:     {len(train_value):,} "
        f"({len(train_value) / len(train):.2%})"
    )
    print(
        f"External-season overlap:  {len(external_value):,} "
        f"({len(external_value) / len(external):.2%})"
    )

    print()
    print("Season sample summary")
    print(
        sample_summary.round(
            {"observed_swing_rate": 3}
        ).to_string(index=False)
    )

    print()
    print("Training-state transport coverage")
    print(
        state_coverage.round(
            {"exact_state_pct": 3}
        ).to_string(index=False)
    )

    print()
    print("Cross-fit training propensity folds")
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
    print("External propensity transport")
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
    print("2024 training weight diagnostics")
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
    print("External factual transition-value metrics")
    print(
        factual_table.round(
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
    print("External model-action policy sensitivity")
    print(
        policy_table.round(
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
    print(f"Saved sample summary:      {DEFAULT_SAMPLE_SUMMARY}")
    print(f"Saved state coverage:      {DEFAULT_STATE_COVERAGE}")
    print(f"Saved training folds:      {DEFAULT_FOLD_METRICS}")
    print(f"Saved propensity metrics:  {DEFAULT_PROPENSITY_METRICS}")
    print(f"Saved calibration bins:    {DEFAULT_PROPENSITY_BINS}")
    print(f"Saved weight diagnostics:  {DEFAULT_WEIGHT_DIAGNOSTICS}")
    print(f"Saved factual metrics:     {DEFAULT_FACTUAL_METRICS}")
    print(f"Saved policy sensitivity:  {DEFAULT_POLICY_SENSITIVITY}")
    print(f"Saved figure:              {DEFAULT_FIGURE}")
    print(f"Saved external predictions:{DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
