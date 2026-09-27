from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
from northstar.metrics.execution import (
    EXECUTION_RESIDUAL,
    attach_execution_residual,
    build_execution_half_reliability,
    build_player_execution_summary,
    fit_swing_execution_baseline,
    merge_selection_execution_profile,
    predict_swing_execution_baseline,
)
from northstar.metrics.run_expectancy import build_run_expectancy_table
from northstar.models.baselines import (
    evaluate_regression_predictions,
    make_pitch_value_xgboost_model,
)
from northstar.models.pitch_decision import (
    PITCH_DECISION_CATEGORICAL_FEATURES,
    PITCH_DECISION_NUMERIC_FEATURES,
    prepare_pitch_decision_model_frame,
)


DEFAULT_TRAIN_RAW = Path("data/raw/statcast_2024")
DEFAULT_EXTERNAL_RAW = Path("data/raw/statcast")
DEFAULT_TRAIN_PA = Path("data/processed/plate_appearances_2024.parquet")
DEFAULT_EXTERNAL_PA = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_SELECTION = Path(
    "reports/tables/player_decision_metric_prototype_2025.csv"
)

DEFAULT_MODEL_METRICS = Path(
    "reports/tables/swing_execution_model_metrics_2024_2025.csv"
)
DEFAULT_PLAYER_SUMMARY = Path(
    "reports/tables/player_swing_execution_2025.csv"
)
DEFAULT_HALF_TABLE = Path(
    "reports/tables/player_swing_execution_half_splits_2025.csv"
)
DEFAULT_RELIABILITY = Path(
    "reports/tables/player_swing_execution_reliability_2025.csv"
)
DEFAULT_PROFILE = Path(
    "reports/tables/selection_execution_profile_2025.csv"
)
DEFAULT_RELIABILITY_FIGURE = Path(
    "reports/figures/player_swing_execution_reliability_2025.png"
)
DEFAULT_PROFILE_FIGURE = Path(
    "reports/figures/selection_execution_profile_2025.png"
)
DEFAULT_PREDICTIONS = Path(
    "data/processed/swing_execution_test_predictions_2024_2025.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 2024 league swing-execution model and evaluate hitter "
            "execution above pitch/context expectation on the 2025 season."
        )
    )
    parser.add_argument("--train-raw-dir", type=Path, default=DEFAULT_TRAIN_RAW)
    parser.add_argument(
        "--external-raw-dir",
        type=Path,
        default=DEFAULT_EXTERNAL_RAW,
    )
    parser.add_argument("--train-pa", type=Path, default=DEFAULT_TRAIN_PA)
    parser.add_argument(
        "--external-pa",
        type=Path,
        default=DEFAULT_EXTERNAL_PA,
    )
    parser.add_argument(
        "--selection-table",
        type=Path,
        default=DEFAULT_SELECTION,
    )
    parser.add_argument("--min-swings", type=int, default=300)
    parser.add_argument("--min-half-swings", type=int, default=150)
    parser.add_argument("--split-date", default="2025-07-01")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def _load_decision_chunks(
    raw_dir: Path,
    label: str,
) -> tuple[pd.DataFrame, int]:
    paths = sorted(raw_dir.glob("statcast_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No cached Statcast Parquet files found under {raw_dir}."
        )

    projected = sorted(PITCH_DECISION_REQUIRED_COLUMNS)
    frames: list[pd.DataFrame] = []
    raw_count = 0

    print(
        f"Reading {label}: {len(paths)} cached chunks "
        f"with {len(projected)} projected columns..."
    )
    for index, path in enumerate(paths, start=1):
        chunk = pd.read_parquet(path, columns=projected)
        raw_count += len(chunk)
        frames.append(build_pitch_decision_frame(chunk))

        if index == 1 or index % 5 == 0 or index == len(paths):
            print(f"  prepared chunk {index}/{len(paths)}")

    return pd.concat(frames, ignore_index=True, copy=False), raw_count


def _prepare_external_execution_frames(
    train_decisions: pd.DataFrame,
    external_decisions: pd.DataFrame,
    train_pa: pd.DataFrame,
    external_pa: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use only 2024 RE/state values to define 2024 and 2025 swing targets."""
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
        _,
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

    train_swings = train_transition.loc[
        train_transition["clean_one_step_transition"]
        & train_transition["is_swing"].eq(1)
    ].reset_index(drop=True)
    external_swings = external_transition.loc[
        external_transition["clean_one_step_transition"]
        & external_transition["is_swing"].eq(1)
    ].reset_index(drop=True)

    return train_swings, external_swings


def _metric_row(model: str, metrics) -> dict[str, object]:
    return {
        "train_season": 2024,
        "external_season": 2025,
        "model": model,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "pearson_r": metrics.pearson_r,
        "spearman_rho": metrics.spearman_rho,
    }


def _save_reliability_figure(
    half_table: pd.DataFrame,
    output: Path,
) -> None:
    pivot = half_table.pivot(
        index="batter",
        columns="season_half",
        values="execution_value_per_100_swings",
    ).dropna()
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pivot["early"], pivot["late"], alpha=0.65)

    lower = min(pivot["early"].min(), pivot["late"].min())
    upper = max(pivot["early"].max(), pivot["late"].max())
    ax.plot([lower, upper], [lower, upper], linestyle="--")

    ax.set_xlabel("Early execution value / 100 swings")
    ax.set_ylabel("Late execution value / 100 swings")
    ax.set_title("2025 swing-execution split-half reliability")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_profile_figure(
    profile: pd.DataFrame,
    output: Path,
) -> None:
    required = {
        "selection_centered_per_100",
        "centered_execution_per_100_swings",
    }
    if profile.empty or not required.issubset(profile.columns):
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        profile["selection_centered_per_100"],
        profile["centered_execution_per_100_swings"],
        alpha=0.65,
    )
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Selection value / 100 vs profile average")
    ax.set_ylabel("Execution value / 100 swings vs league average")
    ax.set_title("NorthStar selection / execution profile — 2025")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.min_swings < 1:
        raise ValueError("--min-swings must be >= 1")
    if args.min_half_swings < 1:
        raise ValueError("--min-half-swings must be >= 1")
    if args.top < 1:
        raise ValueError("--top must be >= 1")

    for label, path in [
        ("2024 plate appearances", args.train_pa),
        ("2025 plate appearances", args.external_pa),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    train_decisions, train_raw_count = _load_decision_chunks(
        args.train_raw_dir,
        "2024 training season",
    )
    external_decisions, external_raw_count = _load_decision_chunks(
        args.external_raw_dir,
        "2025 external season",
    )

    train_pa = pd.read_parquet(args.train_pa)
    external_pa = pd.read_parquet(args.external_pa)

    train_swings, external_swings = _prepare_external_execution_frames(
        train_decisions,
        external_decisions,
        train_pa,
        external_pa,
    )

    features = (
        PITCH_DECISION_CATEGORICAL_FEATURES
        + PITCH_DECISION_NUMERIC_FEATURES
    )

    baseline_map, pitch_fallback, overall_fallback = (
        fit_swing_execution_baseline(train_swings)
    )
    baseline_prediction = predict_swing_execution_baseline(
        external_swings,
        baseline_map,
        pitch_fallback,
        overall_fallback,
    )
    baseline_metrics = evaluate_regression_predictions(
        external_swings[DECISION_TRANSITION_TARGET],
        baseline_prediction,
    )

    print()
    print(
        "Fitting 2024 swing-execution XGBoost model "
        "(no batter identity, no post-swing features)..."
    )
    model = make_pitch_value_xgboost_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )
    model.fit(
        train_swings[features],
        train_swings[DECISION_TRANSITION_TARGET],
    )
    external_prediction = model.predict(external_swings[features])
    model_metrics = evaluate_regression_predictions(
        external_swings[DECISION_TRANSITION_TARGET],
        external_prediction,
    )

    metric_table = pd.DataFrame(
        [
            _metric_row("count_zone_pitch_type", baseline_metrics),
            _metric_row("xgboost_execution", model_metrics),
        ]
    )

    scored = attach_execution_residual(
        external_swings,
        external_prediction,
    )
    player_summary = build_player_execution_summary(
        scored,
        min_swings=args.min_swings,
    )
    half_table, reliability = build_execution_half_reliability(
        scored,
        split_date=args.split_date,
        min_half_swings=args.min_half_swings,
    )

    profile = pd.DataFrame()
    if args.selection_table.exists():
        selection = pd.read_csv(args.selection_table)
        profile = merge_selection_execution_profile(
            player_summary,
            selection,
        )

        if (
            "player_name" in profile.columns
            and "player_name" not in player_summary.columns
        ):
            player_summary = player_summary.merge(
                profile[["batter", "player_name"]].drop_duplicates("batter"),
                on="batter",
                how="left",
                validate="one_to_one",
            )
    else:
        print(
            f"Warning: {args.selection_table} was not found; "
            "two-axis profile will be skipped."
        )

    prediction_columns = [
        column
        for column in [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "batter",
            "pitcher",
            "description",
            "balls",
            "strikes",
            "pitch_type",
            "zone",
            "base_out_state_before",
            DECISION_TRANSITION_TARGET,
            "expected_swing_transition_value",
            EXECUTION_RESIDUAL,
            "positive_execution",
        ]
        if column in scored.columns
    ]

    DEFAULT_MODEL_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metric_table.to_csv(DEFAULT_MODEL_METRICS, index=False)
    player_summary.to_csv(DEFAULT_PLAYER_SUMMARY, index=False)
    half_table.to_csv(DEFAULT_HALF_TABLE, index=False)
    reliability.to_csv(DEFAULT_RELIABILITY, index=False)
    profile.to_csv(DEFAULT_PROFILE, index=False)

    DEFAULT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    scored[prediction_columns].to_parquet(
        DEFAULT_PREDICTIONS,
        index=False,
    )

    _save_reliability_figure(
        half_table,
        DEFAULT_RELIABILITY_FIGURE,
    )
    _save_profile_figure(
        profile,
        DEFAULT_PROFILE_FIGURE,
    )

    print()
    print("NorthStar swing-execution component V1")
    print(f"2024 raw pitches:          {train_raw_count:,}")
    print(f"2025 raw pitches:          {external_raw_count:,}")
    print(f"2024 clean swing rows:     {len(train_swings):,}")
    print(f"2025 clean swing rows:     {len(external_swings):,}")
    print(
        "Execution target: actual one-step transition value minus "
        "2024-trained expected swing value."
    )

    print()
    print("External 2025 swing-value prediction")
    print(
        metric_table.round(
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
    print("Player swing-execution split-half reliability")
    print(
        reliability.round(
            {
                "pearson_r": 3,
                "spearman_rho": 3,
                "early_mean": 3,
                "late_mean": 3,
                "mean_absolute_half_difference": 3,
            }
        ).to_string(index=False)
    )

    if not profile.empty:
        selection_execution_pearson = profile[
            "selection_centered_per_100"
        ].corr(
            profile["centered_execution_per_100_swings"],
            method="pearson",
        )
        selection_execution_spearman = profile[
            "selection_centered_per_100"
        ].corr(
            profile["centered_execution_per_100_swings"],
            method="spearman",
        )

        print()
        print("Selection / execution relationship")
        print(
            f"Players: {len(profile):,} | "
            f"Pearson {selection_execution_pearson:.3f} | "
            f"Spearman {selection_execution_spearman:.3f}"
        )

    display_columns = [
        column
        for column in [
            "player_name",
            "batter",
            "swings",
            "execution_value_total",
            "execution_value_per_100_swings",
            "centered_execution_per_100_swings",
            "positive_execution_rate",
        ]
        if column in player_summary.columns
    ]
    print()
    print(f"Top {args.top} by centered execution value / 100 swings")
    print(
        player_summary.head(args.top)[display_columns]
        .round(
            {
                "execution_value_total": 3,
                "execution_value_per_100_swings": 3,
                "centered_execution_per_100_swings": 3,
                "positive_execution_rate": 3,
            }
        )
        .to_string(index=False)
    )

    print()
    print(f"Saved model metrics:       {DEFAULT_MODEL_METRICS}")
    print(f"Saved player execution:    {DEFAULT_PLAYER_SUMMARY}")
    print(f"Saved half splits:         {DEFAULT_HALF_TABLE}")
    print(f"Saved reliability:         {DEFAULT_RELIABILITY}")
    print(f"Saved two-axis profile:    {DEFAULT_PROFILE}")
    print(f"Saved reliability figure:  {DEFAULT_RELIABILITY_FIGURE}")
    print(f"Saved profile figure:      {DEFAULT_PROFILE_FIGURE}")
    print(f"Saved local predictions:   {DEFAULT_PREDICTIONS}")


if __name__ == "__main__":
    main()
