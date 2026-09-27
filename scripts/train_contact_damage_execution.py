from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from northstar.features.pitches import (
    PITCH_DECISION_REQUIRED_COLUMNS,
    build_pitch_decision_frame,
)
from northstar.metrics.execution_v2 import (
    CONTACT_PROBABILITY,
    CONTACT_RESIDUAL,
    CONTACT_TARGET,
    DAMAGE_PREDICTION,
    DAMAGE_RESIDUAL,
    DAMAGE_TARGET,
    attach_contact_residual,
    attach_damage_residual,
    build_contact_frame,
    build_contact_half_reliability,
    build_damage_frame,
    build_damage_half_reliability,
    build_player_contact_summary,
    build_player_damage_summary,
    build_selection_contact_damage_profile,
    fit_context_mean_baseline,
    predict_context_mean_baseline,
)
from northstar.models.baselines import (
    evaluate_probabilities,
    evaluate_regression_predictions,
    make_pitch_value_xgboost_model,
    make_pitch_xgboost_model,
)
from northstar.models.pitch_decision import (
    PITCH_DECISION_CATEGORICAL_FEATURES,
    PITCH_DECISION_NUMERIC_FEATURES,
    prepare_pitch_decision_model_frame,
)
from northstar.validation.crossfit import (
    apply_propensity_calibrator,
    crossfit_pitch_propensity,
    fit_isotonic_calibrator,
)


DEFAULT_TRAIN_RAW = Path("data/raw/statcast_2024")
DEFAULT_EXTERNAL_RAW = Path("data/raw/statcast")
DEFAULT_SELECTION = Path(
    "reports/tables/player_decision_metric_prototype_2025.csv"
)

DEFAULT_CONTACT_MODEL_METRICS = Path(
    "reports/tables/contact_execution_model_metrics_2024_2025.csv"
)
DEFAULT_DAMAGE_MODEL_METRICS = Path(
    "reports/tables/damage_execution_model_metrics_2024_2025.csv"
)
DEFAULT_CONTACT_PLAYERS = Path(
    "reports/tables/player_contact_execution_2025.csv"
)
DEFAULT_DAMAGE_PLAYERS = Path(
    "reports/tables/player_damage_execution_2025.csv"
)
DEFAULT_CONTACT_HALF = Path(
    "reports/tables/player_contact_execution_half_splits_2025.csv"
)
DEFAULT_DAMAGE_HALF = Path(
    "reports/tables/player_damage_execution_half_splits_2025.csv"
)
DEFAULT_CONTACT_RELIABILITY = Path(
    "reports/tables/player_contact_execution_reliability_2025.csv"
)
DEFAULT_DAMAGE_RELIABILITY = Path(
    "reports/tables/player_damage_execution_reliability_2025.csv"
)
DEFAULT_PROFILE = Path(
    "reports/tables/selection_contact_damage_profile_2025.csv"
)
DEFAULT_CONTACT_FIGURE = Path(
    "reports/figures/player_contact_execution_reliability_2025.png"
)
DEFAULT_DAMAGE_FIGURE = Path(
    "reports/figures/player_damage_execution_reliability_2025.png"
)
DEFAULT_PROFILE_FIGURE = Path(
    "reports/figures/selection_contact_damage_profile_2025.png"
)
DEFAULT_CONTACT_PREDICTIONS = Path(
    "data/processed/contact_execution_test_predictions_2024_2025.parquet"
)
DEFAULT_DAMAGE_PREDICTIONS = Path(
    "data/processed/damage_execution_test_predictions_2024_2025.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose NorthStar swing execution into contact execution and "
            "damage execution using 2024-trained models transported to 2025."
        )
    )
    parser.add_argument("--train-raw-dir", type=Path, default=DEFAULT_TRAIN_RAW)
    parser.add_argument(
        "--external-raw-dir",
        type=Path,
        default=DEFAULT_EXTERNAL_RAW,
    )
    parser.add_argument(
        "--selection-table",
        type=Path,
        default=DEFAULT_SELECTION,
    )
    parser.add_argument("--contact-folds", type=int, default=3)
    parser.add_argument("--min-swings", type=int, default=300)
    parser.add_argument("--min-half-swings", type=int, default=150)
    parser.add_argument("--min-batted-balls", type=int, default=100)
    parser.add_argument("--min-half-batted-balls", type=int, default=50)
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

    projected = sorted(
        PITCH_DECISION_REQUIRED_COLUMNS | {DAMAGE_TARGET}
    )
    frames: list[pd.DataFrame] = []
    raw_count = 0

    print(
        f"Reading {label}: {len(paths)} cached chunks "
        f"with {len(projected)} projected columns..."
    )
    for index, path in enumerate(paths, start=1):
        try:
            chunk = pd.read_parquet(path, columns=projected)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read required contact/damage columns from {path}. "
                f"The cached Statcast schema must include {DAMAGE_TARGET}."
            ) from exc

        raw_count += len(chunk)
        decision_frame = build_pitch_decision_frame(chunk)
        frames.append(
            prepare_pitch_decision_model_frame(decision_frame)
        )

        if index == 1 or index % 5 == 0 or index == len(paths):
            print(f"  prepared chunk {index}/{len(paths)}")

    return pd.concat(frames, ignore_index=True, copy=False), raw_count


def _classification_row(model: str, metrics) -> dict[str, object]:
    return {
        "train_season": 2024,
        "external_season": 2025,
        "model": model,
        "roc_auc": metrics.roc_auc,
        "log_loss": metrics.log_loss,
        "brier": metrics.brier,
    }


def _regression_row(model: str, metrics) -> dict[str, object]:
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


def _save_half_scatter(
    half_table: pd.DataFrame,
    metric: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    pivot = half_table.pivot(
        index="batter",
        columns="season_half",
        values=metric,
    ).dropna()
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pivot["early"], pivot["late"], alpha=0.65)

    lower = min(pivot["early"].min(), pivot["late"].min())
    upper = max(pivot["early"].max(), pivot["late"].max())
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
        "contact_above_expected_per_100_swings",
        "damage_xwoba_points_above_expected",
    }
    if profile.empty or not required.issubset(profile.columns):
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        profile["contact_above_expected_per_100_swings"],
        profile["damage_xwoba_points_above_expected"],
        alpha=0.65,
    )
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Contact above expected per 100 swings")
    ax.set_ylabel("Damage above expected (xwOBA points)")
    ax.set_title("NorthStar contact / damage execution profile — 2025")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _add_names(
    frame: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty or "player_name" not in selection.columns:
        return frame

    names = selection[
        ["batter", "player_name"]
    ].drop_duplicates("batter")
    out = frame.merge(
        names,
        on="batter",
        how="left",
        validate="one_to_one",
    )
    columns = ["batter", "player_name"] + [
        column
        for column in out.columns
        if column not in {"batter", "player_name"}
    ]
    return out[columns]


def main() -> None:
    args = parse_args()

    for name, value in [
        ("--contact-folds", args.contact_folds),
        ("--min-swings", args.min_swings),
        ("--min-half-swings", args.min_half_swings),
        ("--min-batted-balls", args.min_batted_balls),
        ("--min-half-batted-balls", args.min_half_batted_balls),
        ("--top", args.top),
    ]:
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    if args.contact_folds < 2:
        raise ValueError("--contact-folds must be >= 2")

    train_decisions, train_raw_count = _load_decision_chunks(
        args.train_raw_dir,
        "2024 training season",
    )
    external_decisions, external_raw_count = _load_decision_chunks(
        args.external_raw_dir,
        "2025 external season",
    )

    train_contact = build_contact_frame(train_decisions)
    external_contact = build_contact_frame(external_decisions)
    train_damage = build_damage_frame(train_decisions)
    external_damage = build_damage_frame(external_decisions)

    features = (
        PITCH_DECISION_CATEGORICAL_FEATURES
        + PITCH_DECISION_NUMERIC_FEATURES
    )

    print()
    print(
        f"Cross-fitting 2024 contact model across "
        f"{args.contact_folds} game-grouped folds..."
    )
    contact_crossfit = crossfit_pitch_propensity(
        train_contact,
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
        target=CONTACT_TARGET,
        group_column="game_pk",
        folds=args.contact_folds,
        seed=52,
        progress=True,
    )
    contact_calibrator = fit_isotonic_calibrator(
        contact_crossfit.oof_raw,
        train_contact[CONTACT_TARGET],
    )

    contact_baseline_map, contact_fallback, contact_overall = (
        fit_context_mean_baseline(
            train_contact,
            CONTACT_TARGET,
        )
    )
    contact_baseline_probability = predict_context_mean_baseline(
        external_contact,
        contact_baseline_map,
        contact_fallback,
        contact_overall,
    )
    contact_baseline_metrics = evaluate_probabilities(
        external_contact[CONTACT_TARGET],
        contact_baseline_probability,
    )

    print(
        "Fitting final 2024 contact XGBoost model "
        "(no batter identity, no outcome features)..."
    )
    contact_model = make_pitch_xgboost_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )
    contact_model.fit(
        train_contact[features],
        train_contact[CONTACT_TARGET],
    )
    contact_probability_raw = contact_model.predict_proba(
        external_contact[features]
    )[:, 1]
    contact_probability = apply_propensity_calibrator(
        contact_calibrator,
        contact_probability_raw,
    )
    contact_metrics = evaluate_probabilities(
        external_contact[CONTACT_TARGET],
        contact_probability,
    )

    contact_metric_table = pd.DataFrame(
        [
            _classification_row(
                "count_zone_pitch_type",
                contact_baseline_metrics,
            ),
            _classification_row(
                "xgboost_contact_calibrated",
                contact_metrics,
            ),
        ]
    )

    damage_baseline_map, damage_fallback, damage_overall = (
        fit_context_mean_baseline(
            train_damage,
            DAMAGE_TARGET,
        )
    )
    damage_baseline_prediction = predict_context_mean_baseline(
        external_damage,
        damage_baseline_map,
        damage_fallback,
        damage_overall,
    )
    damage_baseline_metrics = evaluate_regression_predictions(
        external_damage[DAMAGE_TARGET],
        damage_baseline_prediction,
    )

    print()
    print(
        "Fitting 2024 damage XGBoost model "
        "(xwOBA contact quality is target only)..."
    )
    damage_model = make_pitch_value_xgboost_model(
        PITCH_DECISION_CATEGORICAL_FEATURES,
        PITCH_DECISION_NUMERIC_FEATURES,
    )
    damage_model.fit(
        train_damage[features],
        train_damage[DAMAGE_TARGET],
    )
    damage_prediction = damage_model.predict(
        external_damage[features]
    )
    damage_metrics = evaluate_regression_predictions(
        external_damage[DAMAGE_TARGET],
        damage_prediction,
    )

    damage_metric_table = pd.DataFrame(
        [
            _regression_row(
                "count_zone_pitch_type",
                damage_baseline_metrics,
            ),
            _regression_row(
                "xgboost_damage",
                damage_metrics,
            ),
        ]
    )

    scored_contact = attach_contact_residual(
        external_contact,
        contact_probability,
    )
    scored_damage = attach_damage_residual(
        external_damage,
        damage_prediction,
    )

    contact_players = build_player_contact_summary(
        scored_contact,
        min_swings=args.min_swings,
    )
    damage_players = build_player_damage_summary(
        scored_damage,
        min_batted_balls=args.min_batted_balls,
    )
    contact_half, contact_reliability = build_contact_half_reliability(
        scored_contact,
        split_date=args.split_date,
        min_half_swings=args.min_half_swings,
    )
    damage_half, damage_reliability = build_damage_half_reliability(
        scored_damage,
        split_date=args.split_date,
        min_half_batted_balls=args.min_half_batted_balls,
    )

    selection = pd.DataFrame()
    profile = pd.DataFrame()
    if args.selection_table.exists():
        selection = pd.read_csv(args.selection_table)
        contact_players = _add_names(contact_players, selection)
        damage_players = _add_names(damage_players, selection)

        profile = build_selection_contact_damage_profile(
            selection,
            contact_players,
            damage_players,
        )
    else:
        print(
            f"Warning: {args.selection_table} was not found; "
            "selection/contact/damage profile will be skipped."
        )

    DEFAULT_CONTACT_MODEL_METRICS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    contact_metric_table.to_csv(
        DEFAULT_CONTACT_MODEL_METRICS,
        index=False,
    )
    damage_metric_table.to_csv(
        DEFAULT_DAMAGE_MODEL_METRICS,
        index=False,
    )
    contact_players.to_csv(DEFAULT_CONTACT_PLAYERS, index=False)
    damage_players.to_csv(DEFAULT_DAMAGE_PLAYERS, index=False)
    contact_half.to_csv(DEFAULT_CONTACT_HALF, index=False)
    damage_half.to_csv(DEFAULT_DAMAGE_HALF, index=False)
    contact_reliability.to_csv(
        DEFAULT_CONTACT_RELIABILITY,
        index=False,
    )
    damage_reliability.to_csv(
        DEFAULT_DAMAGE_RELIABILITY,
        index=False,
    )
    profile.to_csv(DEFAULT_PROFILE, index=False)

    DEFAULT_CONTACT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    contact_columns = [
        column
        for column in [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "batter",
            "pitcher",
            "balls",
            "strikes",
            "pitch_type",
            "zone",
            "description",
            CONTACT_TARGET,
            CONTACT_PROBABILITY,
            CONTACT_RESIDUAL,
        ]
        if column in scored_contact.columns
    ]
    damage_columns = [
        column
        for column in [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "batter",
            "pitcher",
            "balls",
            "strikes",
            "pitch_type",
            "zone",
            "description",
            DAMAGE_TARGET,
            DAMAGE_PREDICTION,
            DAMAGE_RESIDUAL,
        ]
        if column in scored_damage.columns
    ]
    scored_contact[contact_columns].to_parquet(
        DEFAULT_CONTACT_PREDICTIONS,
        index=False,
    )
    scored_damage[damage_columns].to_parquet(
        DEFAULT_DAMAGE_PREDICTIONS,
        index=False,
    )

    _save_half_scatter(
        contact_half,
        metric="contact_above_expected_per_100_swings",
        xlabel="Early contact above expected / 100 swings",
        ylabel="Late contact above expected / 100 swings",
        title="2025 contact-execution split-half reliability",
        output=DEFAULT_CONTACT_FIGURE,
    )
    _save_half_scatter(
        damage_half,
        metric="damage_xwoba_points_above_expected",
        xlabel="Early damage above expected (xwOBA points)",
        ylabel="Late damage above expected (xwOBA points)",
        title="2025 damage-execution split-half reliability",
        output=DEFAULT_DAMAGE_FIGURE,
    )
    _save_profile_figure(profile, DEFAULT_PROFILE_FIGURE)

    print()
    print("NorthStar contact / damage execution V2")
    print(f"2024 raw pitches:             {train_raw_count:,}")
    print(f"2025 raw pitches:             {external_raw_count:,}")
    print(f"2024 observed swings:         {len(train_contact):,}")
    print(f"2025 observed swings:         {len(external_contact):,}")
    print(f"2024 damage-target BIP rows:  {len(train_damage):,}")
    print(f"2025 damage-target BIP rows:  {len(external_damage):,}")
    print(
        "Contact target: observed contact vs calibrated 2024 league "
        "expectation on the swing."
    )
    print(
        "Damage target: Statcast estimated wOBA using launch speed/angle "
        "vs 2024 league expectation on the batted ball."
    )

    print()
    print("External 2025 contact prediction")
    print(
        contact_metric_table.round(
            {
                "roc_auc": 3,
                "log_loss": 4,
                "brier": 4,
            }
        ).to_string(index=False)
    )

    print()
    print("External 2025 damage prediction")
    print(
        damage_metric_table.round(
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
    print("Player contact-execution split-half reliability")
    print(
        contact_reliability.round(
            {
                "pearson_r": 3,
                "spearman_rho": 3,
                "early_mean": 3,
                "late_mean": 3,
                "mean_absolute_half_difference": 3,
            }
        ).to_string(index=False)
    )

    print()
    print("Player damage-execution split-half reliability")
    print(
        damage_reliability.round(
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
        pairs = [
            (
                "selection vs contact",
                "selection_centered_per_100",
                "contact_above_expected_per_100_swings",
            ),
            (
                "selection vs damage",
                "selection_centered_per_100",
                "damage_xwoba_points_above_expected",
            ),
            (
                "contact vs damage",
                "contact_above_expected_per_100_swings",
                "damage_xwoba_points_above_expected",
            ),
        ]
        print()
        print("Selection / contact / damage relationships")
        for label, left, right in pairs:
            if {left, right}.issubset(profile.columns):
                print(
                    f"{label}: "
                    f"Pearson {profile[left].corr(profile[right], method='pearson'):.3f} | "
                    f"Spearman {profile[left].corr(profile[right], method='spearman'):.3f}"
                )

    print()
    contact_columns_display = [
        column
        for column in [
            "player_name",
            "batter",
            "swings",
            "observed_contact_rate",
            "expected_contact_rate",
            "contact_above_expected_per_100_swings",
        ]
        if column in contact_players.columns
    ]
    print(f"Top {args.top} contact execution")
    print(
        contact_players.head(args.top)[contact_columns_display]
        .round(
            {
                "observed_contact_rate": 3,
                "expected_contact_rate": 3,
                "contact_above_expected_per_100_swings": 3,
            }
        )
        .to_string(index=False)
    )

    print()
    damage_columns_display = [
        column
        for column in [
            "player_name",
            "batter",
            "batted_balls",
            "observed_damage_xwoba",
            "expected_damage_xwoba",
            "damage_xwoba_points_above_expected",
        ]
        if column in damage_players.columns
    ]
    print(f"Top {args.top} damage execution")
    print(
        damage_players.head(args.top)[damage_columns_display]
        .round(
            {
                "observed_damage_xwoba": 3,
                "expected_damage_xwoba": 3,
                "damage_xwoba_points_above_expected": 1,
            }
        )
        .to_string(index=False)
    )

    print()
    print(f"Saved contact metrics:     {DEFAULT_CONTACT_MODEL_METRICS}")
    print(f"Saved damage metrics:      {DEFAULT_DAMAGE_MODEL_METRICS}")
    print(f"Saved contact players:     {DEFAULT_CONTACT_PLAYERS}")
    print(f"Saved damage players:      {DEFAULT_DAMAGE_PLAYERS}")
    print(f"Saved contact reliability: {DEFAULT_CONTACT_RELIABILITY}")
    print(f"Saved damage reliability:  {DEFAULT_DAMAGE_RELIABILITY}")
    print(f"Saved three-axis profile:  {DEFAULT_PROFILE}")
    print(f"Saved contact figure:      {DEFAULT_CONTACT_FIGURE}")
    print(f"Saved damage figure:       {DEFAULT_DAMAGE_FIGURE}")
    print(f"Saved profile figure:      {DEFAULT_PROFILE_FIGURE}")
    print(f"Saved local contact preds: {DEFAULT_CONTACT_PREDICTIONS}")
    print(f"Saved local damage preds:  {DEFAULT_DAMAGE_PREDICTIONS}")


if __name__ == "__main__":
    main()
