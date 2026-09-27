from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from northstar.validation.execution_constructs import (
    CONTACT_CONSTRUCT_FEATURE_SETS,
    DAMAGE_CONSTRUCT_FEATURE_SETS,
    build_contact_skill_summary,
    build_damage_skill_summary,
    merge_execution_construct_data,
    merge_half_execution_constructs,
)
from northstar.validation.player_metric import (
    add_incremental_oof_residuals,
    construct_correlation_summary,
    incremental_half_reliability,
    residual_outcome_correlations,
)


DEFAULT_RAW_DIR = Path("data/raw/statcast")
DEFAULT_CONTACT = Path(
    "reports/tables/player_contact_execution_2025.csv"
)
DEFAULT_DAMAGE = Path(
    "reports/tables/player_damage_execution_2025.csv"
)
DEFAULT_CONTACT_HALF = Path(
    "reports/tables/player_contact_execution_half_splits_2025.csv"
)
DEFAULT_DAMAGE_HALF = Path(
    "reports/tables/player_damage_execution_half_splits_2025.csv"
)
DEFAULT_OUTCOME = Path(
    "reports/tables/player_run_value_validation_2025.csv"
)

DEFAULT_CONTACT_DATA = Path(
    "reports/tables/contact_execution_construct_dataset_2025.csv"
)
DEFAULT_DAMAGE_DATA = Path(
    "reports/tables/damage_execution_construct_dataset_2025.csv"
)
DEFAULT_CONTACT_CORRELATIONS = Path(
    "reports/tables/contact_execution_construct_correlations_2025.csv"
)
DEFAULT_DAMAGE_CORRELATIONS = Path(
    "reports/tables/damage_execution_construct_correlations_2025.csv"
)
DEFAULT_CONTACT_RESIDUALS = Path(
    "reports/tables/contact_execution_incremental_residuals_2025.csv"
)
DEFAULT_DAMAGE_RESIDUALS = Path(
    "reports/tables/damage_execution_incremental_residuals_2025.csv"
)
DEFAULT_CONTACT_SUMMARY = Path(
    "reports/tables/contact_execution_incremental_summary_2025.csv"
)
DEFAULT_DAMAGE_SUMMARY = Path(
    "reports/tables/damage_execution_incremental_summary_2025.csv"
)
DEFAULT_CONTACT_HALF_DATASET = Path(
    "reports/tables/contact_execution_incremental_half_dataset_2025.csv"
)
DEFAULT_DAMAGE_HALF_DATASET = Path(
    "reports/tables/damage_execution_incremental_half_dataset_2025.csv"
)
DEFAULT_CONTACT_HALF_RESIDUALS = Path(
    "reports/tables/contact_execution_incremental_half_residuals_2025.csv"
)
DEFAULT_DAMAGE_HALF_RESIDUALS = Path(
    "reports/tables/damage_execution_incremental_half_residuals_2025.csv"
)
DEFAULT_CONTACT_RELIABILITY = Path(
    "reports/tables/contact_execution_incremental_reliability_2025.csv"
)
DEFAULT_DAMAGE_RELIABILITY = Path(
    "reports/tables/damage_execution_incremental_reliability_2025.csv"
)
DEFAULT_CONTACT_OUTCOMES = Path(
    "reports/tables/contact_execution_incremental_outcome_correlations_2025.csv"
)
DEFAULT_DAMAGE_OUTCOMES = Path(
    "reports/tables/damage_execution_incremental_outcome_correlations_2025.csv"
)
DEFAULT_CONTACT_FIGURE = Path(
    "reports/figures/contact_execution_incremental_reliability_2025.png"
)
DEFAULT_DAMAGE_FIGURE = Path(
    "reports/figures/damage_execution_incremental_reliability_2025.png"
)


CONTACT_TARGET = "contact_above_expected_per_100_swings"
DAMAGE_TARGET = "damage_xwoba_points_above_expected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether NorthStar contact and damage execution retain "
            "repeatable hitter information beyond conventional metrics."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--contact", type=Path, default=DEFAULT_CONTACT)
    parser.add_argument("--damage", type=Path, default=DEFAULT_DAMAGE)
    parser.add_argument(
        "--contact-half",
        type=Path,
        default=DEFAULT_CONTACT_HALF,
    )
    parser.add_argument(
        "--damage-half",
        type=Path,
        default=DEFAULT_DAMAGE_HALF,
    )
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    parser.add_argument("--split-date", default="2025-07-01")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def _load_raw(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("statcast_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No cached Statcast Parquet files found under {raw_dir}."
        )

    columns = [
        "game_date",
        "batter",
        "description",
        "zone",
        "strikes",
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "estimated_woba_using_speedangle",
    ]

    frames: list[pd.DataFrame] = []
    print(
        f"Reading {len(paths)} cached 2025 Statcast chunks "
        "for contact/damage construct validation..."
    )
    for index, path in enumerate(paths, start=1):
        frames.append(pd.read_parquet(path, columns=columns))
        if index == 1 or index % 5 == 0 or index == len(paths):
            print(f"  prepared chunk {index}/{len(paths)}")

    frame = pd.concat(frames, ignore_index=True, copy=False)
    frame["game_date"] = pd.to_datetime(
        frame["game_date"],
        errors="coerce",
    )
    return frame


def _build_half_constructs(
    raw: pd.DataFrame,
    split_date: str,
    builder,
) -> pd.DataFrame:
    boundary = pd.Timestamp(split_date)
    pieces: list[pd.DataFrame] = []

    for half, mask in [
        ("early", raw["game_date"] < boundary),
        ("late", raw["game_date"] >= boundary),
    ]:
        summary = builder(raw.loc[mask].copy())
        summary["season_half"] = half
        pieces.append(summary)

    return pd.concat(pieces, ignore_index=True)


def _save_reliability_figure(
    reliability: pd.DataFrame,
    title: str,
    output: Path,
) -> None:
    if reliability.empty:
        return

    current = reliability.sort_values("pearson_r")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(current["construct_set"], current["pearson_r"])
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Early vs late Pearson reliability")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _print_block(
    heading: str,
    frame: pd.DataFrame,
    rounding: dict[str, int],
) -> None:
    print()
    print(heading)
    if frame.empty:
        print("No rows available.")
        return
    print(frame.round(rounding).to_string(index=False))


def main() -> None:
    args = parse_args()

    if args.folds < 3:
        raise ValueError("--folds must be >= 3")

    required_paths = [
        args.contact,
        args.damage,
        args.contact_half,
        args.damage_half,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    raw = _load_raw(args.raw_dir)

    contact_constructs = build_contact_skill_summary(raw)
    damage_constructs = build_damage_skill_summary(raw)
    contact_half_constructs = _build_half_constructs(
        raw,
        args.split_date,
        build_contact_skill_summary,
    )
    damage_half_constructs = _build_half_constructs(
        raw,
        args.split_date,
        build_damage_skill_summary,
    )

    contact = pd.read_csv(args.contact)
    damage = pd.read_csv(args.damage)
    contact_half = pd.read_csv(args.contact_half)
    damage_half = pd.read_csv(args.damage_half)

    outcome = (
        pd.read_csv(args.outcome)
        if args.outcome.exists()
        else pd.DataFrame()
    )

    contact_data = merge_execution_construct_data(
        contact,
        contact_constructs,
        outcome_reference=outcome,
    )
    damage_data = merge_execution_construct_data(
        damage,
        damage_constructs,
        outcome_reference=outcome,
    )

    contact_half_data = merge_half_execution_constructs(
        contact_half,
        contact_half_constructs,
    )
    damage_half_data = merge_half_execution_constructs(
        damage_half,
        damage_half_constructs,
    )

    contact_reference_metrics = [
        "statcast_contact_rate",
        "statcast_whiff_per_swing",
        "statcast_zone_contact_rate",
        "statcast_chase_contact_rate",
        "statcast_two_strike_contact_rate",
        "re24_per_100_pa",
        "woba",
        "est_woba",
    ]
    damage_reference_metrics = [
        "statcast_xwoba_on_contact",
        "statcast_avg_exit_velocity",
        "statcast_hard_hit_rate",
        "statcast_barrel_rate",
        "statcast_sweet_spot_rate",
        "re24_per_100_pa",
        "woba",
        "est_woba",
    ]

    contact_correlations = construct_correlation_summary(
        contact_data,
        targets=[CONTACT_TARGET],
        references=contact_reference_metrics,
    )
    damage_correlations = construct_correlation_summary(
        damage_data,
        targets=[DAMAGE_TARGET],
        references=damage_reference_metrics,
    )

    contact_residuals, contact_summary = add_incremental_oof_residuals(
        contact_data,
        target=CONTACT_TARGET,
        feature_sets=CONTACT_CONSTRUCT_FEATURE_SETS,
        folds=args.folds,
        seed=61,
    )
    damage_residuals, damage_summary = add_incremental_oof_residuals(
        damage_data,
        target=DAMAGE_TARGET,
        feature_sets=DAMAGE_CONSTRUCT_FEATURE_SETS,
        folds=args.folds,
        seed=71,
    )

    (
        contact_half_residuals,
        contact_reliability,
    ) = incremental_half_reliability(
        contact_half_data,
        target=CONTACT_TARGET,
        feature_sets=CONTACT_CONSTRUCT_FEATURE_SETS,
        folds=args.folds,
        seed=61,
    )
    (
        damage_half_residuals,
        damage_reliability,
    ) = incremental_half_reliability(
        damage_half_data,
        target=DAMAGE_TARGET,
        feature_sets=DAMAGE_CONSTRUCT_FEATURE_SETS,
        folds=args.folds,
        seed=71,
    )

    contact_outcomes = residual_outcome_correlations(
        contact_residuals,
        contact_data,
    )
    damage_outcomes = residual_outcome_correlations(
        damage_residuals,
        damage_data,
    )

    DEFAULT_CONTACT_DATA.parent.mkdir(parents=True, exist_ok=True)
    contact_data.to_csv(DEFAULT_CONTACT_DATA, index=False)
    damage_data.to_csv(DEFAULT_DAMAGE_DATA, index=False)
    contact_correlations.to_csv(
        DEFAULT_CONTACT_CORRELATIONS,
        index=False,
    )
    damage_correlations.to_csv(
        DEFAULT_DAMAGE_CORRELATIONS,
        index=False,
    )
    contact_residuals.to_csv(DEFAULT_CONTACT_RESIDUALS, index=False)
    damage_residuals.to_csv(DEFAULT_DAMAGE_RESIDUALS, index=False)
    contact_summary.to_csv(DEFAULT_CONTACT_SUMMARY, index=False)
    damage_summary.to_csv(DEFAULT_DAMAGE_SUMMARY, index=False)
    contact_half_data.to_csv(
        DEFAULT_CONTACT_HALF_DATASET,
        index=False,
    )
    damage_half_data.to_csv(
        DEFAULT_DAMAGE_HALF_DATASET,
        index=False,
    )
    contact_half_residuals.to_csv(
        DEFAULT_CONTACT_HALF_RESIDUALS,
        index=False,
    )
    damage_half_residuals.to_csv(
        DEFAULT_DAMAGE_HALF_RESIDUALS,
        index=False,
    )
    contact_reliability.to_csv(
        DEFAULT_CONTACT_RELIABILITY,
        index=False,
    )
    damage_reliability.to_csv(
        DEFAULT_DAMAGE_RELIABILITY,
        index=False,
    )
    contact_outcomes.to_csv(DEFAULT_CONTACT_OUTCOMES, index=False)
    damage_outcomes.to_csv(DEFAULT_DAMAGE_OUTCOMES, index=False)

    _save_reliability_figure(
        contact_reliability,
        "Incremental contact-execution reliability",
        DEFAULT_CONTACT_FIGURE,
    )
    _save_reliability_figure(
        damage_reliability,
        "Incremental damage-execution reliability",
        DEFAULT_DAMAGE_FIGURE,
    )

    print()
    print("NorthStar contact / damage construct validation V1")
    print(f"Contact players: {len(contact_data):,}")
    print(f"Damage players:  {len(damage_data):,}")
    print(
        "Question: after conventional contact or quality-of-contact "
        "metrics explain what they can, is the NorthStar residual "
        "still repeatable?"
    )

    _print_block(
        "Contact construct correlations",
        contact_correlations,
        {"pearson_r": 3, "spearman_rho": 3},
    )
    _print_block(
        "Contact full-season OOF reconstruction",
        contact_summary,
        {
            "oof_r2": 3,
            "residual_std": 3,
            "residual_mean_abs": 3,
        },
    )
    _print_block(
        "Contact incremental early/late reliability",
        contact_reliability,
        {
            "pearson_r": 3,
            "spearman_rho": 3,
            "early_residual_std": 3,
            "late_residual_std": 3,
            "mean_absolute_half_difference": 3,
        },
    )
    _print_block(
        "Contact residual association with offensive outcomes",
        contact_outcomes,
        {"pearson_r": 3, "spearman_rho": 3},
    )

    _print_block(
        "Damage construct correlations",
        damage_correlations,
        {"pearson_r": 3, "spearman_rho": 3},
    )
    _print_block(
        "Damage full-season OOF reconstruction",
        damage_summary,
        {
            "oof_r2": 3,
            "residual_std": 3,
            "residual_mean_abs": 3,
        },
    )
    _print_block(
        "Damage incremental early/late reliability",
        damage_reliability,
        {
            "pearson_r": 3,
            "spearman_rho": 3,
            "early_residual_std": 3,
            "late_residual_std": 3,
            "mean_absolute_half_difference": 3,
        },
    )
    _print_block(
        "Damage residual association with offensive outcomes",
        damage_outcomes,
        {"pearson_r": 3, "spearman_rho": 3},
    )

    print()
    print(f"Saved contact dataset:       {DEFAULT_CONTACT_DATA}")
    print(f"Saved damage dataset:        {DEFAULT_DAMAGE_DATA}")
    print(f"Saved contact correlations:  {DEFAULT_CONTACT_CORRELATIONS}")
    print(f"Saved damage correlations:   {DEFAULT_DAMAGE_CORRELATIONS}")
    print(f"Saved contact summary:       {DEFAULT_CONTACT_SUMMARY}")
    print(f"Saved damage summary:        {DEFAULT_DAMAGE_SUMMARY}")
    print(f"Saved contact reliability:   {DEFAULT_CONTACT_RELIABILITY}")
    print(f"Saved damage reliability:    {DEFAULT_DAMAGE_RELIABILITY}")
    print(f"Saved contact outcomes:      {DEFAULT_CONTACT_OUTCOMES}")
    print(f"Saved damage outcomes:       {DEFAULT_DAMAGE_OUTCOMES}")
    print(f"Saved contact figure:        {DEFAULT_CONTACT_FIGURE}")
    print(f"Saved damage figure:         {DEFAULT_DAMAGE_FIGURE}")


if __name__ == "__main__":
    main()
