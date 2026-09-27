from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from northstar.validation.player_metric import (
    DEFAULT_INCREMENTAL_FEATURE_SETS,
    add_incremental_oof_residuals,
    build_plate_discipline_summary,
    incremental_half_reliability,
    residual_outcome_correlations,
)


DEFAULT_CONSTRUCT = Path(
    "reports/tables/player_metric_construct_dataset_2025.csv"
)
DEFAULT_HALF = Path(
    "reports/tables/player_decision_metric_half_splits_2025.csv"
)
DEFAULT_RAW_DIR = Path("data/raw/statcast")

DEFAULT_FULL_RESIDUALS = Path(
    "reports/tables/player_metric_incremental_residuals_2025.csv"
)
DEFAULT_FULL_SUMMARY = Path(
    "reports/tables/player_metric_incremental_summary_2025.csv"
)
DEFAULT_HALF_DATASET = Path(
    "reports/tables/player_metric_incremental_half_dataset_2025.csv"
)
DEFAULT_HALF_RESIDUALS = Path(
    "reports/tables/player_metric_incremental_half_residuals_2025.csv"
)
DEFAULT_RELIABILITY = Path(
    "reports/tables/player_metric_incremental_reliability_2025.csv"
)
DEFAULT_OUTCOMES = Path(
    "reports/tables/player_metric_incremental_outcome_correlations_2025.csv"
)
DEFAULT_FIGURE = Path(
    "reports/figures/player_metric_incremental_reliability_2025.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether NorthStar contains repeatable hitter information "
            "beyond simple Statcast plate-discipline summaries."
        )
    )
    parser.add_argument("--construct", type=Path, default=DEFAULT_CONSTRUCT)
    parser.add_argument("--half-table", type=Path, default=DEFAULT_HALF)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--split-date", default="2025-07-01")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def _load_half_discipline(
    raw_dir: Path,
    split_date: str,
) -> pd.DataFrame:
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
        "balls",
        "strikes",
    ]
    frames: list[pd.DataFrame] = []

    print(
        f"Reading {len(paths)} cached 2025 Statcast chunks "
        "for early/late discipline summaries..."
    )
    for index, path in enumerate(paths, start=1):
        frames.append(pd.read_parquet(path, columns=columns))
        if index == 1 or index % 5 == 0 or index == len(paths):
            print(f"  prepared chunk {index}/{len(paths)}")

    pitches = pd.concat(frames, ignore_index=True, copy=False)
    pitches["game_date"] = pd.to_datetime(
        pitches["game_date"],
        errors="coerce",
    )
    boundary = pd.Timestamp(split_date)

    rows: list[pd.DataFrame] = []
    for half, mask in [
        ("early", pitches["game_date"] < boundary),
        ("late", pitches["game_date"] >= boundary),
    ]:
        summary = build_plate_discipline_summary(
            pitches.loc[mask].copy()
        )
        summary = summary.rename(
            columns={
                column: f"statcast_{column}"
                for column in summary.columns
                if column != "batter"
            }
        )
        summary["season_half"] = half
        rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def _save_reliability_figure(
    reliability: pd.DataFrame,
    output: Path,
) -> None:
    if reliability.empty:
        return

    current = reliability.sort_values("pearson_r")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(current["construct_set"], current["pearson_r"])
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Early vs late Pearson reliability")
    ax.set_title(
        "Incremental NorthStar player signal after plate-discipline removal"
    )
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.folds < 3:
        raise ValueError("--folds must be >= 3")
    if not args.construct.exists():
        raise FileNotFoundError(
            f"Missing construct dataset: {args.construct}. "
            "Run validate_player_decision_metric.py first."
        )
    if not args.half_table.exists():
        raise FileNotFoundError(
            f"Missing half-split table: {args.half_table}."
        )

    construct = pd.read_csv(args.construct)
    half_metric = pd.read_csv(args.half_table)
    half_discipline = _load_half_discipline(
        args.raw_dir,
        args.split_date,
    )

    half_dataset = half_metric.merge(
        half_discipline,
        on=["batter", "season_half"],
        how="left",
        validate="one_to_one",
    )

    full_residuals, full_summary = add_incremental_oof_residuals(
        construct,
        target="decision_value_per_100",
        feature_sets=DEFAULT_INCREMENTAL_FEATURE_SETS,
        folds=args.folds,
    )

    half_residuals, reliability = incremental_half_reliability(
        half_dataset,
        target="decision_value_per_100",
        feature_sets=DEFAULT_INCREMENTAL_FEATURE_SETS,
        folds=args.folds,
    )

    outcome_correlations = residual_outcome_correlations(
        full_residuals,
        construct,
    )

    DEFAULT_FULL_RESIDUALS.parent.mkdir(parents=True, exist_ok=True)
    full_residuals.to_csv(DEFAULT_FULL_RESIDUALS, index=False)
    full_summary.to_csv(DEFAULT_FULL_SUMMARY, index=False)
    half_dataset.to_csv(DEFAULT_HALF_DATASET, index=False)
    half_residuals.to_csv(DEFAULT_HALF_RESIDUALS, index=False)
    reliability.to_csv(DEFAULT_RELIABILITY, index=False)
    outcome_correlations.to_csv(DEFAULT_OUTCOMES, index=False)
    _save_reliability_figure(reliability, DEFAULT_FIGURE)

    print()
    print("NorthStar incremental player-signal audit V1")
    print(
        "Question: after removing what simple plate-discipline summaries "
        "already explain, is the remaining player signal repeatable?"
    )

    print()
    print("Full-season out-of-fold residualization")
    print(
        full_summary.round(
            {
                "oof_r2": 3,
                "residual_std": 3,
                "residual_mean_abs": 3,
            }
        ).to_string(index=False)
    )

    print()
    print("Early/late residual reliability")
    print(
        reliability.round(
            {
                "pearson_r": 3,
                "spearman_rho": 3,
                "early_residual_std": 3,
                "late_residual_std": 3,
                "mean_absolute_half_difference": 3,
            }
        ).to_string(index=False)
    )

    if not outcome_correlations.empty:
        print()
        print("Incremental residual association with offensive outcomes")
        print(
            outcome_correlations.round(
                {
                    "pearson_r": 3,
                    "spearman_rho": 3,
                }
            ).to_string(index=False)
        )

    print()
    print(f"Saved full residuals:    {DEFAULT_FULL_RESIDUALS}")
    print(f"Saved full summary:      {DEFAULT_FULL_SUMMARY}")
    print(f"Saved half dataset:      {DEFAULT_HALF_DATASET}")
    print(f"Saved half residuals:    {DEFAULT_HALF_RESIDUALS}")
    print(f"Saved reliability:       {DEFAULT_RELIABILITY}")
    print(f"Saved outcome checks:    {DEFAULT_OUTCOMES}")
    print(f"Saved figure:            {DEFAULT_FIGURE}")


if __name__ == "__main__":
    main()
