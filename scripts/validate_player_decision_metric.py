from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from northstar.metrics.player_decision import add_player_decision_values
from northstar.validation.player_metric import (
    build_plate_discipline_summary,
    construct_correlation_summary,
    construct_cv_summary,
    empirical_bayes_player_shrinkage,
    merge_player_construct_data,
)


DEFAULT_RAW_DIR = Path("data/raw/statcast")
DEFAULT_PLAYER_METRIC = Path(
    "reports/tables/player_decision_metric_prototype_2025.csv"
)
DEFAULT_PREDICTIONS = Path(
    "data/processed/external_season_test_predictions_2024_2025.parquet"
)
DEFAULT_OUTCOME_REFERENCE = Path(
    "reports/tables/player_run_value_validation_2025.csv"
)

DEFAULT_CONSTRUCT_DATA = Path(
    "reports/tables/player_metric_construct_dataset_2025.csv"
)
DEFAULT_CORRELATIONS = Path(
    "reports/tables/player_metric_construct_correlations_2025.csv"
)
DEFAULT_CV = Path(
    "reports/tables/player_metric_construct_cv_2025.csv"
)
DEFAULT_SHRUNK = Path(
    "reports/tables/player_metric_shrunk_2025.csv"
)
DEFAULT_CORRELATION_FIGURE = Path(
    "reports/figures/player_metric_construct_correlations_2025.png"
)
DEFAULT_SHRINKAGE_FIGURE = Path(
    "reports/figures/player_metric_shrunk_2025_top20.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the NorthStar player decision-value prototype against "
            "independent Statcast plate-discipline and offensive constructs, "
            "then build a sample-aware shrunken ranking."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--player-metric",
        type=Path,
        default=DEFAULT_PLAYER_METRIC,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
    )
    parser.add_argument(
        "--outcome-reference",
        type=Path,
        default=DEFAULT_OUTCOME_REFERENCE,
    )
    parser.add_argument("--min-decisions", type=int, default=500)
    parser.add_argument("--min-discretionary", type=int, default=300)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def _load_statcast_discipline(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("statcast_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No cached Statcast Parquet files found under {raw_dir}."
        )

    columns = ["batter", "description", "zone", "balls", "strikes"]
    frames: list[pd.DataFrame] = []

    print(
        f"Reading {len(paths)} cached 2025 Statcast chunks "
        "for plate-discipline constructs..."
    )
    for index, path in enumerate(paths, start=1):
        frames.append(pd.read_parquet(path, columns=columns))
        if index == 1 or index % 5 == 0 or index == len(paths):
            print(f"  prepared chunk {index}/{len(paths)}")

    pitches = pd.concat(frames, ignore_index=True, copy=False)
    return build_plate_discipline_summary(pitches)


def _save_correlation_figure(
    correlations: pd.DataFrame,
    output: Path,
) -> None:
    current = correlations.loc[
        correlations["northstar_metric"].eq("decision_value_per_100")
    ].copy()
    if current.empty:
        return

    current["abs_spearman"] = current["spearman_rho"].abs()
    current = (
        current.sort_values("abs_spearman", ascending=False)
        .head(12)
        .sort_values("spearman_rho")
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(current["reference_metric"], current["spearman_rho"])
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Spearman correlation with decision value / 100")
    ax.set_title("NorthStar player metric construct associations — 2025")
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_shrinkage_figure(
    shrunk: pd.DataFrame,
    output: Path,
    top_n: int,
) -> None:
    if shrunk.empty:
        return

    current = shrunk.head(top_n).copy().iloc[::-1]
    label = (
        "player_name"
        if "player_name" in current.columns
        else "batter"
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.34)))
    ax.barh(
        current[label].astype(str),
        current["shrunk_centered_per_100"],
    )
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Shrunken decision value / 100 vs league average")
    ax.set_title("NorthStar player decision metric — sample-aware prototype")
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.min_decisions < 1:
        raise ValueError("--min-decisions must be >= 1")
    if args.min_discretionary < 1:
        raise ValueError("--min-discretionary must be >= 1")
    if args.top < 1:
        raise ValueError("--top must be >= 1")
    if not args.player_metric.exists():
        raise FileNotFoundError(
            f"Missing player metric table: {args.player_metric}"
        )
    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Missing external prediction file: {args.predictions}"
        )

    player_metric = pd.read_csv(args.player_metric)
    discipline = _load_statcast_discipline(args.raw_dir)

    outcome_reference = None
    if args.outcome_reference.exists():
        outcome_reference = pd.read_csv(args.outcome_reference)
    else:
        print(
            f"Warning: {args.outcome_reference} was not found; "
            "outcome-value construct checks will be skipped."
        )

    construct_data = merge_player_construct_data(
        player_metric,
        discipline,
        outcome_reference=outcome_reference,
    )

    correlations = construct_correlation_summary(construct_data)
    cv_primary = construct_cv_summary(
        construct_data,
        target="decision_value_per_100",
    )
    cv_discretionary = construct_cv_summary(
        construct_data,
        target="discretionary_decision_value_per_100",
    )
    cv_summary = pd.concat(
        [cv_primary, cv_discretionary],
        ignore_index=True,
    )

    predictions = pd.read_parquet(args.predictions)
    decisions = add_player_decision_values(predictions)

    shrunk, shrinkage_summary = empirical_bayes_player_shrinkage(
        decisions,
        min_decisions=args.min_decisions,
        min_discretionary=args.min_discretionary,
    )

    names = player_metric.loc[
        :, [column for column in ["batter", "player_name"] if column in player_metric.columns]
    ].drop_duplicates("batter")
    if "player_name" in names.columns and not shrunk.empty:
        shrunk = shrunk.merge(
            names,
            on="batter",
            how="left",
            validate="one_to_one",
        )
        columns = ["batter", "player_name"] + [
            column
            for column in shrunk.columns
            if column not in {"batter", "player_name"}
        ]
        shrunk = shrunk[columns]

    DEFAULT_CONSTRUCT_DATA.parent.mkdir(parents=True, exist_ok=True)
    construct_data.to_csv(DEFAULT_CONSTRUCT_DATA, index=False)
    correlations.to_csv(DEFAULT_CORRELATIONS, index=False)
    cv_summary.to_csv(DEFAULT_CV, index=False)
    shrunk.to_csv(DEFAULT_SHRUNK, index=False)

    _save_correlation_figure(
        correlations,
        DEFAULT_CORRELATION_FIGURE,
    )
    _save_shrinkage_figure(
        shrunk,
        DEFAULT_SHRINKAGE_FIGURE,
        top_n=args.top,
    )

    print()
    print("NorthStar player metric validation V1")
    print(f"Prototype players: {len(player_metric):,}")
    print(f"Statcast discipline players: {len(discipline):,}")
    print(
        f"Shrinkage eligibility: >= {args.min_decisions:,} decisions "
        f"and >= {args.min_discretionary:,} discretionary decisions"
    )
    print(
        "Pitch-weighted league mean decision value / 100: "
        f"{shrinkage_summary.league_mean * 100.0:.4f}"
    )
    print(
        "Estimated between-player variance: "
        f"{shrinkage_summary.between_player_variance:.8f}"
    )
    print(
        f"Players retained for shrinkage: "
        f"{shrinkage_summary.eligible_players:,}"
    )

    if not correlations.empty:
        print()
        print("Construct correlations with decision value / 100")
        primary = correlations.loc[
            correlations["northstar_metric"].eq(
                "decision_value_per_100"
            )
        ].copy()
        primary["abs_spearman"] = primary["spearman_rho"].abs()
        print(
            primary.sort_values(
                "abs_spearman",
                ascending=False,
            )
            .drop(columns=["abs_spearman"])
            .round(
                {
                    "pearson_r": 3,
                    "spearman_rho": 3,
                }
            )
            .to_string(index=False)
        )

    if not cv_summary.empty:
        print()
        print("Out-of-fold construct explanation")
        print(
            cv_summary.round(
                {
                    "oof_r2": 3,
                    "oof_pearson_r": 3,
                    "oof_spearman_rho": 3,
                }
            ).to_string(index=False)
        )

    if not shrunk.empty:
        print()
        print(f"Top {args.top} sample-aware centered prototype")
        columns = [
            column
            for column in [
                "player_name",
                "batter",
                "eligible_decisions",
                "discretionary_decisions",
                "raw_centered_per_100",
                "shrunk_centered_per_100",
                "shrinkage_factor",
                "cluster_se",
                "approx_ci_low_per_100",
                "approx_ci_high_per_100",
            ]
            if column in shrunk.columns
        ]
        print(
            shrunk.head(args.top)[columns]
            .round(
                {
                    "raw_centered_per_100": 3,
                    "shrunk_centered_per_100": 3,
                    "shrinkage_factor": 3,
                    "cluster_se": 4,
                    "approx_ci_low_per_100": 3,
                    "approx_ci_high_per_100": 3,
                }
            )
            .to_string(index=False)
        )

    print()
    print(f"Saved construct dataset: {DEFAULT_CONSTRUCT_DATA}")
    print(f"Saved correlations:      {DEFAULT_CORRELATIONS}")
    print(f"Saved construct CV:      {DEFAULT_CV}")
    print(f"Saved shrunken table:    {DEFAULT_SHRUNK}")
    print(f"Saved correlation fig:   {DEFAULT_CORRELATION_FIGURE}")
    print(f"Saved shrinkage fig:     {DEFAULT_SHRINKAGE_FIGURE}")


if __name__ == "__main__":
    main()
