from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from northstar.data.fangraphs import fetch_fangraphs_batting
from northstar.validation.offense import (
    DEFAULT_REFERENCE_METRICS,
    build_correlation_summary,
    merge_with_reference_stats,
)


DEFAULT_INPUT = Path("reports/tables/player_run_value_2025.csv")
DEFAULT_MERGED = Path("reports/tables/player_run_value_validation_2025.csv")
DEFAULT_CORRELATIONS = Path("reports/tables/player_run_value_correlations_2025.csv")
DEFAULT_FIGURE = Path("reports/figures/re24_vs_wrcplus_2025.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate NorthStar RE24 against established FanGraphs batting metrics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--min-pa", type=int, default=300)
    parser.add_argument("--annotate", type=int, default=8)
    return parser.parse_args()


def save_scatter(
    merged: pd.DataFrame,
    output: Path,
    annotate: int = 8,
) -> None:
    required = {"re24_per_100_pa", "wRC+", "Name"}
    if not required.issubset(merged.columns):
        print("wRC+ scatter skipped because required columns were unavailable.")
        return

    plot = merged.loc[:, ["Name", "re24_per_100_pa", "wRC+"]].copy()
    plot["re24_per_100_pa"] = pd.to_numeric(plot["re24_per_100_pa"], errors="coerce")
    plot["wRC+"] = pd.to_numeric(plot["wRC+"], errors="coerce")
    plot = plot.dropna()

    if plot.empty:
        print("wRC+ scatter skipped because there were no matched rows.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot["wRC+"], plot["re24_per_100_pa"], alpha=0.7)
    ax.set_xlabel("FanGraphs wRC+")
    ax.set_ylabel("NorthStar RE24 per 100 PA")
    ax.set_title("2025 MLB: NorthStar RE24 vs wRC+")
    ax.grid(alpha=0.2)

    if annotate > 0:
        labels = plot.nlargest(annotate, "re24_per_100_pa")
        for _, row in labels.iterrows():
            ax.annotate(
                str(row["Name"]),
                (row["wRC+"], row["re24_per_100_pa"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.min_pa < 1:
        raise ValueError("--min-pa must be >= 1")
    if args.annotate < 0:
        raise ValueError("--annotate must be >= 0")
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Build the player leaderboard first."
        )

    leaderboard = pd.read_csv(args.input)

    print(f"Downloading {args.season} FanGraphs batting stats from the modern API...")
    reference = fetch_fangraphs_batting(
        season=args.season,
        min_pa=args.min_pa,
    )

    merged = merge_with_reference_stats(
        leaderboard=leaderboard,
        reference_stats=reference,
    )

    correlations = build_correlation_summary(
        merged,
        northstar_metric="re24_per_100_pa",
        reference_metrics=DEFAULT_REFERENCE_METRICS,
    )

    DEFAULT_MERGED.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(DEFAULT_MERGED, index=False)
    correlations.to_csv(DEFAULT_CORRELATIONS, index=False)

    save_scatter(merged, DEFAULT_FIGURE, annotate=args.annotate)

    print()
    print(
        f"Matched {len(merged):,} NorthStar hitters to FanGraphs "
        f"(minimum {args.min_pa} PA)."
    )

    if correlations.empty:
        print("No correlation results were available.")
    else:
        print()
        print("Correlation with NorthStar RE24 per 100 PA")
        print(
            correlations.round(
                {"pearson_r": 3, "spearman_rho": 3}
            ).to_string(index=False)
        )

    print()
    print(f"Saved matched data: {DEFAULT_MERGED}")
    print(f"Saved correlations: {DEFAULT_CORRELATIONS}")
    if DEFAULT_FIGURE.exists():
        print(f"Saved chart: {DEFAULT_FIGURE}")


if __name__ == "__main__":
    main()
