from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pybaseball import statcast_batter_expected_stats

from northstar.validation.offense import (
    DEFAULT_REFERENCE_METRICS,
    build_correlation_summary,
    merge_with_reference_stats,
)


DEFAULT_INPUT = Path("reports/tables/player_run_value_2025.csv")
DEFAULT_MERGED = Path("reports/tables/player_run_value_validation_2025.csv")
DEFAULT_CORRELATIONS = Path("reports/tables/player_run_value_correlations_2025.csv")
DEFAULT_FIGURE = Path("reports/figures/re24_vs_xwoba_2025.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate NorthStar RE24 against Baseball Savant batting metrics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--min-pa", type=int, default=300)
    parser.add_argument("--annotate", type=int, default=8)
    return parser.parse_args()


def _player_name_column(frame: pd.DataFrame) -> str | None:
    for column in ("last_name, first_name", "player_name"):
        if column in frame.columns:
            return column
    return None


def _display_player_name(value: object) -> str:
    text = str(value).strip()
    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        if first and last:
            return f"{first} {last}"
    return text


def save_scatter(
    merged: pd.DataFrame,
    output: Path,
    annotate: int = 4,
) -> None:
    required = {"re24_per_100_pa", "est_woba"}
    if not required.issubset(merged.columns):
        print("xwOBA scatter skipped because required columns were unavailable.")
        return

    name_column = _player_name_column(merged)
    columns = ["re24_per_100_pa", "est_woba"]
    if name_column:
        columns.append(name_column)

    plot = merged.loc[:, columns].copy()
    plot["re24_per_100_pa"] = pd.to_numeric(
        plot["re24_per_100_pa"], errors="coerce"
    )
    plot["est_woba"] = pd.to_numeric(plot["est_woba"], errors="coerce")
    plot = plot.dropna(subset=["re24_per_100_pa", "est_woba"])

    if plot.empty:
        print("xwOBA scatter skipped because there were no matched rows.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        plot["est_woba"],
        plot["re24_per_100_pa"],
        alpha=0.68,
        s=34,
    )

    ax.set_xlabel("Baseball Savant xwOBA")
    ax.set_ylabel("NorthStar RE24 per 100 PA")
    ax.set_title("2025 MLB: NorthStar RE24 vs xwOBA")
    ax.grid(alpha=0.18)

    if annotate > 0 and name_column:
        labels = plot.nlargest(annotate, "re24_per_100_pa").copy()

        offsets = [
            (12, 12),
            (12, -18),
            (-58, 12),
            (-58, -18),
            (20, 24),
            (-70, 24),
        ]

        for index, (_, row) in enumerate(labels.iterrows()):
            dx, dy = offsets[index % len(offsets)]
            label = _display_player_name(row[name_column])

            ax.annotate(
                label,
                (row["est_woba"], row["re24_per_100_pa"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.9,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "linewidth": 0.7,
                    "alpha": 0.55,
                },
            )

    fig.tight_layout(pad=1.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
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

    print(f"Downloading {args.season} Baseball Savant expected batting stats...")
    reference = statcast_batter_expected_stats(
        args.season,
        minPA=args.min_pa,
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
        f"Matched {len(merged):,} NorthStar hitters to Baseball Savant "
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
