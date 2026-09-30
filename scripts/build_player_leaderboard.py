from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pybaseball import playerid_reverse_lookup

from northstar.metrics.player_value import build_player_run_value


DEFAULT_INPUT = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_TOTAL_TABLE = Path("reports/tables/player_run_value_2025.csv")
DEFAULT_RATE_TABLE = Path("reports/tables/player_run_value_2025_rate.csv")
DEFAULT_FIGURE = Path("reports/figures/player_run_value_2025_top20.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NorthStar's 2025 hitter run-value leaderboard."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--min-pa", type=int, default=300)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--sort",
        choices=["total", "rate"],
        default="total",
        help="Rank by seasonal RE24 total or RE24 per 100 PA.",
    )
    parser.add_argument(
        "--skip-names",
        action="store_true",
        help="Skip MLBAM player-name lookup and keep batter IDs only.",
    )
    return parser.parse_args()


def add_player_names(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Resolve MLBAM batter IDs with pybaseball's Chadwick lookup."""
    if leaderboard.empty:
        out = leaderboard.copy()
        out.insert(1, "player_name", pd.Series(dtype="string"))
        return out

    batter_ids = leaderboard["batter"].astype(int).tolist()

    try:
        lookup = playerid_reverse_lookup(batter_ids, key_type="mlbam")
    except Exception as exc:
        print(f"Warning: player-name lookup failed: {exc}")
        out = leaderboard.copy()
        out.insert(1, "player_name", out["batter"].map(lambda value: f"MLBAM {value}"))
        return out

    required = {"key_mlbam", "name_first", "name_last"}
    if not required.issubset(lookup.columns):
        print("Warning: player lookup returned an unexpected schema; keeping MLBAM IDs.")
        out = leaderboard.copy()
        out.insert(1, "player_name", out["batter"].map(lambda value: f"MLBAM {value}"))
        return out

    names = lookup.loc[:, ["key_mlbam", "name_first", "name_last"]].copy()
    names["batter"] = pd.to_numeric(names["key_mlbam"], errors="coerce")
    names = names.dropna(subset=["batter"])
    names["batter"] = names["batter"].astype("int64")
    names["player_name"] = (
        names["name_first"].fillna("").astype(str).str.strip()
        + " "
        + names["name_last"].fillna("").astype(str).str.strip()
    ).str.strip().str.title()

    out = leaderboard.merge(
        names[["batter", "player_name"]].drop_duplicates("batter"),
        on="batter",
        how="left",
    )
    out["player_name"] = out["player_name"].fillna(
        out["batter"].map(lambda value: f"MLBAM {value}")
    )

    columns = ["batter", "player_name"] + [
        column for column in out.columns if column not in {"batter", "player_name"}
    ]
    return out[columns]


def save_chart(
    leaderboard: pd.DataFrame,
    output: Path,
    top_n: int,
    sort_column: str,
) -> None:
    if leaderboard.empty:
        print("No players met the minimum-PA threshold; chart not created.")
        return

    top = leaderboard.head(top_n).copy().iloc[::-1]
    label_column = "player_name" if "player_name" in top.columns else "batter"

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.32)))

    ax.barh(top[label_column].astype(str), top[sort_column])

    if sort_column == "re24_total":
        ax.set_xlabel("RE24 total")
        ax.set_title("2025 MLB Hitter Run Value — Total RE24")
    else:
        ax.set_xlabel("RE24 per 100 plate appearances")
        ax.set_title("2025 MLB Hitter Run Value — RE24 per 100 PA")

    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.top < 1:
        raise ValueError("--top must be >= 1")
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Run scripts/build_baseline.py first."
        )

    plate_appearances = pd.read_parquet(args.input)
    sort_column = "re24_total" if args.sort == "total" else "re24_per_100_pa"

    leaderboard = build_player_run_value(
        plate_appearances,
        min_pa=args.min_pa,
        sort_by=sort_column,
    )

    if not args.skip_names:
        leaderboard = add_player_names(leaderboard)

    table_output = (
        DEFAULT_TOTAL_TABLE
        if args.sort == "total"
        else DEFAULT_RATE_TABLE
    )
    table_output.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(table_output, index=False)

    figure_output = DEFAULT_FIGURE
    if args.sort == "rate":
        figure_output = Path(
            "reports/figures/player_run_value_2025_rate_top20.png"
        )

    save_chart(
        leaderboard,
        figure_output,
        top_n=args.top,
        sort_column=sort_column,
    )

    display_columns = [
        column
        for column in [
            "player_name",
            "batter",
            "plate_appearances",
            "re24_total",
            "re24_per_100_pa",
            "positive_pa_pct",
        ]
        if column in leaderboard.columns
    ]

    print()
    print(f"2025 MLB hitter run-value leaderboard (minimum {args.min_pa} PA)")
    print(
        leaderboard.head(args.top)[display_columns]
        .round(
            {
                "re24_total": 2,
                "re24_per_100_pa": 2,
                "positive_pa_pct": 1,
            }
        )
        .to_string(index=False)
    )

    print()
    print(f"Saved table: {table_output}")
    if not leaderboard.empty:
        print(f"Saved chart: {figure_output}")


if __name__ == "__main__":
    main()
