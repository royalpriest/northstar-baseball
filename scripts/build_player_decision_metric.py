from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pybaseball import playerid_reverse_lookup

from northstar.metrics.player_decision import (
    add_player_decision_values,
    build_player_decision_metric,
    build_player_half_reliability,
    decision_value_identity_error,
)


DEFAULT_INPUT = Path(
    "data/processed/external_season_test_predictions_2024_2025.parquet"
)
DEFAULT_TABLE = Path(
    "reports/tables/player_decision_metric_prototype_2025.csv"
)
DEFAULT_HALF_TABLE = Path(
    "reports/tables/player_decision_metric_half_splits_2025.csv"
)
DEFAULT_RELIABILITY = Path(
    "reports/tables/player_decision_metric_reliability_2025.csv"
)
DEFAULT_FIGURE = Path(
    "reports/figures/player_decision_metric_prototype_2025_top20.png"
)
DEFAULT_RELIABILITY_FIGURE = Path(
    "reports/figures/player_decision_metric_reliability_2025.png"
)


SORT_MAP = {
    "rate": "decision_value_per_100",
    "total": "decision_value_total",
    "missed": "missed_value_per_100",
    "optimal": "value_weighted_optimal_choice_rate",
    "discretionary": "discretionary_decision_value_per_100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build NorthStar's first external-season player decision-value "
            "prototype from 2024-trained / 2025-scored pitch predictions."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--min-decisions", type=int, default=500)
    parser.add_argument("--min-half-decisions", type=int, default=200)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--split-date",
        default="2025-07-01",
        help="Boundary between early and late 2025 reliability samples.",
    )
    parser.add_argument(
        "--sort",
        choices=sorted(SORT_MAP),
        default="rate",
    )
    parser.add_argument("--discretionary-lower", type=float, default=0.20)
    parser.add_argument("--discretionary-upper", type=float, default=0.80)
    parser.add_argument(
        "--skip-names",
        action="store_true",
        help="Keep MLBAM batter IDs instead of attempting player-name lookup.",
    )
    return parser.parse_args()


def add_player_names(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "batter" not in frame.columns:
        return frame

    out = frame.copy()
    batter_ids = out["batter"].astype(int).drop_duplicates().tolist()

    try:
        lookup = playerid_reverse_lookup(batter_ids, key_type="mlbam")
    except Exception as exc:
        print(f"Warning: player-name lookup failed: {exc}")
        out.insert(
            1,
            "player_name",
            out["batter"].map(lambda value: f"MLBAM {value}"),
        )
        return out

    required = {"key_mlbam", "name_first", "name_last"}
    if not required.issubset(lookup.columns):
        print(
            "Warning: player lookup returned an unexpected schema; "
            "keeping MLBAM IDs."
        )
        out.insert(
            1,
            "player_name",
            out["batter"].map(lambda value: f"MLBAM {value}"),
        )
        return out

    names = lookup.loc[
        :, ["key_mlbam", "name_first", "name_last"]
    ].copy()
    names["batter"] = pd.to_numeric(
        names["key_mlbam"], errors="coerce"
    )
    names = names.dropna(subset=["batter"])
    names["batter"] = names["batter"].astype("int64")
    names["player_name"] = (
        names["name_first"].fillna("").astype(str).str.strip()
        + " "
        + names["name_last"].fillna("").astype(str).str.strip()
    ).str.strip().str.title()

    out = out.merge(
        names[["batter", "player_name"]].drop_duplicates("batter"),
        on="batter",
        how="left",
    )
    out["player_name"] = out["player_name"].fillna(
        out["batter"].map(lambda value: f"MLBAM {value}")
    )

    columns = ["batter", "player_name"] + [
        column
        for column in out.columns
        if column not in {"batter", "player_name"}
    ]
    return out[columns]


def save_leaderboard_chart(
    leaderboard: pd.DataFrame,
    output: Path,
    top_n: int,
    sort_column: str,
) -> None:
    if leaderboard.empty:
        return

    top = leaderboard.head(top_n).copy().iloc[::-1]
    label = (
        "player_name"
        if "player_name" in top.columns
        else "batter"
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.34)))
    ax.barh(top[label].astype(str), top[sort_column])
    ax.set_xlabel(sort_column.replace("_", " "))
    ax.set_title(
        "NorthStar player decision-value prototype — 2025 external season"
    )
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_reliability_chart(
    half_table: pd.DataFrame,
    output: Path,
) -> None:
    if half_table.empty:
        return

    pivot = half_table.pivot(
        index="batter",
        columns="season_half",
        values="decision_value_per_100",
    ).dropna()
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pivot["early"], pivot["late"], alpha=0.65)

    lower = min(pivot["early"].min(), pivot["late"].min())
    upper = max(pivot["early"].max(), pivot["late"].max())
    ax.plot([lower, upper], [lower, upper], linestyle="--")

    ax.set_xlabel("Early-season decision value / 100")
    ax.set_ylabel("Late-season decision value / 100")
    ax.set_title("2025 player decision-value split-half reliability")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.min_decisions < 1:
        raise ValueError("--min-decisions must be >= 1")
    if args.min_half_decisions < 1:
        raise ValueError("--min-half-decisions must be >= 1")
    if args.top < 1:
        raise ValueError("--top must be >= 1")
    if not (
        0.0
        < args.discretionary_lower
        < args.discretionary_upper
        < 1.0
    ):
        raise ValueError(
            "Discretionary bounds must satisfy 0 < lower < upper < 1."
        )
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Run external-season validation first."
        )

    predictions = pd.read_parquet(args.input)
    sort_column = SORT_MAP[args.sort]

    decisions = add_player_decision_values(
        predictions,
        discretionary_lower=args.discretionary_lower,
        discretionary_upper=args.discretionary_upper,
    )
    identity_error = decision_value_identity_error(decisions).abs()

    leaderboard = build_player_decision_metric(
        predictions,
        min_decisions=args.min_decisions,
        discretionary_lower=args.discretionary_lower,
        discretionary_upper=args.discretionary_upper,
        sort_by=sort_column,
    )
    half_table, reliability = build_player_half_reliability(
        predictions,
        split_date=args.split_date,
        min_half_decisions=args.min_half_decisions,
        discretionary_lower=args.discretionary_lower,
        discretionary_upper=args.discretionary_upper,
    )

    if not args.skip_names:
        leaderboard = add_player_names(leaderboard)

    DEFAULT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(DEFAULT_TABLE, index=False)
    half_table.to_csv(DEFAULT_HALF_TABLE, index=False)
    reliability.to_csv(DEFAULT_RELIABILITY, index=False)

    save_leaderboard_chart(
        leaderboard,
        DEFAULT_FIGURE,
        top_n=args.top,
        sort_column=sort_column,
    )
    save_reliability_chart(
        half_table,
        DEFAULT_RELIABILITY_FIGURE,
    )

    print()
    print("NorthStar player decision-value prototype V1")
    print(f"External 2025 decision rows: {len(decisions):,}")
    print(
        "Decision-value accounting max absolute identity error: "
        f"{identity_error.max():.12f}"
    )
    print(
        f"Leaderboard threshold: {args.min_decisions:,} eligible decisions"
    )
    print(
        f"Reliability threshold: {args.min_half_decisions:,} decisions per half"
    )
    print(
        "Discretionary opportunity window: "
        f"{args.discretionary_lower:.0%}-"
        f"{args.discretionary_upper:.0%}"
    )

    display_columns = [
        column
        for column in [
            "player_name",
            "batter",
            "eligible_decisions",
            "discretionary_decisions",
            "decision_value_total",
            "decision_value_per_100",
            "missed_value_per_100",
            "optimal_choice_rate",
            "value_weighted_optimal_choice_rate",
            "discretionary_decision_value_per_100",
        ]
        if column in leaderboard.columns
    ]

    print()
    print(f"Top {args.top} by {sort_column}")
    print(
        leaderboard.head(args.top)[display_columns]
        .round(
            {
                "decision_value_total": 3,
                "decision_value_per_100": 3,
                "missed_value_per_100": 3,
                "optimal_choice_rate": 3,
                "value_weighted_optimal_choice_rate": 3,
                "discretionary_decision_value_per_100": 3,
            }
        )
        .to_string(index=False)
    )

    print()
    print("Full-season split-half reliability")
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

    print()
    print(f"Saved prototype table: {DEFAULT_TABLE}")
    print(f"Saved half splits:     {DEFAULT_HALF_TABLE}")
    print(f"Saved reliability:     {DEFAULT_RELIABILITY}")
    print(f"Saved leaderboard fig: {DEFAULT_FIGURE}")
    print(f"Saved reliability fig: {DEFAULT_RELIABILITY_FIGURE}")


if __name__ == "__main__":
    main()
