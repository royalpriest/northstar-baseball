from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from northstar.profiles.comparison import (
    build_hitter_comparison,
    comparison_percentile_long,
)


DEFAULT_PROFILE = Path(
    "reports/tables/hitter_profile_with_canada_2025.csv"
)
DEFAULT_TABLE_DIR = Path("reports/tables")
DEFAULT_FIGURE_DIR = Path("reports/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a side-by-side NorthStar hitter comparison for 2-4 players."
        )
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
    )
    parser.add_argument(
        "--players",
        nargs="+",
        required=True,
        help="Exact player names or MLBAM IDs. Supply 2 to 4 players.",
    )
    return parser.parse_args()


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(
        r"[^a-z0-9]+",
        "-",
        ascii_value.strip().lower(),
    ).strip("-") or "player"


def _comparison_stem(comparison: pd.DataFrame) -> str:
    names = []
    for _, row in comparison.iterrows():
        name = row.get("player_name")
        if pd.isna(name):
            name = str(int(row["batter"]))
        names.append(_slug(str(name)))
    return "hitter_comparison_" + "_vs_".join(names) + "_2025"


def _save_figure(
    comparison: pd.DataFrame,
    long: pd.DataFrame,
    output: Path,
) -> None:
    if long.empty:
        return

    order = [
        "Selection context",
        "Contact context",
        "Damage context",
        "Chase avoidance",
        "Contact rate",
        "xwOBA on contact",
        "Barrel rate",
        "xwOBA",
        "RE24 / 100 PA",
    ]
    metrics = [
        metric for metric in order if metric in set(long["metric"])
    ]
    players = comparison["player_name"].fillna(
        comparison["batter"].astype(str)
    ).astype(str).tolist()

    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    y = np.arange(len(metrics), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(players))

    for offset, player in zip(offsets, players, strict=True):
        current = long.loc[long["player_name"].astype(str).eq(player)]
        values = []
        for metric in metrics:
            matches = current.loc[current["metric"].eq(metric), "percentile"]
            values.append(
                float(matches.iloc[0]) if not matches.empty else np.nan
            )
        ax.scatter(
            values,
            y + offset,
            s=70,
            label=player,
            zorder=3,
        )

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlabel("League-relative percentile")
    ax.set_title("NorthStar Hitter Comparison — 2025")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(len(players), 4),
        frameon=False,
    )

    # Visually separate NorthStar diagnostics from established measures.
    if len(metrics) > 3:
        ax.axhline(2.5, linewidth=1, alpha=0.25)

    fig.text(
        0.5,
        0.02,
        (
            "Selection, contact, and damage are contextual diagnostics. "
            "No composite player score is calculated."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.08, 0.98, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.profile.exists():
        raise FileNotFoundError(
            f"Missing enriched hitter profile: {args.profile}. "
            "Run build_canada_layer.py first."
        )

    profile = pd.read_csv(args.profile)
    parsed_players: list[str | int] = []
    for player in args.players:
        value = player.strip()
        parsed_players.append(int(value) if value.isdigit() else value)

    comparison = build_hitter_comparison(
        profile,
        parsed_players,
    )
    long = comparison_percentile_long(comparison)

    stem = _comparison_stem(comparison)
    table_output = DEFAULT_TABLE_DIR / f"{stem}.csv"
    figure_output = DEFAULT_FIGURE_DIR / f"{stem}.png"

    table_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(table_output, index=False)
    _save_figure(comparison, long, figure_output)

    print()
    print("NorthStar hitter comparison V1")
    print("Players:")
    for _, row in comparison.iterrows():
        name = row.get("player_name")
        label = (
            str(name)
            if not pd.isna(name)
            else f"MLBAM {int(row['batter'])}"
        )
        team = row.get("primary_team_2025")
        birth = row.get("birth_country")
        details = [
            item
            for item in [
                f"team {team}" if not pd.isna(team) else None,
                f"born {birth}" if not pd.isna(birth) else None,
            ]
            if item is not None
        ]
        suffix = f" ({'; '.join(details)})" if details else ""
        print(f"  {label}{suffix}")

    display = [
        column
        for column in [
            "player_name",
            "selection_context_percentile",
            "contact_context_percentile",
            "damage_context_percentile",
            "contact_rate_percentile",
            "chase_avoidance_percentile",
            "xwoba_on_contact_percentile",
            "re24_percentile",
        ]
        if column in comparison.columns
    ]
    print()
    print(
        comparison.loc[:, display]
        .round(0)
        .to_string(index=False)
    )

    print()
    print(f"Saved comparison table:  {table_output}")
    print(f"Saved comparison figure: {figure_output}")


if __name__ == "__main__":
    main()
