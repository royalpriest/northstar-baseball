from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from northstar.profiles.hitter import (
    build_hitter_profile,
    select_hitter_profile,
)


DEFAULT_SELECTION = Path(
    "reports/tables/player_metric_shrunk_2025.csv"
)
DEFAULT_CONTACT = Path(
    "reports/tables/contact_execution_construct_dataset_2025.csv"
)
DEFAULT_DAMAGE = Path(
    "reports/tables/damage_execution_construct_dataset_2025.csv"
)
DEFAULT_DISCIPLINE = Path(
    "reports/tables/player_metric_construct_dataset_2025.csv"
)
DEFAULT_OUTCOMES = Path(
    "reports/tables/player_run_value_validation_2025.csv"
)
DEFAULT_OUTPUT = Path(
    "reports/tables/hitter_profile_2025.csv"
)
DEFAULT_FIGURE_DIR = Path("reports/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical NorthStar hitter profile without creating "
            "a composite overall score."
        )
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--contact", type=Path, default=DEFAULT_CONTACT)
    parser.add_argument("--damage", type=Path, default=DEFAULT_DAMAGE)
    parser.add_argument("--discipline", type=Path, default=DEFAULT_DISCIPLINE)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--player",
        default=None,
        help="Optional exact player name or MLBAM batter ID for a profile card.",
    )
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "player"


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _save_player_card(
    row: pd.Series,
    output: Path,
) -> None:
    axes_data = [
        (
            "Selection context",
            row.get("selection_context_percentile"),
            row.get("shrunk_centered_per_100"),
            "value / 100",
        ),
        (
            "Contact context",
            row.get("contact_context_percentile"),
            row.get("contact_above_expected_per_100_swings"),
            "pp / 100 swings",
        ),
        (
            "Damage context",
            row.get("damage_context_percentile"),
            row.get("damage_xwoba_points_above_expected"),
            "xwOBA pts",
        ),
    ]
    axes_data = [
        item for item in axes_data if not pd.isna(item[1])
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]

    labels = [item[0] for item in axes_data]
    percentiles = [float(item[1]) for item in axes_data]
    ax.barh(labels, percentiles)
    ax.set_xlim(0, 100)
    ax.set_xlabel("League-relative percentile")
    ax.set_title("NorthStar contextual diagnostics")
    ax.grid(axis="x", alpha=0.2)

    for index, (_, percentile, raw, unit) in enumerate(axes_data):
        ax.text(
            min(float(percentile) + 1.5, 96.0),
            index,
            f"{float(percentile):.0f}th | {_fmt(raw)} {unit}",
            va="center",
            fontsize=9,
        )

    detail = axes[1]
    detail.axis("off")

    player_name = row.get("player_name")
    if pd.isna(player_name):
        player_name = f"MLBAM {int(row['batter'])}"

    lines = [
        f"{player_name}",
        f"MLBAM: {int(row['batter'])}",
        "",
        "Conventional context",
        (
            "Contact rate: "
            f"{_fmt(row.get('statcast_contact_rate'))} "
            f"({ _fmt(row.get('contact_rate_percentile'), 0) } pct)"
        ),
        (
            "Chase rate: "
            f"{_fmt(row.get('statcast_chase_rate'))} "
            f"({ _fmt(row.get('chase_avoidance_percentile'), 0) } "
            "chase-avoidance pct)"
        ),
        (
            "xwOBA on contact: "
            f"{_fmt(row.get('statcast_xwoba_on_contact'))} "
            f"({ _fmt(row.get('xwoba_on_contact_percentile'), 0) } pct)"
        ),
        (
            "Barrel rate: "
            f"{_fmt(row.get('statcast_barrel_rate'))} "
            f"({ _fmt(row.get('barrel_percentile'), 0) } pct)"
        ),
        (
            "wOBA: "
            f"{_fmt(row.get('woba'))} "
            f"({ _fmt(row.get('woba_percentile'), 0) } pct)"
        ),
        (
            "xwOBA: "
            f"{_fmt(row.get('est_woba'))} "
            f"({ _fmt(row.get('xwoba_percentile'), 0) } pct)"
        ),
        (
            "RE24 / 100 PA: "
            f"{_fmt(row.get('re24_per_100_pa'))} "
            f"({ _fmt(row.get('re24_percentile'), 0) } pct)"
        ),
        "",
        "Samples",
        f"Selection decisions: {_fmt(row.get('eligible_decisions'), 0)}",
        f"Swings: {_fmt(row.get('swings'), 0)}",
        f"Batted balls: {_fmt(row.get('batted_balls'), 0)}",
        "",
        "Interpretation",
        "NorthStar axes are contextual diagnostics.",
        "No overall score or standalone predictive-skill claim is made.",
    ]

    detail.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.35,
    )
    detail.set_title("Hitter profile", loc="left")

    fig.suptitle(f"NorthStar Hitter Profile — {player_name} — 2025")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _print_top(
    profile: pd.DataFrame,
    metric: str,
    label: str,
    top: int,
) -> None:
    if metric not in profile.columns:
        return

    columns = [
        column
        for column in [
            "player_name",
            "batter",
            metric,
        ]
        if column in profile.columns
    ]
    current = (
        profile.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .head(top)
    )

    print()
    print(label)
    print(current[columns].round({metric: 3}).to_string(index=False))


def main() -> None:
    args = parse_args()

    if args.top < 1:
        raise ValueError("--top must be >= 1")

    required = [
        args.selection,
        args.contact,
        args.damage,
        args.discipline,
        args.outcomes,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing profile input: {path}")

    profile = build_hitter_profile(
        selection=pd.read_csv(args.selection),
        contact=pd.read_csv(args.contact),
        damage=pd.read_csv(args.damage),
        discipline=pd.read_csv(args.discipline),
        outcomes=pd.read_csv(args.outcomes),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.output, index=False)

    print()
    print("NorthStar Hitter Profile V1")
    print(f"Players in canonical table: {len(profile):,}")
    print(
        "Complete selection/contact/damage profiles: "
        f"{int(profile['complete_three_axis_profile'].sum()):,}"
    )
    print("No composite overall score is created.")
    print(
        "Percentiles are descriptive league-relative positions within the "
        "available 2025 profile sample."
    )

    _print_top(
        profile,
        "shrunk_centered_per_100",
        f"Top {args.top} selection context",
        args.top,
    )
    _print_top(
        profile,
        "contact_above_expected_per_100_swings",
        f"Top {args.top} contact context",
        args.top,
    )
    _print_top(
        profile,
        "damage_xwoba_points_above_expected",
        f"Top {args.top} damage context",
        args.top,
    )

    print()
    print(f"Saved canonical profile: {args.output}")

    if args.player is not None:
        player_value: str | int = args.player
        if str(args.player).strip().isdigit():
            player_value = int(str(args.player).strip())

        row = select_hitter_profile(profile, player_value)
        name = row.get("player_name")
        label = (
            str(name)
            if not pd.isna(name)
            else str(int(row["batter"]))
        )
        output = DEFAULT_FIGURE_DIR / (
            f"hitter_profile_{_slug(label)}_2025.png"
        )
        _save_player_card(row, output)
        print(f"Saved player profile card: {output}")


if __name__ == "__main__":
    main()
