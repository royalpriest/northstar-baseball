from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from northstar.profiles.hitter import select_hitter_profile


COMPARISON_FIELDS = [
    "batter",
    "player_name",
    "birth_country",
    "birth_city",
    "birth_state_province",
    "primary_team_2025",
    "teams_2025",
    "plate_appearances",
    "eligible_decisions",
    "swings",
    "batted_balls",
    "shrunk_centered_per_100",
    "selection_context_percentile",
    "contact_above_expected_per_100_swings",
    "contact_context_percentile",
    "damage_xwoba_points_above_expected",
    "damage_context_percentile",
    "statcast_chase_rate",
    "chase_avoidance_percentile",
    "statcast_contact_rate",
    "contact_rate_percentile",
    "statcast_xwoba_on_contact",
    "xwoba_on_contact_percentile",
    "statcast_barrel_rate",
    "barrel_percentile",
    "woba",
    "woba_percentile",
    "est_woba",
    "xwoba_percentile",
    "re24_per_100_pa",
    "re24_percentile",
]


def build_hitter_comparison(
    profile: pd.DataFrame,
    players: Sequence[str | int],
) -> pd.DataFrame:
    """Select 2-4 hitters for a side-by-side NorthStar comparison."""
    if len(players) < 2:
        raise ValueError("At least two players are required")
    if len(players) > 4:
        raise ValueError("At most four players can be compared at once")

    rows: list[pd.Series] = []
    seen: set[int] = set()

    for player in players:
        row = select_hitter_profile(profile, player)
        batter = int(row["batter"])
        if batter in seen:
            raise ValueError(
                f"Player {player!r} resolves to a duplicate hitter"
            )
        seen.add(batter)
        rows.append(row)

    comparison = pd.DataFrame(rows).reset_index(drop=True)
    columns = [
        column
        for column in COMPARISON_FIELDS
        if column in comparison.columns
    ]
    return comparison.loc[:, columns]


def comparison_percentile_long(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Convert comparison percentiles into a plotting-friendly long table."""
    metrics = [
        (
            "NorthStar",
            "Selection context",
            "selection_context_percentile",
        ),
        (
            "NorthStar",
            "Contact context",
            "contact_context_percentile",
        ),
        (
            "NorthStar",
            "Damage context",
            "damage_context_percentile",
        ),
        (
            "Established",
            "Chase avoidance",
            "chase_avoidance_percentile",
        ),
        (
            "Established",
            "Contact rate",
            "contact_rate_percentile",
        ),
        (
            "Established",
            "xwOBA on contact",
            "xwoba_on_contact_percentile",
        ),
        (
            "Established",
            "Barrel rate",
            "barrel_percentile",
        ),
        (
            "Established",
            "xwOBA",
            "xwoba_percentile",
        ),
        (
            "Established",
            "RE24 / 100 PA",
            "re24_percentile",
        ),
    ]

    rows: list[dict[str, object]] = []
    for _, player in comparison.iterrows():
        for group, label, column in metrics:
            if column not in comparison.columns:
                continue
            value = pd.to_numeric(
                pd.Series([player.get(column)]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "batter": int(player["batter"]),
                    "player_name": player.get("player_name"),
                    "metric_group": group,
                    "metric": label,
                    "percentile": float(value),
                }
            )

    return pd.DataFrame(rows)
