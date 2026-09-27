from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from pybaseball import playerid_reverse_lookup

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


def _resolve_missing_player_names(
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing profile names from MLBAM IDs when possible."""
    out = profile.copy()
    if "player_name" not in out.columns:
        out["player_name"] = pd.NA

    missing_mask = (
        out["player_name"].isna()
        | out["player_name"].astype("string").str.strip().eq("")
    )
    missing_ids = (
        pd.to_numeric(
            out.loc[missing_mask, "batter"],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )

    if missing_ids:
        try:
            lookup = playerid_reverse_lookup(
                missing_ids,
                key_type="mlbam",
            )
            required = {"key_mlbam", "name_first", "name_last"}
            if required.issubset(lookup.columns):
                names = lookup.loc[
                    :,
                    ["key_mlbam", "name_first", "name_last"],
                ].copy()
                names["batter"] = pd.to_numeric(
                    names["key_mlbam"],
                    errors="coerce",
                )
                names = names.dropna(subset=["batter"])
                names["batter"] = names["batter"].astype("int64")
                names["resolved_name"] = (
                    names["name_first"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    + " "
                    + names["name_last"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                ).str.strip().str.title()

                name_map = (
                    names.loc[
                        names["resolved_name"].ne(""),
                        ["batter", "resolved_name"],
                    ]
                    .drop_duplicates("batter")
                    .set_index("batter")["resolved_name"]
                    .to_dict()
                )
                out.loc[missing_mask, "player_name"] = (
                    out.loc[missing_mask, "batter"].map(name_map)
                )
        except Exception as exc:
            print(
                "Warning: fallback player-name lookup failed: "
                f"{exc}"
            )

    unresolved = (
        out["player_name"].isna()
        | out["player_name"].astype("string").str.strip().eq("")
    )
    out.loc[unresolved, "player_name"] = out.loc[
        unresolved,
        "batter",
    ].map(lambda value: f"MLBAM {int(value)}")

    return out


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        ascii_value.strip().lower(),
    ).strip("-")
    return slug or "player"


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _ordinal_percentile(value: object) -> str:
    if pd.isna(value):
        return "n/a"

    number = max(1, min(100, int(round(float(value)))))
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(
            number % 10,
            "th",
        )
    return f"{number}{suffix}"


def _fmt_rate(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def _fmt_decimal(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    text = f"{float(value):.{digits}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def _fmt_signed(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):+.{digits}f}"


def _card_box(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=ax.transAxes,
        linewidth=0.9,
        edgecolor="#dbe3ec",
        facecolor="#f8fafc",
        clip_on=False,
    )
    ax.add_patch(box)


def _draw_axis_card(
    ax,
    *,
    title: str,
    description: str,
    percentile: object,
    raw_text: str,
) -> None:
    _card_box(ax)

    pct = (
        float(percentile)
        if not pd.isna(percentile)
        else 0.0
    )
    pct = max(0.0, min(100.0, pct))

    ax.text(
        0.06,
        0.82,
        title.upper(),
        fontsize=10,
        fontweight="bold",
        color="#334155",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.06,
        0.57,
        _ordinal_percentile(percentile),
        fontsize=27,
        fontweight="bold",
        color="#0f172a",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.06,
        0.38,
        raw_text,
        fontsize=10.5,
        color="#334155",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.06,
        0.20,
        description,
        fontsize=8.7,
        color="#64748b",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    ax.add_patch(
        Rectangle(
            (0.06, 0.08),
            0.88,
            0.035,
            transform=ax.transAxes,
            facecolor="#e2e8f0",
            edgecolor="none",
        )
    )
    ax.add_patch(
        Rectangle(
            (0.06, 0.08),
            0.88 * (pct / 100.0),
            0.035,
            transform=ax.transAxes,
            facecolor="#2563eb",
            edgecolor="none",
        )
    )


def _draw_metric_panel(
    ax,
    *,
    title: str,
    rows: list[tuple[str, str, str]],
) -> None:
    _card_box(ax)
    ax.text(
        0.06,
        0.91,
        title.upper(),
        fontsize=9.5,
        fontweight="bold",
        color="#334155",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    if not rows:
        ax.text(
            0.06,
            0.76,
            "No eligible data",
            fontsize=9,
            color="#64748b",
            transform=ax.transAxes,
        )
        return

    top = 0.78
    bottom = 0.10
    step = (
        (top - bottom) / max(len(rows) - 1, 1)
        if len(rows) > 1
        else 0.0
    )

    for index, (label, value, context) in enumerate(rows):
        y = top - index * step
        ax.text(
            0.06,
            y,
            label,
            fontsize=9.1,
            color="#334155",
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.61,
            y,
            value,
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
            ha="right",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.94,
            y,
            context,
            fontsize=8.4,
            color="#64748b",
            ha="right",
            va="center",
            transform=ax.transAxes,
        )


def _save_player_card(
    row: pd.Series,
    output: Path,
) -> None:
    player_name = row.get("player_name")
    if pd.isna(player_name):
        player_name = f"MLBAM {int(row['batter'])}"

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        nrows=4,
        ncols=12,
        height_ratios=[0.8, 2.15, 3.45, 0.75],
        hspace=0.24,
        wspace=0.22,
    )

    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    header.text(
        0.0,
        0.76,
        "NORTHSTAR / HITTER PROFILE",
        fontsize=10,
        fontweight="bold",
        color="#2563eb",
        ha="left",
        va="center",
        transform=header.transAxes,
    )
    header.text(
        0.0,
        0.27,
        str(player_name),
        fontsize=25,
        fontweight="bold",
        color="#0f172a",
        ha="left",
        va="center",
        transform=header.transAxes,
    )
    header.text(
        1.0,
        0.66,
        "2025",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        ha="right",
        va="center",
        transform=header.transAxes,
    )
    header.text(
        1.0,
        0.27,
        f"MLBAM {int(row['batter'])}",
        fontsize=9.5,
        color="#64748b",
        ha="right",
        va="center",
        transform=header.transAxes,
    )

    _draw_axis_card(
        fig.add_subplot(grid[1, 0:4]),
        title="Selection",
        description="Decision quality relative to context",
        percentile=row.get("selection_context_percentile"),
        raw_text=(
            f"{_fmt_signed(row.get('shrunk_centered_per_100'))} "
            "value / 100"
        ),
    )
    _draw_axis_card(
        fig.add_subplot(grid[1, 4:8]),
        title="Contact",
        description="Contact relative to swing difficulty",
        percentile=row.get("contact_context_percentile"),
        raw_text=(
            f"{_fmt_signed(row.get('contact_above_expected_per_100_swings'))} "
            "pp / 100 swings"
        ),
    )
    _draw_axis_card(
        fig.add_subplot(grid[1, 8:12]),
        title="Damage",
        description="Contact quality relative to opportunity",
        percentile=row.get("damage_context_percentile"),
        raw_text=(
            f"{_fmt_signed(row.get('damage_xwoba_points_above_expected'))} "
            "xwOBA pts"
        ),
    )

    plate_rows = [
        (
            "Chase rate",
            _fmt_rate(row.get("statcast_chase_rate")),
            (
                f"{_ordinal_percentile(row.get('chase_avoidance_percentile'))} "
                "avoidance"
            ),
        ),
        (
            "Swing rate",
            _fmt_rate(row.get("statcast_swing_rate")),
            (
                f"{_ordinal_percentile(row.get('swing_rate_percentile'))} "
                "swing-rate pct"
            ),
        ),
        (
            "First-pitch swing",
            _fmt_rate(row.get("statcast_first_pitch_swing_rate")),
            (
                f"{_ordinal_percentile(row.get('first_pitch_swing_rate_percentile'))} "
                "swing-rate pct"
            ),
        ),
        (
            "Two-strike chase",
            _fmt_rate(row.get("statcast_two_strike_chase_rate")),
            (
                f"{_ordinal_percentile(row.get('two_strike_chase_avoidance_percentile'))} "
                "avoidance"
            ),
        ),
    ]

    bat_to_ball_rows = [
        (
            "Contact rate",
            _fmt_rate(row.get("statcast_contact_rate")),
            _ordinal_percentile(row.get("contact_rate_percentile")),
        ),
        (
            "Zone contact",
            _fmt_rate(row.get("statcast_zone_contact_rate")),
            _ordinal_percentile(row.get("zone_contact_percentile")),
        ),
        (
            "Chase contact",
            _fmt_rate(row.get("statcast_chase_contact_rate")),
            _ordinal_percentile(row.get("chase_contact_percentile")),
        ),
        (
            "Two-strike contact",
            _fmt_rate(row.get("statcast_two_strike_contact_rate")),
            _ordinal_percentile(row.get("two_strike_contact_percentile")),
        ),
    ]

    quality_rows = [
        (
            "xwOBA on contact",
            _fmt_decimal(row.get("statcast_xwoba_on_contact")),
            _ordinal_percentile(row.get("xwoba_on_contact_percentile")),
        ),
        (
            "Barrel rate",
            _fmt_rate(row.get("statcast_barrel_rate")),
            _ordinal_percentile(row.get("barrel_percentile")),
        ),
        (
            "Hard-hit rate",
            _fmt_rate(row.get("statcast_hard_hit_rate")),
            _ordinal_percentile(row.get("hard_hit_percentile")),
        ),
        (
            "Avg exit velocity",
            (
                f"{_fmt(row.get('statcast_avg_exit_velocity'), 1)} mph"
                if not pd.isna(row.get("statcast_avg_exit_velocity"))
                else "n/a"
            ),
            _ordinal_percentile(row.get("exit_velocity_percentile")),
        ),
        (
            "wOBA",
            _fmt_decimal(row.get("woba")),
            _ordinal_percentile(row.get("woba_percentile")),
        ),
        (
            "xwOBA",
            _fmt_decimal(row.get("est_woba")),
            _ordinal_percentile(row.get("xwoba_percentile")),
        ),
        (
            "RE24 / 100 PA",
            _fmt_signed(row.get("re24_per_100_pa")),
            _ordinal_percentile(row.get("re24_percentile")),
        ),
    ]

    _draw_metric_panel(
        fig.add_subplot(grid[2, 0:4]),
        title="Plate discipline",
        rows=plate_rows,
    )
    _draw_metric_panel(
        fig.add_subplot(grid[2, 4:8]),
        title="Bat-to-ball",
        rows=bat_to_ball_rows,
    )
    _draw_metric_panel(
        fig.add_subplot(grid[2, 8:12]),
        title="Contact quality / outcomes",
        rows=quality_rows,
    )

    footer = fig.add_subplot(grid[3, :])
    footer.axis("off")
    sample_parts = [
        (
            f"{int(row['eligible_decisions']):,} decisions"
            if not pd.isna(row.get("eligible_decisions"))
            else None
        ),
        (
            f"{int(row['swings']):,} swings"
            if not pd.isna(row.get("swings"))
            else None
        ),
        (
            f"{int(row['batted_balls']):,} batted balls"
            if not pd.isna(row.get("batted_balls"))
            else None
        ),
        (
            f"{int(row['plate_appearances']):,} PA"
            if not pd.isna(row.get("plate_appearances"))
            else None
        ),
    ]
    sample_text = " · ".join(
        part for part in sample_parts if part is not None
    )
    footer.text(
        0.0,
        0.68,
        sample_text,
        fontsize=9.5,
        fontweight="bold",
        color="#334155",
        ha="left",
        va="center",
        transform=footer.transAxes,
    )
    footer.text(
        0.0,
        0.24,
        (
            "NorthStar axes are contextual diagnostics, not a composite "
            "player rating or standalone predictive-skill claim."
        ),
        fontsize=8.8,
        color="#64748b",
        ha="left",
        va="center",
        transform=footer.transAxes,
    )
    footer.text(
        1.0,
        0.24,
        "Percentiles are league-relative within the available 2025 sample.",
        fontsize=8.8,
        color="#64748b",
        ha="right",
        va="center",
        transform=footer.transAxes,
    )

    fig.subplots_adjust(
        left=0.045,
        right=0.955,
        top=0.965,
        bottom=0.045,
    )
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
    profile = _resolve_missing_player_names(profile)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.output, index=False)

    print()
    print("NorthStar Hitter Profile V2")
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
