from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from northstar.data.mlb_people import fetch_mlb_people
from northstar.profiles.canada import (
    attach_canada_context,
    build_blue_jays_hitter_view,
    build_canadian_hitter_view,
    build_canadian_leaderboard,
    build_player_team_summary,
)


DEFAULT_PROFILE = Path("reports/tables/hitter_profile_2025.csv")
DEFAULT_PA = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_IDENTITY = Path("reports/tables/player_identity_2025.csv")
DEFAULT_TEAM_CONTEXT = Path("reports/tables/player_team_context_2025.csv")
DEFAULT_ENRICHED = Path(
    "reports/tables/hitter_profile_with_canada_2025.csv"
)
DEFAULT_CANADIAN = Path(
    "reports/tables/canadian_hitter_profile_2025.csv"
)
DEFAULT_CANADIAN_LEADERBOARD = Path(
    "reports/tables/canadian_hitter_leaderboard_2025.csv"
)
DEFAULT_BLUE_JAYS = Path(
    "reports/tables/blue_jays_hitter_profile_2025.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build NorthStar's Canadian-born and Toronto Blue Jays hitter views."
        )
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--plate-appearances", type=Path, default=DEFAULT_PA)
    parser.add_argument("--identity-cache", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument(
        "--refresh-identities",
        action="store_true",
        help="Refetch all MLB player identity records instead of reusing the cache.",
    )
    parser.add_argument(
        "--canada-min-pa",
        type=int,
        default=0,
        help="Optional minimum full-season PA for the Canadian-born view.",
    )
    parser.add_argument(
        "--leaderboard-sort",
        default="re24_per_100_pa",
        help="Canonical profile column used to order the Canadian leaderboard.",
    )
    return parser.parse_args()


def _load_or_fetch_identities(
    profile: pd.DataFrame,
    cache: Path,
    *,
    refresh: bool,
) -> pd.DataFrame:
    ids = (
        pd.to_numeric(profile["batter"], errors="coerce")
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )

    cached = pd.DataFrame()
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache)
        if "batter" in cached.columns:
            cached["batter"] = pd.to_numeric(
                cached["batter"],
                errors="coerce",
            )
            cached = cached.dropna(subset=["batter"]).copy()
            cached["batter"] = cached["batter"].astype("int64")
        else:
            cached = pd.DataFrame()

    if refresh:
        missing_ids = ids
        cached = pd.DataFrame()
    else:
        known_ids = (
            set(cached["batter"].tolist())
            if not cached.empty and "batter" in cached.columns
            else set()
        )
        missing_ids = sorted(set(ids).difference(known_ids))

    if missing_ids:
        print(
            f"Fetching MLB identity data for {len(missing_ids):,} "
            "profile hitter(s)..."
        )
        fetched = fetch_mlb_people(missing_ids)
        identities = pd.concat(
            [cached, fetched],
            ignore_index=True,
        )
    else:
        print("Reusing complete MLB player identity cache.")
        identities = cached

    identities = (
        identities.drop_duplicates("batter", keep="last")
        .sort_values("batter")
        .reset_index(drop=True)
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    identities.to_csv(cache, index=False)
    return identities


def main() -> None:
    args = parse_args()

    if args.canada_min_pa < 0:
        raise ValueError("--canada-min-pa must be >= 0")
    if not args.profile.exists():
        raise FileNotFoundError(
            f"Missing canonical hitter profile: {args.profile}. "
            "Run build_hitter_profiles.py first."
        )
    if not args.plate_appearances.exists():
        raise FileNotFoundError(
            f"Missing plate-appearance dataset: {args.plate_appearances}."
        )

    profile = pd.read_csv(args.profile)
    if "batter" not in profile.columns:
        raise ValueError("Canonical hitter profile must contain batter")

    identities = _load_or_fetch_identities(
        profile,
        args.identity_cache,
        refresh=args.refresh_identities,
    )

    plate_appearances = pd.read_parquet(
        args.plate_appearances,
        columns=["batter", "batting_team"],
    )
    team_context = build_player_team_summary(
        plate_appearances,
        focus_team="TOR",
    )

    enriched = attach_canada_context(
        profile,
        identities,
        team_context,
    )
    canadian = build_canadian_hitter_view(
        enriched,
        min_pa=args.canada_min_pa,
    )
    leaderboard = build_canadian_leaderboard(
        canadian,
        sort_by=args.leaderboard_sort,
    )
    blue_jays = build_blue_jays_hitter_view(
        enriched,
        team_code="TOR",
    )

    DEFAULT_TEAM_CONTEXT.parent.mkdir(parents=True, exist_ok=True)
    team_context.to_csv(DEFAULT_TEAM_CONTEXT, index=False)
    enriched.to_csv(DEFAULT_ENRICHED, index=False)
    canadian.to_csv(DEFAULT_CANADIAN, index=False)
    leaderboard.to_csv(DEFAULT_CANADIAN_LEADERBOARD, index=False)
    blue_jays.to_csv(DEFAULT_BLUE_JAYS, index=False)

    print()
    print("NorthStar Canada layer V1")
    print(f"Canonical hitters: {len(enriched):,}")
    print(
        "Identity records resolved: "
        f"{int(enriched['birth_country'].notna().sum()):,}"
    )
    print(f"Canadian-born hitters: {len(canadian):,}")
    print(
        "Hitters with at least one Toronto PA in 2025: "
        f"{len(blue_jays):,}"
    )
    print(
        "Canadian definition: birth_country == Canada. "
        "This is birthplace, not citizenship."
    )
    print(
        "Blue Jays view: NorthStar diagnostics are full-season MLB values; "
        "TOR PA columns show team exposure."
    )

    if not leaderboard.empty:
        display = [
            column
            for column in [
                "canada_rank",
                "player_name",
                "primary_team_2025",
                "plate_appearances",
                args.leaderboard_sort,
                "selection_context_percentile",
                "contact_context_percentile",
                "damage_context_percentile",
            ]
            if column in leaderboard.columns
        ]
        print()
        print(
            f"Canadian-born leaderboard by {args.leaderboard_sort}"
        )
        print(
            leaderboard.loc[:, display]
            .head(20)
            .round(
                {
                    args.leaderboard_sort: 3,
                    "selection_context_percentile": 0,
                    "contact_context_percentile": 0,
                    "damage_context_percentile": 0,
                }
            )
            .to_string(index=False)
        )

    print()
    print(f"Saved identities:           {args.identity_cache}")
    print(f"Saved team context:         {DEFAULT_TEAM_CONTEXT}")
    print(f"Saved enriched profile:     {DEFAULT_ENRICHED}")
    print(f"Saved Canadian-born view:   {DEFAULT_CANADIAN}")
    print(f"Saved Canadian leaderboard: {DEFAULT_CANADIAN_LEADERBOARD}")
    print(f"Saved Blue Jays view:       {DEFAULT_BLUE_JAYS}")


if __name__ == "__main__":
    main()
