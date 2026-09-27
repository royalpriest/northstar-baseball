from __future__ import annotations

import pandas as pd


def build_player_team_summary(
    plate_appearances: pd.DataFrame,
    *,
    focus_team: str = "TOR",
) -> pd.DataFrame:
    """Summarize each hitter's 2025 batting-team exposure from PA data."""
    required = {"batter", "batting_team"}
    missing = required.difference(plate_appearances.columns)
    if missing:
        raise ValueError(
            f"Plate-appearance data is missing: {sorted(missing)}"
        )

    frame = plate_appearances.loc[:, ["batter", "batting_team"]].copy()
    frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce")
    frame["batting_team"] = (
        frame["batting_team"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    frame = frame.dropna(subset=["batter", "batting_team"])
    frame = frame.loc[frame["batting_team"].ne("")].copy()
    frame["batter"] = frame["batter"].astype("int64")

    counts = (
        frame.groupby(["batter", "batting_team"], as_index=False)
        .size()
        .rename(columns={"size": "team_plate_appearances"})
    )
    totals = (
        counts.groupby("batter", as_index=False)["team_plate_appearances"]
        .sum()
        .rename(
            columns={
                "team_plate_appearances": "statcast_plate_appearances"
            }
        )
    )

    ranked = counts.sort_values(
        ["batter", "team_plate_appearances", "batting_team"],
        ascending=[True, False, True],
    )
    primary = (
        ranked.drop_duplicates("batter")
        .rename(
            columns={
                "batting_team": "primary_team_2025",
                "team_plate_appearances": "primary_team_pa",
            }
        )
        .loc[:, ["batter", "primary_team_2025", "primary_team_pa"]]
    )

    team_lists = (
        ranked.groupby("batter", sort=False)
        .apply(
            lambda group: ";".join(
                group["batting_team"].astype(str).tolist()
            ),
            include_groups=False,
        )
        .rename("teams_2025")
        .reset_index()
    )

    focus = focus_team.strip().upper()
    focus_counts = (
        counts.loc[counts["batting_team"].eq(focus)]
        .rename(
            columns={
                "team_plate_appearances": f"{focus.lower()}_plate_appearances"
            }
        )
        .loc[:, ["batter", f"{focus.lower()}_plate_appearances"]]
    )

    summary = (
        totals.merge(primary, on="batter", how="left")
        .merge(team_lists, on="batter", how="left")
        .merge(focus_counts, on="batter", how="left")
    )
    focus_column = f"{focus.lower()}_plate_appearances"
    summary[focus_column] = (
        pd.to_numeric(summary[focus_column], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    summary[f"{focus.lower()}_pa_share"] = (
        summary[focus_column]
        / summary["statcast_plate_appearances"].replace(0, pd.NA)
    )
    summary[f"appeared_for_{focus.lower()}"] = (
        summary[focus_column] > 0
    )
    summary["primary_team_pa_share"] = (
        summary["primary_team_pa"]
        / summary["statcast_plate_appearances"].replace(0, pd.NA)
    )

    return summary.sort_values("batter").reset_index(drop=True)


def attach_canada_context(
    profile: pd.DataFrame,
    identities: pd.DataFrame,
    team_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach birthplace and season-team context to the canonical profile."""
    if "batter" not in profile.columns:
        raise ValueError("profile must contain batter")
    if "batter" not in identities.columns:
        raise ValueError("identities must contain batter")
    if "batter" not in team_summary.columns:
        raise ValueError("team_summary must contain batter")

    out = profile.copy()
    for frame in [out, identities, team_summary]:
        frame["batter"] = pd.to_numeric(
            frame["batter"],
            errors="coerce",
        )

    out = out.dropna(subset=["batter"]).copy()
    out["batter"] = out["batter"].astype("int64")

    identity = identities.dropna(subset=["batter"]).copy()
    identity["batter"] = identity["batter"].astype("int64")
    identity = identity.drop_duplicates("batter")

    teams = team_summary.dropna(subset=["batter"]).copy()
    teams["batter"] = teams["batter"].astype("int64")
    teams = teams.drop_duplicates("batter")

    out = (
        out.merge(identity, on="batter", how="left", validate="one_to_one")
        .merge(teams, on="batter", how="left", validate="one_to_one")
    )

    if "player_name" in out.columns and "mlb_full_name" in out.columns:
        missing_name = (
            out["player_name"].isna()
            | out["player_name"].astype("string").str.strip().eq("")
        )
        out.loc[missing_name, "player_name"] = out.loc[
            missing_name,
            "mlb_full_name",
        ]

    birth_country = out.get(
        "birth_country",
        pd.Series(index=out.index, dtype="string"),
    )
    out["canadian_born"] = (
        birth_country.astype("string").str.strip().str.casefold().eq("canada")
    )

    return out


def build_canadian_hitter_view(
    profile: pd.DataFrame,
    *,
    min_pa: int = 0,
) -> pd.DataFrame:
    """Return Canadian-born hitters, using birth country rather than citizenship."""
    if "canadian_born" not in profile.columns:
        raise ValueError("profile must contain canadian_born")

    out = profile.loc[profile["canadian_born"].fillna(False)].copy()

    if min_pa > 0 and "plate_appearances" in out.columns:
        pa = pd.to_numeric(out["plate_appearances"], errors="coerce")
        out = out.loc[pa.ge(min_pa)].copy()

    sort_columns = [
        column
        for column in ["re24_per_100_pa", "plate_appearances", "player_name"]
        if column in out.columns
    ]
    if sort_columns:
        ascending = [
            False if column != "player_name" else True
            for column in sort_columns
        ]
        out = out.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
        )

    out.insert(0, "canada_scope", "Canadian-born")
    return out.reset_index(drop=True)


def build_blue_jays_hitter_view(
    profile: pd.DataFrame,
    *,
    team_code: str = "TOR",
) -> pd.DataFrame:
    """Return hitters with at least one PA for the selected team in 2025.

    NorthStar diagnostic fields remain full-season MLB values. Team exposure
    columns make that scope explicit for traded players.
    """
    team = team_code.strip().upper()
    flag = f"appeared_for_{team.lower()}"
    if flag not in profile.columns:
        raise ValueError(f"profile must contain {flag}")

    out = profile.loc[profile[flag].fillna(False)].copy()
    team_pa = f"{team.lower()}_plate_appearances"
    sort_columns = [
        column
        for column in [team_pa, "plate_appearances", "player_name"]
        if column in out.columns
    ]
    if sort_columns:
        ascending = [
            False if column != "player_name" else True
            for column in sort_columns
        ]
        out = out.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
        )

    out.insert(
        0,
        "team_profile_scope",
        "Full-season MLB profile; team columns show 2025 exposure",
    )
    return out.reset_index(drop=True)


def build_canadian_leaderboard(
    canadian_profile: pd.DataFrame,
    *,
    sort_by: str = "re24_per_100_pa",
) -> pd.DataFrame:
    """Build a transparent Canadian-born leaderboard without a composite score."""
    if sort_by not in canadian_profile.columns:
        raise ValueError(f"Unknown leaderboard field: {sort_by}")

    out = canadian_profile.copy()
    out[sort_by] = pd.to_numeric(out[sort_by], errors="coerce")
    out = out.loc[out[sort_by].notna()].copy()
    out = out.sort_values(
        [sort_by, "player_name", "batter"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    out.insert(0, "canada_rank", range(1, len(out) + 1))
    out.insert(1, "ranked_by", sort_by)

    preferred = [
        "canada_rank",
        "ranked_by",
        "player_name",
        "batter",
        "birth_city",
        "birth_state_province",
        "birth_country",
        "primary_team_2025",
        "teams_2025",
        "plate_appearances",
        "re24_per_100_pa",
        "woba",
        "est_woba",
        "selection_context_percentile",
        "contact_context_percentile",
        "damage_context_percentile",
        "statcast_contact_rate",
        "statcast_xwoba_on_contact",
    ]
    columns = [
        column for column in preferred if column in out.columns
    ]
    return out.loc[:, columns]
