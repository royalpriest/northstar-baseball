from __future__ import annotations

import pandas as pd


def normalize_player_name(value: object) -> str:
    """Normalize player names for forgiving user-facing lookup."""
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    words = "".join(
        character.casefold() if character.isalnum() else " "
        for character in without_marks
    )
    return " ".join(words.split())


def player_name_matches(name: object, query: object) -> bool:
    """Match a full, first, last, or partial player-name query."""
    normalized_name = normalize_player_name(name)
    normalized_query = normalize_player_name(query)
    if not normalized_query:
        return False
    if normalized_name == normalized_query:
        return True

    compact_name = normalized_name.replace(" ", "")
    compact_query = normalized_query.replace(" ", "")
    if compact_query and compact_query in compact_name:
        return True

    name_tokens = normalized_name.split()
    query_tokens = normalized_query.split()
    return all(
        any(query_token in name_token for name_token in name_tokens)
        for query_token in query_tokens
    )


def _percentile(
    series: pd.Series,
    *,
    higher_is_better: bool = True,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    percentile = numeric.rank(method="average", pct=True) * 100.0

    if not higher_is_better:
        percentile = 100.0 - percentile + (
            100.0 / numeric.notna().sum()
            if numeric.notna().sum()
            else 0.0
        )

    return percentile.clip(lower=0.0, upper=100.0)


def _coalesce_player_names(frame: pd.DataFrame) -> pd.DataFrame:
    name_columns = [
        column
        for column in frame.columns
        if column == "player_name" or column.startswith("player_name_")
    ]
    if not name_columns:
        frame["player_name"] = pd.NA
        return frame

    name = frame[name_columns[0]].copy()
    for column in name_columns[1:]:
        name = name.fillna(frame[column])

    frame["player_name"] = name

    drop_columns = [
        column
        for column in name_columns
        if column != "player_name"
    ]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)

    return frame


def build_hitter_profile(
    selection: pd.DataFrame,
    contact: pd.DataFrame,
    damage: pd.DataFrame,
    discipline: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the canonical NorthStar hitter-profile table.

    The profile deliberately keeps selection, contact, and damage as separate
    diagnostics. No overall composite score is created.
    """
    for label, frame in [
        ("selection", selection),
        ("contact", contact),
        ("damage", damage),
    ]:
        if "batter" not in frame.columns:
            raise ValueError(f"{label} table must contain batter")

    selection_columns = [
        column
        for column in [
            "batter",
            "player_name",
            "eligible_decisions",
            "discretionary_decisions",
            "raw_centered_per_100",
            "shrunk_centered_per_100",
            "shrinkage_factor",
            "approx_posterior_se_per_100",
            "approx_ci_low_per_100",
            "approx_ci_high_per_100",
        ]
        if column in selection.columns
    ]
    contact_columns = [
        column
        for column in [
            "batter",
            "player_name",
            "swings",
            "expected_contact_rate",
            "contact_above_expected_per_100_swings",
            "statcast_contact_rate",
            "statcast_whiff_per_swing",
            "statcast_zone_contact_rate",
            "statcast_chase_contact_rate",
            "statcast_two_strike_contact_rate",
        ]
        if column in contact.columns
    ]
    damage_columns = [
        column
        for column in [
            "batter",
            "player_name",
            "batted_balls",
            "expected_damage_xwoba",
            "damage_xwoba_points_above_expected",
            "statcast_xwoba_on_contact",
            "statcast_avg_exit_velocity",
            "statcast_hard_hit_rate",
            "statcast_barrel_rate",
            "statcast_sweet_spot_rate",
        ]
        if column in damage.columns
    ]

    profile = (
        selection.loc[:, selection_columns]
        .drop_duplicates("batter")
        .merge(
            contact.loc[:, contact_columns].drop_duplicates("batter"),
            on="batter",
            how="outer",
            suffixes=("_selection", "_contact"),
        )
        .merge(
            damage.loc[:, damage_columns].drop_duplicates("batter"),
            on="batter",
            how="outer",
            suffixes=("", "_damage"),
        )
    )
    profile = _coalesce_player_names(profile)

    if discipline is not None and not discipline.empty:
        if "batter" not in discipline.columns:
            raise ValueError("discipline must contain batter")

        discipline_columns = [
            column
            for column in [
                "batter",
                "player_name",
                "statcast_swing_rate",
                "statcast_zone_swing_rate",
                "statcast_chase_rate",
                "statcast_first_pitch_swing_rate",
                "statcast_two_strike_swing_rate",
                "statcast_two_strike_chase_rate",
            ]
            if column in discipline.columns
        ]
        profile = profile.merge(
            discipline.loc[:, discipline_columns].drop_duplicates("batter"),
            on="batter",
            how="left",
            suffixes=("", "_discipline"),
        )
        profile = _coalesce_player_names(profile)

    if outcomes is not None and not outcomes.empty:
        reference = outcomes.copy()
        if "batter" not in reference.columns:
            if "player_id" not in reference.columns:
                raise ValueError(
                    "outcomes must contain batter or player_id"
                )
            reference["batter"] = pd.to_numeric(
                reference["player_id"],
                errors="coerce",
            )

        reference["batter"] = pd.to_numeric(
            reference["batter"],
            errors="coerce",
        )
        reference = reference.dropna(subset=["batter"]).copy()
        reference["batter"] = reference["batter"].astype("int64")

        outcome_columns = [
            column
            for column in [
                "batter",
                "player_name",
                "plate_appearances",
                "re24_per_100_pa",
                "positive_pa_pct",
                "ba",
                "est_ba",
                "slg",
                "est_slg",
                "woba",
                "est_woba",
            ]
            if column in reference.columns
        ]

        profile = profile.merge(
            reference.loc[:, outcome_columns].drop_duplicates("batter"),
            on="batter",
            how="left",
            suffixes=("", "_outcome"),
        )
        profile = _coalesce_player_names(profile)

    profile["batter"] = pd.to_numeric(
        profile["batter"],
        errors="coerce",
    ).astype("Int64")
    profile = profile.loc[profile["batter"].notna()].copy()
    profile["batter"] = profile["batter"].astype("int64")

    profile["has_selection"] = profile.get(
        "shrunk_centered_per_100",
        pd.Series(index=profile.index, dtype="float64"),
    ).notna()
    profile["has_contact"] = profile.get(
        "contact_above_expected_per_100_swings",
        pd.Series(index=profile.index, dtype="float64"),
    ).notna()
    profile["has_damage"] = profile.get(
        "damage_xwoba_points_above_expected",
        pd.Series(index=profile.index, dtype="float64"),
    ).notna()
    profile["complete_three_axis_profile"] = (
        profile["has_selection"]
        & profile["has_contact"]
        & profile["has_damage"]
    )

    percentile_specs = {
        "selection_context_percentile": (
            "shrunk_centered_per_100",
            True,
        ),
        "contact_context_percentile": (
            "contact_above_expected_per_100_swings",
            True,
        ),
        "damage_context_percentile": (
            "damage_xwoba_points_above_expected",
            True,
        ),
        "swing_rate_percentile": (
            "statcast_swing_rate",
            True,
        ),
        "zone_swing_rate_percentile": (
            "statcast_zone_swing_rate",
            True,
        ),
        "first_pitch_swing_rate_percentile": (
            "statcast_first_pitch_swing_rate",
            True,
        ),
        "two_strike_swing_rate_percentile": (
            "statcast_two_strike_swing_rate",
            True,
        ),
        "contact_rate_percentile": (
            "statcast_contact_rate",
            True,
        ),
        "zone_contact_percentile": (
            "statcast_zone_contact_rate",
            True,
        ),
        "chase_avoidance_percentile": (
            "statcast_chase_rate",
            False,
        ),
        "two_strike_chase_avoidance_percentile": (
            "statcast_two_strike_chase_rate",
            False,
        ),
        "chase_contact_percentile": (
            "statcast_chase_contact_rate",
            True,
        ),
        "two_strike_contact_percentile": (
            "statcast_two_strike_contact_rate",
            True,
        ),
        "xwoba_on_contact_percentile": (
            "statcast_xwoba_on_contact",
            True,
        ),
        "exit_velocity_percentile": (
            "statcast_avg_exit_velocity",
            True,
        ),
        "hard_hit_percentile": (
            "statcast_hard_hit_rate",
            True,
        ),
        "barrel_percentile": (
            "statcast_barrel_rate",
            True,
        ),
        "sweet_spot_percentile": (
            "statcast_sweet_spot_rate",
            True,
        ),
        "woba_percentile": (
            "woba",
            True,
        ),
        "xwoba_percentile": (
            "est_woba",
            True,
        ),
        "re24_percentile": (
            "re24_per_100_pa",
            True,
        ),
        "expected_contact_percentile": (
            "expected_contact_rate",
            True,
        ),
        "expected_damage_percentile": (
            "expected_damage_xwoba",
            True,
        ),
    }

    for output, (source, higher_is_better) in percentile_specs.items():
        if source in profile.columns:
            profile[output] = _percentile(
                profile[source],
                higher_is_better=higher_is_better,
            )

    front = [
        column
        for column in [
            "batter",
            "player_name",
            "complete_three_axis_profile",
            "eligible_decisions",
            "swings",
            "batted_balls",
            "plate_appearances",
            "shrunk_centered_per_100",
            "selection_context_percentile",
            "contact_above_expected_per_100_swings",
            "contact_context_percentile",
            "damage_xwoba_points_above_expected",
            "damage_context_percentile",
            "statcast_contact_rate",
            "contact_rate_percentile",
            "statcast_chase_rate",
            "chase_avoidance_percentile",
            "statcast_chase_contact_rate",
            "chase_contact_percentile",
            "statcast_xwoba_on_contact",
            "xwoba_on_contact_percentile",
            "woba",
            "woba_percentile",
            "est_woba",
            "xwoba_percentile",
            "re24_per_100_pa",
            "re24_percentile",
        ]
        if column in profile.columns
    ]
    remainder = [
        column for column in profile.columns if column not in front
    ]

    return profile.loc[:, front + remainder].sort_values(
        ["complete_three_axis_profile", "player_name", "batter"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def select_hitter_profile(
    profile: pd.DataFrame,
    player: str | int,
) -> pd.Series:
    """Select one hitter by MLBAM ID or a unique first/last/full-name query."""
    if "batter" not in profile.columns:
        raise ValueError("profile must contain batter")

    if isinstance(player, int) or str(player).strip().isdigit():
        batter = int(player)
        matches = profile.loc[profile["batter"].eq(batter)]
    else:
        if "player_name" not in profile.columns:
            raise ValueError("profile must contain player_name")

        query = normalize_player_name(player)
        names = profile["player_name"].fillna("").map(normalize_player_name)

        exact = profile.loc[names.eq(query)]
        if len(exact) == 1:
            return exact.iloc[0]

        matches = profile.loc[
            profile["player_name"]
            .fillna("")
            .map(lambda name: player_name_matches(name, query))
        ]

    if matches.empty:
        raise KeyError(f"No hitter matched {player!r}")
    if len(matches) > 1:
        options = ", ".join(
            matches["player_name"].astype(str).head(5).tolist()
        )
        raise ValueError(
            f"Multiple hitters matched {player!r}: {options}. "
            "Use more of the name or the MLBAM ID."
        )

    return matches.iloc[0]
