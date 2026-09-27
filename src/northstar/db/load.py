from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from northstar.db.models import HitterProfile, HitterTeamMembership


def _json_value(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _record(row: pd.Series) -> dict[str, object]:
    return {
        str(key): _json_value(value)
        for key, value in row.to_dict().items()
    }


def _as_int(value: object) -> int | None:
    value = _json_value(value)
    return None if value is None else int(value)


def _as_float(value: object) -> float | None:
    value = _json_value(value)
    return None if value is None else float(value)


def _as_text(value: object) -> str | None:
    value = _json_value(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: object) -> bool:
    value = _json_value(value)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _teams(value: object) -> list[str]:
    text = _as_text(value)
    if text is None:
        return []
    return list(
        dict.fromkeys(
            token.strip().upper()
            for token in text.split(";")
            if token.strip()
        )
    )


def replace_hitter_profiles(
    session: Session,
    profile: pd.DataFrame,
    *,
    season: int = 2025,
) -> int:
    """Replace one season of product profiles in a single transaction."""
    required = {"batter", "player_name"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"profile is missing required columns: {sorted(missing)}")

    session.execute(
        delete(HitterTeamMembership).where(
            HitterTeamMembership.season == season
        )
    )
    session.execute(
        delete(HitterProfile).where(HitterProfile.season == season)
    )

    inserted = 0
    for _, row in profile.iterrows():
        batter = _as_int(row.get("batter"))
        player_name = _as_text(row.get("player_name"))
        if batter is None or player_name is None:
            continue

        payload = _record(row)
        payload["batter"] = batter
        payload["player_name"] = player_name

        current = HitterProfile(
            batter=batter,
            season=season,
            player_name=player_name,
            birth_country=_as_text(row.get("birth_country")),
            birth_city=_as_text(row.get("birth_city")),
            birth_state_province=_as_text(row.get("birth_state_province")),
            canadian_born=_as_bool(row.get("canadian_born")),
            primary_team_2025=_as_text(row.get("primary_team_2025")),
            teams_2025=_as_text(row.get("teams_2025")),
            season_plate_appearances=_as_int(
                row.get("season_plate_appearances")
            ),
            selection_context_percentile=_as_float(
                row.get("selection_context_percentile")
            ),
            contact_context_percentile=_as_float(
                row.get("contact_context_percentile")
            ),
            damage_context_percentile=_as_float(
                row.get("damage_context_percentile")
            ),
            woba=_as_float(row.get("woba")),
            est_woba=_as_float(row.get("est_woba")),
            re24_per_100_pa=_as_float(row.get("re24_per_100_pa")),
            profile_json=payload,
        )
        session.add(current)

        primary = _as_text(row.get("primary_team_2025"))
        for team_code in _teams(row.get("teams_2025")):
            session.add(
                HitterTeamMembership(
                    batter=batter,
                    season=season,
                    team_code=team_code,
                    is_primary=(
                        primary is not None
                        and team_code == primary.upper()
                    ),
                )
            )
        inserted += 1

    session.commit()
    return inserted


def load_profile_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def available_teams(records: Iterable[dict[str, object]]) -> set[str]:
    teams: set[str] = set()
    for record in records:
        teams.update(_teams(record.get("teams_2025")))
    return teams
