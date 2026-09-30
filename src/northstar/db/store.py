from __future__ import annotations

from collections.abc import Iterable, Sequence

from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

from northstar.db.models import HitterProfile, HitterTeamMembership
from northstar.profiles.comparison import COMPARISON_FIELDS
from northstar.profiles.hitter import (
    normalize_player_name,
    player_name_matches,
)


PLAYER_SUMMARY_FIELDS = [
    "batter",
    "player_name",
    "birth_country",
    "birth_city",
    "birth_state_province",
    "primary_team_2025",
    "teams_2025",
    "season_plate_appearances",
    "plate_appearances",
    "canadian_born",
    "selection_context_percentile",
    "contact_context_percentile",
    "damage_context_percentile",
    "woba",
    "est_woba",
    "re24_per_100_pa",
]


SORT_COLUMNS = {
    "season_plate_appearances": HitterProfile.season_plate_appearances,
    "selection_context_percentile": HitterProfile.selection_context_percentile,
    "contact_context_percentile": HitterProfile.contact_context_percentile,
    "damage_context_percentile": HitterProfile.damage_context_percentile,
    "woba": HitterProfile.woba,
    "est_woba": HitterProfile.est_woba,
    "re24_per_100_pa": HitterProfile.re24_per_100_pa,
}


def _summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        field: payload.get(field)
        for field in PLAYER_SUMMARY_FIELDS
        if field in payload
    }


class DatabaseProfileStore:
    """SQL-backed implementation of the NorthStar profile query interface."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        season: int = 2025,
    ) -> None:
        self._sessions = sessions
        self._season = season

    @property
    def size(self) -> int:
        with self._sessions() as session:
            value = session.scalar(
                select(func.count())
                .select_from(HitterProfile)
                .where(HitterProfile.season == self._season)
            )
        return int(value or 0)

    def get_player(self, batter: int) -> dict[str, object] | None:
        with self._sessions() as session:
            model = session.get(
                HitterProfile,
                {"batter": int(batter), "season": self._season},
            )
            if model is None:
                return None
            return dict(model.profile_json)

    def _team_batters(
        self,
        session: Session,
        team: str,
    ) -> select:
        code = team.strip().upper()
        return (
            select(HitterTeamMembership.batter)
            .where(HitterTeamMembership.season == self._season)
            .where(HitterTeamMembership.team_code == code)
        )

    def search_players(
        self,
        *,
        query: str | None = None,
        team: str | None = None,
        canadian_born: bool | None = None,
        limit: int = 25,
    ) -> list[dict[str, object]]:
        statement = select(HitterProfile).where(
            HitterProfile.season == self._season
        )

        name_query: str | None = None
        if query is not None and query.strip():
            needle = query.strip()
            if needle.isdigit():
                statement = statement.where(
                    HitterProfile.batter == int(needle)
                )
            else:
                name_query = needle

        if canadian_born is not None:
            statement = statement.where(
                HitterProfile.canadian_born == canadian_born
            )

        with self._sessions() as session:
            if team is not None and team.strip():
                statement = statement.where(
                    HitterProfile.batter.in_(
                        self._team_batters(session, team)
                    )
                )

            statement = statement.order_by(
                desc(HitterProfile.season_plate_appearances),
                asc(HitterProfile.player_name),
            )

            if name_query is None:
                models = session.scalars(statement.limit(limit)).all()
            else:
                candidates = session.scalars(statement).all()
                models = [
                    model
                    for model in candidates
                    if player_name_matches(model.player_name, name_query)
                ][:limit]

            return [_summary(dict(model.profile_json)) for model in models]

    def canadian_players(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        return self.search_players(
            canadian_born=True,
            limit=limit,
        )

    def team_players(
        self,
        team: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        return self.search_players(team=team, limit=limit)

    def leaderboard(
        self,
        *,
        scope: str = "all",
        sort_by: str = "re24_per_100_pa",
        ascending: bool = False,
        min_pa: int = 0,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        if sort_by not in SORT_COLUMNS:
            raise KeyError(sort_by)

        metric = SORT_COLUMNS[sort_by]
        statement = (
            select(HitterProfile)
            .where(HitterProfile.season == self._season)
            .where(metric.is_not(None))
        )
        scope_normalized = scope.strip().casefold()

        with self._sessions() as session:
            if scope_normalized == "canadian":
                statement = statement.where(
                    HitterProfile.canadian_born.is_(True)
                )
            elif scope_normalized.startswith("team:"):
                code = scope.split(":", maxsplit=1)[1].strip().upper()
                if not code:
                    raise ValueError("team scope requires a team code")
                statement = statement.where(
                    HitterProfile.batter.in_(
                        self._team_batters(session, code)
                    )
                )
            elif scope_normalized != "all":
                raise ValueError(
                    "scope must be one of: all, canadian, team:<CODE>"
                )

            if min_pa > 0:
                statement = statement.where(
                    HitterProfile.season_plate_appearances >= min_pa
                )

            direction = asc(metric) if ascending else desc(metric)
            statement = statement.order_by(
                direction,
                asc(HitterProfile.player_name),
                asc(HitterProfile.batter),
            ).limit(limit)

            models = session.scalars(statement).all()

        records: list[dict[str, object]] = []
        for rank, model in enumerate(models, start=1):
            payload = _summary(dict(model.profile_json))
            payload["rank"] = rank
            payload["ranked_by"] = sort_by
            payload["scope"] = scope
            records.append(payload)
        return records

    def compare(
        self,
        players: Sequence[str | int],
    ) -> list[dict[str, object]]:
        if len(players) < 2:
            raise ValueError("At least two players are required")
        if len(players) > 4:
            raise ValueError("At most four players can be compared at once")

        records: list[dict[str, object]] = []
        seen: set[int] = set()

        with self._sessions() as session:
            for player in players:
                if isinstance(player, int) or str(player).strip().isdigit():
                    model = session.get(
                        HitterProfile,
                        {
                            "batter": int(player),
                            "season": self._season,
                        },
                    )
                else:
                    query = str(player).strip().casefold()
                    model = session.scalar(
                        select(HitterProfile)
                        .where(HitterProfile.season == self._season)
                        .where(
                            func.lower(HitterProfile.player_name) == query
                        )
                    )

                if model is None:
                    raise KeyError(f"No hitter matched {player!r}")
                if model.batter in seen:
                    raise ValueError(
                        f"Player {player!r} resolves to a duplicate hitter"
                    )

                seen.add(model.batter)
                payload = dict(model.profile_json)
                records.append(
                    {
                        field: payload.get(field)
                        for field in COMPARISON_FIELDS
                        if field in payload
                    }
                )

        return records

    def resolve_players(
        self,
        players: Iterable[str],
    ) -> list[str | int]:
        resolved: list[str | int] = []

        with self._sessions() as session:
            all_players = session.scalars(
                select(HitterProfile)
                .where(HitterProfile.season == self._season)
                .order_by(asc(HitterProfile.player_name))
            ).all()

            for player in players:
                value = str(player).strip()
                if not value:
                    continue
                if value.isdigit():
                    resolved.append(int(value))
                    continue

                normalized = normalize_player_name(value)
                exact = [
                    model
                    for model in all_players
                    if normalize_player_name(model.player_name) == normalized
                ]
                if len(exact) == 1:
                    resolved.append(int(exact[0].batter))
                    continue

                matches = [
                    model
                    for model in all_players
                    if player_name_matches(model.player_name, value)
                ]
                if not matches:
                    raise KeyError(f"No hitter matched {player!r}")
                if len(matches) > 1:
                    options = ", ".join(
                        model.player_name for model in matches[:5]
                    )
                    raise ValueError(
                        f"Multiple hitters matched {player!r}: {options}. "
                        "Use more of the name or the MLBAM ID."
                    )

                resolved.append(int(matches[0].batter))

        return resolved
