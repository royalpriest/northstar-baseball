from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from northstar.profiles.comparison import build_hitter_comparison


DEFAULT_PROFILE_PATH = Path(
    "reports/tables/hitter_profile_with_canada_2025.csv"
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


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        records.append(
            {
                str(key): _json_value(value)
                for key, value in record.items()
            }
        )
    return records


def row_to_record(row: pd.Series) -> dict[str, object]:
    return {
        str(key): _json_value(value)
        for key, value in row.to_dict().items()
    }


def _team_tokens(value: object) -> set[str]:
    if value is None:
        return set()
    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass

    return {
        token.strip().upper()
        for token in str(value).split(";")
        if token.strip()
    }


class ProfileStore:
    """In-memory query layer over NorthStar's canonical enriched hitter profile."""

    def __init__(self, profile: pd.DataFrame) -> None:
        if "batter" not in profile.columns:
            raise ValueError("profile must contain batter")
        if "player_name" not in profile.columns:
            raise ValueError("profile must contain player_name")

        frame = profile.copy()
        frame["batter"] = pd.to_numeric(
            frame["batter"],
            errors="coerce",
        )
        frame = frame.dropna(subset=["batter"]).copy()
        frame["batter"] = frame["batter"].astype("int64")
        frame["player_name"] = frame["player_name"].astype("string")
        frame = frame.drop_duplicates("batter", keep="last")
        self._profile = frame.reset_index(drop=True)

    @classmethod
    def from_csv(cls, path: Path = DEFAULT_PROFILE_PATH) -> "ProfileStore":
        if not path.exists():
            raise FileNotFoundError(
                f"Missing enriched hitter profile: {path}. "
                "Run build_canada_layer.py first."
            )
        return cls(pd.read_csv(path))

    @property
    def size(self) -> int:
        return len(self._profile)

    def profile_frame(self) -> pd.DataFrame:
        return self._profile.copy()

    def get_player(self, batter: int) -> dict[str, object] | None:
        matches = self._profile.loc[
            self._profile["batter"].eq(int(batter))
        ]
        if matches.empty:
            return None
        return row_to_record(matches.iloc[0])

    def search_players(
        self,
        *,
        query: str | None = None,
        team: str | None = None,
        canadian_born: bool | None = None,
        limit: int = 25,
    ) -> list[dict[str, object]]:
        frame = self._profile.copy()

        if query is not None and query.strip():
            needle = query.strip().casefold()
            id_match = pd.Series(False, index=frame.index)
            if needle.isdigit():
                id_match = frame["batter"].eq(int(needle))

            name_match = (
                frame["player_name"]
                .fillna("")
                .astype(str)
                .str.casefold()
                .str.contains(needle, regex=False)
            )
            frame = frame.loc[id_match | name_match].copy()

        if canadian_born is not None:
            if "canadian_born" not in frame.columns:
                frame = frame.iloc[0:0].copy()
            else:
                mask = (
                    frame["canadian_born"]
                    .fillna(False)
                    .astype(bool)
                    .eq(canadian_born)
                )
                frame = frame.loc[mask].copy()

        if team is not None and team.strip():
            code = team.strip().upper()
            if "teams_2025" not in frame.columns:
                frame = frame.iloc[0:0].copy()
            else:
                mask = frame["teams_2025"].map(
                    lambda value: code in _team_tokens(value)
                )
                frame = frame.loc[mask].copy()

        sort_columns = [
            column
            for column in [
                "season_plate_appearances",
                "plate_appearances",
                "player_name",
            ]
            if column in frame.columns
        ]
        if sort_columns:
            ascending = [
                column == "player_name"
                for column in sort_columns
            ]
            frame = frame.sort_values(
                sort_columns,
                ascending=ascending,
                na_position="last",
            )

        fields = [
            column
            for column in PLAYER_SUMMARY_FIELDS
            if column in frame.columns
        ]
        return frame_to_records(frame.loc[:, fields].head(limit))

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
        return self.search_players(
            team=team,
            limit=limit,
        )

    def leaderboard(
        self,
        *,
        scope: str = "all",
        sort_by: str = "re24_per_100_pa",
        ascending: bool = False,
        min_pa: int = 0,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        if sort_by not in self._profile.columns:
            raise KeyError(sort_by)

        frame = self._profile.copy()
        scope_normalized = scope.strip().casefold()

        if scope_normalized == "canadian":
            if "canadian_born" not in frame.columns:
                frame = frame.iloc[0:0].copy()
            else:
                frame = frame.loc[
                    frame["canadian_born"].fillna(False).astype(bool)
                ].copy()
        elif scope_normalized.startswith("team:"):
            team_code = scope.split(":", maxsplit=1)[1].strip().upper()
            if not team_code:
                raise ValueError("team scope requires a team code")
            if "teams_2025" not in frame.columns:
                frame = frame.iloc[0:0].copy()
            else:
                frame = frame.loc[
                    frame["teams_2025"].map(
                        lambda value: team_code in _team_tokens(value)
                    )
                ].copy()
        elif scope_normalized != "all":
            raise ValueError(
                "scope must be one of: all, canadian, team:<CODE>"
            )

        pa_column = (
            "season_plate_appearances"
            if "season_plate_appearances" in frame.columns
            else "plate_appearances"
        )
        if min_pa > 0 and pa_column in frame.columns:
            pa = pd.to_numeric(frame[pa_column], errors="coerce")
            frame = frame.loc[pa.ge(min_pa)].copy()

        frame[sort_by] = pd.to_numeric(
            frame[sort_by],
            errors="coerce",
        )
        frame = frame.loc[frame[sort_by].notna()].copy()
        frame = frame.sort_values(
            [sort_by, "player_name", "batter"],
            ascending=[ascending, True, True],
            na_position="last",
        ).head(limit)

        fields = [
            column
            for column in PLAYER_SUMMARY_FIELDS + [sort_by]
            if column in frame.columns
        ]
        # Preserve order while de-duplicating fields.
        fields = list(dict.fromkeys(fields))
        records = frame_to_records(frame.loc[:, fields])

        for rank, record in enumerate(records, start=1):
            record["rank"] = rank
            record["ranked_by"] = sort_by
            record["scope"] = scope

        return records

    def compare(
        self,
        players: Sequence[str | int],
    ) -> list[dict[str, object]]:
        comparison = build_hitter_comparison(
            self._profile,
            players,
        )
        return frame_to_records(comparison)

    def resolve_players(
        self,
        players: Iterable[str],
    ) -> list[str | int]:
        resolved: list[str | int] = []
        for player in players:
            value = str(player).strip()
            if not value:
                continue
            resolved.append(int(value) if value.isdigit() else value)
        return resolved
