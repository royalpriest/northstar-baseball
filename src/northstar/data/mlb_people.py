from __future__ import annotations

import json
import time
from collections.abc import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
DEFAULT_FIELDS = (
    "people,id,fullName,birthCity,birthStateProvince,birthCountry"
)


def parse_people_payload(payload: dict[str, object]) -> pd.DataFrame:
    """Normalize an MLB Stats API people response into a small identity table."""
    people = payload.get("people", [])
    if not isinstance(people, list):
        raise ValueError("MLB people payload must contain a people list")

    rows: list[dict[str, object]] = []
    for person in people:
        if not isinstance(person, dict):
            continue

        player_id = pd.to_numeric(person.get("id"), errors="coerce")
        if pd.isna(player_id):
            continue

        rows.append(
            {
                "batter": int(player_id),
                "mlb_full_name": person.get("fullName"),
                "birth_city": person.get("birthCity"),
                "birth_state_province": person.get("birthStateProvince"),
                "birth_country": person.get("birthCountry"),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "batter",
            "mlb_full_name",
            "birth_city",
            "birth_state_province",
            "birth_country",
        ],
    )


def _chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_mlb_people(
    player_ids: Iterable[int],
    *,
    batch_size: int = 50,
    timeout: float = 30.0,
    retries: int = 2,
) -> pd.DataFrame:
    """Fetch MLB player identity/birthplace data for MLBAM IDs.

    The API call is intentionally isolated from the profile logic so tests can
    exercise the transformations without requiring network access.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if retries < 0:
        raise ValueError("retries must be >= 0")

    ids = sorted(
        {
            int(player_id)
            for player_id in player_ids
            if not pd.isna(player_id)
        }
    )
    if not ids:
        return parse_people_payload({"people": []})

    frames: list[pd.DataFrame] = []

    for batch_number, batch in enumerate(
        _chunks(ids, batch_size),
        start=1,
    ):
        query = urlencode(
            {
                "personIds": ",".join(str(value) for value in batch),
                "fields": DEFAULT_FIELDS,
            }
        )
        url = f"{MLB_PEOPLE_URL}?{query}"
        request = Request(
            url,
            headers={
                "User-Agent": "northstar-baseball/0.1",
                "Accept": "application/json",
            },
        )

        payload: dict[str, object] | None = None
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with urlopen(request, timeout=timeout) as response:
                    payload = json.loads(
                        response.read().decode("utf-8")
                    )
                break
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(0.75 * (attempt + 1))

        if payload is None:
            raise RuntimeError(
                "MLB people lookup failed for batch "
                f"{batch_number} after {retries + 1} attempt(s)."
            ) from last_error

        frames.append(parse_people_payload(payload))

    if not frames:
        return parse_people_payload({"people": []})

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("batter")
        .sort_values("batter")
        .reset_index(drop=True)
    )
