from __future__ import annotations

from typing import Any

import pandas as pd
import requests


FANGRAPHS_LEADERS_URL = "https://www.fangraphs.com/api/leaders/major-league/data"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36"
    ),
}


def fetch_fangraphs_batting(
    season: int,
    min_pa: int = 300,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch season-level MLB batting metrics from FanGraphs' modern JSON leaderboard."""
    if season < 1876:
        raise ValueError("season must be a valid major-league season")
    if min_pa < 1:
        raise ValueError("min_pa must be >= 1")

    params: dict[str, Any] = {
        "age": "",
        "pos": "all",
        "stats": "bat",
        "lg": "all",
        "qual": str(min_pa),
        "season": str(season),
        "season1": str(season),
        "startdate": "",
        "enddate": "",
        "month": "0",
        "hand": "",
        "team": "0",
        "pageitems": "2000000000",
        "pagenum": "1",
        "ind": "1",
        "rost": "0",
        "players": "",
        "type": "1",
        "postseason": "",
        "sortdir": "default",
        "sortstat": "wRC+",
    }

    client = session or requests.Session()
    response = client.get(
        FANGRAPHS_LEADERS_URL,
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=timeout,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            "FanGraphs modern leaderboard request failed "
            f"with HTTP {response.status_code}."
        ) from exc

    payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("FanGraphs response did not contain a data array.")

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("FanGraphs returned no batting rows.")

    # Current FanGraphs JSON includes the MLBAM ID directly. Prefer the clean
    # PlayerName field over the HTML-oriented Name field when available.
    if "PlayerName" in frame.columns:
        frame["Name"] = frame["PlayerName"]

    required = {"xMLBAMID", "Name", "PA", "OPS", "wOBA", "wRC+"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "FanGraphs response is missing required columns: "
            f"{sorted(missing)}"
        )

    frame["xMLBAMID"] = pd.to_numeric(frame["xMLBAMID"], errors="coerce")
    frame["PA"] = pd.to_numeric(frame["PA"], errors="coerce")
    frame = frame.dropna(subset=["xMLBAMID", "PA"])
    frame["xMLBAMID"] = frame["xMLBAMID"].astype("int64")

    # Keep the local qualification rule explicit even if FanGraphs also applies it.
    frame = frame.loc[frame["PA"].ge(min_pa)].copy()

    return frame.reset_index(drop=True)
