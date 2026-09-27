import pandas as pd
from fastapi.testclient import TestClient

from northstar.api.app import create_app
from northstar.api.store import ProfileStore


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batter": [10, 20, 30],
            "player_name": ["Alpha Canada", "Beta Toronto", "Gamma"],
            "birth_country": ["Canada", "USA", "Canada"],
            "birth_city": ["Toronto", "Boston", "Vancouver"],
            "birth_state_province": ["ON", "MA", "BC"],
            "primary_team_2025": ["NYY", "TOR", "SEA"],
            "teams_2025": ["NYY", "TOR;MIN", "SEA"],
            "season_plate_appearances": [600, 550, 500],
            "plate_appearances": [600, 550, 500],
            "canadian_born": [True, False, True],
            "selection_context_percentile": [90.0, 50.0, 20.0],
            "contact_context_percentile": [20.0, 60.0, 95.0],
            "damage_context_percentile": [98.0, 55.0, 10.0],
            "chase_avoidance_percentile": [85.0, 45.0, 30.0],
            "contact_rate_percentile": [25.0, 65.0, 99.0],
            "xwoba_on_contact_percentile": [99.0, 60.0, 5.0],
            "barrel_percentile": [98.0, 50.0, 4.0],
            "woba_percentile": [97.0, 55.0, 30.0],
            "xwoba_percentile": [98.0, 58.0, 25.0],
            "re24_percentile": [99.0, 50.0, 20.0],
            "woba": [0.420, 0.340, 0.310],
            "est_woba": [0.430, 0.350, 0.305],
            "re24_per_100_pa": [7.0, 2.0, -1.0],
        }
    )


def _client() -> TestClient:
    store = ProfileStore(_profile())
    return TestClient(create_app(store=store))


def test_health_and_meta():
    client = _client()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    meta = client.get("/api/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["players"] == 3
    assert "composite player score" in meta.json()["scope_notes"]["score"]


def test_player_search_supports_name_team_and_canadian_scope():
    client = _client()

    search = client.get(
        "/api/v1/players",
        params={"query": "alpha"},
    )
    assert search.status_code == 200
    assert search.json()["items"][0]["batter"] == 10

    toronto = client.get(
        "/api/v1/players",
        params={"team": "TOR"},
    )
    assert [item["batter"] for item in toronto.json()["items"]] == [20]

    canadian = client.get(
        "/api/v1/players",
        params={"canadian_born": "true"},
    )
    assert {item["batter"] for item in canadian.json()["items"]} == {10, 30}


def test_player_detail_and_unknown_player():
    client = _client()

    detail = client.get("/api/v1/players/10")
    assert detail.status_code == 200
    assert detail.json()["player_name"] == "Alpha Canada"

    missing = client.get("/api/v1/players/999")
    assert missing.status_code == 404


def test_canada_and_team_endpoints_keep_scopes_distinct():
    client = _client()

    canadian = client.get("/api/v1/canada")
    assert canadian.status_code == 200
    assert {item["batter"] for item in canadian.json()["items"]} == {10, 30}

    toronto = client.get("/api/v1/teams/TOR")
    assert toronto.status_code == 200
    payload = toronto.json()
    assert payload["metric_scope"] == "full-season MLB"
    assert [item["batter"] for item in payload["items"]] == [20]


def test_leaderboard_uses_named_metric_and_scope():
    client = _client()

    response = client.get(
        "/api/v1/leaderboards",
        params={
            "scope": "canadian",
            "sort_by": "re24_per_100_pa",
            "limit": 10,
        },
    )
    assert response.status_code == 200
    items = response.json()["items"]

    assert [item["batter"] for item in items] == [10, 30]
    assert items[0]["rank"] == 1
    assert items[0]["ranked_by"] == "re24_per_100_pa"


def test_compare_returns_two_to_four_profiles_without_winner():
    client = _client()

    response = client.get(
        "/api/v1/compare",
        params=[
            ("player", "10"),
            ("player", "Gamma"),
        ],
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["count"] == 2
    assert [item["batter"] for item in payload["players"]] == [10, 30]
    assert "no comparison winner" in payload["interpretation"]

    invalid = client.get(
        "/api/v1/compare",
        params={"player": "10"},
    )
    assert invalid.status_code == 422
