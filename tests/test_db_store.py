import pandas as pd
import pytest
from sqlalchemy import select

from northstar.db.database import (
    create_engine_from_url,
    create_schema,
    session_factory,
)
from northstar.db.load import replace_hitter_profiles
from northstar.db.models import HitterProfile, HitterTeamMembership
from northstar.db.store import DatabaseProfileStore


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batter": [10, 20, 30],
            "player_name": ["Alpha Canada", "Beta Toronto", "Luis Arráez"],
            "birth_country": ["Canada", "USA", "Venezuela"],
            "birth_city": ["Toronto", "Boston", "San Felipe"],
            "birth_state_province": ["ON", "MA", pd.NA],
            "primary_team_2025": ["NYY", "TOR", "SD"],
            "teams_2025": ["NYY", "TOR;MIN", "SD"],
            "season_plate_appearances": [600, 550, 500],
            "plate_appearances": [600.0, 550.0, 500.0],
            "canadian_born": [True, False, False],
            "selection_context_percentile": [90.0, 50.0, 4.0],
            "contact_context_percentile": [20.0, 60.0, 100.0],
            "damage_context_percentile": [98.0, 55.0, 6.0],
            "chase_avoidance_percentile": [85.0, 45.0, 16.0],
            "contact_rate_percentile": [25.0, 65.0, 100.0],
            "xwoba_on_contact_percentile": [99.0, 60.0, 3.0],
            "barrel_percentile": [98.0, 50.0, 2.0],
            "woba_percentile": [97.0, 55.0, 44.0],
            "xwoba_percentile": [98.0, 58.0, 28.0],
            "re24_percentile": [99.0, 50.0, 27.0],
            "woba": [0.420, 0.340, 0.313],
            "est_woba": [0.430, 0.350, 0.303],
            "re24_per_100_pa": [7.0, 2.0, -1.47],
        }
    )


def _store() -> tuple[DatabaseProfileStore, object]:
    engine = create_engine_from_url("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    sessions = session_factory(engine)

    with sessions() as session:
        inserted = replace_hitter_profiles(session, _profile(), season=2025)
        assert inserted == 3

    return DatabaseProfileStore(sessions, season=2025), sessions


def test_sql_loader_creates_profiles_and_normalized_team_memberships():
    store, sessions = _store()

    assert store.size == 3

    with sessions() as session:
        profiles = session.scalars(
            select(HitterProfile).order_by(HitterProfile.batter)
        ).all()
        memberships = session.scalars(
            select(HitterTeamMembership).order_by(
                HitterTeamMembership.batter,
                HitterTeamMembership.team_code,
            )
        ).all()

    assert [profile.batter for profile in profiles] == [10, 20, 30]
    assert [
        (item.batter, item.team_code, item.is_primary)
        for item in memberships
    ] == [
        (10, "NYY", True),
        (20, "MIN", False),
        (20, "TOR", True),
        (30, "SD", True),
    ]


def test_database_store_searches_name_team_and_canadian_scope():
    store, _ = _store()

    assert store.search_players(query="alpha")[0]["batter"] == 10
    assert store.search_players(team="TOR")[0]["batter"] == 20
    assert store.canadian_players()[0]["batter"] == 10


def test_database_store_returns_unicode_profile_and_comparison():
    store, _ = _store()

    player = store.get_player(30)
    assert player is not None
    assert player["player_name"] == "Luis Arráez"

    comparison = store.compare([10, "Luis Arráez"])
    assert [item["batter"] for item in comparison] == [10, 30]


def test_database_store_leaderboard_preserves_named_metric_scope():
    store, _ = _store()

    items = store.leaderboard(
        scope="all",
        sort_by="re24_per_100_pa",
        limit=10,
    )
    assert [item["batter"] for item in items] == [10, 20, 30]
    assert items[0]["rank"] == 1
    assert items[0]["ranked_by"] == "re24_per_100_pa"

    toronto = store.leaderboard(
        scope="team:TOR",
        sort_by="est_woba",
    )
    assert [item["batter"] for item in toronto] == [20]


def test_replace_hitter_profiles_is_idempotent_for_one_season():
    store, sessions = _store()
    updated = _profile().copy()
    updated.loc[updated["batter"].eq(10), "woba"] = 0.500

    with sessions() as session:
        inserted = replace_hitter_profiles(session, updated, season=2025)

    assert inserted == 3
    assert store.size == 3
    assert store.get_player(10)["woba"] == pytest.approx(0.500)
