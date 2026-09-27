import pandas as pd
import pytest

from northstar.data.mlb_people import parse_people_payload
from northstar.profiles.canada import (
    attach_canada_context,
    build_blue_jays_hitter_view,
    build_canadian_hitter_view,
    build_canadian_leaderboard,
    build_player_team_summary,
)


def test_parse_people_payload_normalizes_identity_fields():
    frame = parse_people_payload(
        {
            "people": [
                {
                    "id": 10,
                    "fullName": "Alpha Hitter",
                    "birthCity": "Toronto",
                    "birthStateProvince": "Ontario",
                    "birthCountry": "Canada",
                },
                {
                    "id": 20,
                    "fullName": "Beta Hitter",
                    "birthCountry": "USA",
                },
            ]
        }
    )

    assert frame["batter"].tolist() == [10, 20]
    assert frame.loc[0, "birth_country"] == "Canada"
    assert frame.loc[0, "birth_state_province"] == "Ontario"


def test_build_player_team_summary_tracks_traded_players_and_toronto_exposure():
    pa = pd.DataFrame(
        {
            "batter": [10, 10, 10, 20, 20, 30],
            "batting_team": ["TOR", "TOR", "NYY", "BOS", "BOS", "TOR"],
        }
    )

    summary = build_player_team_summary(pa, focus_team="TOR")

    player_10 = summary.loc[summary["batter"].eq(10)].iloc[0]
    assert player_10["primary_team_2025"] == "TOR"
    assert player_10["primary_team_pa"] == 2
    assert player_10["tor_plate_appearances"] == 2
    assert player_10["tor_pa_share"] == pytest.approx(2 / 3)
    assert player_10["teams_2025"] == "TOR;NYY"

    player_20 = summary.loc[summary["batter"].eq(20)].iloc[0]
    assert not bool(player_20["appeared_for_tor"])


def test_attach_canada_context_uses_birth_country_not_name_or_team():
    profile = pd.DataFrame(
        {
            "batter": [10, 20],
            "player_name": ["Alpha", "Beta"],
            "re24_per_100_pa": [4.0, 2.0],
        }
    )
    identities = pd.DataFrame(
        {
            "batter": [10, 20],
            "mlb_full_name": ["Alpha", "Beta"],
            "birth_country": ["Canada", "USA"],
            "birth_city": ["Toronto", "Boston"],
            "birth_state_province": ["Ontario", "Massachusetts"],
        }
    )
    teams = pd.DataFrame(
        {
            "batter": [10, 20],
            "primary_team_2025": ["NYY", "TOR"],
            "primary_team_pa": [500, 500],
            "statcast_plate_appearances": [500, 500],
            "teams_2025": ["NYY", "TOR"],
            "tor_plate_appearances": [0, 500],
            "tor_pa_share": [0.0, 1.0],
            "appeared_for_tor": [False, True],
            "primary_team_pa_share": [1.0, 1.0],
        }
    )

    enriched = attach_canada_context(profile, identities, teams)

    alpha = enriched.loc[enriched["batter"].eq(10)].iloc[0]
    beta = enriched.loc[enriched["batter"].eq(20)].iloc[0]

    assert bool(alpha["canadian_born"])
    assert not bool(beta["canadian_born"])


def test_canadian_and_blue_jays_views_have_distinct_membership_rules():
    profile = pd.DataFrame(
        {
            "batter": [10, 20, 30],
            "player_name": ["Canada Elsewhere", "US Toronto", "Canada Toronto"],
            "birth_country": ["Canada", "USA", "Canada"],
            "canadian_born": [True, False, True],
            "appeared_for_tor": [False, True, True],
            "tor_plate_appearances": [0, 400, 150],
            "plate_appearances": [600, 550, 300],
            "re24_per_100_pa": [4.0, 3.0, 1.0],
        }
    )

    canadian = build_canadian_hitter_view(profile)
    blue_jays = build_blue_jays_hitter_view(profile)

    assert set(canadian["batter"]) == {10, 30}
    assert set(blue_jays["batter"]) == {20, 30}
    assert canadian["canada_scope"].eq("Canadian-born").all()
    assert blue_jays["team_profile_scope"].str.contains(
        "Full-season MLB"
    ).all()


def test_canadian_leaderboard_uses_named_metric_not_composite_score():
    canadian = pd.DataFrame(
        {
            "player_name": ["Alpha", "Beta"],
            "batter": [10, 20],
            "birth_country": ["Canada", "Canada"],
            "primary_team_2025": ["TOR", "SEA"],
            "plate_appearances": [600, 500],
            "re24_per_100_pa": [2.0, 5.0],
            "selection_context_percentile": [80.0, 20.0],
            "contact_context_percentile": [40.0, 90.0],
            "damage_context_percentile": [70.0, 30.0],
        }
    )

    leaderboard = build_canadian_leaderboard(
        canadian,
        sort_by="re24_per_100_pa",
    )

    assert leaderboard.loc[0, "player_name"] == "Beta"
    assert leaderboard.loc[0, "canada_rank"] == 1
    assert leaderboard.loc[0, "ranked_by"] == "re24_per_100_pa"
    assert "overall_score" not in leaderboard.columns
