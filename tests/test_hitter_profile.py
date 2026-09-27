import pandas as pd
import pytest

from northstar.profiles.hitter import (
    build_hitter_profile,
    select_hitter_profile,
)


def _frames():
    selection = pd.DataFrame(
        {
            "batter": [10, 20, 30],
            "player_name": ["Alpha", "Beta", "Gamma"],
            "eligible_decisions": [1000, 900, 800],
            "discretionary_decisions": [600, 550, 500],
            "shrunk_centered_per_100": [0.8, 0.1, -0.5],
            "approx_ci_low_per_100": [0.6, -0.1, -0.8],
            "approx_ci_high_per_100": [1.0, 0.3, -0.2],
        }
    )
    contact = pd.DataFrame(
        {
            "batter": [10, 20, 30],
            "player_name": ["Alpha", "Beta", "Gamma"],
            "swings": [700, 650, 600],
            "expected_contact_rate": [0.75, 0.78, 0.80],
            "contact_above_expected_per_100_swings": [4.0, 1.0, -2.0],
            "statcast_contact_rate": [0.82, 0.80, 0.76],
            "statcast_chase_contact_rate": [0.70, 0.65, 0.60],
        }
    )
    damage = pd.DataFrame(
        {
            "batter": [10, 20, 30],
            "player_name": ["Alpha", "Beta", "Gamma"],
            "batted_balls": [300, 280, 260],
            "expected_damage_xwoba": [0.36, 0.37, 0.38],
            "damage_xwoba_points_above_expected": [80.0, 20.0, -30.0],
            "statcast_xwoba_on_contact": [0.44, 0.39, 0.35],
            "statcast_barrel_rate": [0.16, 0.10, 0.05],
        }
    )
    discipline = pd.DataFrame(
        {
            "batter": [10, 20, 30],
            "player_name": ["Alpha", "Beta", "Gamma"],
            "statcast_swing_rate": [0.45, 0.48, 0.52],
            "statcast_chase_rate": [0.18, 0.25, 0.35],
        }
    )
    outcomes = pd.DataFrame(
        {
            "player_id": [10, 20, 30],
            "player_name": ["Alpha", "Beta", "Gamma"],
            "plate_appearances": [650, 620, 590],
            "re24_per_100_pa": [6.0, 2.0, -1.0],
            "woba": [0.400, 0.350, 0.310],
            "est_woba": [0.410, 0.345, 0.305],
        }
    )
    return selection, contact, damage, discipline, outcomes


def test_build_hitter_profile_keeps_axes_separate_and_adds_percentiles():
    selection, contact, damage, discipline, outcomes = _frames()

    profile = build_hitter_profile(
        selection,
        contact,
        damage,
        discipline=discipline,
        outcomes=outcomes,
    )

    assert len(profile) == 3
    assert profile["complete_three_axis_profile"].all()
    assert "selection_context_percentile" in profile.columns
    assert "contact_context_percentile" in profile.columns
    assert "damage_context_percentile" in profile.columns
    assert "chase_avoidance_percentile" in profile.columns
    assert "swing_rate_percentile" in profile.columns
    assert "overall_score" not in profile.columns

    alpha = profile.loc[profile["batter"].eq(10)].iloc[0]
    gamma = profile.loc[profile["batter"].eq(30)].iloc[0]

    assert alpha["selection_context_percentile"] > gamma[
        "selection_context_percentile"
    ]
    assert alpha["chase_avoidance_percentile"] > gamma[
        "chase_avoidance_percentile"
    ]


def test_build_hitter_profile_allows_partial_profiles():
    selection, contact, damage, discipline, outcomes = _frames()
    damage = damage.loc[damage["batter"].ne(30)].copy()

    profile = build_hitter_profile(
        selection,
        contact,
        damage,
        discipline=discipline,
        outcomes=outcomes,
    )

    gamma = profile.loc[profile["batter"].eq(30)].iloc[0]
    assert not bool(gamma["complete_three_axis_profile"])
    assert pd.isna(gamma["damage_xwoba_points_above_expected"])


def test_select_hitter_profile_supports_name_and_id():
    selection, contact, damage, discipline, outcomes = _frames()
    profile = build_hitter_profile(
        selection,
        contact,
        damage,
        discipline=discipline,
        outcomes=outcomes,
    )

    by_name = select_hitter_profile(profile, "alpha")
    by_id = select_hitter_profile(profile, 10)

    assert by_name["batter"] == 10
    assert by_id["player_name"] == "Alpha"


def test_select_hitter_profile_raises_for_unknown_player():
    selection, contact, damage, discipline, outcomes = _frames()
    profile = build_hitter_profile(
        selection,
        contact,
        damage,
        discipline=discipline,
        outcomes=outcomes,
    )

    with pytest.raises(KeyError):
        select_hitter_profile(profile, "Nobody")
