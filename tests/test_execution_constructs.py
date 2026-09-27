import numpy as np
import pandas as pd
import pytest

from northstar.validation.execution_constructs import (
    build_contact_skill_summary,
    build_damage_skill_summary,
    merge_execution_construct_data,
    merge_half_execution_constructs,
)


def _sample_pitches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_date": [
                "2025-04-01",
                "2025-04-01",
                "2025-04-02",
                "2025-04-02",
                "2025-08-01",
                "2025-08-01",
            ],
            "batter": [10, 10, 10, 20, 20, 20],
            "description": [
                "foul",
                "swinging_strike",
                "hit_into_play",
                "swinging_strike_blocked",
                "hit_into_play",
                "foul_tip",
            ],
            "zone": [5, 12, 13, 5, 6, 14],
            "strikes": [0, 2, 2, 1, 0, 2],
            "launch_speed": [
                np.nan,
                np.nan,
                100.0,
                np.nan,
                92.0,
                np.nan,
            ],
            "launch_angle": [
                np.nan,
                np.nan,
                20.0,
                np.nan,
                10.0,
                np.nan,
            ],
            "launch_speed_angle": [
                np.nan,
                np.nan,
                6.0,
                np.nan,
                4.0,
                np.nan,
            ],
            "estimated_woba_using_speedangle": [
                np.nan,
                np.nan,
                0.70,
                np.nan,
                0.30,
                np.nan,
            ],
        }
    )


def test_contact_skill_summary_builds_situational_rates():
    summary = build_contact_skill_summary(_sample_pitches())
    player = summary.loc[summary["batter"].eq(10)].iloc[0]

    assert player["statcast_swings"] == 3
    assert player["statcast_contact_rate"] == pytest.approx(2 / 3)
    assert player["statcast_zone_contact_rate"] == pytest.approx(1.0)
    assert player["statcast_chase_contact_rate"] == pytest.approx(0.5)
    assert player["statcast_two_strike_contact_rate"] == pytest.approx(0.5)


def test_damage_skill_summary_builds_quality_metrics():
    summary = build_damage_skill_summary(_sample_pitches())

    player_10 = summary.loc[summary["batter"].eq(10)].iloc[0]
    assert player_10["statcast_batted_balls"] == 1
    assert player_10["statcast_avg_exit_velocity"] == pytest.approx(100.0)
    assert player_10["statcast_hard_hit_rate"] == pytest.approx(1.0)
    assert player_10["statcast_sweet_spot_rate"] == pytest.approx(1.0)
    assert player_10["statcast_barrel_rate"] == pytest.approx(1.0)
    assert player_10["statcast_xwoba_on_contact"] == pytest.approx(0.70)

    player_20 = summary.loc[summary["batter"].eq(20)].iloc[0]
    assert player_20["statcast_hard_hit_rate"] == pytest.approx(0.0)
    assert player_20["statcast_barrel_rate"] == pytest.approx(0.0)


def test_merge_execution_construct_data_adds_outcome_reference():
    execution = pd.DataFrame(
        {
            "batter": [10],
            "contact_above_expected_per_100_swings": [2.5],
        }
    )
    constructs = pd.DataFrame(
        {
            "batter": [10],
            "statcast_contact_rate": [0.85],
        }
    )
    outcomes = pd.DataFrame(
        {
            "player_id": [10],
            "re24_per_100_pa": [4.0],
            "woba": [0.350],
            "est_woba": [0.360],
        }
    )

    merged = merge_execution_construct_data(
        execution,
        constructs,
        outcome_reference=outcomes,
    )

    assert merged.loc[0, "statcast_contact_rate"] == pytest.approx(0.85)
    assert merged.loc[0, "re24_per_100_pa"] == pytest.approx(4.0)
    assert merged.loc[0, "est_woba"] == pytest.approx(0.360)


def test_merge_half_execution_constructs_uses_batter_and_half():
    execution = pd.DataFrame(
        {
            "batter": [10, 10],
            "season_half": ["early", "late"],
            "contact_above_expected_per_100_swings": [3.0, 2.0],
        }
    )
    constructs = pd.DataFrame(
        {
            "batter": [10, 10],
            "season_half": ["early", "late"],
            "statcast_contact_rate": [0.80, 0.85],
        }
    )

    merged = merge_half_execution_constructs(
        execution,
        constructs,
    )

    assert len(merged) == 2
    assert merged.loc[
        merged["season_half"].eq("late"),
        "statcast_contact_rate",
    ].iloc[0] == pytest.approx(0.85)
