import pandas as pd

from northstar.features.plate_appearances import build_plate_appearances


def test_pa_uses_first_pitch_for_start_state_and_last_pitch_for_result():
    pitches = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2025-04-01",
                "inning": 1,
                "inning_topbot": "Top",
                "at_bat_number": 1,
                "pitch_number": 1,
                "outs_when_up": 0,
                "on_1b": 100,
                "on_2b": None,
                "on_3b": None,
                "bat_score": 0,
                "post_bat_score": 0,
                "batter": 10,
                "pitcher": 20,
                "events": None,
            },
            {
                "game_pk": 1,
                "game_date": "2025-04-01",
                "inning": 1,
                "inning_topbot": "Top",
                "at_bat_number": 1,
                "pitch_number": 2,
                "outs_when_up": 0,
                "on_1b": None,
                "on_2b": 100,
                "on_3b": None,
                "bat_score": 0,
                "post_bat_score": 1,
                "batter": 10,
                "pitcher": 20,
                "events": "double",
            },
            {
                "game_pk": 1,
                "game_date": "2025-04-01",
                "inning": 1,
                "inning_topbot": "Top",
                "at_bat_number": 2,
                "pitch_number": 1,
                "outs_when_up": 0,
                "on_1b": None,
                "on_2b": 10,
                "on_3b": None,
                "bat_score": 1,
                "post_bat_score": 1,
                "batter": 11,
                "pitcher": 20,
                "events": "field_out",
            },
        ]
    )

    pa = build_plate_appearances(pitches)
    first_pa = pa.iloc[0]

    assert first_pa["base_state_before"] == "100"
    assert first_pa["events"] == "double"
    assert first_pa["runs_scored_on_pa"] == 1
    assert first_pa["base_state_after"] == "010"



def test_pa_retains_optional_pre_pa_context_when_available():
    pitches = pd.DataFrame(
        [
            {
                "game_pk": 2,
                "game_date": "2025-04-02",
                "inning": 1,
                "inning_topbot": "Bottom",
                "at_bat_number": 1,
                "pitch_number": 1,
                "outs_when_up": 0,
                "on_1b": None,
                "on_2b": None,
                "on_3b": None,
                "bat_score": 2,
                "fld_score": 1,
                "post_bat_score": 2,
                "batter": 30,
                "pitcher": 40,
                "events": "field_out",
                "stand": "R",
                "p_throws": "L",
                "home_team": "TOR",
                "away_team": "BOS",
            }
        ]
    )

    pa = build_plate_appearances(pitches)
    row = pa.iloc[0]

    assert row["batter_hand"] == "R"
    assert row["pitcher_hand"] == "L"
    assert row["score_diff_before"] == 1
    assert row["batting_team"] == "TOR"
    assert row["fielding_team"] == "BOS"
    assert row["is_home_batting"] == 1
    assert row["park_team"] == "TOR"
