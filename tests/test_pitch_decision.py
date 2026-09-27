import pandas as pd
import pytest

from northstar.features.pitches import (
    build_pitch_decision_frame,
    classify_pitch_decision,
)
from northstar.models.pitch_decision import (
    chronological_pitch_split,
    decision_surprise,
    fit_count_zone_swing_baseline,
    predict_count_zone_swing_baseline,
)


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("swinging_strike", "swing"),
        ("foul", "swing"),
        ("hit_into_play", "swing"),
        ("missed_bunt", "swing"),
        ("ball", "take"),
        ("called_strike", "take"),
        ("hit_by_pitch", "take"),
        ("automatic_strike", None),
        ("automatic_ball", None),
    ],
)
def test_classify_pitch_decision(description, expected):
    assert classify_pitch_decision(description) == expected


def _sample_raw_pitches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [1, 1, 2],
            "game_date": ["2025-07-20", "2025-08-10", "2025-09-10"],
            "at_bat_number": [1, 2, 1],
            "pitch_number": [1, 1, 1],
            "batter": [10, 11, 12],
            "pitcher": [20, 21, 22],
            "balls": [0, 1, 2],
            "strikes": [0, 1, 2],
            "description": ["called_strike", "foul", "ball"],
            "stand": ["R", "L", "R"],
            "p_throws": ["R", "L", "R"],
            "pitch_type": ["FF", "SL", "CH"],
            "release_speed": [95.0, 85.0, 88.0],
            "plate_x": [0.4, 0.5, -0.2],
            "plate_z": [2.5, 2.0, 3.0],
            "sz_top": [3.5, 3.4, 3.6],
            "sz_bot": [1.5, 1.4, 1.6],
            "pfx_x": [0.2, -0.4, 0.1],
            "pfx_z": [1.2, 0.3, 0.7],
            "zone": [5, 7, 12],
            "inning": [1, 4, 7],
            "inning_topbot": ["Top", "Bottom", "Top"],
            "outs_when_up": [0, 1, 2],
            "on_1b": [None, 100, None],
            "on_2b": [None, None, 200],
            "on_3b": [None, None, None],
            "bat_score": [0, 2, 3],
            "fld_score": [0, 1, 4],
        }
    )


def test_build_pitch_decision_frame_derives_opportunity_features():
    frame = build_pitch_decision_frame(_sample_raw_pitches())

    assert frame["is_swing"].tolist() == [0, 1, 0]
    assert frame["base_state_before"].tolist() == ["000", "100", "010"]
    assert frame["score_diff_before"].tolist() == [0, 1, -1]
    assert frame.loc[1, "plate_x_batter"] == pytest.approx(-0.5)
    assert frame.loc[1, "pfx_x_batter"] == pytest.approx(0.4)


def test_chronological_pitch_split():
    frame = build_pitch_decision_frame(_sample_raw_pitches())
    split = chronological_pitch_split(frame)

    assert len(split.train) == 1
    assert len(split.validation) == 1
    assert len(split.test) == 1


def test_count_zone_baseline_uses_training_mapping():
    frame = build_pitch_decision_frame(_sample_raw_pitches())
    train = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
    mapping, fallback = fit_count_zone_swing_baseline(train)

    prediction = predict_count_zone_swing_baseline(frame.iloc[[0]], mapping, fallback)

    assert prediction.iloc[0] == pytest.approx(0.0)
    assert fallback == pytest.approx(0.0)


def test_decision_surprise_is_behavioral_residual():
    surprise = decision_surprise(
        pd.Series([1, 0]),
        pd.Series([0.7, 0.2]),
    )
    assert surprise.tolist() == pytest.approx([0.3, -0.2])
