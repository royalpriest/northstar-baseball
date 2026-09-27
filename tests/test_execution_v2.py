import numpy as np
import pandas as pd
import pytest

from northstar.metrics.execution_v2 import (
    CONTACT_RESIDUAL,
    DAMAGE_RESIDUAL,
    attach_contact_residual,
    attach_damage_residual,
    build_contact_frame,
    build_contact_half_reliability,
    build_damage_frame,
    build_damage_half_reliability,
    build_player_contact_summary,
    build_player_damage_summary,
    fit_context_mean_baseline,
    predict_context_mean_baseline,
)


def _sample_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_date": [
                "2025-04-01",
                "2025-04-02",
                "2025-08-01",
                "2025-08-02",
                "2025-04-01",
                "2025-04-02",
                "2025-08-01",
                "2025-08-02",
            ],
            "game_pk": [1, 2, 3, 4, 5, 6, 7, 8],
            "batter": [10, 10, 10, 10, 20, 20, 20, 20],
            "is_swing": [1] * 8,
            "description": [
                "foul",
                "swinging_strike",
                "hit_into_play",
                "foul_tip",
                "swinging_strike_blocked",
                "hit_into_play",
                "foul",
                "hit_into_play_score",
            ],
            "balls": [0, 1, 0, 1, 0, 1, 0, 1],
            "strikes": [0, 1, 0, 1, 0, 1, 0, 1],
            "zone": [5, 5, 5, 5, 12, 12, 12, 12],
            "pitch_type": ["FF"] * 4 + ["SL"] * 4,
            "estimated_woba_using_speedangle": [
                np.nan,
                np.nan,
                0.45,
                np.nan,
                np.nan,
                0.25,
                np.nan,
                0.60,
            ],
        }
    )


def test_contact_frame_labels_whiffs_and_contact():
    contact = build_contact_frame(_sample_decisions())

    assert contact["is_contact"].tolist() == [
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        1,
    ]


def test_damage_frame_keeps_batted_balls_with_target():
    damage = build_damage_frame(_sample_decisions())

    assert len(damage) == 3
    assert damage["estimated_woba_using_speedangle"].tolist() == pytest.approx(
        [0.45, 0.25, 0.60]
    )


def test_context_baseline_falls_back_by_pitch_type():
    frame = _sample_decisions().copy()
    frame["target"] = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]

    mapping, fallback, overall = fit_context_mean_baseline(
        frame,
        "target",
    )
    novel = pd.DataFrame(
        {
            "balls": [3],
            "strikes": [2],
            "zone": [14],
            "pitch_type": ["FF"],
        }
    )
    prediction = predict_context_mean_baseline(
        novel,
        mapping,
        fallback,
        overall,
    )

    assert prediction[0] == pytest.approx(0.75)


def test_contact_residual_and_player_summary():
    contact = build_contact_frame(_sample_decisions())
    scored = attach_contact_residual(
        contact,
        np.array([0.5] * len(contact)),
    )

    assert scored.loc[0, CONTACT_RESIDUAL] == pytest.approx(0.5)
    assert scored.loc[1, CONTACT_RESIDUAL] == pytest.approx(-0.5)

    summary = build_player_contact_summary(scored, min_swings=1)
    player_10 = summary.loc[summary["batter"].eq(10)].iloc[0]

    assert player_10["swings"] == 4
    assert player_10["observed_contact_rate"] == pytest.approx(0.75)
    assert player_10[
        "contact_above_expected_per_100_swings"
    ] == pytest.approx(25.0)


def test_damage_residual_and_player_summary():
    damage = build_damage_frame(_sample_decisions())
    scored = attach_damage_residual(
        damage,
        np.array([0.30, 0.30, 0.30]),
    )

    assert scored.iloc[0][DAMAGE_RESIDUAL] == pytest.approx(0.15)

    summary = build_player_damage_summary(
        scored,
        min_batted_balls=1,
    )
    player_20 = summary.loc[summary["batter"].eq(20)].iloc[0]

    assert player_20["batted_balls"] == 2
    assert player_20[
        "damage_xwoba_points_above_expected"
    ] == pytest.approx(125.0)


def test_contact_and_damage_half_reliability_run_without_constant_warnings():
    contact = build_contact_frame(_sample_decisions())
    contact_scored = attach_contact_residual(
        contact,
        np.array([0.45, 0.55, 0.50, 0.60, 0.40, 0.50, 0.45, 0.55]),
    )
    contact_half, contact_rel = build_contact_half_reliability(
        contact_scored,
        split_date="2025-07-01",
        min_half_swings=2,
    )

    assert contact_half["batter"].nunique() == 2
    primary_contact = contact_rel.loc[
        contact_rel["metric"].eq(
            "contact_above_expected_per_100_swings"
        )
    ].iloc[0]
    assert primary_contact["eligible_players"] == 2

    damage = build_damage_frame(_sample_decisions())
    scored_damage = attach_damage_residual(
        damage,
        np.array([0.30, 0.30, 0.30]),
    )

    damage_half, damage_rel = build_damage_half_reliability(
        scored_damage,
        split_date="2025-07-01",
        min_half_batted_balls=1,
    )
    assert not damage_half.empty
    primary_damage = damage_rel.loc[
        damage_rel["metric"].eq(
            "damage_xwoba_points_above_expected"
        )
    ].iloc[0]
    assert primary_damage["eligible_players"] >= 1
