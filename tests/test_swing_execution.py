import numpy as np
import pandas as pd
import pytest

from northstar.metrics.execution import (
    EXECUTION_RESIDUAL,
    attach_execution_residual,
    build_execution_half_reliability,
    build_player_execution_summary,
    fit_swing_execution_baseline,
    merge_selection_execution_profile,
    predict_swing_execution_baseline,
)


def _sample_swings() -> pd.DataFrame:
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
            "balls": [0, 1, 0, 1, 0, 1, 0, 1],
            "strikes": [0, 1, 0, 1, 0, 1, 0, 1],
            "zone": [5, 5, 5, 5, 12, 12, 12, 12],
            "pitch_type": ["FF"] * 4 + ["SL"] * 4,
            "decision_transition_value": [
                0.20,
                0.10,
                0.30,
                0.20,
                -0.10,
                -0.20,
                0.00,
                -0.10,
            ],
        }
    )


def test_swing_execution_baseline_predicts_group_means():
    train = _sample_swings()
    mapping, fallback, overall = fit_swing_execution_baseline(train)

    prediction = predict_swing_execution_baseline(
        train.iloc[[0, 4]],
        mapping,
        fallback,
        overall,
    )

    assert prediction[0] == pytest.approx(0.25)
    assert prediction[1] == pytest.approx(-0.05)


def test_attach_execution_residual_is_actual_minus_expected():
    swings = _sample_swings().iloc[:2].copy()
    scored = attach_execution_residual(
        swings,
        np.array([0.15, 0.12]),
    )

    assert scored.loc[0, EXECUTION_RESIDUAL] == pytest.approx(0.05)
    assert scored.loc[1, EXECUTION_RESIDUAL] == pytest.approx(-0.02)
    assert bool(scored.loc[0, "positive_execution"])
    assert not bool(scored.loc[1, "positive_execution"])


def test_player_execution_summary_centers_on_league_mean():
    swings = _sample_swings()
    scored = attach_execution_residual(
        swings,
        np.array([0.10] * len(swings)),
    )

    summary = build_player_execution_summary(
        scored,
        min_swings=1,
    )

    player_10 = summary.loc[summary["batter"].eq(10)].iloc[0]
    assert player_10["swings"] == 4
    assert player_10["execution_value_total"] == pytest.approx(0.4)
    assert player_10["execution_value_per_100_swings"] == pytest.approx(10.0)


def test_execution_half_reliability_uses_both_halves():
    swings = _sample_swings()
    scored = attach_execution_residual(
        swings,
        np.array([0.05] * len(swings)),
    )

    half_table, reliability = build_execution_half_reliability(
        scored,
        split_date="2025-07-01",
        min_half_swings=2,
    )

    assert half_table["batter"].nunique() == 2
    assert set(half_table["season_half"]) == {"early", "late"}
    primary = reliability.loc[
        reliability["metric"].eq("execution_value_per_100_swings")
    ].iloc[0]
    assert primary["eligible_players"] == 2


def test_merge_selection_execution_profile_keeps_axes_separate():
    execution = pd.DataFrame(
        {
            "batter": [10, 20],
            "swings": [500, 600],
            "centered_execution_per_100_swings": [0.4, -0.3],
        }
    )
    selection = pd.DataFrame(
        {
            "batter": [10, 20],
            "player_name": ["A", "B"],
            "eligible_decisions": [1000, 1200],
            "decision_value_per_100": [0.8, -0.2],
        }
    )

    profile = merge_selection_execution_profile(execution, selection)

    assert len(profile) == 2
    assert "selection_centered_per_100" in profile.columns
    weighted_mean = np.average(
        profile["selection_centered_per_100"],
        weights=profile["eligible_decisions"],
    )
    assert weighted_mean == pytest.approx(0.0)
