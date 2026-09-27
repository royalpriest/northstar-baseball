import pandas as pd
import pytest

from northstar.metrics.player_decision import (
    add_player_decision_values,
    build_player_decision_metric,
    build_player_half_reliability,
    decision_value_identity_error,
)


def _sample_predictions() -> pd.DataFrame:
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
            "batter": [10, 10, 10, 10, 20, 20, 20, 20],
            "is_swing": [1, 0, 1, 0, 1, 1, 0, 0],
            "swing_propensity_calibrated": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.4,
                0.4,
                0.6,
                0.6,
            ],
            "q_take": [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 0.2],
            "q_swing": [1.0, 1.0, 1.0, 1.0, 0.4, 0.4, 0.0, 0.0],
        }
    )


def test_add_player_decision_values_has_expected_accounting():
    frame = add_player_decision_values(_sample_predictions())

    first = frame.iloc[0]
    second = frame.iloc[1]

    assert first["expected_action_value"] == pytest.approx(0.5)
    assert first["chosen_action_value"] == pytest.approx(1.0)
    assert first["decision_value_vs_expected"] == pytest.approx(0.5)
    assert first["missed_decision_value"] == pytest.approx(0.0)

    assert second["chosen_action_value"] == pytest.approx(0.0)
    assert second["decision_value_vs_expected"] == pytest.approx(-0.5)
    assert second["missed_decision_value"] == pytest.approx(1.0)

    error = decision_value_identity_error(frame)
    assert error.abs().max() == pytest.approx(0.0)


def test_build_player_decision_metric_aggregates_rate_and_choice_quality():
    leaderboard = build_player_decision_metric(
        _sample_predictions(),
        min_decisions=1,
    )

    player_10 = leaderboard.loc[leaderboard["batter"].eq(10)].iloc[0]

    assert player_10["eligible_decisions"] == 4
    assert player_10["decision_value_total"] == pytest.approx(0.0)
    assert player_10["decision_value_per_100"] == pytest.approx(0.0)
    assert player_10["optimal_choice_rate"] == pytest.approx(0.5)
    assert player_10["value_weighted_optimal_choice_rate"] == pytest.approx(0.5)
    assert player_10["missed_value_per_100"] == pytest.approx(50.0)


def test_discretionary_window_is_applied():
    predictions = _sample_predictions()
    predictions.loc[0, "swing_propensity_calibrated"] = 0.95

    leaderboard = build_player_decision_metric(
        predictions,
        min_decisions=1,
        discretionary_lower=0.20,
        discretionary_upper=0.80,
    )

    player_10 = leaderboard.loc[leaderboard["batter"].eq(10)].iloc[0]
    assert player_10["eligible_decisions"] == 4
    assert player_10["discretionary_decisions"] == 3


def test_player_half_reliability_requires_both_halves():
    half_table, reliability = build_player_half_reliability(
        _sample_predictions(),
        split_date="2025-07-01",
        min_half_decisions=2,
    )

    assert half_table["batter"].nunique() == 2
    assert set(half_table["season_half"]) == {"early", "late"}
    assert (
        reliability["eligible_players"].min()
        == 2
    )
    assert {
        "decision_value_per_100",
        "optimal_choice_rate",
        "discretionary_decision_value_per_100",
    }.issubset(set(reliability["metric"]))
