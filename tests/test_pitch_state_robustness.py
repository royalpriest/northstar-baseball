import numpy as np
import pandas as pd
import pytest

from northstar.validation.pitch_state import (
    cluster_bootstrap_mean,
    doubly_robust_policy_score,
    overlap_policy_robustness,
    player_half_stability,
)


def test_doubly_robust_policy_score_perfect_case():
    score = doubly_robust_policy_score(
        outcome=np.array([1.0, 0.0]),
        actual_swing=np.array([1, 0]),
        swing_probability=np.array([0.5, 0.5]),
        q_take=np.array([0.0, 0.0]),
        q_swing=np.array([1.0, 1.0]),
        policy_action=np.array([1, 1]),
    )

    assert score.tolist() == pytest.approx([1.0, 1.0])


def test_cluster_bootstrap_mean_returns_clustered_interval():
    result = cluster_bootstrap_mean(
        values=np.array([1.0, 1.0, -1.0, -1.0]),
        clusters=np.array([10, 10, 20, 20]),
        reps=200,
        seed=7,
    )

    assert result.estimate == pytest.approx(0.0)
    assert result.clusters == 2
    assert result.rows == 4
    assert result.ci_low <= result.estimate <= result.ci_high


def test_overlap_policy_robustness_tightens_sample():
    frame = pd.DataFrame(
        {
            "game_pk": [1, 1, 2, 2],
            "is_swing": [0, 1, 0, 1],
            "swing_propensity": [0.06, 0.30, 0.70, 0.94],
            "q_take": [0.2, 0.1, 0.2, 0.1],
            "q_swing": [0.1, 0.3, 0.1, 0.4],
            "decision_transition_value": [0.2, 0.3, 0.2, 0.4],
        }
    )

    result = overlap_policy_robustness(
        frame,
        windows=[(0.05, 0.95), (0.20, 0.80)],
        bootstrap_reps=100,
        seed=3,
    )

    assert result["pitches"].tolist() == [4, 2]
    assert result["games"].tolist() == [2, 2]


def test_player_half_stability_filters_minimum_sample():
    frame = pd.DataFrame(
        {
            "game_date": [
                "2025-09-05",
                "2025-09-06",
                "2025-09-20",
                "2025-09-21",
            ],
            "batter": [10, 10, 10, 10],
            "is_swing": [1, 0, 1, 0],
            "q_take": [0.0, 0.2, 0.0, 0.2],
            "q_swing": [0.3, 0.1, 0.3, 0.1],
        }
    )

    players, summary = player_half_stability(
        frame,
        split_date="2025-09-15",
        min_pitches_per_half=2,
    )

    assert len(players) == 1
    assert players.iloc[0]["batter"] == 10
    assert summary["eligible_players"].tolist() == [1, 1]
