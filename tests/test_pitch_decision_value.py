import numpy as np
import pandas as pd
import pytest

from northstar.features.decision_value import (
    PITCH_VALUE_TARGET,
    attach_pitch_value_target,
    overlap_mask,
    stabilized_action_weights,
)
from northstar.models.pitch_value import (
    doubly_robust_policy_value,
    modeled_action_advantage,
    optimal_action,
)


def test_attach_pitch_value_target_uses_training_expectancy():
    pitches = pd.DataFrame(
        {
            "game_pk": [1],
            "at_bat_number": [5],
            "bat_score": [2],
            "base_out_state_before": ["0_100"],
        }
    )
    pa = pd.DataFrame(
        {
            "game_pk": [1],
            "at_bat_number": [5],
            "bat_score_after": [3],
            "base_out_state_after": ["1_010"],
            "outs_after": [1],
        }
    )
    expectancy = pd.DataFrame(
        {
            "base_out_state_before": ["0_100", "1_010"],
            "expected_runs": [0.9, 0.6],
        }
    )

    result = attach_pitch_value_target(pitches, pa, expectancy)

    assert result.loc[0, "runs_remaining_in_pa"] == pytest.approx(1.0)
    assert result.loc[0, PITCH_VALUE_TARGET] == pytest.approx(0.7)


def test_overlap_mask_and_stabilized_weights():
    propensity = np.array([0.02, 0.2, 0.8, 0.98])
    action = np.array([0, 0, 1, 1])

    mask = overlap_mask(propensity, 0.05, 0.95)
    weights = stabilized_action_weights(action[mask], propensity[mask])

    assert mask.tolist() == [False, True, True, False]
    assert weights.tolist() == pytest.approx([0.625, 0.625])


def test_modeled_action_advantage_and_optimal_action():
    actual = np.array([1, 0])
    q_take = np.array([0.1, 0.3])
    q_swing = np.array([0.4, 0.2])

    advantage = modeled_action_advantage(actual, q_take, q_swing)
    policy = optimal_action(q_take, q_swing)

    assert advantage.tolist() == pytest.approx([0.3, 0.1])
    assert policy.tolist() == [1, 0]


def test_doubly_robust_policy_value_matches_simple_perfect_case():
    outcome = np.array([1.0, 0.0])
    actual = np.array([1, 0])
    propensity = np.array([0.5, 0.5])
    q_take = np.array([0.0, 0.0])
    q_swing = np.array([1.0, 1.0])
    policy = np.array([1, 1])

    estimate = doubly_robust_policy_value(
        outcome,
        actual,
        propensity,
        q_take,
        q_swing,
        policy,
        policy_name="always_swing",
    )

    assert estimate.value == pytest.approx(1.0)
    assert estimate.swing_share == pytest.approx(1.0)
