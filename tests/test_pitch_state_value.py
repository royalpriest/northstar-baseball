import pandas as pd
import pytest

from northstar.features.state_value import (
    ABSOLUTE_STATE_RETURN,
    DECISION_TRANSITION_TARGET,
    attach_absolute_future_value,
    attach_decision_transition_values,
    build_telescoping_audit,
    fit_decision_state_values,
)


def _sample_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [1, 1],
            "at_bat_number": [5, 5],
            "pitch_number": [1, 2],
            "balls": [0, 1],
            "strikes": [0, 0],
            "base_out_state_before": ["0_000", "0_000"],
            "bat_score": [0, 0],
        }
    )


def _sample_pa() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [1],
            "at_bat_number": [5],
            "bat_score_after": [1],
            "base_out_state_after": ["1_100"],
            "outs_after": [1],
        }
    )


def _sample_expectancy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "base_out_state_before": ["0_000", "1_100"],
            "expected_runs": [0.5, 0.4],
        }
    )


def test_attach_absolute_future_value():
    result = attach_absolute_future_value(
        _sample_decisions(),
        _sample_pa(),
        _sample_expectancy(),
    )

    assert result[ABSOLUTE_STATE_RETURN].tolist() == pytest.approx([1.4, 1.4])
    assert result["decision_state"].tolist() == ["0-0|0_000", "1-0|0_000"]


def test_transition_values_telescope():
    frame = attach_absolute_future_value(
        _sample_decisions(),
        _sample_pa(),
        _sample_expectancy(),
    )

    training = pd.concat(
        [
            frame.assign(**{ABSOLUTE_STATE_RETURN: [0.8, 0.9]}),
            frame.assign(**{ABSOLUTE_STATE_RETURN: [1.0, 1.1]}),
        ],
        ignore_index=True,
    )
    mapping, base_out, overall, _ = fit_decision_state_values(training)

    transitions = attach_decision_transition_values(
        frame,
        mapping,
        base_out,
        overall,
    )
    audit = build_telescoping_audit(transitions)

    assert transitions.loc[0, "current_state_value"] == pytest.approx(0.9)
    assert transitions.loc[0, "next_state_value"] == pytest.approx(1.0)
    assert transitions.loc[0, DECISION_TRANSITION_TARGET] == pytest.approx(0.1)

    assert transitions.loc[1, "next_state_value"] == pytest.approx(0.4)
    assert transitions.loc[1, "interval_runs"] == pytest.approx(1.0)
    assert transitions.loc[1, DECISION_TRANSITION_TARGET] == pytest.approx(0.4)

    assert audit.loc[0, "transition_total"] == pytest.approx(0.5)
    assert audit.loc[0, "expected_telescoping_total"] == pytest.approx(0.5)
    assert audit.loc[0, "telescoping_error"] == pytest.approx(0.0)


def test_pitch_number_gap_is_not_clean_one_step_transition():
    frame = _sample_decisions()
    frame.loc[1, "pitch_number"] = 3
    frame = attach_absolute_future_value(
        frame,
        _sample_pa(),
        _sample_expectancy(),
    )

    mapping, base_out, overall, _ = fit_decision_state_values(frame)
    transitions = attach_decision_transition_values(
        frame,
        mapping,
        base_out,
        overall,
    )

    assert not bool(transitions.loc[0, "clean_one_step_transition"])
    assert bool(transitions.loc[1, "clean_one_step_transition"])
