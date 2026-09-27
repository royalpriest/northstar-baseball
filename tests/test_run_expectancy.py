import pytest
import pandas as pd

from northstar.metrics.run_expectancy import add_re24, build_run_expectancy_table


def test_run_expectancy_and_re24():
    pa = pd.DataFrame(
        [
            {
                "outs_before": 0,
                "base_state_before": "000",
                "base_out_state_before": "0_000",
                "runs_to_end": 1,
                "runs_scored_on_pa": 0,
                "outs_after": 0,
                "base_state_after": "100",
                "base_out_state_after": "0_100",
            },
            {
                "outs_before": 0,
                "base_state_before": "000",
                "base_out_state_before": "0_000",
                "runs_to_end": 0,
                "runs_scored_on_pa": 0,
                "outs_after": 1,
                "base_state_after": "000",
                "base_out_state_after": "1_000",
            },
            {
                "outs_before": 0,
                "base_state_before": "100",
                "base_out_state_before": "0_100",
                "runs_to_end": 1,
                "runs_scored_on_pa": 1,
                "outs_after": 3,
                "base_state_after": "000",
                "base_out_state_after": "3_000",
            },
            {
                "outs_before": 1,
                "base_state_before": "000",
                "base_out_state_before": "1_000",
                "runs_to_end": 0,
                "runs_scored_on_pa": 0,
                "outs_after": 3,
                "base_state_after": "000",
                "base_out_state_after": "3_000",
            },
        ]
    )

    table = build_run_expectancy_table(pa)
    mapping = table.set_index("base_out_state_before")["expected_runs"]

    assert mapping["0_000"] == 0.5
    assert mapping["0_100"] == 1.0
    assert mapping["1_000"] == 0.0

    enriched = add_re24(pa, table)
    assert enriched.loc[0, "re24"] == 0.5
    assert enriched.loc[2, "re24"] == 0.0



def test_terminal_event_re24_excludes_earlier_state_change():
    pa = pd.DataFrame(
        [
            {
                "outs_before": 0,
                "base_state_before": "100",
                "base_out_state_before": "0_100",
                "runs_to_end": 1,
                "runs_scored_on_pa": 1,
                "action_base_out_state_before": "0_010",
                "runs_scored_on_terminal_event": 1,
                "outs_after": 0,
                "base_state_after": "010",
                "base_out_state_after": "0_010",
            }
        ]
    )

    expectancy = pd.DataFrame(
        [
            {
                "base_out_state_before": "0_100",
                "expected_runs": 0.9,
            },
            {
                "base_out_state_before": "0_010",
                "expected_runs": 1.1,
            },
        ]
    )

    enriched = add_re24(pa, expectancy)

    assert enriched.loc[0, "re24"] == pytest.approx(1.2)
    assert enriched.loc[0, "terminal_event_re24"] == pytest.approx(1.0)
    assert enriched.loc[0, "within_pa_state_re24"] == pytest.approx(0.2)
