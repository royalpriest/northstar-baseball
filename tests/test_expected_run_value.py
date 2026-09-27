import pandas as pd
import pytest

from northstar.models.expected_run_value import (
    chronological_run_value_split,
    fit_state_run_value_baseline,
    predict_state_run_value,
    prepare_expected_re24_frame,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_date": [
                "2025-07-15",
                "2025-07-20",
                "2025-08-10",
                "2025-08-20",
                "2025-09-05",
                "2025-09-10",
            ],
            "base_state_before": ["000", "100", "010", "111", "000", "100"],
            "outs_before": [0, 1, 1, 0, 2, 0],
            "inning_topbot": ["Top", "Bottom", "Top", "Bottom", "Top", "Bottom"],
            "batter_hand": ["R", "L", "R", "L", "R", "L"],
            "pitcher_hand": ["R", "R", "L", "L", "R", "L"],
            "park_team": ["TOR"] * 6,
            "inning": [1, 2, 3, 4, 5, 6],
            "score_diff_before": [0, 1, -1, 0, 2, -2],
            "is_home_batting": [0, 1, 0, 1, 0, 1],
            "batter_terminal_prior_value_per_pa": [None, 0.1, 0.2, 0.1, 0.3, 0.2],
            "batter_terminal_prior_positive_pa_pct": [None, 50, 55, 45, 60, 52],
            "batter_terminal_prior_log_count": [0, 1, 2, 2.5, 3, 3.2],
            "pitcher_terminal_prior_value_per_pa": [None, -0.1, 0.0, 0.1, -0.2, 0.05],
            "pitcher_terminal_prior_positive_pa_pct": [None, 45, 50, 55, 40, 51],
            "pitcher_terminal_prior_log_count": [0, 1, 2, 2.4, 3, 3.1],
            "base_out_state_before": [
                "0_000",
                "1_100",
                "1_010",
                "0_111",
                "2_000",
                "0_100",
            ],
            "terminal_event_re24": [0.2, -0.1, 0.4, 0.8, -0.2, 0.3],
            "within_pa_state_re24": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_prepare_expected_re24_frame_and_split():
    frame = prepare_expected_re24_frame(_sample_frame())
    split = chronological_run_value_split(frame)

    assert len(split.train) == 2
    assert len(split.validation) == 2
    assert len(split.test) == 2
    assert split.train["game_date"].max() < pd.Timestamp("2025-08-01")
    assert split.test["game_date"].min() >= pd.Timestamp("2025-09-01")


def test_state_run_value_baseline_uses_train_only_means():
    frame = prepare_expected_re24_frame(_sample_frame())
    split = chronological_run_value_split(frame)
    mapping, fallback = fit_state_run_value_baseline(split.train)

    assert mapping["0_000"] == pytest.approx(0.2)
    assert mapping["1_100"] == pytest.approx(-0.1)
    assert fallback == pytest.approx(0.05)

    predictions = predict_state_run_value(split.test, mapping, fallback)
    assert predictions.tolist() == pytest.approx([0.05, 0.05])


def test_prepare_expected_re24_requires_terminal_attribution_fields():
    incomplete = _sample_frame().drop(columns=["terminal_event_re24"])
    with pytest.raises(ValueError, match="Expected-RE24 columns are missing"):
        prepare_expected_re24_frame(incomplete)



def test_prepare_expected_re24_filters_shifted_pa_by_default():
    sample = _sample_frame()
    sample.loc[0, "within_pa_state_re24"] = 0.25

    clean = prepare_expected_re24_frame(sample)
    all_pa = prepare_expected_re24_frame(sample, clean_attribution_only=False)

    assert len(clean) == len(sample) - 1
    assert len(all_pa) == len(sample)
