import numpy as np
import pandas as pd
import pytest

from northstar.features.outcomes import (
    OUTCOME_CLASSES,
    OUTCOME_TO_CODE,
    add_backward_looking_outcome_context,
    classify_pa_outcome,
)
from northstar.models.pa_outcome import (
    expected_re24_from_outcome_probabilities,
    fit_class_prior,
    fit_outcome_run_value_table,
    prepare_outcome_frame,
    repeat_class_prior,
)


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("strikeout", "strikeout"),
        ("walk", "walk_hbp"),
        ("hit_by_pitch", "walk_hbp"),
        ("single", "single"),
        ("double", "double_triple"),
        ("triple", "double_triple"),
        ("home_run", "home_run"),
        ("field_out", "field_out"),
        ("sac_fly", "sacrifice"),
        ("field_error", "reach_other"),
    ],
)
def test_classify_pa_outcome(event, expected):
    assert classify_pa_outcome(event) == expected


def test_prior_outcome_rates_exclude_same_day():
    frame = pd.DataFrame(
        {
            "game_date": ["2025-04-01", "2025-04-01", "2025-04-02"],
            "batter": [10, 10, 10],
            "pitcher": [20, 21, 20],
            "events": ["single", "strikeout", "walk"],
        }
    )

    enriched = add_backward_looking_outcome_context(frame)
    day_one = enriched[enriched["game_date"].eq(pd.Timestamp("2025-04-01"))]
    assert day_one["batter_outcome_prior_count"].eq(0).all()

    day_two = enriched[
        enriched["game_date"].eq(pd.Timestamp("2025-04-02"))
    ].iloc[0]
    assert day_two["batter_outcome_prior_count"] == 2
    assert day_two["batter_outcome_prior_single_rate"] == pytest.approx(0.5)
    assert day_two["batter_outcome_prior_strikeout_rate"] == pytest.approx(0.5)


def test_class_prior_repeats_probability_distribution():
    target = pd.DataFrame(
        {
            "outcome_code": [
                OUTCOME_TO_CODE["single"],
                OUTCOME_TO_CODE["single"],
                OUTCOME_TO_CODE["strikeout"],
            ]
        }
    )
    prior = fit_class_prior(target)
    repeated = repeat_class_prior(prior, 2)

    assert repeated.shape == (2, len(OUTCOME_CLASSES))
    assert prior[OUTCOME_TO_CODE["single"]] == pytest.approx(2 / 3)
    assert prior[OUTCOME_TO_CODE["strikeout"]] == pytest.approx(1 / 3)


def test_outcome_probabilities_translate_to_expected_re24():
    single = OUTCOME_TO_CODE["single"]
    strikeout = OUTCOME_TO_CODE["strikeout"]

    train = pd.DataFrame(
        {
            "base_out_state_before": ["0_000", "0_000"],
            "outcome_code": [single, strikeout],
            "terminal_event_re24": [0.4, -0.2],
        }
    )
    state_values, fallbacks, overall = fit_outcome_run_value_table(train)

    probabilities = np.zeros((1, len(OUTCOME_CLASSES)))
    probabilities[0, single] = 0.75
    probabilities[0, strikeout] = 0.25

    frame = pd.DataFrame({"base_out_state_before": ["0_000"]})
    predicted = expected_re24_from_outcome_probabilities(
        frame,
        probabilities,
        state_values,
        fallbacks,
        overall,
    )

    assert predicted[0] == pytest.approx(0.25)


def test_prepare_outcome_frame_filters_shifted_pa():
    row = {
        "game_date": "2025-04-01",
        "base_state_before": "000",
        "outs_before": 0,
        "inning_topbot": "Top",
        "batter_hand": "R",
        "pitcher_hand": "R",
        "park_team": "TOR",
        "inning": 1,
        "score_diff_before": 0,
        "is_home_batting": 0,
        "batter_outcome_prior_log_count": 0.0,
        "pitcher_outcome_prior_log_count": 0.0,
        "base_out_state_before": "0_000",
        "terminal_event_re24": 0.1,
        "within_pa_state_re24": 0.2,
        "outcome_class": "single",
        "outcome_code": OUTCOME_TO_CODE["single"],
    }
    for outcome in OUTCOME_CLASSES:
        row[f"batter_outcome_prior_{outcome}_rate"] = np.nan
        row[f"pitcher_outcome_prior_{outcome}_rate"] = np.nan

    sample = pd.DataFrame([row])
    clean = prepare_outcome_frame(sample)
    all_pa = prepare_outcome_frame(sample, clean_attribution_only=False)

    assert clean.empty
    assert len(all_pa) == 1
