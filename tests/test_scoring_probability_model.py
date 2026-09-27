import pandas as pd

from northstar.models.scoring_probability import (
    chronological_split,
    fit_state_probability_baseline,
    predict_state_probability,
    prepare_scoring_frame,
)


def _sample_plate_appearances() -> pd.DataFrame:
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
            "runs_to_end": [1, 0, 2, 0, 1, 0],
            "base_state_before": ["000", "000", "100", "010", "000", "111"],
            "outs_before": [0, 1, 0, 1, 2, 0],
            "inning_topbot": ["Top", "Bottom", "Top", "Bottom", "Top", "Bottom"],
            "inning": [1, 2, 3, 4, 5, 6],
            "base_out_state_before": [
                "0_000",
                "1_000",
                "0_100",
                "1_010",
                "2_000",
                "0_111",
            ],
        }
    )


def test_prepare_scoring_frame_builds_binary_target():
    frame = prepare_scoring_frame(_sample_plate_appearances())

    assert frame["scored_before_inning_end"].tolist() == [1, 0, 1, 0, 1, 0]


def test_chronological_split_keeps_future_games_out_of_training():
    frame = prepare_scoring_frame(_sample_plate_appearances())
    split = chronological_split(frame)

    assert split.train["game_date"].max() < pd.Timestamp("2025-08-01")
    assert split.validation["game_date"].min() >= pd.Timestamp("2025-08-01")
    assert split.validation["game_date"].max() < pd.Timestamp("2025-09-01")
    assert split.test["game_date"].min() >= pd.Timestamp("2025-09-01")


def test_state_probability_baseline_uses_train_only_mapping():
    frame = prepare_scoring_frame(_sample_plate_appearances())
    split = chronological_split(frame)

    mapping, fallback = fit_state_probability_baseline(split.train)
    probabilities = predict_state_probability(split.test, mapping, fallback)

    assert probabilities.between(0.0, 1.0).all()
    assert len(probabilities) == len(split.test)
