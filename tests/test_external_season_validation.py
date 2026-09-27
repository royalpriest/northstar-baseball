import pandas as pd
import pytest

from northstar.validation.external_season import (
    season_sample_summary,
    state_transport_coverage,
)


def test_state_transport_coverage_counts_unseen_states():
    mapping = {
        "0-0|0_000": 0.5,
        "1-0|0_000": 0.6,
    }
    external = pd.DataFrame(
        {
            "decision_state": [
                "0-0|0_000",
                "1-0|0_000",
                "2-0|0_000",
                "2-0|0_000",
            ]
        }
    )

    result = state_transport_coverage(mapping, external).iloc[0]

    assert result["external_rows"] == 4
    assert result["exact_state_rows"] == 2
    assert result["fallback_state_rows"] == 2
    assert result["exact_state_pct"] == pytest.approx(50.0)
    assert result["unseen_unique_states"] == 1


def test_season_sample_summary_labels_train_and_external():
    train = pd.DataFrame(
        {
            "is_swing": [0, 1, 1],
            "game_pk": [1, 1, 2],
            "batter": [10, 10, 11],
            "pitcher": [20, 21, 20],
        }
    )
    external = pd.DataFrame(
        {
            "is_swing": [0, 0],
            "game_pk": [3, 4],
            "batter": [12, 13],
            "pitcher": [22, 23],
        }
    )

    result = season_sample_summary(
        train,
        external,
        train_season=2024,
        external_season=2025,
    )

    assert result["role"].tolist() == ["train", "external"]
    assert result["season"].tolist() == [2024, 2025]
    assert result.loc[0, "observed_swing_rate"] == pytest.approx(2 / 3)
    assert result.loc[1, "observed_swing_rate"] == pytest.approx(0.0)
