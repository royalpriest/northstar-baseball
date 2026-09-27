import pandas as pd
import pytest

from northstar.metrics.player_value import build_player_run_value


def test_build_player_run_value_aggregates_and_normalizes():
    plate_appearances = pd.DataFrame(
        {
            "batter": [101, 101, 101, 202],
            "re24": [1.0, -0.5, 0.5, 2.0],
        }
    )

    result = build_player_run_value(
        plate_appearances,
        min_pa=2,
        sort_by="re24_total",
    )

    assert result["batter"].tolist() == [101]
    row = result.iloc[0]

    assert row["plate_appearances"] == 3
    assert row["re24_total"] == pytest.approx(1.0)
    assert row["re24_per_pa"] == pytest.approx(1.0 / 3.0)
    assert row["re24_per_100_pa"] == pytest.approx(100.0 / 3.0)
    assert row["positive_pa_pct"] == pytest.approx(200.0 / 3.0)


def test_build_player_run_value_can_rank_by_rate():
    plate_appearances = pd.DataFrame(
        {
            "batter": [1, 1, 2, 2],
            "re24": [0.5, 0.5, 0.8, 0.8],
        }
    )

    result = build_player_run_value(
        plate_appearances,
        min_pa=2,
        sort_by="re24_per_100_pa",
    )

    assert result["batter"].tolist() == [2, 1]


def test_build_player_run_value_validates_inputs():
    with pytest.raises(ValueError, match="Missing plate-appearance columns"):
        build_player_run_value(pd.DataFrame({"batter": [1]}))

    with pytest.raises(ValueError, match="min_pa"):
        build_player_run_value(
            pd.DataFrame({"batter": [1], "re24": [0.1]}),
            min_pa=0,
        )
