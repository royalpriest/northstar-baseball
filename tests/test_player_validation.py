import pandas as pd
import pytest

from northstar.validation.offense import (
    build_correlation_summary,
    merge_with_reference_stats,
)


def test_merge_with_reference_stats_uses_mlbam_id():
    leaderboard = pd.DataFrame(
        {
            "batter": [1001, 1002],
            "re24_total": [20.0, 10.0],
            "re24_per_100_pa": [5.0, 2.5],
        }
    )
    reference = pd.DataFrame(
        {
            "player_id": [1001, 1002],
            "last_name, first_name": ["One, Player", "Two, Player"],
            "pa": [600, 500],
            "woba": [0.400, 0.340],
            "est_woba": [0.410, 0.350],
            "ba": [0.310, 0.270],
            "est_ba": [0.315, 0.275],
            "slg": [0.590, 0.450],
            "est_slg": [0.610, 0.470],
        }
    )

    merged = merge_with_reference_stats(leaderboard, reference)

    assert merged["last_name, first_name"].tolist() == ["One, Player", "Two, Player"]
    assert merged["est_woba"].tolist() == [0.410, 0.350]


def test_build_correlation_summary_detects_positive_relationship():
    merged = pd.DataFrame(
        {
            "re24_per_100_pa": [1.0, 2.0, 3.0, 4.0],
            "woba": [0.300, 0.330, 0.360, 0.400],
            "est_woba": [0.305, 0.335, 0.365, 0.405],
            "ba": [0.220, 0.240, 0.260, 0.290],
            "est_ba": [0.225, 0.245, 0.265, 0.295],
            "slg": [0.350, 0.410, 0.470, 0.550],
            "est_slg": [0.360, 0.420, 0.480, 0.560],
        }
    )

    summary = build_correlation_summary(merged)

    assert set(summary["reference_metric"]) == {
        "woba",
        "est_woba",
        "ba",
        "est_ba",
        "slg",
        "est_slg",
    }
    assert (summary["pearson_r"] > 0.95).all()
    assert all(value == pytest.approx(1.0) for value in summary["spearman_rho"])


def test_validation_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="Leaderboard is missing columns"):
        merge_with_reference_stats(
            pd.DataFrame({"batter": [1]}),
            pd.DataFrame({"player_id": [1], "pa": [300]}),
        )
