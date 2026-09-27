import pandas as pd
import pytest

from northstar.validation.offense import (
    build_correlation_summary,
    merge_with_reference_stats,
)


def test_merge_with_reference_stats_uses_id_crosswalk():
    leaderboard = pd.DataFrame(
        {
            "batter": [1001, 1002],
            "re24_total": [20.0, 10.0],
            "re24_per_100_pa": [5.0, 2.5],
        }
    )
    lookup = pd.DataFrame(
        {
            "key_mlbam": [1001, 1002],
            "key_fangraphs": [501, 502],
        }
    )
    reference = pd.DataFrame(
        {
            "IDfg": [501, 502],
            "Name": ["Player One", "Player Two"],
            "PA": [600, 500],
            "OPS": [0.950, 0.800],
            "wOBA": [0.400, 0.340],
            "wRC+": [150, 115],
        }
    )

    merged = merge_with_reference_stats(leaderboard, lookup, reference)

    assert merged["Name"].tolist() == ["Player One", "Player Two"]
    assert merged["wRC+"].tolist() == [150, 115]


def test_build_correlation_summary_detects_positive_relationship():
    merged = pd.DataFrame(
        {
            "re24_per_100_pa": [1.0, 2.0, 3.0, 4.0],
            "OPS": [0.700, 0.760, 0.820, 0.900],
            "wOBA": [0.300, 0.330, 0.360, 0.400],
            "wRC+": [90, 105, 120, 145],
        }
    )

    summary = build_correlation_summary(merged)

    assert set(summary["reference_metric"]) == {"OPS", "wOBA", "wRC+"}
    assert (summary["pearson_r"] > 0.95).all()
    assert all(value == pytest.approx(1.0) for value in summary["spearman_rho"])


def test_validation_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="Leaderboard is missing columns"):
        merge_with_reference_stats(
            pd.DataFrame({"batter": [1]}),
            pd.DataFrame({"key_mlbam": [1], "key_fangraphs": [10]}),
            pd.DataFrame({"IDfg": [10], "Name": ["A"], "PA": [300]}),
        )
