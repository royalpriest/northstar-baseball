import pandas as pd
import pytest

from northstar.profiles.comparison import (
    build_hitter_comparison,
    comparison_percentile_long,
)


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batter": [10, 20, 30],
            "player_name": ["Alpha", "Beta", "Gamma"],
            "birth_country": ["Canada", "USA", "Canada"],
            "primary_team_2025": ["TOR", "NYY", "SEA"],
            "selection_context_percentile": [80.0, 40.0, 20.0],
            "contact_context_percentile": [30.0, 70.0, 95.0],
            "damage_context_percentile": [75.0, 55.0, 10.0],
            "chase_avoidance_percentile": [85.0, 50.0, 25.0],
            "contact_rate_percentile": [35.0, 65.0, 98.0],
            "xwoba_on_contact_percentile": [90.0, 60.0, 5.0],
            "barrel_percentile": [88.0, 55.0, 3.0],
            "xwoba_percentile": [92.0, 58.0, 20.0],
            "re24_percentile": [95.0, 52.0, 18.0],
        }
    )


def test_build_hitter_comparison_supports_mixed_name_and_id():
    comparison = build_hitter_comparison(
        _profile(),
        ["alpha", 20],
    )

    assert comparison["batter"].tolist() == [10, 20]
    assert comparison["player_name"].tolist() == ["Alpha", "Beta"]
    assert "overall_score" not in comparison.columns


def test_build_hitter_comparison_requires_two_to_four_unique_players():
    profile = _profile()

    with pytest.raises(ValueError):
        build_hitter_comparison(profile, ["Alpha"])

    with pytest.raises(ValueError):
        build_hitter_comparison(
            profile,
            ["Alpha", "Beta", "Gamma", 10, 20],
        )

    with pytest.raises(ValueError):
        build_hitter_comparison(profile, ["Alpha", 10])


def test_comparison_percentile_long_separates_northstar_and_established():
    comparison = build_hitter_comparison(
        _profile(),
        ["Alpha", "Gamma"],
    )
    long = comparison_percentile_long(comparison)

    assert set(long["metric_group"]) == {"NorthStar", "Established"}
    assert set(
        long.loc[
            long["metric_group"].eq("NorthStar"),
            "metric",
        ]
    ) == {
        "Selection context",
        "Contact context",
        "Damage context",
    }

    alpha_selection = long.loc[
        long["player_name"].eq("Alpha")
        & long["metric"].eq("Selection context"),
        "percentile",
    ].iloc[0]
    assert alpha_selection == pytest.approx(80.0)
