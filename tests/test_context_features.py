import pandas as pd
import pytest

from northstar.features.context import (
    add_backward_looking_player_context,
    add_backward_looking_terminal_context,
)


def test_player_context_excludes_current_day_results():
    frame = pd.DataFrame(
        {
            "game_date": [
                "2025-04-01",
                "2025-04-01",
                "2025-04-02",
                "2025-04-03",
            ],
            "batter": [10, 10, 10, 10],
            "pitcher": [20, 21, 20, 22],
            "re24": [1.0, -0.5, 0.5, 2.0],
        }
    )

    enriched = add_backward_looking_player_context(frame)

    day_one = enriched[enriched["game_date"].eq(pd.Timestamp("2025-04-01"))]
    assert day_one["batter_prior_count"].eq(0).all()
    assert day_one["batter_prior_re24_per_pa"].isna().all()

    day_two = enriched[enriched["game_date"].eq(pd.Timestamp("2025-04-02"))].iloc[0]
    assert day_two["batter_prior_count"] == 2
    assert day_two["batter_prior_re24_per_pa"] == pytest.approx(0.25)

    day_three = enriched[enriched["game_date"].eq(pd.Timestamp("2025-04-03"))].iloc[0]
    assert day_three["batter_prior_count"] == 3
    assert day_three["batter_prior_re24_per_pa"] == pytest.approx(1.0 / 3.0)


def test_pitcher_context_tracks_prior_value_allowed():
    frame = pd.DataFrame(
        {
            "game_date": [
                "2025-04-01",
                "2025-04-02",
                "2025-04-03",
            ],
            "batter": [1, 2, 3],
            "pitcher": [99, 99, 99],
            "re24": [0.4, -0.2, 0.8],
        }
    )

    enriched = add_backward_looking_player_context(frame)
    day_three = enriched[enriched["game_date"].eq(pd.Timestamp("2025-04-03"))].iloc[0]

    assert day_three["pitcher_prior_count"] == 2
    assert day_three["pitcher_prior_re24_per_pa"] == pytest.approx(0.1)


def test_player_context_validates_required_columns():
    with pytest.raises(ValueError, match="Missing plate-appearance columns"):
        add_backward_looking_player_context(
            pd.DataFrame(
                {
                    "game_date": ["2025-04-01"],
                    "batter": [1],
                }
            )
        )



def test_terminal_context_uses_prior_dates_only():
    frame = pd.DataFrame(
        {
            "game_date": [
                "2025-04-01",
                "2025-04-01",
                "2025-04-02",
            ],
            "batter": [10, 10, 10],
            "pitcher": [20, 21, 20],
            "terminal_event_re24": [0.6, -0.2, 0.4],
        }
    )

    enriched = add_backward_looking_terminal_context(frame)

    first_day = enriched[enriched["game_date"].eq(pd.Timestamp("2025-04-01"))]
    assert first_day["batter_terminal_prior_count"].eq(0).all()

    day_two = enriched[
        enriched["game_date"].eq(pd.Timestamp("2025-04-02"))
    ].iloc[0]
    assert day_two["batter_terminal_prior_count"] == 2
    assert day_two["batter_terminal_prior_value_per_pa"] == pytest.approx(0.2)
    assert day_two["pitcher_terminal_prior_count"] == 1
    assert day_two["pitcher_terminal_prior_value_per_pa"] == pytest.approx(0.6)
