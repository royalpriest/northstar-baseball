import numpy as np
import pandas as pd
import pytest

from northstar.validation.player_metric import (
    build_plate_discipline_summary,
    construct_correlation_summary,
    empirical_bayes_player_shrinkage,
    merge_player_construct_data,
    add_incremental_oof_residuals,
    incremental_half_reliability,
    residual_outcome_correlations,
)


def test_plate_discipline_summary_uses_zone_and_swing_definitions():
    pitches = pd.DataFrame(
        {
            "batter": [10, 10, 10, 10, 20, 20],
            "description": [
                "swinging_strike",
                "foul",
                "ball",
                "called_strike",
                "hit_into_play",
                "ball",
            ],
            "zone": [5, 12, 13, 4, 6, 14],
            "balls": [0, 0, 1, 2, 0, 1],
            "strikes": [0, 1, 2, 2, 0, 2],
        }
    )

    result = build_plate_discipline_summary(pitches)
    batter_10 = result.loc[result["batter"].eq(10)].iloc[0]

    assert batter_10["decision_pitches"] == 4
    assert batter_10["swing_rate"] == pytest.approx(0.5)
    assert batter_10["zone_swing_rate"] == pytest.approx(0.5)
    assert batter_10["chase_rate"] == pytest.approx(0.5)
    assert batter_10["whiff_per_swing"] == pytest.approx(0.5)
    assert batter_10["contact_rate"] == pytest.approx(0.5)
    assert batter_10["two_strike_chase_rate"] == pytest.approx(0.0)


def test_merge_construct_data_prefixes_statcast_metrics():
    player = pd.DataFrame(
        {
            "batter": [10],
            "decision_value_per_100": [0.5],
            "swing_rate": [0.45],
        }
    )
    discipline = pd.DataFrame(
        {
            "batter": [10],
            "decision_pitches": [100],
            "swing_rate": [0.50],
            "chase_rate": [0.20],
        }
    )
    outcome = pd.DataFrame(
        {
            "batter": [10],
            "re24_per_100_pa": [3.2],
            "woba": [0.350],
            "est_woba": [0.360],
        }
    )

    merged = merge_player_construct_data(
        player,
        discipline,
        outcome_reference=outcome,
    )

    assert merged.loc[0, "swing_rate"] == pytest.approx(0.45)
    assert merged.loc[0, "statcast_swing_rate"] == pytest.approx(0.50)
    assert merged.loc[0, "statcast_chase_rate"] == pytest.approx(0.20)
    assert merged.loc[0, "est_woba"] == pytest.approx(0.360)


def test_construct_correlation_summary_detects_monotonic_association():
    frame = pd.DataFrame(
        {
            "decision_value_per_100": np.arange(40, dtype=float),
            "statcast_chase_rate": np.arange(40, dtype=float) * -1.0,
        }
    )

    result = construct_correlation_summary(
        frame,
        targets=["decision_value_per_100"],
        references=["statcast_chase_rate"],
        min_pairs=30,
    )

    assert len(result) == 1
    assert result.loc[0, "pearson_r"] == pytest.approx(-1.0)
    assert result.loc[0, "spearman_rho"] == pytest.approx(-1.0)


def test_empirical_bayes_shrinkage_centers_and_shrinks():
    decisions = pd.DataFrame(
        {
            "batter": [10] * 4 + [20] * 4 + [30] * 4,
            "game_pk": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                6,
                6,
            ],
            "decision_value_vs_expected": [
                0.10,
                0.08,
                0.12,
                0.09,
                -0.08,
                -0.10,
                -0.06,
                -0.09,
                0.01,
                -0.01,
                0.02,
                -0.02,
            ],
            "discretionary_opportunity": [True] * 12,
        }
    )

    result, summary = empirical_bayes_player_shrinkage(
        decisions,
        min_decisions=4,
        min_discretionary=4,
    )

    assert summary.eligible_players == 3
    assert len(result) == 3
    assert result["shrinkage_factor"].between(0.0, 1.0).all()

    top = result.iloc[0]
    assert top["batter"] == 10
    assert abs(top["shrunk_centered_per_100"]) <= abs(
        top["raw_centered_per_100"]
    ) + 1e-12



def test_incremental_oof_residuals_remove_simple_construct_signal():
    players = 60
    chase = np.linspace(0.1, 0.5, players)
    frame = pd.DataFrame(
        {
            "batter": np.arange(players),
            "decision_value_per_100": 2.0 - 4.0 * chase,
            "statcast_chase_rate": chase,
        }
    )

    residuals, summary = add_incremental_oof_residuals(
        frame,
        feature_sets={"chase_only": ["statcast_chase_rate"]},
        folds=5,
        seed=7,
    )

    assert len(residuals) == players
    assert summary.loc[0, "oof_r2"] > 0.99
    assert residuals["incremental_residual"].abs().mean() < 0.02


def test_incremental_half_reliability_recovers_stable_residual_trait():
    rng = np.random.default_rng(7)
    players = 80
    batter = np.arange(players)
    trait = rng.normal(0.0, 0.3, players)
    early_chase = rng.uniform(0.15, 0.45, players)
    late_chase = rng.uniform(0.15, 0.45, players)

    early = pd.DataFrame(
        {
            "batter": batter,
            "season_half": "early",
            "decision_value_per_100": 1.0 - 3.0 * early_chase + trait,
            "statcast_chase_rate": early_chase,
        }
    )
    late = pd.DataFrame(
        {
            "batter": batter,
            "season_half": "late",
            "decision_value_per_100": 1.0 - 3.0 * late_chase + trait,
            "statcast_chase_rate": late_chase,
        }
    )
    frame = pd.concat([early, late], ignore_index=True)

    residuals, reliability = incremental_half_reliability(
        frame,
        feature_sets={"chase_only": ["statcast_chase_rate"]},
        folds=5,
        seed=7,
    )

    chase_row = reliability.loc[
        reliability["construct_set"].eq("chase_only")
    ].iloc[0]

    assert len(residuals) == players * 2
    assert chase_row["players"] == players
    assert chase_row["pearson_r"] > 0.8


def test_residual_outcome_correlations():
    residuals = pd.DataFrame(
        {
            "batter": np.arange(40),
            "construct_set": "plate_discipline",
            "incremental_residual": np.arange(40, dtype=float),
        }
    )
    construct = pd.DataFrame(
        {
            "batter": np.arange(40),
            "re24_per_100_pa": np.arange(40, dtype=float) * 2.0,
        }
    )

    result = residual_outcome_correlations(
        residuals,
        construct,
        references=["re24_per_100_pa"],
        min_pairs=30,
    )

    assert len(result) == 1
    assert result.loc[0, "pearson_r"] == pytest.approx(1.0)
    assert result.loc[0, "spearman_rho"] == pytest.approx(1.0)
