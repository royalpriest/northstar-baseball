import numpy as np
import pandas as pd

from northstar.validation.future_incremental import (
    compare_future_incremental_prediction,
)


def test_future_incremental_prediction_detects_added_signal():
    rng = np.random.default_rng(14)
    players = 120
    batter = np.arange(players)

    early_contact = rng.normal(0.75, 0.04, players)
    stable_extra = rng.normal(0.0, 0.03, players)
    late_contact = (
        0.55
        + 0.28 * early_contact
        + 0.85 * stable_extra
        + rng.normal(0.0, 0.008, players)
    )

    early = pd.DataFrame(
        {
            "batter": batter,
            "season_half": "early",
            "statcast_contact_rate": early_contact,
            "contact_above_expected_per_100_swings": (
                stable_extra * 100.0
            ),
        }
    )
    late = pd.DataFrame(
        {
            "batter": batter,
            "season_half": "late",
            "statcast_contact_rate": late_contact,
            "contact_above_expected_per_100_swings": rng.normal(
                0.0,
                1.0,
                players,
            ),
        }
    )
    frame = pd.concat([early, late], ignore_index=True)

    predictions, summary = compare_future_incremental_prediction(
        frame,
        future_target="statcast_contact_rate",
        northstar_feature="contact_above_expected_per_100_swings",
        feature_sets={"raw_contact": ["statcast_contact_rate"]},
        component="contact",
        folds=5,
        seed=4,
    )

    assert len(predictions) == players
    assert len(summary) == 1
    assert summary.loc[0, "delta_oof_r2"] > 0.25
    assert summary.loc[0, "augmented_pearson_r"] > summary.loc[
        0, "base_pearson_r"
    ]


def test_future_incremental_prediction_uses_same_complete_case_sample():
    players = 60
    early = pd.DataFrame(
        {
            "batter": np.arange(players),
            "season_half": "early",
            "statcast_contact_rate": np.linspace(0.6, 0.9, players),
            "statcast_zone_contact_rate": np.linspace(
                0.7,
                0.95,
                players,
            ),
            "contact_above_expected_per_100_swings": np.linspace(
                -3.0,
                3.0,
                players,
            ),
        }
    )
    early.loc[0, "statcast_zone_contact_rate"] = np.nan

    late = pd.DataFrame(
        {
            "batter": np.arange(players),
            "season_half": "late",
            "statcast_contact_rate": np.linspace(0.62, 0.88, players),
            "statcast_zone_contact_rate": np.linspace(
                0.72,
                0.93,
                players,
            ),
            "contact_above_expected_per_100_swings": np.linspace(
                -2.0,
                2.0,
                players,
            ),
        }
    )
    frame = pd.concat([early, late], ignore_index=True)

    predictions, summary = compare_future_incremental_prediction(
        frame,
        future_target="statcast_contact_rate",
        northstar_feature="contact_above_expected_per_100_swings",
        feature_sets={
            "situational_contact": [
                "statcast_contact_rate",
                "statcast_zone_contact_rate",
            ]
        },
        component="contact",
        folds=5,
        seed=3,
    )

    assert len(predictions) == players - 1
    assert summary.loc[0, "players"] == players - 1


def test_future_incremental_prediction_can_show_no_increment():
    rng = np.random.default_rng(9)
    players = 100
    batter = np.arange(players)

    early_damage = rng.normal(0.36, 0.05, players)
    late_damage = (
        0.15
        + 0.60 * early_damage
        + rng.normal(0.0, 0.01, players)
    )

    early = pd.DataFrame(
        {
            "batter": batter,
            "season_half": "early",
            "statcast_xwoba_on_contact": early_damage,
            "damage_xwoba_points_above_expected": rng.normal(
                0.0,
                20.0,
                players,
            ),
        }
    )
    late = pd.DataFrame(
        {
            "batter": batter,
            "season_half": "late",
            "statcast_xwoba_on_contact": late_damage,
            "damage_xwoba_points_above_expected": rng.normal(
                0.0,
                20.0,
                players,
            ),
        }
    )
    frame = pd.concat([early, late], ignore_index=True)

    _, summary = compare_future_incremental_prediction(
        frame,
        future_target="statcast_xwoba_on_contact",
        northstar_feature="damage_xwoba_points_above_expected",
        feature_sets={
            "raw_xwoba_contact": ["statcast_xwoba_on_contact"]
        },
        component="damage",
        folds=5,
        seed=6,
    )

    assert abs(summary.loc[0, "delta_oof_r2"]) < 0.05
