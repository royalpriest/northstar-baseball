from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PITCH_DECISION_TARGET = "is_swing"

PITCH_DECISION_CATEGORICAL_FEATURES = [
    "balls",
    "strikes",
    "pitch_type",
    "batter_hand",
    "pitcher_hand",
    "zone",
    "base_state_before",
    "outs_when_up",
    "inning_topbot",
]

PITCH_DECISION_NUMERIC_FEATURES = [
    "release_speed",
    "plate_x_batter",
    "plate_z_norm",
    "pfx_x_batter",
    "pfx_z",
    "inning",
    "score_diff_before",
    "is_home_batting",
]

PITCH_DECISION_REQUIRED_COLUMNS = set(
    PITCH_DECISION_CATEGORICAL_FEATURES
    + PITCH_DECISION_NUMERIC_FEATURES
    + ["game_date", PITCH_DECISION_TARGET]
)


@dataclass(frozen=True)
class PitchDecisionSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def prepare_pitch_decision_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = PITCH_DECISION_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Pitch-decision model columns are missing: {sorted(missing)}"
        )

    out = frame.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out[PITCH_DECISION_TARGET] = pd.to_numeric(
        out[PITCH_DECISION_TARGET], errors="coerce"
    )
    out = out.loc[
        out["game_date"].notna()
        & out[PITCH_DECISION_TARGET].isin([0, 1])
    ].copy()
    out[PITCH_DECISION_TARGET] = out[PITCH_DECISION_TARGET].astype("int64")
    return out.reset_index(drop=True)


def chronological_pitch_split(
    frame: pd.DataFrame,
    validation_start: str | pd.Timestamp = "2025-08-01",
    test_start: str | pd.Timestamp = "2025-09-01",
) -> PitchDecisionSplit:
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)

    if validation_start >= test_start:
        raise ValueError("validation_start must be earlier than test_start")

    train = frame.loc[frame["game_date"] < validation_start].copy()
    validation = frame.loc[
        frame["game_date"].ge(validation_start)
        & frame["game_date"].lt(test_start)
    ].copy()
    test = frame.loc[frame["game_date"].ge(test_start)].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Chronological pitch-decision split produced an empty partition."
        )

    return PitchDecisionSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def fit_count_zone_swing_baseline(
    train: pd.DataFrame,
) -> tuple[dict[tuple[int, int, str], float], float]:
    """Fit a simple training-only swing-rate baseline by count and Statcast zone."""
    required = {"balls", "strikes", "zone", PITCH_DECISION_TARGET}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training pitch data is missing: {sorted(missing)}")

    work = train.copy()
    work["zone_key"] = work["zone"].astype("string").fillna("NA")

    grouped = (
        work.groupby(["balls", "strikes", "zone_key"])[PITCH_DECISION_TARGET]
        .mean()
        .astype(float)
    )
    mapping = {
        (int(balls), int(strikes), str(zone)): float(value)
        for (balls, strikes, zone), value in grouped.items()
    }
    fallback = float(work[PITCH_DECISION_TARGET].mean())
    return mapping, fallback


def predict_count_zone_swing_baseline(
    frame: pd.DataFrame,
    mapping: dict[tuple[int, int, str], float],
    fallback: float,
) -> pd.Series:
    zone_key = frame["zone"].astype("string").fillna("NA")
    keys = zip(frame["balls"], frame["strikes"], zone_key)
    values = [
        mapping.get((int(balls), int(strikes), str(zone)), fallback)
        for balls, strikes, zone in keys
    ]
    return pd.Series(values, index=frame.index, dtype="float64").clip(0.0, 1.0)


def decision_surprise(
    actual_swing: pd.Series,
    predicted_swing_probability: np.ndarray | pd.Series,
) -> pd.Series:
    """Observed swing minus expected swing probability.

    This is a behavioral deviation measure, not a run-value metric.
    """
    actual = pd.Series(actual_swing, dtype="float64").reset_index(drop=True)
    expected = pd.Series(
        predicted_swing_probability, dtype="float64"
    ).reset_index(drop=True)
    return actual - expected
