from __future__ import annotations

import numpy as np
import pandas as pd

from northstar.features.state_value import DECISION_TRANSITION_TARGET


def fit_count_zone_transition_baseline(
    train: pd.DataFrame,
) -> tuple[
    dict[tuple[int, int, str, int], float],
    dict[int, float],
    float,
]:
    """Training-only mean transition value by count, zone, and action."""
    required = {
        "balls",
        "strikes",
        "zone",
        "is_swing",
        DECISION_TRANSITION_TARGET,
    }
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(
            f"Training transition data is missing: {sorted(missing)}"
        )

    work = train.copy()
    work["zone_key"] = work["zone"].astype("string").fillna("NA")

    grouped = (
        work.groupby(["balls", "strikes", "zone_key", "is_swing"])[
            DECISION_TRANSITION_TARGET
        ]
        .mean()
        .astype(float)
    )
    mapping = {
        (int(balls), int(strikes), str(zone), int(action)): float(value)
        for (balls, strikes, zone, action), value in grouped.items()
    }

    action_fallback = (
        work.groupby("is_swing")[DECISION_TRANSITION_TARGET]
        .mean()
        .astype(float)
        .to_dict()
    )
    overall = float(work[DECISION_TRANSITION_TARGET].mean())
    return mapping, action_fallback, overall


def predict_count_zone_transition_values(
    frame: pd.DataFrame,
    mapping: dict[tuple[int, int, str, int], float],
    action_fallback: dict[int, float],
    overall_fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict take/swing one-step values from the empirical baseline."""
    zone_key = frame["zone"].astype("string").fillna("NA")
    take_values: list[float] = []
    swing_values: list[float] = []

    for balls, strikes, zone in zip(
        frame["balls"],
        frame["strikes"],
        zone_key,
    ):
        key = (int(balls), int(strikes), str(zone))
        take_values.append(
            mapping.get(
                (*key, 0),
                action_fallback.get(0, overall_fallback),
            )
        )
        swing_values.append(
            mapping.get(
                (*key, 1),
                action_fallback.get(1, overall_fallback),
            )
        )

    return (
        np.asarray(take_values, dtype="float64"),
        np.asarray(swing_values, dtype="float64"),
    )
