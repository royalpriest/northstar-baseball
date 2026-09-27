from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from northstar.features.decision_value import PITCH_VALUE_TARGET


@dataclass(frozen=True)
class PolicyValue:
    policy: str
    value: float
    standard_error: float
    swing_share: float
    pitches: int


def fit_count_zone_action_value_baseline(
    train: pd.DataFrame,
) -> tuple[dict[tuple[int, int, str, int], float], dict[int, float], float]:
    """Training-only mean run value by count, zone, and observed action."""
    required = {"balls", "strikes", "zone", "is_swing", PITCH_VALUE_TARGET}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training pitch data is missing: {sorted(missing)}")

    work = train.copy()
    work["zone_key"] = work["zone"].astype("string").fillna("NA")

    grouped = (
        work.groupby(["balls", "strikes", "zone_key", "is_swing"])[
            PITCH_VALUE_TARGET
        ]
        .mean()
        .astype(float)
    )
    mapping = {
        (int(balls), int(strikes), str(zone), int(action)): float(value)
        for (balls, strikes, zone, action), value in grouped.items()
    }
    action_fallback = (
        work.groupby("is_swing")[PITCH_VALUE_TARGET]
        .mean()
        .astype(float)
        .to_dict()
    )
    overall = float(work[PITCH_VALUE_TARGET].mean())
    return mapping, action_fallback, overall


def predict_count_zone_action_values(
    frame: pd.DataFrame,
    mapping: dict[tuple[int, int, str, int], float],
    action_fallback: dict[int, float],
    overall_fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    zone_key = frame["zone"].astype("string").fillna("NA")
    swing_values: list[float] = []
    take_values: list[float] = []

    for balls, strikes, zone in zip(
        frame["balls"],
        frame["strikes"],
        zone_key,
    ):
        key_base = (int(balls), int(strikes), str(zone))
        take_values.append(
            mapping.get(
                (*key_base, 0),
                action_fallback.get(0, overall_fallback),
            )
        )
        swing_values.append(
            mapping.get(
                (*key_base, 1),
                action_fallback.get(1, overall_fallback),
            )
        )

    return (
        np.asarray(take_values, dtype="float64"),
        np.asarray(swing_values, dtype="float64"),
    )


def factual_action_prediction(
    actual_swing: pd.Series | np.ndarray,
    q_take: np.ndarray,
    q_swing: np.ndarray,
) -> np.ndarray:
    action = np.asarray(actual_swing, dtype="int8")
    return np.where(action == 1, q_swing, q_take)


def modeled_action_advantage(
    actual_swing: pd.Series | np.ndarray,
    q_take: np.ndarray,
    q_swing: np.ndarray,
) -> np.ndarray:
    """Modeled value of the chosen action relative to the alternative."""
    action = np.asarray(actual_swing, dtype="int8")
    chosen = np.where(action == 1, q_swing, q_take)
    alternative = np.where(action == 1, q_take, q_swing)
    return chosen - alternative


def optimal_action(q_take: np.ndarray, q_swing: np.ndarray) -> np.ndarray:
    return (q_swing >= q_take).astype("int8")


def doubly_robust_policy_value(
    outcome: pd.Series | np.ndarray,
    actual_swing: pd.Series | np.ndarray,
    swing_probability: pd.Series | np.ndarray,
    q_take: np.ndarray,
    q_swing: np.ndarray,
    policy_action: pd.Series | np.ndarray,
    policy_name: str,
) -> PolicyValue:
    """Estimate an observational policy value with a doubly robust score.

    This is a diagnostic, not a causal guarantee. It assumes the observed
    opportunity features are sufficient to adjust action selection.
    """
    y = np.asarray(outcome, dtype="float64")
    action = np.asarray(actual_swing, dtype="int8")
    propensity = np.clip(
        np.asarray(swing_probability, dtype="float64"),
        1e-4,
        1.0 - 1e-4,
    )
    policy = np.asarray(policy_action, dtype="int8")

    q_policy = np.where(policy == 1, q_swing, q_take)
    q_observed = np.where(action == 1, q_swing, q_take)
    observed_probability = np.where(
        action == 1,
        propensity,
        1.0 - propensity,
    )
    matches = action == policy

    score = q_policy + (
        matches.astype("float64")
        * (y - q_observed)
        / observed_probability
    )

    return PolicyValue(
        policy=policy_name,
        value=float(score.mean()),
        standard_error=float(score.std(ddof=1) / np.sqrt(len(score))),
        swing_share=float(policy.mean()),
        pitches=int(len(score)),
    )
