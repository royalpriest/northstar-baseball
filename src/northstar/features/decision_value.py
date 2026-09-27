from __future__ import annotations

import numpy as np
import pandas as pd


PITCH_VALUE_TARGET = "pitch_to_pa_end_re24"


def attach_pitch_value_target(
    pitch_frame: pd.DataFrame,
    plate_appearances: pd.DataFrame,
    run_expectancy: pd.DataFrame,
) -> pd.DataFrame:
    """Attach realized run value from the current pitch state to PA completion.

    The target is:

        runs scored from this pitch through PA end
        + RE(after PA)
        - RE(current pitch pre-state)

    Run expectancy comes from a training-only table supplied by the caller.
    """
    pitch_required = {
        "game_pk",
        "at_bat_number",
        "bat_score",
        "base_out_state_before",
    }
    pa_required = {
        "game_pk",
        "at_bat_number",
        "bat_score_after",
        "base_out_state_after",
        "outs_after",
    }
    expectancy_required = {"base_out_state_before", "expected_runs"}

    missing = pitch_required.difference(pitch_frame.columns)
    if missing:
        raise ValueError(f"Pitch data is missing columns: {sorted(missing)}")
    missing = pa_required.difference(plate_appearances.columns)
    if missing:
        raise ValueError(f"PA data is missing columns: {sorted(missing)}")
    missing = expectancy_required.difference(run_expectancy.columns)
    if missing:
        raise ValueError(f"Run expectancy is missing columns: {sorted(missing)}")

    mapping = run_expectancy.set_index("base_out_state_before")[
        "expected_runs"
    ].to_dict()

    pa = plate_appearances[
        [
            "game_pk",
            "at_bat_number",
            "bat_score_after",
            "base_out_state_after",
            "outs_after",
        ]
    ].copy()
    pa["pa_re_after"] = pa["base_out_state_after"].map(mapping)
    pa.loc[pd.to_numeric(pa["outs_after"], errors="coerce").eq(3), "pa_re_after"] = 0.0

    out = pitch_frame.merge(
        pa,
        on=["game_pk", "at_bat_number"],
        how="inner",
        validate="many_to_one",
    )

    out["pitch_re_before"] = out["base_out_state_before"].map(mapping)
    out["bat_score"] = pd.to_numeric(out["bat_score"], errors="coerce")
    out["bat_score_after"] = pd.to_numeric(
        out["bat_score_after"], errors="coerce"
    )
    out["runs_remaining_in_pa"] = (
        out["bat_score_after"] - out["bat_score"]
    ).clip(lower=0)

    out[PITCH_VALUE_TARGET] = (
        out["runs_remaining_in_pa"]
        + out["pa_re_after"]
        - out["pitch_re_before"]
    )

    out = out.dropna(
        subset=[
            PITCH_VALUE_TARGET,
            "pitch_re_before",
            "pa_re_after",
        ]
    ).copy()

    return out.reset_index(drop=True)


def overlap_mask(
    swing_probability: pd.Series | np.ndarray,
    lower: float = 0.05,
    upper: float = 0.95,
) -> np.ndarray:
    probabilities = np.asarray(swing_probability, dtype="float64")
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("Overlap bounds must satisfy 0 < lower < upper < 1.")
    return (probabilities >= lower) & (probabilities <= upper)


def stabilized_action_weights(
    actual_swing: pd.Series | np.ndarray,
    swing_probability: pd.Series | np.ndarray,
    clip: float = 10.0,
) -> np.ndarray:
    """Compute stabilized inverse-propensity weights for observed actions."""
    action = np.asarray(actual_swing, dtype="float64")
    propensity = np.asarray(swing_probability, dtype="float64")
    propensity = np.clip(propensity, 1e-6, 1.0 - 1e-6)

    swing_rate = float(action.mean())
    weights = np.where(
        action == 1.0,
        swing_rate / propensity,
        (1.0 - swing_rate) / (1.0 - propensity),
    )
    return np.clip(weights, 0.0, clip)
