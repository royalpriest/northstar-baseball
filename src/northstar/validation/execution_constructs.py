from __future__ import annotations

import numpy as np
import pandas as pd

from northstar.features.pitches import SWING_DESCRIPTIONS
from northstar.metrics.execution_v2 import (
    BALL_IN_PLAY_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
)


CONTACT_CONSTRUCT_FEATURE_SETS = {
    "raw_contact": [
        "statcast_contact_rate",
    ],
    "situational_contact": [
        "statcast_contact_rate",
        "statcast_zone_contact_rate",
        "statcast_chase_contact_rate",
        "statcast_two_strike_contact_rate",
    ],
}

DAMAGE_CONSTRUCT_FEATURE_SETS = {
    "raw_xwoba_contact": [
        "statcast_xwoba_on_contact",
    ],
    "batted_ball_quality": [
        "statcast_avg_exit_velocity",
        "statcast_hard_hit_rate",
        "statcast_barrel_rate",
        "statcast_sweet_spot_rate",
    ],
    "full_contact_quality": [
        "statcast_xwoba_on_contact",
        "statcast_avg_exit_velocity",
        "statcast_hard_hit_rate",
        "statcast_barrel_rate",
        "statcast_sweet_spot_rate",
    ],
}


def build_contact_skill_summary(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate conventional contact-skill constructs by batter."""
    required = {
        "batter",
        "description",
        "zone",
        "strikes",
    }
    missing = required.difference(pitches.columns)
    if missing:
        raise ValueError(
            f"Contact construct data is missing: {sorted(missing)}"
        )

    frame = pitches.loc[:, sorted(required)].copy()
    frame["description"] = (
        frame["description"].astype("string").str.strip().str.lower()
    )
    frame["batter"] = pd.to_numeric(
        frame["batter"],
        errors="coerce",
    )
    frame["zone"] = pd.to_numeric(
        frame["zone"],
        errors="coerce",
    )
    frame["strikes"] = pd.to_numeric(
        frame["strikes"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["batter", "strikes"])
    frame["batter"] = frame["batter"].astype("int64")

    frame = frame.loc[
        frame["description"].isin(SWING_DESCRIPTIONS)
    ].copy()
    frame["is_contact"] = ~frame["description"].isin(
        WHIFF_DESCRIPTIONS
    )
    frame["in_zone"] = frame["zone"].between(
        1,
        9,
        inclusive="both",
    )
    frame["out_zone"] = frame["zone"].between(
        11,
        14,
        inclusive="both",
    )
    frame["two_strike"] = frame["strikes"].eq(2)

    rows: list[dict[str, object]] = []
    for batter, group in frame.groupby("batter", sort=False):
        in_zone = group["in_zone"]
        out_zone = group["out_zone"]
        two_strike = group["two_strike"]

        swings = len(group)
        contacts = int(group["is_contact"].sum())

        rows.append(
            {
                "batter": int(batter),
                "statcast_swings": swings,
                "statcast_contact_rate": (
                    contacts / swings if swings else np.nan
                ),
                "statcast_whiff_per_swing": (
                    1.0 - contacts / swings if swings else np.nan
                ),
                "statcast_zone_swings": int(in_zone.sum()),
                "statcast_zone_contact_rate": (
                    float(group.loc[in_zone, "is_contact"].mean())
                    if in_zone.any()
                    else np.nan
                ),
                "statcast_chase_swings": int(out_zone.sum()),
                "statcast_chase_contact_rate": (
                    float(group.loc[out_zone, "is_contact"].mean())
                    if out_zone.any()
                    else np.nan
                ),
                "statcast_two_strike_swings": int(two_strike.sum()),
                "statcast_two_strike_contact_rate": (
                    float(group.loc[two_strike, "is_contact"].mean())
                    if two_strike.any()
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_damage_skill_summary(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate conventional quality-of-contact constructs by batter."""
    required = {
        "batter",
        "description",
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "estimated_woba_using_speedangle",
    }
    missing = required.difference(pitches.columns)
    if missing:
        raise ValueError(
            f"Damage construct data is missing: {sorted(missing)}"
        )

    frame = pitches.loc[:, sorted(required)].copy()
    frame["description"] = (
        frame["description"].astype("string").str.strip().str.lower()
    )
    frame["batter"] = pd.to_numeric(
        frame["batter"],
        errors="coerce",
    )
    for column in (
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "estimated_woba_using_speedangle",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.loc[
        frame["description"].isin(BALL_IN_PLAY_DESCRIPTIONS)
        & frame["batter"].notna()
    ].copy()
    frame["batter"] = frame["batter"].astype("int64")

    rows: list[dict[str, object]] = []
    for batter, group in frame.groupby("batter", sort=False):
        exit_velocity = group["launch_speed"].dropna()
        launch_angle = group["launch_angle"].dropna()
        xwoba_contact = group[
            "estimated_woba_using_speedangle"
        ].dropna()
        launch_speed_angle = group["launch_speed_angle"].dropna()

        rows.append(
            {
                "batter": int(batter),
                "statcast_batted_balls": len(group),
                "statcast_ev_batted_balls": len(exit_velocity),
                "statcast_avg_exit_velocity": (
                    float(exit_velocity.mean())
                    if len(exit_velocity)
                    else np.nan
                ),
                "statcast_hard_hit_rate": (
                    float(exit_velocity.ge(95.0).mean())
                    if len(exit_velocity)
                    else np.nan
                ),
                "statcast_sweet_spot_rate": (
                    float(
                        launch_angle.between(
                            8.0,
                            32.0,
                            inclusive="both",
                        ).mean()
                    )
                    if len(launch_angle)
                    else np.nan
                ),
                "statcast_barrel_rate": (
                    float(launch_speed_angle.eq(6.0).mean())
                    if len(launch_speed_angle)
                    else np.nan
                ),
                "statcast_xwoba_on_contact": (
                    float(xwoba_contact.mean())
                    if len(xwoba_contact)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def merge_execution_construct_data(
    execution: pd.DataFrame,
    constructs: pd.DataFrame,
    outcome_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge one NorthStar execution component with conventional constructs."""
    if "batter" not in execution.columns:
        raise ValueError("execution must contain batter")
    if "batter" not in constructs.columns:
        raise ValueError("constructs must contain batter")

    merged = execution.merge(
        constructs,
        on="batter",
        how="left",
        validate="one_to_one",
    )

    if outcome_reference is not None and not outcome_reference.empty:
        reference = outcome_reference.copy()
        if "batter" not in reference.columns:
            if "player_id" not in reference.columns:
                raise ValueError(
                    "outcome_reference must contain batter or player_id"
                )
            reference["batter"] = pd.to_numeric(
                reference["player_id"],
                errors="coerce",
            )

        reference["batter"] = pd.to_numeric(
            reference["batter"],
            errors="coerce",
        )
        reference = reference.dropna(subset=["batter"])
        reference["batter"] = reference["batter"].astype("int64")

        keep = [
            column
            for column in [
                "batter",
                "re24_per_100_pa",
                "woba",
                "est_woba",
            ]
            if column in reference.columns
        ]
        merged = merged.merge(
            reference.loc[:, keep].drop_duplicates("batter"),
            on="batter",
            how="left",
            validate="one_to_one",
        )

    return merged


def merge_half_execution_constructs(
    half_execution: pd.DataFrame,
    half_constructs: pd.DataFrame,
) -> pd.DataFrame:
    required = {"batter", "season_half"}
    for label, frame in [
        ("half_execution", half_execution),
        ("half_constructs", half_constructs),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{label} is missing: {sorted(missing)}"
            )

    return half_execution.merge(
        half_constructs,
        on=["batter", "season_half"],
        how="left",
        validate="one_to_one",
    )
