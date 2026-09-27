from __future__ import annotations

import numpy as np
import pandas as pd


SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "swinging_pitchout",
    "foul",
    "foul_tip",
    "foul_bunt",
    "foul_pitchout",
    "bunt_foul_tip",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

TAKE_DESCRIPTIONS = {
    "ball",
    "blocked_ball",
    "ball_in_dirt",
    "called_strike",
    "pitchout",
    "hit_by_pitch",
}

PITCH_DECISION_REQUIRED_COLUMNS = {
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "balls",
    "strikes",
    "description",
    "stand",
    "p_throws",
    "pitch_type",
    "release_speed",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "pfx_x",
    "pfx_z",
    "zone",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "fld_score",
}


def classify_pitch_decision(description: object) -> str | None:
    """Classify a Statcast pitch description as a batter swing or take."""
    if pd.isna(description):
        return None

    normalized = str(description).strip().lower()
    if normalized in SWING_DESCRIPTIONS:
        return "swing"
    if normalized in TAKE_DESCRIPTIONS:
        return "take"
    # Clock violations / automatic balls and strikes are not physical pitch
    # opportunities, so they are intentionally excluded rather than called takes.
    return None


def _base_state(row: pd.Series) -> str:
    return "".join(
        [
            "1" if pd.notna(row["on_1b"]) else "0",
            "1" if pd.notna(row["on_2b"]) else "0",
            "1" if pd.notna(row["on_3b"]) else "0",
        ]
    )


def build_pitch_decision_frame(pitches: pd.DataFrame) -> pd.DataFrame:
    """Prepare pitch-level batter-decision features from raw Statcast rows.

    This is an *opportunity-conditioned* model, not a pre-pitch forecast. Exact
    pitch location and movement are allowed because the question is how a hitter
    responds to the pitch that was actually presented.

    Post-contact variables (launch speed, launch angle, events, etc.) are not used
    as decision features.
    """
    missing = PITCH_DECISION_REQUIRED_COLUMNS.difference(pitches.columns)
    if missing:
        raise ValueError(f"Missing Statcast pitch columns: {sorted(missing)}")

    frame = pitches.copy()
    frame["decision"] = frame["description"].map(classify_pitch_decision)
    frame = frame.loc[frame["decision"].notna()].copy()
    frame["is_swing"] = frame["decision"].eq("swing").astype(int)

    numeric_columns = [
        "balls",
        "strikes",
        "release_speed",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "pfx_x",
        "pfx_z",
        "zone",
        "inning",
        "outs_when_up",
        "bat_score",
        "fld_score",
        "pitch_number",
        "at_bat_number",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["base_state_before"] = frame.apply(_base_state, axis=1)
    frame["base_out_state_before"] = (
        frame["outs_when_up"].astype("Int64").astype(str)
        + "_"
        + frame["base_state_before"]
    )
    frame["score_diff_before"] = frame["bat_score"] - frame["fld_score"]
    frame["is_home_batting"] = (
        frame["inning_topbot"].astype(str).str.lower().eq("bottom")
    ).astype(int)

    frame["batter_hand"] = frame["stand"].astype("string")
    frame["pitcher_hand"] = frame["p_throws"].astype("string")

    # Re-orient horizontal features from the hitter's perspective so positive
    # values have the same inside/outside meaning for R/L hitters.
    left_handed = frame["batter_hand"].eq("L")
    frame["plate_x_batter"] = frame["plate_x"].where(
        ~left_handed, -frame["plate_x"]
    )
    frame["pfx_x_batter"] = frame["pfx_x"].where(
        ~left_handed, -frame["pfx_x"]
    )

    zone_height = (frame["sz_top"] - frame["sz_bot"]).replace(0, np.nan)
    frame["plate_z_norm"] = (
        frame["plate_z"] - frame["sz_bot"]
    ) / zone_height

    frame = frame.loc[
        frame["game_date"].notna()
        & frame["balls"].between(0, 3)
        & frame["strikes"].between(0, 2)
        & frame["outs_when_up"].between(0, 2)
    ].copy()

    frame["balls"] = frame["balls"].astype("int64")
    frame["strikes"] = frame["strikes"].astype("int64")
    frame["outs_when_up"] = frame["outs_when_up"].astype("int64")
    frame["inning"] = frame["inning"].astype("Int64")
    frame["zone"] = frame["zone"].astype("Int64")

    return frame.reset_index(drop=True)
