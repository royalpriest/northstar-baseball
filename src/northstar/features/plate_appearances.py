from __future__ import annotations

import pandas as pd

CORE_COLUMNS = {
    "game_pk",
    "game_date",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "post_bat_score",
    "batter",
    "pitcher",
    "events",
}


def _validate_columns(frame: pd.DataFrame) -> None:
    missing = CORE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required Statcast columns: {sorted(missing)}")


def _base_state(row: pd.Series) -> str:
    return "".join(
        [
            "1" if pd.notna(row["on_1b"]) else "0",
            "1" if pd.notna(row["on_2b"]) else "0",
            "1" if pd.notna(row["on_3b"]) else "0",
        ]
    )


def build_plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    """Collapse pitch-level Statcast rows into plate appearances.

    The first pitch supplies the *start* base/out state. The final pitch supplies
    the PA result and post-play score. Using the final pitch's pre-pitch state as
    the PA start would be wrong when steals, wild pitches, or other movement occur
    during the plate appearance.
    """
    _validate_columns(pitches)
    if pitches.empty:
        return pd.DataFrame()

    data = pitches.copy()
    data["pitch_number"] = pd.to_numeric(data["pitch_number"], errors="coerce")
    data["at_bat_number"] = pd.to_numeric(data["at_bat_number"], errors="coerce")
    data = data.dropna(subset=["game_pk", "at_bat_number", "pitch_number"])
    data = data.sort_values(["game_pk", "at_bat_number", "pitch_number"])

    group_cols = ["game_pk", "at_bat_number"]
    first = data.groupby(group_cols, sort=False, group_keys=False).head(1).copy()
    last = data.groupby(group_cols, sort=False, group_keys=False).tail(1).copy()

    start_cols = [
        "game_pk",
        "at_bat_number",
        "game_date",
        "inning",
        "inning_topbot",
        "outs_when_up",
        "on_1b",
        "on_2b",
        "on_3b",
        "bat_score",
        "batter",
        "pitcher",
    ]
    end_cols = ["game_pk", "at_bat_number", "post_bat_score", "events"]

    pa = first[start_cols].merge(
        last[end_cols], on=["game_pk", "at_bat_number"], how="inner", validate="one_to_one"
    )

    pa = pa.rename(
        columns={
            "outs_when_up": "outs_before",
            "bat_score": "bat_score_before",
            "post_bat_score": "bat_score_after",
        }
    )

    for column in ["outs_before", "bat_score_before", "bat_score_after", "inning"]:
        pa[column] = pd.to_numeric(pa[column], errors="coerce")

    pa["base_state_before"] = pa.apply(_base_state, axis=1)
    pa["base_out_state_before"] = (
        pa["outs_before"].astype("Int64").astype(str) + "_" + pa["base_state_before"]
    )
    pa["runs_scored_on_pa"] = (pa["bat_score_after"] - pa["bat_score_before"]).clip(lower=0)

    half_cols = ["game_pk", "inning", "inning_topbot"]
    inning_final = (
        pa.groupby(half_cols, dropna=False)["bat_score_after"]
        .transform("max")
        .rename("inning_final_bat_score")
    )
    pa["inning_final_bat_score"] = inning_final
    pa["runs_to_end"] = (pa["inning_final_bat_score"] - pa["bat_score_before"]).clip(lower=0)

    pa = pa.sort_values(half_cols + ["at_bat_number"]).reset_index(drop=True)
    grouped = pa.groupby(half_cols, sort=False, dropna=False)
    pa["outs_after"] = grouped["outs_before"].shift(-1)
    pa["base_state_after"] = grouped["base_state_before"].shift(-1)

    is_last_pa = grouped.cumcount(ascending=False).eq(0)
    pa.loc[is_last_pa, "outs_after"] = 3
    pa.loc[is_last_pa, "base_state_after"] = "000"
    pa["outs_after"] = pd.to_numeric(pa["outs_after"], errors="coerce").astype("Int64")
    pa["base_out_state_after"] = (
        pa["outs_after"].astype(str) + "_" + pa["base_state_after"].fillna("000")
    )

    return pa
