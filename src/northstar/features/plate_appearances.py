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

OPTIONAL_CONTEXT_COLUMNS = {
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "fld_score",
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


def _action_base_state(row: pd.Series) -> str:
    return "".join(
        [
            "1" if pd.notna(row["action_on_1b"]) else "0",
            "1" if pd.notna(row["action_on_2b"]) else "0",
            "1" if pd.notna(row["action_on_3b"]) else "0",
        ]
    )


def _add_optional_context(first: pd.DataFrame, start_cols: list[str]) -> list[str]:
    """Include context fields when the Statcast payload provides them."""
    for column in sorted(OPTIONAL_CONTEXT_COLUMNS):
        if column in first.columns:
            start_cols.append(column)
    return start_cols


def build_plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    """Collapse pitch-level Statcast rows into plate appearances.

    The literal first pitch supplies the pre-PA state. The literal final pitch
    supplies the terminal batting-event state and result.

    Retaining both states matters for attribution. Runner movement can occur
    between the first and final pitch because of steals, wild pitches, balks,
    caught stealing, or similar events. NorthStar therefore keeps a separate
    terminal-event pre-state so those earlier state changes do not have to be
    credited to the batter's final PA event.

    Optional pre-PA context fields (handedness, teams, fielding score) are
    retained when present so later models can use information known before the PA.
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
    start_cols = _add_optional_context(first, start_cols)

    terminal = last[
        [
            "game_pk",
            "at_bat_number",
            "outs_when_up",
            "on_1b",
            "on_2b",
            "on_3b",
            "bat_score",
            "post_bat_score",
            "events",
        ]
    ].copy()
    terminal = terminal.rename(
        columns={
            "outs_when_up": "action_outs_before",
            "on_1b": "action_on_1b",
            "on_2b": "action_on_2b",
            "on_3b": "action_on_3b",
            "bat_score": "action_bat_score_before",
            "post_bat_score": "bat_score_after",
        }
    )

    pa = first[start_cols].merge(
        terminal,
        on=["game_pk", "at_bat_number"],
        how="inner",
        validate="one_to_one",
    )

    pa = pa.rename(
        columns={
            "outs_when_up": "outs_before",
            "bat_score": "bat_score_before",
            "stand": "batter_hand",
            "p_throws": "pitcher_hand",
            "fld_score": "field_score_before",
        }
    )

    for column in [
        "outs_before",
        "action_outs_before",
        "bat_score_before",
        "action_bat_score_before",
        "bat_score_after",
        "field_score_before",
        "inning",
    ]:
        if column in pa.columns:
            pa[column] = pd.to_numeric(pa[column], errors="coerce")

    pa["base_state_before"] = pa.apply(_base_state, axis=1)
    pa["base_out_state_before"] = (
        pa["outs_before"].astype("Int64").astype(str) + "_" + pa["base_state_before"]
    )

    pa["action_base_state_before"] = pa.apply(_action_base_state, axis=1)
    pa["action_base_out_state_before"] = (
        pa["action_outs_before"].astype("Int64").astype(str)
        + "_"
        + pa["action_base_state_before"]
    )

    # Whole-PA scoring change preserves the traditional RE24 benchmark.
    pa["runs_scored_on_pa"] = (
        pa["bat_score_after"] - pa["bat_score_before"]
    ).clip(lower=0)

    # Terminal-event scoring change excludes runs that occurred on earlier pitches.
    pa["runs_scored_on_terminal_event"] = (
        pa["bat_score_after"] - pa["action_bat_score_before"]
    ).clip(lower=0)

    pa["runs_scored_before_terminal_event"] = (
        pa["action_bat_score_before"] - pa["bat_score_before"]
    ).clip(lower=0)

    if "field_score_before" in pa.columns:
        pa["score_diff_before"] = (
            pa["bat_score_before"] - pa["field_score_before"]
        )

    if {"home_team", "away_team"}.issubset(pa.columns):
        is_top = pa["inning_topbot"].astype(str).str.lower().eq("top")
        pa["batting_team"] = pa["home_team"].where(~is_top, pa["away_team"])
        pa["fielding_team"] = pa["away_team"].where(~is_top, pa["home_team"])
        pa["is_home_batting"] = (~is_top).astype(int)
        pa["park_team"] = pa["home_team"]

    half_cols = ["game_pk", "inning", "inning_topbot"]
    inning_final = (
        pa.groupby(half_cols, dropna=False)["bat_score_after"]
        .transform("max")
        .rename("inning_final_bat_score")
    )
    pa["inning_final_bat_score"] = inning_final
    pa["runs_to_end"] = (
        pa["inning_final_bat_score"] - pa["bat_score_before"]
    ).clip(lower=0)

    pa = pa.sort_values(half_cols + ["at_bat_number"]).reset_index(drop=True)
    grouped = pa.groupby(half_cols, sort=False, dropna=False)
    pa["outs_after"] = grouped["outs_before"].shift(-1)
    pa["base_state_after"] = grouped["base_state_before"].shift(-1)

    is_last_pa = grouped.cumcount(ascending=False).eq(0)
    pa.loc[is_last_pa, "outs_after"] = 3
    pa.loc[is_last_pa, "base_state_after"] = "000"
    pa["outs_after"] = pd.to_numeric(
        pa["outs_after"], errors="coerce"
    ).astype("Int64")
    pa["base_out_state_after"] = (
        pa["outs_after"].astype(str)
        + "_"
        + pa["base_state_after"].fillna("000")
    )

    return pa
