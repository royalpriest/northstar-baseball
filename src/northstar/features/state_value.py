from __future__ import annotations

import numpy as np
import pandas as pd


DECISION_TRANSITION_TARGET = "decision_transition_value"
ABSOLUTE_STATE_RETURN = "absolute_future_pa_value"


def _state_key(
    balls: object,
    strikes: object,
    base_out_state: object,
) -> str:
    return f"{int(balls)}-{int(strikes)}|{base_out_state}"


def attach_absolute_future_value(
    decision_frame: pd.DataFrame,
    plate_appearances: pd.DataFrame,
    run_expectancy: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the absolute future offensive value from each decision opportunity.

    Absolute future value is:

        runs from current decision opportunity through PA end + RE(after PA)

    The supplied run-expectancy table must be training-only.
    """
    decision_required = {
        "game_pk",
        "at_bat_number",
        "balls",
        "strikes",
        "base_out_state_before",
        "bat_score",
    }
    pa_required = {
        "game_pk",
        "at_bat_number",
        "bat_score_after",
        "base_out_state_after",
        "outs_after",
    }
    expectancy_required = {"base_out_state_before", "expected_runs"}

    missing = decision_required.difference(decision_frame.columns)
    if missing:
        raise ValueError(f"Decision frame is missing: {sorted(missing)}")
    missing = pa_required.difference(plate_appearances.columns)
    if missing:
        raise ValueError(f"PA data is missing: {sorted(missing)}")
    missing = expectancy_required.difference(run_expectancy.columns)
    if missing:
        raise ValueError(f"Run expectancy is missing: {sorted(missing)}")

    re_mapping = run_expectancy.set_index("base_out_state_before")[
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
    pa["pa_re_after"] = pa["base_out_state_after"].map(re_mapping)
    pa.loc[
        pd.to_numeric(pa["outs_after"], errors="coerce").eq(3),
        "pa_re_after",
    ] = 0.0

    out = decision_frame.merge(
        pa,
        on=["game_pk", "at_bat_number"],
        how="inner",
        validate="many_to_one",
    )

    out["bat_score"] = pd.to_numeric(out["bat_score"], errors="coerce")
    out["bat_score_after"] = pd.to_numeric(
        out["bat_score_after"], errors="coerce"
    )
    out["runs_remaining_in_pa"] = (
        out["bat_score_after"] - out["bat_score"]
    ).clip(lower=0)
    out[ABSOLUTE_STATE_RETURN] = (
        out["runs_remaining_in_pa"] + out["pa_re_after"]
    )

    out["decision_state"] = [
        _state_key(balls, strikes, base_out)
        for balls, strikes, base_out in zip(
            out["balls"],
            out["strikes"],
            out["base_out_state_before"],
        )
    ]

    return out.dropna(
        subset=[ABSOLUTE_STATE_RETURN, "pa_re_after"]
    ).reset_index(drop=True)


def fit_decision_state_values(
    training_frame: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float], float, pd.DataFrame]:
    """Fit training-only count/base-out state values.

    Returns exact count+base/out values, base/out fallbacks, an overall fallback,
    and a table suitable for audit/reporting.
    """
    required = {
        "decision_state",
        "base_out_state_before",
        ABSOLUTE_STATE_RETURN,
    }
    missing = required.difference(training_frame.columns)
    if missing:
        raise ValueError(f"Training state data is missing: {sorted(missing)}")

    exact_table = (
        training_frame.groupby(
            ["decision_state", "base_out_state_before"],
            as_index=False,
        )[ABSOLUTE_STATE_RETURN]
        .agg(state_value="mean", samples="size")
    )
    exact_mapping = exact_table.set_index("decision_state")[
        "state_value"
    ].astype(float).to_dict()

    base_out_mapping = (
        training_frame.groupby("base_out_state_before")[
            ABSOLUTE_STATE_RETURN
        ]
        .mean()
        .astype(float)
        .to_dict()
    )
    overall = float(training_frame[ABSOLUTE_STATE_RETURN].mean())

    return exact_mapping, base_out_mapping, overall, exact_table


def lookup_decision_state_value(
    state: str,
    base_out_state: str,
    exact_mapping: dict[str, float],
    base_out_mapping: dict[str, float],
    overall_fallback: float,
) -> float:
    return float(
        exact_mapping.get(
            str(state),
            base_out_mapping.get(str(base_out_state), overall_fallback),
        )
    )


def attach_decision_transition_values(
    frame: pd.DataFrame,
    exact_mapping: dict[str, float],
    base_out_mapping: dict[str, float],
    overall_fallback: float,
) -> pd.DataFrame:
    """Attach one-step decision-opportunity transition values.

    A step runs from one recognized physical swing/take opportunity to the next.
    If the PA terminates, the next value is RE(after PA). The target is:

        runs during the step + V(next decision state) - V(current state)

    Summed over all recognized decision opportunities in a PA, these values
    telescope to the PA's runs plus terminal RE minus the first decision-state
    value. This avoids assigning the entire downstream PA return to every pitch.
    """
    required = {
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "balls",
        "strikes",
        "base_out_state_before",
        "decision_state",
        "bat_score",
        "bat_score_after",
        "pa_re_after",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Decision-state frame is missing: {sorted(missing)}")

    out = frame.copy()
    out = out.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    group_cols = ["game_pk", "at_bat_number"]
    grouped = out.groupby(group_cols, sort=False, dropna=False)

    out["current_state_value"] = [
        lookup_decision_state_value(
            state,
            base_out,
            exact_mapping,
            base_out_mapping,
            overall_fallback,
        )
        for state, base_out in zip(
            out["decision_state"],
            out["base_out_state_before"],
        )
    ]

    out["next_pitch_number"] = grouped["pitch_number"].shift(-1)
    out["next_balls"] = grouped["balls"].shift(-1)
    out["next_strikes"] = grouped["strikes"].shift(-1)
    out["next_base_out_state"] = grouped[
        "base_out_state_before"
    ].shift(-1)
    out["next_bat_score"] = grouped["bat_score"].shift(-1)

    out["has_next_decision"] = grouped.cumcount(ascending=False).gt(0)
    pitch_gap = pd.to_numeric(out["next_pitch_number"], errors="coerce") - pd.to_numeric(
        out["pitch_number"], errors="coerce"
    )
    out["decision_pitch_gap"] = pitch_gap
    out["clean_one_step_transition"] = (
        ~out["has_next_decision"]
        | pitch_gap.eq(1)
    )

    next_state_values: list[float] = []
    interval_runs: list[float] = []

    for row in out.itertuples(index=False):
        if bool(row.has_next_decision):
            next_state = _state_key(
                row.next_balls,
                row.next_strikes,
                row.next_base_out_state,
            )
            next_value = lookup_decision_state_value(
                next_state,
                str(row.next_base_out_state),
                exact_mapping,
                base_out_mapping,
                overall_fallback,
            )
            next_score = float(row.next_bat_score)
        else:
            next_value = float(row.pa_re_after)
            next_score = float(row.bat_score_after)

        next_state_values.append(next_value)
        interval_runs.append(max(next_score - float(row.bat_score), 0.0))

    out["next_state_value"] = np.asarray(
        next_state_values, dtype="float64"
    )
    out["interval_runs"] = np.asarray(interval_runs, dtype="float64")
    out[DECISION_TRANSITION_TARGET] = (
        out["interval_runs"]
        + out["next_state_value"]
        - out["current_state_value"]
    )

    return out


def build_telescoping_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Verify that transition values telescope within each PA."""
    required = {
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "bat_score",
        "bat_score_after",
        "pa_re_after",
        "current_state_value",
        DECISION_TRANSITION_TARGET,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Transition audit is missing: {sorted(missing)}")

    ordered = frame.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).copy()

    records: list[dict[str, float | int]] = []
    for (game_pk, at_bat_number), group in ordered.groupby(
        ["game_pk", "at_bat_number"],
        sort=False,
    ):
        first = group.iloc[0]
        last = group.iloc[-1]
        expected_total = (
            max(
                float(last["bat_score_after"])
                - float(first["bat_score"]),
                0.0,
            )
            + float(last["pa_re_after"])
            - float(first["current_state_value"])
        )
        transition_total = float(
            group[DECISION_TRANSITION_TARGET].sum()
        )
        records.append(
            {
                "game_pk": int(game_pk),
                "at_bat_number": int(at_bat_number),
                "decision_opportunities": int(len(group)),
                "transition_total": transition_total,
                "expected_telescoping_total": expected_total,
                "telescoping_error": transition_total - expected_total,
            }
        )

    return pd.DataFrame.from_records(records)
