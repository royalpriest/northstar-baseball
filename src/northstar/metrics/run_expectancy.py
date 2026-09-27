from __future__ import annotations

from itertools import product

import pandas as pd


def _all_re24_states() -> pd.DataFrame:
    states = []
    for outs, bases in product(range(3), product("01", repeat=3)):
        base_state = "".join(bases)
        states.append(
            {
                "outs_before": outs,
                "base_state_before": base_state,
                "base_out_state_before": f"{outs}_{base_state}",
            }
        )
    return pd.DataFrame(states)


def build_run_expectancy_table(
    plate_appearances: pd.DataFrame, min_samples: int = 1
) -> pd.DataFrame:
    """Build an empirical 24-state run-expectancy table."""
    required = {"outs_before", "base_state_before", "base_out_state_before", "runs_to_end"}
    missing = required.difference(plate_appearances.columns)
    if missing:
        raise ValueError(f"Missing plate-appearance columns: {sorted(missing)}")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    valid = plate_appearances[
        plate_appearances["outs_before"].between(0, 2)
        & plate_appearances["runs_to_end"].notna()
    ].copy()

    summary = (
        valid.groupby(
            ["outs_before", "base_state_before", "base_out_state_before"],
            as_index=False,
        )["runs_to_end"]
        .agg(expected_runs="mean", samples="size")
    )
    summary.loc[summary["samples"] < min_samples, "expected_runs"] = pd.NA

    return _all_re24_states().merge(
        summary,
        on=["outs_before", "base_state_before", "base_out_state_before"],
        how="left",
    )


def add_re24(plate_appearances: pd.DataFrame, expectancy: pd.DataFrame) -> pd.DataFrame:
    """Attach empirical Run Expectancy Added (RE24) to each plate appearance."""
    mapping = expectancy.set_index("base_out_state_before")["expected_runs"].to_dict()
    out = plate_appearances.copy()
    out["re_before"] = out["base_out_state_before"].map(mapping)

    out["re_after"] = out["base_out_state_after"].map(mapping)
    out.loc[out["outs_after"].eq(3), "re_after"] = 0.0

    out["re24"] = out["runs_scored_on_pa"] + out["re_after"] - out["re_before"]
    return out
