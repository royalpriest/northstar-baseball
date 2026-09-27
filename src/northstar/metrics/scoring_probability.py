from __future__ import annotations

import pandas as pd


def build_scoring_probability_table(plate_appearances: pd.DataFrame) -> pd.DataFrame:
    """Probability of at least one run scoring from each base/out state."""
    required = {"base_out_state_before", "outs_before", "base_state_before", "runs_to_end"}
    missing = required.difference(plate_appearances.columns)
    if missing:
        raise ValueError(f"Missing plate-appearance columns: {sorted(missing)}")

    data = plate_appearances.copy()
    data = data[data["outs_before"].between(0, 2) & data["runs_to_end"].notna()]
    data["scored_before_inning_end"] = data["runs_to_end"].gt(0).astype(int)

    return (
        data.groupby(
            ["outs_before", "base_state_before", "base_out_state_before"], as_index=False
        )["scored_before_inning_end"]
        .agg(score_probability="mean", samples="size")
        .sort_values(["outs_before", "base_state_before"])
        .reset_index(drop=True)
    )
