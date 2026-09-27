from __future__ import annotations

import numpy as np
import pandas as pd


OUTCOME_CLASSES = (
    "strikeout",
    "walk_hbp",
    "single",
    "double_triple",
    "home_run",
    "field_out",
    "sacrifice",
    "reach_other",
)

OUTCOME_TO_CODE = {name: index for index, name in enumerate(OUTCOME_CLASSES)}
CODE_TO_OUTCOME = {index: name for name, index in OUTCOME_TO_CODE.items()}

EVENT_TO_OUTCOME = {
    "strikeout": "strikeout",
    "strikeout_double_play": "strikeout",
    "walk": "walk_hbp",
    "intent_walk": "walk_hbp",
    "intentional_walk": "walk_hbp",
    "hit_by_pitch": "walk_hbp",
    "single": "single",
    "double": "double_triple",
    "triple": "double_triple",
    "home_run": "home_run",
    "field_out": "field_out",
    "force_out": "field_out",
    "grounded_into_double_play": "field_out",
    "double_play": "field_out",
    "triple_play": "field_out",
    "fielders_choice_out": "field_out",
    "sac_fly": "sacrifice",
    "sac_fly_double_play": "sacrifice",
    "sac_bunt": "sacrifice",
    "sac_bunt_double_play": "sacrifice",
    "field_error": "reach_other",
    "fielders_choice": "reach_other",
    "catcher_interf": "reach_other",
    "catchers_interf": "reach_other",
}


def classify_pa_outcome(event: object) -> str | None:
    """Map a Statcast terminal PA event into a stable NorthStar outcome class."""
    if pd.isna(event):
        return None
    normalized = str(event).strip().lower()
    if not normalized:
        return None
    return EVENT_TO_OUTCOME.get(normalized, "reach_other")


def add_outcome_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach outcome class/code to plate appearances."""
    if "events" not in frame.columns:
        raise ValueError("Missing required plate-appearance column: events")

    out = frame.copy()
    out["outcome_class"] = out["events"].map(classify_pa_outcome)
    out = out.loc[out["outcome_class"].notna()].copy()
    out["outcome_code"] = out["outcome_class"].map(OUTCOME_TO_CODE).astype("int64")
    return out


def _daily_prior_outcome_rates(
    frame: pd.DataFrame,
    entity_column: str,
    prefix: str,
) -> pd.DataFrame:
    data = frame[[entity_column, "game_date", "outcome_class"]].copy()

    daily = (
        data.groupby([entity_column, "game_date", "outcome_class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for outcome in OUTCOME_CLASSES:
        if outcome not in daily.columns:
            daily[outcome] = 0

    daily = daily[
        [entity_column, "game_date", *OUTCOME_CLASSES]
    ].sort_values([entity_column, "game_date"]).reset_index(drop=True)

    daily["daily_count"] = daily[list(OUTCOME_CLASSES)].sum(axis=1)
    grouped = daily.groupby(entity_column, sort=False)

    daily[f"{prefix}_prior_count"] = (
        grouped["daily_count"].cumsum() - daily["daily_count"]
    )

    for outcome in OUTCOME_CLASSES:
        prior_count_column = f"{prefix}_prior_{outcome}_count"
        daily[prior_count_column] = (
            grouped[outcome].cumsum() - daily[outcome]
        )

        denominator = daily[f"{prefix}_prior_count"].replace(0, np.nan)
        daily[f"{prefix}_prior_{outcome}_rate"] = (
            daily[prior_count_column] / denominator
        )

    daily[f"{prefix}_prior_log_count"] = np.log1p(
        daily[f"{prefix}_prior_count"]
    )

    keep = [
        entity_column,
        "game_date",
        f"{prefix}_prior_count",
        f"{prefix}_prior_log_count",
    ]
    keep.extend(
        f"{prefix}_prior_{outcome}_rate"
        for outcome in OUTCOME_CLASSES
    )

    return daily[keep]


def add_backward_looking_outcome_context(
    plate_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """Add prior-date batter and pitcher terminal-outcome rates.

    Same-day outcomes are deliberately excluded so a later PA on the same
    calendar date cannot learn from an earlier result when doubleheader timing
    is ambiguous in the processed dataset.
    """
    required = {"game_date", "batter", "pitcher", "events"}
    missing = required.difference(plate_appearances.columns)
    if missing:
        raise ValueError(f"Missing plate-appearance columns: {sorted(missing)}")

    frame = add_outcome_class(plate_appearances)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce")
    frame["pitcher"] = pd.to_numeric(frame["pitcher"], errors="coerce")
    frame = frame.dropna(subset=["game_date", "batter", "pitcher"]).copy()
    frame["batter"] = frame["batter"].astype("int64")
    frame["pitcher"] = frame["pitcher"].astype("int64")

    batter_daily = _daily_prior_outcome_rates(
        frame,
        entity_column="batter",
        prefix="batter_outcome",
    )
    pitcher_daily = _daily_prior_outcome_rates(
        frame,
        entity_column="pitcher",
        prefix="pitcher_outcome",
    )

    frame = frame.merge(
        batter_daily,
        on=["batter", "game_date"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        pitcher_daily,
        on=["pitcher", "game_date"],
        how="left",
        validate="many_to_one",
    )

    return frame
