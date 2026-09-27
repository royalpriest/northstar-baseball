from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "game_date",
    "batter",
    "pitcher",
    "re24",
}


def _daily_prior_features(
    frame: pd.DataFrame,
    entity_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Build prior-to-date expanding features for one player entity.

    The current calendar day's results are excluded. This is intentionally
    conservative: it prevents same-day ordering ambiguity (for example,
    doubleheaders) from leaking information into a pre-PA feature.
    """
    data = frame[[entity_column, "game_date", "re24"]].copy()
    data["positive_re24"] = data["re24"].gt(0).astype(int)

    daily = (
        data.groupby([entity_column, "game_date"], as_index=False)
        .agg(
            daily_count=("re24", "size"),
            daily_re24_sum=("re24", "sum"),
            daily_positive=("positive_re24", "sum"),
        )
        .sort_values([entity_column, "game_date"])
        .reset_index(drop=True)
    )

    grouped = daily.groupby(entity_column, sort=False)

    daily[f"{prefix}_prior_count"] = (
        grouped["daily_count"].cumsum() - daily["daily_count"]
    )
    daily[f"{prefix}_prior_re24_sum"] = (
        grouped["daily_re24_sum"].cumsum() - daily["daily_re24_sum"]
    )
    daily[f"{prefix}_prior_positive"] = (
        grouped["daily_positive"].cumsum() - daily["daily_positive"]
    )

    count = daily[f"{prefix}_prior_count"].replace(0, np.nan)
    daily[f"{prefix}_prior_re24_per_pa"] = (
        daily[f"{prefix}_prior_re24_sum"] / count
    )
    daily[f"{prefix}_prior_positive_pa_pct"] = (
        daily[f"{prefix}_prior_positive"] / count
    ) * 100.0
    daily[f"{prefix}_prior_log_count"] = np.log1p(
        daily[f"{prefix}_prior_count"]
    )

    keep = [
        entity_column,
        "game_date",
        f"{prefix}_prior_count",
        f"{prefix}_prior_re24_per_pa",
        f"{prefix}_prior_positive_pa_pct",
        f"{prefix}_prior_log_count",
    ]
    return daily[keep]


def add_backward_looking_player_context(
    plate_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe batter and pitcher history features.

    Features use only results from dates strictly before the current PA date.
    Batter RE24 reflects prior offensive value; pitcher RE24 reflects the
    offensive value previously allowed to opposing hitters.
    """
    missing = REQUIRED_COLUMNS.difference(plate_appearances.columns)
    if missing:
        raise ValueError(f"Missing plate-appearance columns: {sorted(missing)}")

    frame = plate_appearances.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce")
    frame["pitcher"] = pd.to_numeric(frame["pitcher"], errors="coerce")
    frame["re24"] = pd.to_numeric(frame["re24"], errors="coerce")

    frame = frame.dropna(
        subset=["game_date", "batter", "pitcher", "re24"]
    ).copy()
    frame["batter"] = frame["batter"].astype("int64")
    frame["pitcher"] = frame["pitcher"].astype("int64")

    batter_daily = _daily_prior_features(frame, "batter", "batter")
    pitcher_daily = _daily_prior_features(frame, "pitcher", "pitcher")

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
