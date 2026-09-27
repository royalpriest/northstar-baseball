from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"batter", "re24"}


def build_player_run_value(
    plate_appearances: pd.DataFrame,
    min_pa: int = 300,
    sort_by: str = "re24_total",
) -> pd.DataFrame:
    """Aggregate plate-appearance RE24 into a hitter leaderboard.

    Parameters
    ----------
    plate_appearances:
        Plate-appearance level data containing at least batter and re24.
    min_pa:
        Minimum number of plate appearances required for inclusion.
    sort_by:
        re24_total ranks seasonal value; re24_per_100_pa ranks rate value.
    """
    missing = REQUIRED_COLUMNS.difference(plate_appearances.columns)
    if missing:
        raise ValueError(f"Missing plate-appearance columns: {sorted(missing)}")
    if min_pa < 1:
        raise ValueError("min_pa must be >= 1")
    if sort_by not in {"re24_total", "re24_per_100_pa"}:
        raise ValueError("sort_by must be 're24_total' or 're24_per_100_pa'")

    data = plate_appearances.loc[:, ["batter", "re24"]].copy()
    data["batter"] = pd.to_numeric(data["batter"], errors="coerce")
    data["re24"] = pd.to_numeric(data["re24"], errors="coerce")
    data = data.dropna(subset=["batter", "re24"])

    if data.empty:
        return pd.DataFrame(
            columns=[
                "batter",
                "plate_appearances",
                "re24_total",
                "re24_per_pa",
                "re24_per_100_pa",
                "positive_pa_pct",
            ]
        )

    data["batter"] = data["batter"].astype("int64")
    data["positive_re24"] = data["re24"].gt(0)

    leaderboard = (
        data.groupby("batter", as_index=False)
        .agg(
            plate_appearances=("re24", "size"),
            re24_total=("re24", "sum"),
            re24_per_pa=("re24", "mean"),
            positive_pa=("positive_re24", "sum"),
        )
    )

    leaderboard["re24_per_100_pa"] = leaderboard["re24_per_pa"] * 100.0
    leaderboard["positive_pa_pct"] = (
        leaderboard["positive_pa"] / leaderboard["plate_appearances"] * 100.0
    )

    leaderboard = leaderboard.loc[leaderboard["plate_appearances"].ge(min_pa)].copy()
    leaderboard = leaderboard.sort_values(
        [sort_by, "plate_appearances"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return leaderboard[
        [
            "batter",
            "plate_appearances",
            "re24_total",
            "re24_per_pa",
            "re24_per_100_pa",
            "positive_pa_pct",
        ]
    ]
