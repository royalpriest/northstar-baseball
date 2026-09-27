from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEFAULT_REFERENCE_METRICS = ("OPS", "wOBA", "wRC+")


def merge_with_reference_stats(
    leaderboard: pd.DataFrame,
    reference_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Join NorthStar MLBAM batter IDs directly to FanGraphs season metrics."""
    required_leaderboard = {"batter", "re24_total", "re24_per_100_pa"}
    required_reference = {"xMLBAMID", "Name", "PA"}

    missing = required_leaderboard.difference(leaderboard.columns)
    if missing:
        raise ValueError(f"Leaderboard is missing columns: {sorted(missing)}")

    missing = required_reference.difference(reference_stats.columns)
    if missing:
        raise ValueError(f"Reference stats are missing columns: {sorted(missing)}")

    board = leaderboard.copy()
    board["batter"] = pd.to_numeric(board["batter"], errors="coerce")
    board = board.dropna(subset=["batter"])
    board["batter"] = board["batter"].astype("int64")

    reference = reference_stats.copy()
    reference["batter"] = pd.to_numeric(reference["xMLBAMID"], errors="coerce")
    reference = reference.dropna(subset=["batter"])
    reference["batter"] = reference["batter"].astype("int64")
    reference = reference.drop_duplicates("batter")

    merged = board.merge(
        reference,
        on="batter",
        how="inner",
        suffixes=("", "_fg"),
    )

    return merged.reset_index(drop=True)


def build_correlation_summary(
    merged: pd.DataFrame,
    northstar_metric: str = "re24_per_100_pa",
    reference_metrics: Iterable[str] = DEFAULT_REFERENCE_METRICS,
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations against established offensive metrics."""
    if northstar_metric not in merged.columns:
        raise ValueError(f"Missing NorthStar metric: {northstar_metric}")

    rows: list[dict[str, object]] = []

    for metric in reference_metrics:
        if metric not in merged.columns:
            continue

        pair = merged[[northstar_metric, metric]].copy()
        pair[northstar_metric] = pd.to_numeric(pair[northstar_metric], errors="coerce")
        pair[metric] = pd.to_numeric(pair[metric], errors="coerce")
        pair = pair.dropna()

        if len(pair) < 3:
            continue

        rows.append(
            {
                "northstar_metric": northstar_metric,
                "reference_metric": metric,
                "players": len(pair),
                "pearson_r": pair[northstar_metric].corr(pair[metric], method="pearson"),
                "spearman_rho": pair[northstar_metric].corr(pair[metric], method="spearman"),
            }
        )

    return pd.DataFrame(rows)
