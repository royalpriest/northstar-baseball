from __future__ import annotations

import numpy as np
import pandas as pd


PLAYER_DECISION_REQUIRED_COLUMNS = {
    "game_date",
    "batter",
    "is_swing",
    "swing_propensity_calibrated",
    "q_take",
    "q_swing",
}


def add_player_decision_values(
    predictions: pd.DataFrame,
    discretionary_lower: float = 0.20,
    discretionary_upper: float = 0.80,
) -> pd.DataFrame:
    """Derive descriptive player decision-value components.

    The opportunity-conditioned expected action value is the value of a
    league-like action mixture implied by the transported/calibrated swing
    propensity:

        E_action[Q | X] = e(X) * Q_swing(X) + (1 - e(X)) * Q_take(X)

    Player decision value is the modeled value of the observed choice minus
    that opportunity-conditioned expected action value.

    This remains model-based and observational; it is not a causal run value.
    """
    missing = PLAYER_DECISION_REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Player decision predictions are missing: {sorted(missing)}"
        )
    if not 0.0 < discretionary_lower < discretionary_upper < 1.0:
        raise ValueError(
            "Discretionary bounds must satisfy 0 < lower < upper < 1."
        )

    frame = predictions.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce")
    frame["is_swing"] = pd.to_numeric(
        frame["is_swing"], errors="coerce"
    )
    frame["swing_propensity_calibrated"] = pd.to_numeric(
        frame["swing_propensity_calibrated"], errors="coerce"
    )
    frame["q_take"] = pd.to_numeric(frame["q_take"], errors="coerce")
    frame["q_swing"] = pd.to_numeric(frame["q_swing"], errors="coerce")

    frame = frame.dropna(
        subset=[
            "game_date",
            "batter",
            "is_swing",
            "swing_propensity_calibrated",
            "q_take",
            "q_swing",
        ]
    ).copy()
    frame = frame.loc[frame["is_swing"].isin([0, 1])].copy()

    frame["batter"] = frame["batter"].astype("int64")
    frame["is_swing"] = frame["is_swing"].astype("int8")

    propensity = frame["swing_propensity_calibrated"].clip(0.0, 1.0)
    q_take = frame["q_take"]
    q_swing = frame["q_swing"]

    frame["expected_action_value"] = (
        propensity * q_swing + (1.0 - propensity) * q_take
    )
    frame["chosen_action_value"] = np.where(
        frame["is_swing"].eq(1),
        q_swing,
        q_take,
    )
    frame["alternative_action_value"] = np.where(
        frame["is_swing"].eq(1),
        q_take,
        q_swing,
    )
    frame["optimal_action_value"] = np.maximum(q_take, q_swing)
    frame["worst_action_value"] = np.minimum(q_take, q_swing)

    frame["action_value_gap"] = q_swing - q_take
    frame["abs_action_value_gap"] = frame["action_value_gap"].abs()
    frame["optimal_is_swing"] = q_swing.ge(q_take).astype("int8")
    frame["optimal_choice"] = frame["is_swing"].eq(
        frame["optimal_is_swing"]
    )

    frame["decision_value_vs_expected"] = (
        frame["chosen_action_value"] - frame["expected_action_value"]
    )
    frame["missed_decision_value"] = (
        frame["optimal_action_value"] - frame["chosen_action_value"]
    ).clip(lower=0.0)
    frame["opportunity_headroom"] = (
        frame["optimal_action_value"] - frame["expected_action_value"]
    )

    frame["discretionary_opportunity"] = propensity.between(
        discretionary_lower,
        discretionary_upper,
        inclusive="both",
    )

    return frame.reset_index(drop=True)


def _weighted_optimal_choice_rate(group: pd.DataFrame) -> float:
    weights = group["abs_action_value_gap"].to_numpy(dtype="float64")
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan")

    optimal = group["optimal_choice"].to_numpy(dtype="float64")
    return float(np.dot(weights, optimal) / total)


def _aggregate_player_group(group: pd.DataFrame) -> pd.Series:
    discretionary = group.loc[group["discretionary_opportunity"]]

    decisions = len(group)
    discretionary_decisions = len(discretionary)

    decision_value_total = float(
        group["decision_value_vs_expected"].sum()
    )
    missed_value_total = float(group["missed_decision_value"].sum())
    headroom_total = float(group["opportunity_headroom"].sum())

    if discretionary_decisions:
        discretionary_value_per_100 = float(
            discretionary["decision_value_vs_expected"].mean() * 100.0
        )
        discretionary_optimal_choice_rate = float(
            discretionary["optimal_choice"].mean()
        )
        discretionary_weighted_optimal_rate = _weighted_optimal_choice_rate(
            discretionary
        )
    else:
        discretionary_value_per_100 = float("nan")
        discretionary_optimal_choice_rate = float("nan")
        discretionary_weighted_optimal_rate = float("nan")

    return pd.Series(
        {
            "eligible_decisions": decisions,
            "discretionary_decisions": discretionary_decisions,
            "discretionary_share": (
                discretionary_decisions / decisions if decisions else np.nan
            ),
            "swing_rate": float(group["is_swing"].mean()),
            "expected_swing_rate": float(
                group["swing_propensity_calibrated"].mean()
            ),
            "optimal_choice_rate": float(group["optimal_choice"].mean()),
            "value_weighted_optimal_choice_rate": (
                _weighted_optimal_choice_rate(group)
            ),
            "decision_value_total": decision_value_total,
            "decision_value_per_100": (
                decision_value_total / decisions * 100.0
                if decisions
                else np.nan
            ),
            "missed_value_total": missed_value_total,
            "missed_value_per_100": (
                missed_value_total / decisions * 100.0
                if decisions
                else np.nan
            ),
            "opportunity_headroom_total": headroom_total,
            "opportunity_headroom_per_100": (
                headroom_total / decisions * 100.0
                if decisions
                else np.nan
            ),
            "mean_abs_action_value_gap": float(
                group["abs_action_value_gap"].mean()
            ),
            "discretionary_decision_value_per_100": (
                discretionary_value_per_100
            ),
            "discretionary_optimal_choice_rate": (
                discretionary_optimal_choice_rate
            ),
            "discretionary_value_weighted_optimal_choice_rate": (
                discretionary_weighted_optimal_rate
            ),
        }
    )


def build_player_decision_metric(
    predictions: pd.DataFrame,
    min_decisions: int = 500,
    discretionary_lower: float = 0.20,
    discretionary_upper: float = 0.80,
    sort_by: str = "decision_value_per_100",
) -> pd.DataFrame:
    """Aggregate external-season pitch decisions into a player prototype table."""
    if min_decisions < 1:
        raise ValueError("min_decisions must be >= 1")

    allowed_sort = {
        "decision_value_total",
        "decision_value_per_100",
        "missed_value_per_100",
        "optimal_choice_rate",
        "value_weighted_optimal_choice_rate",
        "discretionary_decision_value_per_100",
    }
    if sort_by not in allowed_sort:
        raise ValueError(f"Unsupported sort_by: {sort_by}")

    frame = add_player_decision_values(
        predictions,
        discretionary_lower=discretionary_lower,
        discretionary_upper=discretionary_upper,
    )

    if frame.empty:
        return pd.DataFrame()

    leaderboard = (
        frame.groupby("batter", sort=False)
        .apply(_aggregate_player_group, include_groups=False)
        .reset_index()
    )
    leaderboard["batter"] = leaderboard["batter"].astype("int64")
    leaderboard = leaderboard.loc[
        leaderboard["eligible_decisions"].ge(min_decisions)
    ].copy()

    ascending = sort_by == "missed_value_per_100"
    leaderboard = leaderboard.sort_values(
        [sort_by, "eligible_decisions"],
        ascending=[ascending, False],
    ).reset_index(drop=True)

    return leaderboard


def build_player_half_reliability(
    predictions: pd.DataFrame,
    split_date: str | pd.Timestamp,
    min_half_decisions: int = 200,
    discretionary_lower: float = 0.20,
    discretionary_upper: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare player decision metrics across early/late season halves."""
    if min_half_decisions < 1:
        raise ValueError("min_half_decisions must be >= 1")

    frame = add_player_decision_values(
        predictions,
        discretionary_lower=discretionary_lower,
        discretionary_upper=discretionary_upper,
    )
    boundary = pd.Timestamp(split_date)
    frame["season_half"] = np.where(
        frame["game_date"] < boundary,
        "early",
        "late",
    )

    rows: list[pd.DataFrame] = []
    for half in ("early", "late"):
        current = frame.loc[frame["season_half"].eq(half)]
        summary = (
            current.groupby("batter", sort=False)
            .apply(_aggregate_player_group, include_groups=False)
            .reset_index()
        )
        summary["season_half"] = half
        rows.append(summary)

    half_table = pd.concat(rows, ignore_index=True)
    eligible = half_table.loc[
        half_table["eligible_decisions"].ge(min_half_decisions)
    ].copy()

    metrics = [
        "decision_value_per_100",
        "missed_value_per_100",
        "optimal_choice_rate",
        "value_weighted_optimal_choice_rate",
        "discretionary_decision_value_per_100",
        "discretionary_optimal_choice_rate",
    ]

    reliability_rows: list[dict[str, object]] = []
    for metric in metrics:
        pivot = eligible.pivot(
            index="batter",
            columns="season_half",
            values=metric,
        ).dropna()

        reliability_rows.append(
            {
                "metric": metric,
                "eligible_players": len(pivot),
                "pearson_r": (
                    float(pivot["early"].corr(pivot["late"], method="pearson"))
                    if len(pivot) >= 2
                    else np.nan
                ),
                "spearman_rho": (
                    float(
                        pivot["early"].corr(
                            pivot["late"],
                            method="spearman",
                        )
                    )
                    if len(pivot) >= 2
                    else np.nan
                ),
                "early_mean": (
                    float(pivot["early"].mean()) if len(pivot) else np.nan
                ),
                "late_mean": (
                    float(pivot["late"].mean()) if len(pivot) else np.nan
                ),
                "mean_absolute_half_difference": (
                    float((pivot["early"] - pivot["late"]).abs().mean())
                    if len(pivot)
                    else np.nan
                ),
            }
        )

    return (
        eligible.sort_values(
            ["batter", "season_half"]
        ).reset_index(drop=True),
        pd.DataFrame(reliability_rows),
    )


def decision_value_identity_error(
    decisions: pd.DataFrame,
) -> pd.Series:
    """Audit the per-pitch identity: decision value + missed = headroom."""
    required = {
        "decision_value_vs_expected",
        "missed_decision_value",
        "opportunity_headroom",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(
            f"Decision value identity columns are missing: {sorted(missing)}"
        )

    return (
        decisions["decision_value_vs_expected"]
        + decisions["missed_decision_value"]
        - decisions["opportunity_headroom"]
    )
