from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterBootstrapResult:
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    clusters: int
    rows: int


def doubly_robust_policy_score(
    outcome: pd.Series | np.ndarray,
    actual_swing: pd.Series | np.ndarray,
    swing_probability: pd.Series | np.ndarray,
    q_take: pd.Series | np.ndarray,
    q_swing: pd.Series | np.ndarray,
    policy_action: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return row-level doubly robust scores for a binary action policy."""
    y = np.asarray(outcome, dtype="float64")
    action = np.asarray(actual_swing, dtype="int8")
    propensity = np.clip(
        np.asarray(swing_probability, dtype="float64"),
        1e-4,
        1.0 - 1e-4,
    )
    take = np.asarray(q_take, dtype="float64")
    swing = np.asarray(q_swing, dtype="float64")
    policy = np.asarray(policy_action, dtype="int8")

    q_policy = np.where(policy == 1, swing, take)
    q_observed = np.where(action == 1, swing, take)
    observed_probability = np.where(
        action == 1,
        propensity,
        1.0 - propensity,
    )
    matches = action == policy

    return q_policy + (
        matches.astype("float64")
        * (y - q_observed)
        / observed_probability
    )


def cluster_bootstrap_mean(
    values: pd.Series | np.ndarray,
    clusters: pd.Series | np.ndarray,
    reps: int = 1000,
    seed: int = 42,
) -> ClusterBootstrapResult:
    """Cluster bootstrap a mean by resampling entire clusters with replacement."""
    if reps < 100:
        raise ValueError("reps must be >= 100")

    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype="float64"),
            "cluster": np.asarray(clusters),
        }
    ).dropna()

    if frame.empty:
        raise ValueError("No rows were available for cluster bootstrap.")

    grouped = (
        frame.groupby("cluster", as_index=False)["value"]
        .agg(cluster_sum="sum", cluster_rows="size")
    )
    if len(grouped) < 2:
        raise ValueError("At least two clusters are required.")

    sums = grouped["cluster_sum"].to_numpy(dtype="float64")
    counts = grouped["cluster_rows"].to_numpy(dtype="float64")
    cluster_count = len(grouped)

    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype="float64")

    for index in range(reps):
        sample = rng.integers(0, cluster_count, size=cluster_count)
        boot[index] = sums[sample].sum() / counts[sample].sum()

    return ClusterBootstrapResult(
        estimate=float(frame["value"].mean()),
        standard_error=float(boot.std(ddof=1)),
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        clusters=cluster_count,
        rows=len(frame),
    )


def overlap_policy_robustness(
    predictions: pd.DataFrame,
    windows: list[tuple[float, float]],
    bootstrap_reps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate model-action advantage across progressively tighter overlap."""
    required = {
        "game_pk",
        "is_swing",
        "swing_propensity",
        "q_take",
        "q_swing",
        "decision_transition_value",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction data is missing: {sorted(missing)}")

    rows: list[dict[str, object]] = []

    for index, (lower, upper) in enumerate(windows):
        if not 0.0 < lower < upper < 1.0:
            raise ValueError("Overlap windows must satisfy 0 < lower < upper < 1.")

        current = predictions.loc[
            predictions["swing_propensity"].between(
                lower,
                upper,
                inclusive="both",
            )
        ].copy()

        if current.empty:
            continue

        model_action = (
            current["q_swing"].to_numpy()
            >= current["q_take"].to_numpy()
        ).astype("int8")

        dr_score = doubly_robust_policy_score(
            current["decision_transition_value"],
            current["is_swing"],
            current["swing_propensity"],
            current["q_take"],
            current["q_swing"],
            model_action,
        )
        observed = current["decision_transition_value"].to_numpy(
            dtype="float64"
        )
        difference = dr_score - observed

        bootstrap = cluster_bootstrap_mean(
            difference,
            current["game_pk"],
            reps=bootstrap_reps,
            seed=seed + index,
        )

        chosen = current["is_swing"].to_numpy(dtype="int8")
        rows.append(
            {
                "overlap_lower": lower,
                "overlap_upper": upper,
                "pitches": len(current),
                "games": current["game_pk"].nunique(),
                "observed_swing_share": float(chosen.mean()),
                "model_swing_share": float(model_action.mean()),
                "observed_value": float(observed.mean()),
                "model_dr_value": float(dr_score.mean()),
                "model_advantage": float(difference.mean()),
                "cluster_bootstrap_se": bootstrap.standard_error,
                "ci_low": bootstrap.ci_low,
                "ci_high": bootstrap.ci_high,
                "optimal_choice_rate": float((chosen == model_action).mean()),
                "mean_abs_q_gap": float(
                    (current["q_swing"] - current["q_take"]).abs().mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def propensity_calibration_table(
    actual_swing: pd.Series,
    swing_probability: pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    """Quantile-bin observed versus predicted swing rate."""
    if bins < 3:
        raise ValueError("bins must be >= 3")

    frame = pd.DataFrame(
        {
            "actual_swing": pd.to_numeric(actual_swing, errors="coerce"),
            "swing_probability": pd.to_numeric(
                swing_probability, errors="coerce"
            ),
        }
    ).dropna()

    frame["bin"] = pd.qcut(
        frame["swing_probability"],
        q=min(bins, frame["swing_probability"].nunique()),
        duplicates="drop",
    )

    return (
        frame.groupby("bin", observed=True, as_index=False)
        .agg(
            mean_predicted=("swing_probability", "mean"),
            observed_swing_rate=("actual_swing", "mean"),
            pitches=("actual_swing", "size"),
        )
    )


def subgroup_policy_stability(
    predictions: pd.DataFrame,
    min_pitches: int = 300,
) -> pd.DataFrame:
    """Describe model-action advantage by count and Statcast zone."""
    required = {
        "balls",
        "strikes",
        "zone",
        "is_swing",
        "swing_propensity",
        "q_take",
        "q_swing",
        "decision_transition_value",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction data is missing: {sorted(missing)}")

    frame = predictions.copy()
    frame["model_action"] = (
        frame["q_swing"].to_numpy() >= frame["q_take"].to_numpy()
    ).astype("int8")
    frame["dr_model_score"] = doubly_robust_policy_score(
        frame["decision_transition_value"],
        frame["is_swing"],
        frame["swing_propensity"],
        frame["q_take"],
        frame["q_swing"],
        frame["model_action"],
    )
    frame["dr_advantage"] = (
        frame["dr_model_score"] - frame["decision_transition_value"]
    )
    frame["q_gap"] = frame["q_swing"] - frame["q_take"]

    rows: list[dict[str, object]] = []

    frame["count"] = (
        frame["balls"].astype(int).astype(str)
        + "-"
        + frame["strikes"].astype(int).astype(str)
    )

    for dimension, column in [("count", "count"), ("zone", "zone")]:
        for level, group in frame.groupby(column, dropna=False):
            if len(group) < min_pitches:
                continue

            rows.append(
                {
                    "dimension": dimension,
                    "level": str(level),
                    "pitches": len(group),
                    "observed_swing_share": float(group["is_swing"].mean()),
                    "model_swing_share": float(group["model_action"].mean()),
                    "observed_value": float(
                        group["decision_transition_value"].mean()
                    ),
                    "model_dr_value": float(group["dr_model_score"].mean()),
                    "model_advantage": float(group["dr_advantage"].mean()),
                    "mean_q_gap": float(group["q_gap"].mean()),
                    "mean_abs_q_gap": float(group["q_gap"].abs().mean()),
                    "optimal_choice_rate": float(
                        (
                            group["is_swing"].to_numpy()
                            == group["model_action"].to_numpy()
                        ).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def player_half_stability(
    predictions: pd.DataFrame,
    split_date: str = "2025-09-15",
    min_pitches_per_half: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure early/late September stability of descriptive batter action value."""
    required = {
        "game_date",
        "batter",
        "is_swing",
        "q_take",
        "q_swing",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction data is missing: {sorted(missing)}")

    frame = predictions.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    boundary = pd.Timestamp(split_date)

    frame["half"] = np.where(
        frame["game_date"] < boundary,
        "early",
        "late",
    )
    frame["model_action"] = (
        frame["q_swing"].to_numpy() >= frame["q_take"].to_numpy()
    ).astype("int8")
    action = frame["is_swing"].to_numpy(dtype="int8")
    chosen = np.where(
        action == 1,
        frame["q_swing"].to_numpy(),
        frame["q_take"].to_numpy(),
    )
    alternative = np.where(
        action == 1,
        frame["q_take"].to_numpy(),
        frame["q_swing"].to_numpy(),
    )
    frame["modeled_action_advantage"] = chosen - alternative
    frame["optimal_choice"] = action == frame["model_action"].to_numpy()

    grouped = (
        frame.groupby(["batter", "half"], as_index=False)
        .agg(
            pitches=("is_swing", "size"),
            mean_modeled_action_advantage=(
                "modeled_action_advantage",
                "mean",
            ),
            optimal_choice_rate=("optimal_choice", "mean"),
        )
    )

    eligible = grouped.loc[
        grouped["pitches"] >= min_pitches_per_half
    ].copy()

    advantage = eligible.pivot(
        index="batter",
        columns="half",
        values="mean_modeled_action_advantage",
    ).dropna()
    choices = eligible.pivot(
        index="batter",
        columns="half",
        values="optimal_choice_rate",
    ).dropna()

    player_rows = grouped.merge(
        grouped,
        on="batter",
        suffixes=("_early", "_late"),
    )
    player_rows = player_rows.loc[
        player_rows["half_early"].eq("early")
        & player_rows["half_late"].eq("late")
        & player_rows["pitches_early"].ge(min_pitches_per_half)
        & player_rows["pitches_late"].ge(min_pitches_per_half)
    ].copy()

    summary = pd.DataFrame(
        [
            {
                "metric": "mean_modeled_action_advantage",
                "eligible_players": len(advantage),
                "pearson_r": float(
                    advantage["early"].corr(
                        advantage["late"],
                        method="pearson",
                    )
                )
                if len(advantage) >= 2
                else np.nan,
                "spearman_rho": float(
                    advantage["early"].corr(
                        advantage["late"],
                        method="spearman",
                    )
                )
                if len(advantage) >= 2
                else np.nan,
            },
            {
                "metric": "optimal_choice_rate",
                "eligible_players": len(choices),
                "pearson_r": float(
                    choices["early"].corr(
                        choices["late"],
                        method="pearson",
                    )
                )
                if len(choices) >= 2
                else np.nan,
                "spearman_rho": float(
                    choices["early"].corr(
                        choices["late"],
                        method="spearman",
                    )
                )
                if len(choices) >= 2
                else np.nan,
            },
        ]
    )

    return player_rows.reset_index(drop=True), summary
