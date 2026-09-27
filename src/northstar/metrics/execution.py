from __future__ import annotations

import numpy as np
import pandas as pd


EXECUTION_TARGET = "decision_transition_value"
EXECUTION_PREDICTION = "expected_swing_transition_value"
EXECUTION_RESIDUAL = "swing_execution_value"


def fit_swing_execution_baseline(
    train_swings: pd.DataFrame,
) -> tuple[
    dict[tuple[int, int, str, str], float],
    dict[str, float],
    float,
]:
    """Fit a transparent count/zone/pitch-type mean swing-value baseline."""
    required = {
        "balls",
        "strikes",
        "zone",
        "pitch_type",
        EXECUTION_TARGET,
    }
    missing = required.difference(train_swings.columns)
    if missing:
        raise ValueError(
            f"Swing execution baseline is missing: {sorted(missing)}"
        )

    frame = train_swings.copy()
    frame["zone_key"] = frame["zone"].astype("string").fillna("NA")
    frame["pitch_type_key"] = (
        frame["pitch_type"].astype("string").fillna("NA")
    )

    grouped = (
        frame.groupby(
            ["balls", "strikes", "zone_key", "pitch_type_key"],
            dropna=False,
        )[EXECUTION_TARGET]
        .mean()
        .astype(float)
    )
    mapping = {
        (
            int(balls),
            int(strikes),
            str(zone),
            str(pitch_type),
        ): float(value)
        for (balls, strikes, zone, pitch_type), value in grouped.items()
    }

    pitch_type_fallback = (
        frame.groupby("pitch_type_key")[EXECUTION_TARGET]
        .mean()
        .astype(float)
        .to_dict()
    )
    overall = float(frame[EXECUTION_TARGET].mean())

    return mapping, pitch_type_fallback, overall


def predict_swing_execution_baseline(
    frame: pd.DataFrame,
    mapping: dict[tuple[int, int, str, str], float],
    pitch_type_fallback: dict[str, float],
    overall_fallback: float,
) -> np.ndarray:
    required = {"balls", "strikes", "zone", "pitch_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Swing execution prediction is missing: {sorted(missing)}"
        )

    zone_key = frame["zone"].astype("string").fillna("NA")
    pitch_type_key = frame["pitch_type"].astype("string").fillna("NA")

    values: list[float] = []
    for balls, strikes, zone, pitch_type in zip(
        frame["balls"],
        frame["strikes"],
        zone_key,
        pitch_type_key,
    ):
        values.append(
            mapping.get(
                (
                    int(balls),
                    int(strikes),
                    str(zone),
                    str(pitch_type),
                ),
                pitch_type_fallback.get(
                    str(pitch_type),
                    overall_fallback,
                ),
            )
        )

    return np.asarray(values, dtype="float64")


def attach_execution_residual(
    swings: pd.DataFrame,
    predictions: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Attach actual-minus-expected transition value for observed swings."""
    if EXECUTION_TARGET not in swings.columns:
        raise ValueError(
            f"Swing execution data must contain {EXECUTION_TARGET}"
        )

    values = np.asarray(predictions, dtype="float64")
    if len(values) != len(swings):
        raise ValueError("predictions must match the number of swing rows")

    out = swings.copy()
    out[EXECUTION_PREDICTION] = values
    out[EXECUTION_TARGET] = pd.to_numeric(
        out[EXECUTION_TARGET],
        errors="coerce",
    )
    out = out.dropna(
        subset=[EXECUTION_TARGET, EXECUTION_PREDICTION]
    ).copy()
    out[EXECUTION_RESIDUAL] = (
        out[EXECUTION_TARGET] - out[EXECUTION_PREDICTION]
    )
    out["positive_execution"] = out[EXECUTION_RESIDUAL].gt(0.0)

    return out.reset_index(drop=True)


def _aggregate_execution_group(group: pd.DataFrame) -> pd.Series:
    swings = len(group)
    actual_total = float(group[EXECUTION_TARGET].sum())
    expected_total = float(group[EXECUTION_PREDICTION].sum())
    execution_total = float(group[EXECUTION_RESIDUAL].sum())

    return pd.Series(
        {
            "swings": swings,
            "games": int(group["game_pk"].nunique()),
            "actual_swing_value_total": actual_total,
            "expected_swing_value_total": expected_total,
            "execution_value_total": execution_total,
            "actual_swing_value_per_100": (
                actual_total / swings * 100.0 if swings else np.nan
            ),
            "expected_swing_value_per_100": (
                expected_total / swings * 100.0 if swings else np.nan
            ),
            "execution_value_per_100_swings": (
                execution_total / swings * 100.0 if swings else np.nan
            ),
            "positive_execution_rate": float(
                group["positive_execution"].mean()
            ),
            "execution_std": float(
                group[EXECUTION_RESIDUAL].std(ddof=1)
            ),
        }
    )


def build_player_execution_summary(
    swings: pd.DataFrame,
    min_swings: int = 300,
) -> pd.DataFrame:
    """Aggregate external-season swing execution above league expectation."""
    required = {
        "batter",
        "game_pk",
        EXECUTION_TARGET,
        EXECUTION_PREDICTION,
        EXECUTION_RESIDUAL,
        "positive_execution",
    }
    missing = required.difference(swings.columns)
    if missing:
        raise ValueError(
            f"Player execution data is missing: {sorted(missing)}"
        )
    if min_swings < 1:
        raise ValueError("min_swings must be >= 1")

    summary = (
        swings.groupby("batter", sort=False)
        .apply(_aggregate_execution_group, include_groups=False)
        .reset_index()
    )
    summary["batter"] = pd.to_numeric(
        summary["batter"],
        errors="coerce",
    ).astype("Int64")
    summary = summary.loc[
        summary["swings"].ge(min_swings)
        & summary["batter"].notna()
    ].copy()
    summary["batter"] = summary["batter"].astype("int64")

    league_mean = float(swings[EXECUTION_RESIDUAL].mean())
    summary["league_execution_mean_per_100"] = league_mean * 100.0
    summary["centered_execution_per_100_swings"] = (
        summary["execution_value_per_100_swings"]
        - league_mean * 100.0
    )

    return summary.sort_values(
        ["centered_execution_per_100_swings", "swings"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_execution_half_reliability(
    swings: pd.DataFrame,
    split_date: str | pd.Timestamp = "2025-07-01",
    min_half_swings: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure whether player swing-execution residuals persist across halves."""
    required = {
        "game_date",
        "batter",
        "game_pk",
        EXECUTION_TARGET,
        EXECUTION_PREDICTION,
        EXECUTION_RESIDUAL,
        "positive_execution",
    }
    missing = required.difference(swings.columns)
    if missing:
        raise ValueError(
            f"Execution reliability data is missing: {sorted(missing)}"
        )
    if min_half_swings < 1:
        raise ValueError("min_half_swings must be >= 1")

    frame = swings.copy()
    frame["game_date"] = pd.to_datetime(
        frame["game_date"],
        errors="coerce",
    )
    boundary = pd.Timestamp(split_date)
    frame["season_half"] = np.where(
        frame["game_date"] < boundary,
        "early",
        "late",
    )

    pieces: list[pd.DataFrame] = []
    for half in ("early", "late"):
        current = frame.loc[frame["season_half"].eq(half)]
        summary = (
            current.groupby("batter", sort=False)
            .apply(_aggregate_execution_group, include_groups=False)
            .reset_index()
        )
        summary["season_half"] = half
        pieces.append(summary)

    half_table = pd.concat(pieces, ignore_index=True)
    eligible = half_table.loc[
        half_table["swings"].ge(min_half_swings)
    ].copy()

    metrics = [
        "execution_value_per_100_swings",
        "positive_execution_rate",
        "actual_swing_value_per_100",
        "expected_swing_value_per_100",
    ]

    reliability_rows: list[dict[str, object]] = []
    for metric in metrics:
        pivot = eligible.pivot(
            index="batter",
            columns="season_half",
            values=metric,
        ).dropna()

        can_correlate = (
            len(pivot) >= 2
            and pivot["early"].nunique(dropna=True) >= 2
            and pivot["late"].nunique(dropna=True) >= 2
        )

        reliability_rows.append(
            {
                "metric": metric,
                "eligible_players": len(pivot),
                "pearson_r": (
                    float(
                        pivot["early"].corr(
                            pivot["late"],
                            method="pearson",
                        )
                    )
                    if can_correlate
                    else np.nan
                ),
                "spearman_rho": (
                    float(
                        pivot["early"].corr(
                            pivot["late"],
                            method="spearman",
                        )
                    )
                    if can_correlate
                    else np.nan
                ),
                "early_mean": (
                    float(pivot["early"].mean()) if len(pivot) else np.nan
                ),
                "late_mean": (
                    float(pivot["late"].mean()) if len(pivot) else np.nan
                ),
                "mean_absolute_half_difference": (
                    float(
                        (pivot["early"] - pivot["late"]).abs().mean()
                    )
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


def merge_selection_execution_profile(
    execution: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    """Build a two-axis descriptive profile without combining the components."""
    if "batter" not in execution.columns:
        raise ValueError("execution must contain batter")
    if "batter" not in selection.columns:
        raise ValueError("selection must contain batter")

    selection_columns = [
        column
        for column in [
            "batter",
            "player_name",
            "eligible_decisions",
            "discretionary_decisions",
            "decision_value_per_100",
            "discretionary_decision_value_per_100",
            "value_weighted_optimal_choice_rate",
        ]
        if column in selection.columns
    ]
    profile = execution.merge(
        selection.loc[:, selection_columns].drop_duplicates("batter"),
        on="batter",
        how="inner",
        validate="one_to_one",
    )

    if "decision_value_per_100" in profile.columns:
        profile["selection_centered_per_100"] = (
            profile["decision_value_per_100"]
            - float(
                np.average(
                    profile["decision_value_per_100"],
                    weights=profile.get(
                        "eligible_decisions",
                        pd.Series(
                            np.ones(len(profile)),
                            index=profile.index,
                        ),
                    ),
                )
            )
        )

    return profile
