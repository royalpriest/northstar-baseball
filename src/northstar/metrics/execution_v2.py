from __future__ import annotations

import numpy as np
import pandas as pd


WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "swinging_pitchout",
    "missed_bunt",
}

BALL_IN_PLAY_DESCRIPTIONS = {
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

CONTACT_TARGET = "is_contact"
CONTACT_PROBABILITY = "expected_contact_probability"
CONTACT_RESIDUAL = "contact_execution_residual"

DAMAGE_TARGET = "estimated_woba_using_speedangle"
DAMAGE_PREDICTION = "expected_damage_xwoba"
DAMAGE_RESIDUAL = "damage_execution_residual"


def build_contact_frame(decisions: pd.DataFrame) -> pd.DataFrame:
    """Label contact among observed physical swings."""
    required = {"is_swing", "description"}
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(
            f"Contact execution data is missing: {sorted(missing)}"
        )

    frame = decisions.loc[decisions["is_swing"].eq(1)].copy()
    description = (
        frame["description"].astype("string").str.strip().str.lower()
    )
    frame[CONTACT_TARGET] = (~description.isin(WHIFF_DESCRIPTIONS)).astype(
        "int8"
    )

    return frame.reset_index(drop=True)


def build_damage_frame(decisions: pd.DataFrame) -> pd.DataFrame:
    """Keep balls in play with a valid Statcast contact-quality target."""
    required = {
        "is_swing",
        "description",
        DAMAGE_TARGET,
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(
            f"Damage execution data is missing: {sorted(missing)}"
        )

    frame = decisions.loc[decisions["is_swing"].eq(1)].copy()
    description = (
        frame["description"].astype("string").str.strip().str.lower()
    )
    frame[DAMAGE_TARGET] = pd.to_numeric(
        frame[DAMAGE_TARGET],
        errors="coerce",
    )
    frame = frame.loc[
        description.isin(BALL_IN_PLAY_DESCRIPTIONS)
        & frame[DAMAGE_TARGET].notna()
    ].copy()

    return frame.reset_index(drop=True)


def fit_context_mean_baseline(
    train: pd.DataFrame,
    target: str,
) -> tuple[
    dict[tuple[int, int, str, str], float],
    dict[str, float],
    float,
]:
    """Fit a transparent count/zone/pitch-type empirical baseline."""
    required = {"balls", "strikes", "zone", "pitch_type", target}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(
            f"Context baseline is missing: {sorted(missing)}"
        )

    frame = train.copy()
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target]).copy()
    frame["zone_key"] = frame["zone"].astype("string").fillna("NA")
    frame["pitch_type_key"] = (
        frame["pitch_type"].astype("string").fillna("NA")
    )

    grouped = (
        frame.groupby(
            ["balls", "strikes", "zone_key", "pitch_type_key"],
            dropna=False,
        )[target]
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
        frame.groupby("pitch_type_key")[target]
        .mean()
        .astype(float)
        .to_dict()
    )
    overall = float(frame[target].mean())

    return mapping, pitch_type_fallback, overall


def predict_context_mean_baseline(
    frame: pd.DataFrame,
    mapping: dict[tuple[int, int, str, str], float],
    pitch_type_fallback: dict[str, float],
    overall_fallback: float,
) -> np.ndarray:
    required = {"balls", "strikes", "zone", "pitch_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Context prediction data is missing: {sorted(missing)}"
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


def attach_contact_residual(
    swings: pd.DataFrame,
    contact_probability: np.ndarray | pd.Series,
) -> pd.DataFrame:
    values = np.asarray(contact_probability, dtype="float64")
    if len(values) != len(swings):
        raise ValueError(
            "contact_probability must match the number of swing rows"
        )
    if CONTACT_TARGET not in swings.columns:
        raise ValueError(f"Swing rows must contain {CONTACT_TARGET}")

    out = swings.copy()
    out[CONTACT_PROBABILITY] = np.clip(values, 0.0, 1.0)
    out[CONTACT_RESIDUAL] = (
        out[CONTACT_TARGET].astype("float64")
        - out[CONTACT_PROBABILITY]
    )
    return out.reset_index(drop=True)


def attach_damage_residual(
    batted_balls: pd.DataFrame,
    expected_damage: np.ndarray | pd.Series,
) -> pd.DataFrame:
    values = np.asarray(expected_damage, dtype="float64")
    if len(values) != len(batted_balls):
        raise ValueError(
            "expected_damage must match the number of batted-ball rows"
        )
    if DAMAGE_TARGET not in batted_balls.columns:
        raise ValueError(f"Batted-ball rows must contain {DAMAGE_TARGET}")

    out = batted_balls.copy()
    out[DAMAGE_PREDICTION] = values
    out[DAMAGE_TARGET] = pd.to_numeric(
        out[DAMAGE_TARGET],
        errors="coerce",
    )
    out = out.dropna(subset=[DAMAGE_TARGET, DAMAGE_PREDICTION]).copy()
    out[DAMAGE_RESIDUAL] = (
        out[DAMAGE_TARGET] - out[DAMAGE_PREDICTION]
    )
    return out.reset_index(drop=True)


def _safe_corr(
    early: pd.Series,
    late: pd.Series,
    method: str,
) -> float:
    if len(early) < 2 or len(late) < 2:
        return float("nan")
    if early.nunique(dropna=True) < 2 or late.nunique(dropna=True) < 2:
        return float("nan")
    return float(early.corr(late, method=method))


def build_player_contact_summary(
    scored_swings: pd.DataFrame,
    min_swings: int = 300,
) -> pd.DataFrame:
    required = {
        "batter",
        "game_pk",
        CONTACT_TARGET,
        CONTACT_PROBABILITY,
        CONTACT_RESIDUAL,
    }
    missing = required.difference(scored_swings.columns)
    if missing:
        raise ValueError(
            f"Player contact data is missing: {sorted(missing)}"
        )
    if min_swings < 1:
        raise ValueError("min_swings must be >= 1")

    grouped = scored_swings.groupby("batter", sort=False)
    summary = grouped.agg(
        swings=(CONTACT_TARGET, "size"),
        games=("game_pk", "nunique"),
        observed_contact_rate=(CONTACT_TARGET, "mean"),
        expected_contact_rate=(CONTACT_PROBABILITY, "mean"),
        contact_residual_mean=(CONTACT_RESIDUAL, "mean"),
    ).reset_index()

    summary["contact_above_expected_per_100_swings"] = (
        summary["contact_residual_mean"] * 100.0
    )
    summary = summary.loc[summary["swings"].ge(min_swings)].copy()
    summary["batter"] = pd.to_numeric(
        summary["batter"],
        errors="coerce",
    ).astype("Int64")
    summary = summary.loc[summary["batter"].notna()].copy()
    summary["batter"] = summary["batter"].astype("int64")

    return summary.sort_values(
        [
            "contact_above_expected_per_100_swings",
            "swings",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_player_damage_summary(
    scored_batted_balls: pd.DataFrame,
    min_batted_balls: int = 100,
) -> pd.DataFrame:
    required = {
        "batter",
        "game_pk",
        DAMAGE_TARGET,
        DAMAGE_PREDICTION,
        DAMAGE_RESIDUAL,
    }
    missing = required.difference(scored_batted_balls.columns)
    if missing:
        raise ValueError(
            f"Player damage data is missing: {sorted(missing)}"
        )
    if min_batted_balls < 1:
        raise ValueError("min_batted_balls must be >= 1")

    grouped = scored_batted_balls.groupby("batter", sort=False)
    summary = grouped.agg(
        batted_balls=(DAMAGE_TARGET, "size"),
        games=("game_pk", "nunique"),
        observed_damage_xwoba=(DAMAGE_TARGET, "mean"),
        expected_damage_xwoba=(DAMAGE_PREDICTION, "mean"),
        damage_residual_mean=(DAMAGE_RESIDUAL, "mean"),
    ).reset_index()

    summary["damage_xwoba_points_above_expected"] = (
        summary["damage_residual_mean"] * 1000.0
    )
    summary = summary.loc[
        summary["batted_balls"].ge(min_batted_balls)
    ].copy()
    summary["batter"] = pd.to_numeric(
        summary["batter"],
        errors="coerce",
    ).astype("Int64")
    summary = summary.loc[summary["batter"].notna()].copy()
    summary["batter"] = summary["batter"].astype("int64")

    return summary.sort_values(
        [
            "damage_xwoba_points_above_expected",
            "batted_balls",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)


def _half_reliability(
    frame: pd.DataFrame,
    split_date: str | pd.Timestamp,
    min_half_rows: int,
    count_name: str,
    aggregate_columns: dict[str, tuple[str, str]],
    derived: dict[str, tuple[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = frame.copy()
    current["game_date"] = pd.to_datetime(
        current["game_date"],
        errors="coerce",
    )
    boundary = pd.Timestamp(split_date)
    current["season_half"] = np.where(
        current["game_date"] < boundary,
        "early",
        "late",
    )

    pieces: list[pd.DataFrame] = []
    for half in ("early", "late"):
        subset = current.loc[current["season_half"].eq(half)]
        summary = (
            subset.groupby("batter", sort=False)
            .agg(**aggregate_columns)
            .reset_index()
        )
        summary["season_half"] = half
        for output, (source, multiplier) in derived.items():
            summary[output] = summary[source] * multiplier
        pieces.append(summary)

    half_table = pd.concat(pieces, ignore_index=True)
    eligible = half_table.loc[
        half_table[count_name].ge(min_half_rows)
    ].copy()

    metrics = list(derived) + [
        column
        for column in aggregate_columns
        if column != count_name
        and column not in {source for source, _ in derived.values()}
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
                "pearson_r": _safe_corr(
                    pivot.get("early", pd.Series(dtype=float)),
                    pivot.get("late", pd.Series(dtype=float)),
                    method="pearson",
                ),
                "spearman_rho": _safe_corr(
                    pivot.get("early", pd.Series(dtype=float)),
                    pivot.get("late", pd.Series(dtype=float)),
                    method="spearman",
                ),
                "early_mean": (
                    float(pivot["early"].mean())
                    if "early" in pivot and len(pivot)
                    else np.nan
                ),
                "late_mean": (
                    float(pivot["late"].mean())
                    if "late" in pivot and len(pivot)
                    else np.nan
                ),
                "mean_absolute_half_difference": (
                    float(
                        (
                            pivot["early"]
                            - pivot["late"]
                        ).abs().mean()
                    )
                    if {
                        "early",
                        "late",
                    }.issubset(pivot.columns)
                    and len(pivot)
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


def build_contact_half_reliability(
    scored_swings: pd.DataFrame,
    split_date: str | pd.Timestamp = "2025-07-01",
    min_half_swings: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min_half_swings < 1:
        raise ValueError("min_half_swings must be >= 1")

    return _half_reliability(
        scored_swings,
        split_date=split_date,
        min_half_rows=min_half_swings,
        count_name="swings",
        aggregate_columns={
            "swings": (CONTACT_TARGET, "size"),
            "observed_contact_rate": (CONTACT_TARGET, "mean"),
            "expected_contact_rate": (CONTACT_PROBABILITY, "mean"),
            "contact_residual_mean": (CONTACT_RESIDUAL, "mean"),
        },
        derived={
            "contact_above_expected_per_100_swings": (
                "contact_residual_mean",
                100.0,
            )
        },
    )


def build_damage_half_reliability(
    scored_batted_balls: pd.DataFrame,
    split_date: str | pd.Timestamp = "2025-07-01",
    min_half_batted_balls: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min_half_batted_balls < 1:
        raise ValueError("min_half_batted_balls must be >= 1")

    return _half_reliability(
        scored_batted_balls,
        split_date=split_date,
        min_half_rows=min_half_batted_balls,
        count_name="batted_balls",
        aggregate_columns={
            "batted_balls": (DAMAGE_TARGET, "size"),
            "observed_damage_xwoba": (DAMAGE_TARGET, "mean"),
            "expected_damage_xwoba": (DAMAGE_PREDICTION, "mean"),
            "damage_residual_mean": (DAMAGE_RESIDUAL, "mean"),
        },
        derived={
            "damage_xwoba_points_above_expected": (
                "damage_residual_mean",
                1000.0,
            )
        },
    )


def build_selection_contact_damage_profile(
    selection: pd.DataFrame,
    contact: pd.DataFrame,
    damage: pd.DataFrame,
) -> pd.DataFrame:
    for label, frame in [
        ("selection", selection),
        ("contact", contact),
        ("damage", damage),
    ]:
        if "batter" not in frame.columns:
            raise ValueError(f"{label} frame must contain batter")

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

    profile = (
        selection.loc[:, selection_columns]
        .drop_duplicates("batter")
        .merge(
            contact,
            on="batter",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            damage,
            on="batter",
            how="inner",
            validate="one_to_one",
            suffixes=("_contact", "_damage"),
        )
    )

    if "decision_value_per_100" in profile.columns:
        weights = (
            profile["eligible_decisions"]
            if "eligible_decisions" in profile.columns
            else pd.Series(np.ones(len(profile)), index=profile.index)
        )
        profile["selection_centered_per_100"] = (
            profile["decision_value_per_100"]
            - float(
                np.average(
                    profile["decision_value_per_100"],
                    weights=weights,
                )
            )
        )

    return profile
