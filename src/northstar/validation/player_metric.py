from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from northstar.features.pitches import (
    SWING_DESCRIPTIONS,
    TAKE_DESCRIPTIONS,
)


WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "swinging_pitchout",
    "missed_bunt",
}


DEFAULT_CONSTRUCT_TARGETS = [
    "decision_value_per_100",
    "discretionary_decision_value_per_100",
    "value_weighted_optimal_choice_rate",
    "missed_value_per_100",
]

DEFAULT_DISCIPLINE_METRICS = [
    "swing_rate",
    "zone_swing_rate",
    "chase_rate",
    "whiff_per_swing",
    "contact_rate",
    "first_pitch_swing_rate",
    "two_strike_swing_rate",
    "two_strike_chase_rate",
]

DEFAULT_OUTCOME_METRICS = [
    "re24_per_100_pa",
    "woba",
    "est_woba",
]


@dataclass(frozen=True)
class ShrinkageSummary:
    league_mean: float
    between_player_variance: float
    eligible_players: int


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype("float64") / denominator.replace(0, np.nan)
    return result.astype("float64")


def build_plate_discipline_summary(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    """Build interpretable hitter plate-discipline summaries from Statcast.

    Only physical pitch opportunities classified as swing or take are used.
    Statcast zones 1-9 are treated as in-zone and 11-14 as out-of-zone.
    Zone 10 / missing zones are excluded from zone-specific rates.
    """
    required = {
        "batter",
        "description",
        "zone",
        "balls",
        "strikes",
    }
    missing = required.difference(pitches.columns)
    if missing:
        raise ValueError(
            f"Plate-discipline data is missing columns: {sorted(missing)}"
        )

    frame = pitches.loc[:, sorted(required)].copy()
    frame["description"] = (
        frame["description"].astype("string").str.strip().str.lower()
    )
    frame = frame.loc[
        frame["description"].isin(SWING_DESCRIPTIONS | TAKE_DESCRIPTIONS)
    ].copy()

    frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce")
    frame["zone"] = pd.to_numeric(frame["zone"], errors="coerce")
    frame["balls"] = pd.to_numeric(frame["balls"], errors="coerce")
    frame["strikes"] = pd.to_numeric(frame["strikes"], errors="coerce")
    frame = frame.dropna(subset=["batter", "balls", "strikes"])
    frame["batter"] = frame["batter"].astype("int64")

    frame["is_swing"] = frame["description"].isin(SWING_DESCRIPTIONS)
    frame["is_whiff"] = frame["description"].isin(WHIFF_DESCRIPTIONS)
    frame["in_zone"] = frame["zone"].between(1, 9, inclusive="both")
    frame["out_zone"] = frame["zone"].between(11, 14, inclusive="both")
    frame["first_pitch"] = frame["balls"].eq(0) & frame["strikes"].eq(0)
    frame["two_strike"] = frame["strikes"].eq(2)

    rows: list[dict[str, object]] = []
    for batter, group in frame.groupby("batter", sort=False):
        swings = group["is_swing"]
        in_zone = group["in_zone"]
        out_zone = group["out_zone"]
        first_pitch = group["first_pitch"]
        two_strike = group["two_strike"]
        two_strike_out = two_strike & out_zone

        swing_count = int(swings.sum())
        whiff_count = int(group["is_whiff"].sum())

        rows.append(
            {
                "batter": int(batter),
                "decision_pitches": len(group),
                "swings": swing_count,
                "swing_rate": float(swings.mean()),
                "in_zone_pitches": int(in_zone.sum()),
                "zone_swing_rate": (
                    float(swings.loc[in_zone].mean())
                    if in_zone.any()
                    else np.nan
                ),
                "out_zone_pitches": int(out_zone.sum()),
                "chase_rate": (
                    float(swings.loc[out_zone].mean())
                    if out_zone.any()
                    else np.nan
                ),
                "whiffs": whiff_count,
                "whiff_per_swing": (
                    whiff_count / swing_count if swing_count else np.nan
                ),
                "contact_rate": (
                    1.0 - whiff_count / swing_count
                    if swing_count
                    else np.nan
                ),
                "first_pitch_pitches": int(first_pitch.sum()),
                "first_pitch_swing_rate": (
                    float(swings.loc[first_pitch].mean())
                    if first_pitch.any()
                    else np.nan
                ),
                "two_strike_pitches": int(two_strike.sum()),
                "two_strike_swing_rate": (
                    float(swings.loc[two_strike].mean())
                    if two_strike.any()
                    else np.nan
                ),
                "two_strike_out_zone_pitches": int(two_strike_out.sum()),
                "two_strike_chase_rate": (
                    float(swings.loc[two_strike_out].mean())
                    if two_strike_out.any()
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def merge_player_construct_data(
    player_metric: pd.DataFrame,
    plate_discipline: pd.DataFrame,
    outcome_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge NorthStar player values with independent descriptive constructs."""
    if "batter" not in player_metric.columns:
        raise ValueError("player_metric must contain batter")
    if "batter" not in plate_discipline.columns:
        raise ValueError("plate_discipline must contain batter")

    merged = player_metric.merge(
        plate_discipline,
        on="batter",
        how="left",
        validate="one_to_one",
        suffixes=("", "_discipline"),
    )

    if outcome_reference is not None and not outcome_reference.empty:
        reference = outcome_reference.copy()
        if "batter" not in reference.columns:
            if "player_id" in reference.columns:
                reference["batter"] = pd.to_numeric(
                    reference["player_id"],
                    errors="coerce",
                )
            else:
                raise ValueError(
                    "outcome_reference must contain batter or player_id"
                )

        reference["batter"] = pd.to_numeric(
            reference["batter"],
            errors="coerce",
        )
        reference = reference.dropna(subset=["batter"])
        reference["batter"] = reference["batter"].astype("int64")

        keep = [
            column
            for column in [
                "batter",
                "re24_per_100_pa",
                "positive_pa_pct",
                "woba",
                "est_woba",
                "ba",
                "est_ba",
                "slg",
                "est_slg",
            ]
            if column in reference.columns
        ]
        reference = reference.loc[:, keep].drop_duplicates("batter")

        merged = merged.merge(
            reference,
            on="batter",
            how="left",
            validate="one_to_one",
        )

    return merged


def construct_correlation_summary(
    frame: pd.DataFrame,
    targets: list[str] | None = None,
    references: list[str] | None = None,
    min_pairs: int = 30,
) -> pd.DataFrame:
    """Return Pearson/Spearman construct associations in long form."""
    if min_pairs < 3:
        raise ValueError("min_pairs must be >= 3")

    targets = targets or DEFAULT_CONSTRUCT_TARGETS
    references = references or (
        DEFAULT_DISCIPLINE_METRICS + DEFAULT_OUTCOME_METRICS
    )

    rows: list[dict[str, object]] = []
    for target in targets:
        if target not in frame.columns:
            continue

        for reference in references:
            if reference not in frame.columns:
                continue

            pair = frame.loc[:, [target, reference]].apply(
                pd.to_numeric,
                errors="coerce",
            ).dropna()

            if len(pair) < min_pairs:
                continue

            rows.append(
                {
                    "northstar_metric": target,
                    "reference_metric": reference,
                    "players": len(pair),
                    "pearson_r": float(
                        pair[target].corr(pair[reference], method="pearson")
                    ),
                    "spearman_rho": float(
                        pair[target].corr(pair[reference], method="spearman")
                    ),
                }
            )

    return pd.DataFrame(rows)


def construct_cv_summary(
    frame: pd.DataFrame,
    target: str = "decision_value_per_100",
    folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate how much of NorthStar is explained by simpler constructs.

    This uses player-level out-of-fold Ridge predictions. It is a construct
    diagnostic, not a model-selection benchmark.
    """
    if folds < 3:
        raise ValueError("folds must be >= 3")
    if target not in frame.columns:
        raise ValueError(f"Missing target column: {target}")

    feature_sets = {
        "plate_discipline": [
            column
            for column in DEFAULT_DISCIPLINE_METRICS
            if column in frame.columns
        ],
        "outcome_value": [
            column
            for column in DEFAULT_OUTCOME_METRICS
            if column in frame.columns
        ],
    }
    feature_sets["combined"] = list(
        dict.fromkeys(
            feature_sets["plate_discipline"] + feature_sets["outcome_value"]
        )
    )

    rows: list[dict[str, object]] = []
    for label, features in feature_sets.items():
        if not features:
            continue

        current = frame.loc[:, [target] + features].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()

        if len(current) < max(30, folds * 3):
            continue

        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        prediction = cross_val_predict(
            model,
            current[features],
            current[target],
            cv=splitter,
        )

        rows.append(
            {
                "target": target,
                "construct_set": label,
                "players": len(current),
                "features": ";".join(features),
                "oof_r2": float(r2_score(current[target], prediction)),
                "oof_pearson_r": float(
                    pd.Series(current[target].to_numpy()).corr(
                        pd.Series(prediction),
                        method="pearson",
                    )
                ),
                "oof_spearman_rho": float(
                    pd.Series(current[target].to_numpy()).corr(
                        pd.Series(prediction),
                        method="spearman",
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def _cluster_mean_se(
    group: pd.DataFrame,
    value_column: str,
    cluster_column: str,
) -> float:
    values = pd.to_numeric(group[value_column], errors="coerce")
    clusters = group[cluster_column]
    valid = values.notna() & clusters.notna()
    values = values.loc[valid].to_numpy(dtype="float64")
    clusters = clusters.loc[valid].to_numpy()

    if len(values) < 2:
        return float("nan")

    mean = float(values.mean())
    work = pd.DataFrame(
        {
            "cluster": clusters,
            "residual": values - mean,
        }
    )
    score = work.groupby("cluster")["residual"].sum().to_numpy(
        dtype="float64"
    )
    cluster_count = len(score)
    if cluster_count < 2:
        return float("nan")

    variance = (
        cluster_count
        / (cluster_count - 1.0)
        * float(np.square(score).sum())
        / float(len(values) ** 2)
    )
    return float(np.sqrt(max(variance, 0.0)))


def empirical_bayes_player_shrinkage(
    decisions: pd.DataFrame,
    value_column: str = "decision_value_vs_expected",
    min_decisions: int = 500,
    min_discretionary: int = 300,
) -> tuple[pd.DataFrame, ShrinkageSummary]:
    """Apply sample-aware normal-normal shrinkage to player decision rates.

    Player uncertainty uses a game-cluster robust standard error of the mean.
    The between-player variance is a method-of-moments estimate. This is an
    approximate ranking stabilizer, not a claim that player talent is exactly
    normally distributed.
    """
    required = {
        "batter",
        "game_pk",
        value_column,
        "discretionary_opportunity",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(
            f"Shrinkage input is missing columns: {sorted(missing)}"
        )
    if min_decisions < 1 or min_discretionary < 1:
        raise ValueError("minimum decision thresholds must be >= 1")

    rows: list[dict[str, object]] = []
    for batter, group in decisions.groupby("batter", sort=False):
        values = pd.to_numeric(
            group[value_column],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue

        rows.append(
            {
                "batter": int(batter),
                "eligible_decisions": int(len(group)),
                "discretionary_decisions": int(
                    group["discretionary_opportunity"].sum()
                ),
                "raw_mean": float(values.mean()),
                "cluster_se": _cluster_mean_se(
                    group,
                    value_column=value_column,
                    cluster_column="game_pk",
                ),
                "games": int(group["game_pk"].nunique()),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.loc[
        summary["eligible_decisions"].ge(min_decisions)
        & summary["discretionary_decisions"].ge(min_discretionary)
    ].copy()

    if summary.empty:
        return (
            summary,
            ShrinkageSummary(
                league_mean=float("nan"),
                between_player_variance=float("nan"),
                eligible_players=0,
            ),
        )

    league_mean = float(
        pd.to_numeric(
            decisions[value_column],
            errors="coerce",
        ).mean()
    )

    observed_variance = float(summary["raw_mean"].var(ddof=1))
    mean_sampling_variance = float(
        np.square(summary["cluster_se"].dropna()).mean()
    )
    between_variance = max(
        observed_variance - mean_sampling_variance,
        1e-12,
    )

    sampling_variance = np.square(summary["cluster_se"])
    summary["shrinkage_factor"] = (
        between_variance
        / (between_variance + sampling_variance)
    ).clip(0.0, 1.0)
    summary["raw_centered_per_100"] = (
        summary["raw_mean"] - league_mean
    ) * 100.0
    summary["shrunk_mean"] = (
        league_mean
        + summary["shrinkage_factor"]
        * (summary["raw_mean"] - league_mean)
    )
    summary["shrunk_centered_per_100"] = (
        summary["shrunk_mean"] - league_mean
    ) * 100.0

    posterior_variance = (
        between_variance
        * sampling_variance
        / (between_variance + sampling_variance)
    )
    summary["approx_posterior_se_per_100"] = (
        np.sqrt(posterior_variance) * 100.0
    )
    summary["approx_ci_low_per_100"] = (
        summary["shrunk_centered_per_100"]
        - 1.96 * summary["approx_posterior_se_per_100"]
    )
    summary["approx_ci_high_per_100"] = (
        summary["shrunk_centered_per_100"]
        + 1.96 * summary["approx_posterior_se_per_100"]
    )
    summary["league_mean_per_100"] = league_mean * 100.0

    summary = summary.sort_values(
        ["shrunk_centered_per_100", "eligible_decisions"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return (
        summary,
        ShrinkageSummary(
            league_mean=league_mean,
            between_player_variance=between_variance,
            eligible_players=len(summary),
        ),
    )
