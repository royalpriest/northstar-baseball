from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MODEL_CATEGORICAL_FEATURES = [
    "base_state_before",
    "outs_before",
    "inning_topbot",
]
MODEL_NUMERIC_FEATURES = ["inning"]

CONTEXT_CATEGORICAL_FEATURES = [
    "base_state_before",
    "outs_before",
    "inning_topbot",
    "batter_hand",
    "pitcher_hand",
    "park_team",
]

CONTEXT_NUMERIC_FEATURES = [
    "inning",
    "score_diff_before",
    "is_home_batting",
    "batter_prior_re24_per_pa",
    "batter_prior_positive_pa_pct",
    "batter_prior_log_count",
    "pitcher_prior_re24_per_pa",
    "pitcher_prior_positive_pa_pct",
    "pitcher_prior_log_count",
]

CONTEXT_REQUIRED_COLUMNS = set(CONTEXT_CATEGORICAL_FEATURES + CONTEXT_NUMERIC_FEATURES)


REQUIRED_COLUMNS = {
    "game_date",
    "runs_to_end",
    "base_state_before",
    "outs_before",
    "inning_topbot",
    "inning",
    "base_out_state_before",
}


def validate_context_columns(frame: pd.DataFrame) -> None:
    """Ensure the rebuilt PA dataset contains all V2 context-model fields."""
    missing = CONTEXT_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "Context model columns are missing. Rebuild the 2025 baseline from "
            "the cached raw Statcast files before training V2. Missing: "
            f"{sorted(missing)}"
        )


@dataclass(frozen=True)
class ScoringSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def prepare_scoring_frame(plate_appearances: pd.DataFrame) -> pd.DataFrame:
    """Prepare leakage-safe pre-PA features and the binary scoring target."""
    missing = REQUIRED_COLUMNS.difference(plate_appearances.columns)
    if missing:
        raise ValueError(f"Missing plate-appearance columns: {sorted(missing)}")

    frame = plate_appearances.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["outs_before"] = pd.to_numeric(frame["outs_before"], errors="coerce")
    frame["inning"] = pd.to_numeric(frame["inning"], errors="coerce")
    frame["runs_to_end"] = pd.to_numeric(frame["runs_to_end"], errors="coerce")

    frame = frame[
        frame["game_date"].notna()
        & frame["outs_before"].between(0, 2)
        & frame["inning"].notna()
        & frame["runs_to_end"].notna()
        & frame["base_state_before"].notna()
        & frame["inning_topbot"].notna()
    ].copy()

    frame["scored_before_inning_end"] = frame["runs_to_end"].gt(0).astype(int)
    frame["outs_before"] = frame["outs_before"].astype("int64")
    frame["inning"] = frame["inning"].astype("int64")

    return frame.reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    validation_start: str | pd.Timestamp = "2025-08-01",
    test_start: str | pd.Timestamp = "2025-09-01",
) -> ScoringSplit:
    """Split chronologically so future games never leak into training."""
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)

    if validation_start >= test_start:
        raise ValueError("validation_start must be earlier than test_start")

    train = frame.loc[frame["game_date"] < validation_start].copy()
    validation = frame.loc[
        frame["game_date"].ge(validation_start)
        & frame["game_date"].lt(test_start)
    ].copy()
    test = frame.loc[frame["game_date"].ge(test_start)].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Chronological split produced an empty partition. "
            "Check the date boundaries and input season."
        )

    return ScoringSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def fit_state_probability_baseline(train: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Fit the empirical 24-state scoring-probability benchmark on training data only."""
    required = {"base_out_state_before", "scored_before_inning_end"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")

    mapping = (
        train.groupby("base_out_state_before")["scored_before_inning_end"]
        .mean()
        .astype(float)
        .to_dict()
    )
    fallback = float(train["scored_before_inning_end"].mean())
    return mapping, fallback


def predict_state_probability(
    frame: pd.DataFrame,
    mapping: dict[str, float],
    fallback: float,
) -> pd.Series:
    """Predict scoring probability using only the empirical base/out state."""
    return (
        frame["base_out_state_before"]
        .map(mapping)
        .fillna(fallback)
        .astype(float)
        .clip(0.0, 1.0)
    )
