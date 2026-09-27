from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


EXPECTED_RE24_TARGET = "terminal_event_re24"

EXPECTED_RE24_CATEGORICAL_FEATURES = [
    "base_state_before",
    "outs_before",
    "inning_topbot",
    "batter_hand",
    "pitcher_hand",
    "park_team",
]

EXPECTED_RE24_NUMERIC_FEATURES = [
    "inning",
    "score_diff_before",
    "is_home_batting",
    "batter_terminal_prior_value_per_pa",
    "batter_terminal_prior_positive_pa_pct",
    "batter_terminal_prior_log_count",
    "pitcher_terminal_prior_value_per_pa",
    "pitcher_terminal_prior_positive_pa_pct",
    "pitcher_terminal_prior_log_count",
]

EXPECTED_RE24_REQUIRED_COLUMNS = set(
    EXPECTED_RE24_CATEGORICAL_FEATURES
    + EXPECTED_RE24_NUMERIC_FEATURES
    + [
        "game_date",
        "base_out_state_before",
        EXPECTED_RE24_TARGET,
    ]
)


@dataclass(frozen=True)
class RunValueSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def prepare_expected_re24_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare leakage-safe features for terminal-event RE24 regression."""
    missing = EXPECTED_RE24_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "Expected-RE24 columns are missing. Rebuild the baseline with the "
            "terminal-event attribution fields first. Missing: "
            f"{sorted(missing)}"
        )

    out = frame.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out[EXPECTED_RE24_TARGET] = pd.to_numeric(
        out[EXPECTED_RE24_TARGET], errors="coerce"
    )
    out["outs_before"] = pd.to_numeric(out["outs_before"], errors="coerce")
    out["inning"] = pd.to_numeric(out["inning"], errors="coerce")

    out = out[
        out["game_date"].notna()
        & out[EXPECTED_RE24_TARGET].notna()
        & out["outs_before"].between(0, 2)
        & out["inning"].notna()
    ].copy()

    out["outs_before"] = out["outs_before"].astype("int64")
    out["inning"] = out["inning"].astype("int64")
    return out.reset_index(drop=True)


def chronological_run_value_split(
    frame: pd.DataFrame,
    validation_start: str | pd.Timestamp = "2025-08-01",
    test_start: str | pd.Timestamp = "2025-09-01",
) -> RunValueSplit:
    """Chronological train/validation/test split for run-value regression."""
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
            "Chronological run-value split produced an empty partition. "
            "Check date boundaries and input data."
        )

    return RunValueSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def fit_state_run_value_baseline(
    train: pd.DataFrame,
) -> tuple[dict[str, float], float]:
    """Fit a training-only mean terminal-event RE24 by starting base/out state."""
    required = {"base_out_state_before", EXPECTED_RE24_TARGET}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")

    mapping = (
        train.groupby("base_out_state_before")[EXPECTED_RE24_TARGET]
        .mean()
        .astype(float)
        .to_dict()
    )
    fallback = float(train[EXPECTED_RE24_TARGET].mean())
    return mapping, fallback


def predict_state_run_value(
    frame: pd.DataFrame,
    mapping: dict[str, float],
    fallback: float,
) -> pd.Series:
    """Predict expected terminal-event RE24 from starting base/out state only."""
    return (
        frame["base_out_state_before"]
        .map(mapping)
        .fillna(fallback)
        .astype(float)
    )
