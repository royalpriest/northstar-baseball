from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score

from northstar.features.outcomes import OUTCOME_CLASSES, OUTCOME_TO_CODE


OUTCOME_TARGET = "outcome_code"

OUTCOME_CATEGORICAL_FEATURES = [
    "base_state_before",
    "outs_before",
    "inning_topbot",
    "batter_hand",
    "pitcher_hand",
    "park_team",
]

OUTCOME_NUMERIC_FEATURES = [
    "inning",
    "score_diff_before",
    "is_home_batting",
    "batter_outcome_prior_log_count",
    "pitcher_outcome_prior_log_count",
]

for _outcome in OUTCOME_CLASSES:
    OUTCOME_NUMERIC_FEATURES.append(
        f"batter_outcome_prior_{_outcome}_rate"
    )
    OUTCOME_NUMERIC_FEATURES.append(
        f"pitcher_outcome_prior_{_outcome}_rate"
    )

OUTCOME_REQUIRED_COLUMNS = set(
    OUTCOME_CATEGORICAL_FEATURES
    + OUTCOME_NUMERIC_FEATURES
    + [
        "game_date",
        "base_out_state_before",
        "terminal_event_re24",
        "within_pa_state_re24",
        "outcome_class",
        OUTCOME_TARGET,
    ]
)


@dataclass(frozen=True)
class OutcomeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class MulticlassMetrics:
    log_loss: float
    accuracy: float
    macro_f1: float
    top2_accuracy: float


def prepare_outcome_frame(
    frame: pd.DataFrame,
    clean_attribution_only: bool = True,
    attribution_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Prepare a leakage-safe multiclass PA outcome dataset."""
    missing = OUTCOME_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "PA-outcome columns are missing. Rebuild/prepare the current "
            f"context first. Missing: {sorted(missing)}"
        )

    out = frame.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["terminal_event_re24"] = pd.to_numeric(
        out["terminal_event_re24"], errors="coerce"
    )
    out["within_pa_state_re24"] = pd.to_numeric(
        out["within_pa_state_re24"], errors="coerce"
    )
    out["outs_before"] = pd.to_numeric(out["outs_before"], errors="coerce")
    out["inning"] = pd.to_numeric(out["inning"], errors="coerce")

    out = out[
        out["game_date"].notna()
        & out["terminal_event_re24"].notna()
        & out["within_pa_state_re24"].notna()
        & out["outs_before"].between(0, 2)
        & out["inning"].notna()
        & out[OUTCOME_TARGET].notna()
    ].copy()

    if clean_attribution_only:
        out = out.loc[
            out["within_pa_state_re24"].abs().le(attribution_tolerance)
        ].copy()

    out["outs_before"] = out["outs_before"].astype("int64")
    out["inning"] = out["inning"].astype("int64")
    out[OUTCOME_TARGET] = out[OUTCOME_TARGET].astype("int64")
    return out.reset_index(drop=True)


def chronological_outcome_split(
    frame: pd.DataFrame,
    validation_start: str | pd.Timestamp = "2025-08-01",
    test_start: str | pd.Timestamp = "2025-09-01",
) -> OutcomeSplit:
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
            "Chronological outcome split produced an empty partition."
        )

    return OutcomeSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def fit_class_prior(train: pd.DataFrame) -> np.ndarray:
    """Training-only class-frequency probability baseline."""
    counts = train[OUTCOME_TARGET].value_counts()
    total = counts.sum()
    probabilities = np.zeros(len(OUTCOME_CLASSES), dtype=float)
    for code in range(len(OUTCOME_CLASSES)):
        probabilities[code] = counts.get(code, 0) / total
    return probabilities


def repeat_class_prior(prior: np.ndarray, rows: int) -> np.ndarray:
    return np.tile(prior, (rows, 1))


def evaluate_multiclass_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> MulticlassMetrics:
    labels = list(range(len(OUTCOME_CLASSES)))
    predicted = probabilities.argmax(axis=1)
    return MulticlassMetrics(
        log_loss=float(log_loss(y_true, probabilities, labels=labels)),
        accuracy=float(accuracy_score(y_true, predicted)),
        macro_f1=float(
            f1_score(
                y_true,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        top2_accuracy=float(
            top_k_accuracy_score(
                y_true,
                probabilities,
                k=2,
                labels=labels,
            )
        ),
    )


def align_model_probabilities(model, x: pd.DataFrame) -> np.ndarray:
    """Align model probability columns to NorthStar's fixed outcome-code order."""
    raw = model.predict_proba(x)
    classes = [int(value) for value in model.named_steps["model"].classes_]
    aligned = np.zeros((len(x), len(OUTCOME_CLASSES)), dtype=float)
    for source_index, class_code in enumerate(classes):
        aligned[:, class_code] = raw[:, source_index]
    return aligned


def fit_outcome_run_value_table(
    train: pd.DataFrame,
) -> tuple[dict[tuple[str, int], float], dict[int, float], float]:
    """Fit training-only mean terminal-event RE24 by start state and outcome."""
    grouped = (
        train.groupby(["base_out_state_before", OUTCOME_TARGET])[
            "terminal_event_re24"
        ]
        .mean()
        .astype(float)
    )
    state_outcome = {
        (str(state), int(code)): float(value)
        for (state, code), value in grouped.items()
    }
    outcome_fallback = (
        train.groupby(OUTCOME_TARGET)["terminal_event_re24"]
        .mean()
        .astype(float)
        .to_dict()
    )
    overall = float(train["terminal_event_re24"].mean())
    return state_outcome, outcome_fallback, overall


def expected_re24_from_outcome_probabilities(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    state_outcome_values: dict[tuple[str, int], float],
    outcome_fallback: dict[int, float],
    overall_fallback: float,
) -> np.ndarray:
    """Convert an outcome-probability distribution into expected run value."""
    if probabilities.shape != (len(frame), len(OUTCOME_CLASSES)):
        raise ValueError("Probability matrix has unexpected shape.")

    values = np.empty_like(probabilities, dtype=float)
    states = frame["base_out_state_before"].astype(str).tolist()

    for row_index, state in enumerate(states):
        for class_code in range(len(OUTCOME_CLASSES)):
            values[row_index, class_code] = state_outcome_values.get(
                (state, class_code),
                outcome_fallback.get(class_code, overall_fallback),
            )

    return (probabilities * values).sum(axis=1)
