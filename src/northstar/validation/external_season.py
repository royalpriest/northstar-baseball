from __future__ import annotations

import pandas as pd


def state_transport_coverage(
    training_state_values: dict[str, float],
    external_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how often external decision states were seen in training."""
    if "decision_state" not in external_frame.columns:
        raise ValueError("external_frame must contain decision_state")

    states = external_frame["decision_state"].astype(str)
    exact = states.isin(training_state_values)

    return pd.DataFrame(
        [
            {
                "external_rows": len(external_frame),
                "exact_state_rows": int(exact.sum()),
                "fallback_state_rows": int((~exact).sum()),
                "exact_state_pct": float(exact.mean() * 100.0)
                if len(external_frame)
                else 0.0,
                "external_unique_states": int(states.nunique()),
                "training_unique_states": int(len(training_state_values)),
                "unseen_unique_states": int(
                    states.loc[~exact].nunique()
                ),
            }
        ]
    )


def season_sample_summary(
    train: pd.DataFrame,
    external: pd.DataFrame,
    train_season: int,
    external_season: int,
) -> pd.DataFrame:
    """Basic transport diagnostics before propensity/action modeling."""
    required = {"is_swing", "game_pk", "batter", "pitcher"}
    for label, frame in [("train", train), ("external", external)]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{label} frame is missing columns: {sorted(missing)}"
            )

    rows = []
    for role, season, frame in [
        ("train", train_season, train),
        ("external", external_season, external),
    ]:
        rows.append(
            {
                "role": role,
                "season": int(season),
                "decision_rows": len(frame),
                "games": int(frame["game_pk"].nunique()),
                "batters": int(frame["batter"].nunique()),
                "pitchers": int(frame["pitcher"].nunique()),
                "observed_swing_rate": float(frame["is_swing"].mean()),
            }
        )

    return pd.DataFrame(rows)
