from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from northstar.models.baselines import evaluate_probabilities
from northstar.validation.pitch_state import (
    overlap_policy_robustness,
    player_half_stability,
    propensity_calibration_table,
    subgroup_policy_stability,
)


DEFAULT_INPUT = Path(
    "data/processed/pitch_state_value_test_predictions_2025.parquet"
)
DEFAULT_OVERLAP = Path(
    "reports/tables/pitch_state_robustness_overlap_2025.csv"
)
DEFAULT_PROPENSITY = Path(
    "reports/tables/pitch_state_robustness_propensity_2025.csv"
)
DEFAULT_PROPENSITY_BINS = Path(
    "reports/tables/pitch_state_robustness_propensity_bins_2025.csv"
)
DEFAULT_SUBGROUPS = Path(
    "reports/tables/pitch_state_robustness_subgroups_2025.csv"
)
DEFAULT_PLAYER_STABILITY = Path(
    "reports/tables/pitch_state_robustness_player_stability_2025.csv"
)
DEFAULT_PLAYER_SUMMARY = Path(
    "reports/tables/pitch_state_robustness_player_summary_2025.csv"
)
DEFAULT_FIGURE = Path(
    "reports/figures/pitch_state_robustness_overlap_2025.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit NorthStar pitch-state V2 robustness using the saved "
            "September holdout predictions."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--min-subgroup-pitches", type=int, default=300)
    parser.add_argument("--min-player-half-pitches", type=int, default=50)
    parser.add_argument("--split-date", default="2025-09-15")
    return parser.parse_args()


def _save_overlap_figure(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return

    labels = [
        f"{lower:.0%}-{upper:.0%}"
        for lower, upper in zip(
            frame["overlap_lower"],
            frame["overlap_upper"],
        )
    ]
    estimate = frame["model_advantage"]
    lower_error = estimate - frame["ci_low"]
    upper_error = frame["ci_high"] - estimate

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        labels,
        estimate,
        yerr=[lower_error, upper_error],
        marker="o",
        capsize=4,
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Swing-propensity overlap window")
    ax.set_ylabel("Model-action advantage vs observed (runs / decision)")
    ax.set_title("Pitch-state V2 overlap robustness — September 2025")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.bootstrap_reps < 100:
        raise ValueError("--bootstrap-reps must be >= 100")
    if args.min_subgroup_pitches < 1:
        raise ValueError("--min-subgroup-pitches must be >= 1")
    if args.min_player_half_pitches < 1:
        raise ValueError("--min-player-half-pitches must be >= 1")
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Run train_pitch_state_value.py first."
        )

    predictions = pd.read_parquet(args.input)
    predictions["game_date"] = pd.to_datetime(
        predictions["game_date"], errors="coerce"
    )

    windows = [
        (0.05, 0.95),
        (0.10, 0.90),
        (0.20, 0.80),
    ]
    overlap = overlap_policy_robustness(
        predictions,
        windows=windows,
        bootstrap_reps=args.bootstrap_reps,
        seed=42,
    )

    propensity_metrics = evaluate_probabilities(
        predictions["is_swing"],
        predictions["swing_propensity"],
    )
    propensity_summary = pd.DataFrame(
        [
            {
                "sample": "saved_5_95_overlap_holdout",
                "pitches": len(predictions),
                "roc_auc": propensity_metrics.roc_auc,
                "log_loss": propensity_metrics.log_loss,
                "brier": propensity_metrics.brier,
                "mean_predicted_swing": float(
                    predictions["swing_propensity"].mean()
                ),
                "observed_swing_rate": float(
                    predictions["is_swing"].mean()
                ),
            }
        ]
    )
    propensity_bins = propensity_calibration_table(
        predictions["is_swing"],
        predictions["swing_propensity"],
        bins=10,
    )

    subgroups = subgroup_policy_stability(
        predictions,
        min_pitches=args.min_subgroup_pitches,
    )
    player_stability, player_summary = player_half_stability(
        predictions,
        split_date=args.split_date,
        min_pitches_per_half=args.min_player_half_pitches,
    )

    DEFAULT_OVERLAP.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(DEFAULT_OVERLAP, index=False)
    propensity_summary.to_csv(DEFAULT_PROPENSITY, index=False)
    propensity_bins.to_csv(DEFAULT_PROPENSITY_BINS, index=False)
    subgroups.to_csv(DEFAULT_SUBGROUPS, index=False)
    player_stability.to_csv(DEFAULT_PLAYER_STABILITY, index=False)
    player_summary.to_csv(DEFAULT_PLAYER_SUMMARY, index=False)
    _save_overlap_figure(overlap, DEFAULT_FIGURE)

    print()
    print("NorthStar pitch-state robustness V1")
    print(f"Saved holdout pitches: {len(predictions):,}")
    print(f"Cluster bootstrap reps: {args.bootstrap_reps:,}")

    print()
    print("Overlap sensitivity with game-cluster bootstrap")
    print(
        overlap.round(
            {
                "observed_swing_share": 3,
                "model_swing_share": 3,
                "observed_value": 5,
                "model_dr_value": 5,
                "model_advantage": 5,
                "cluster_bootstrap_se": 5,
                "ci_low": 5,
                "ci_high": 5,
                "optimal_choice_rate": 3,
                "mean_abs_q_gap": 5,
            }
        ).to_string(index=False)
    )

    print()
    print("Propensity diagnostics within saved 5%-95% overlap sample")
    print(
        propensity_summary.round(
            {
                "roc_auc": 3,
                "log_loss": 4,
                "brier": 4,
                "mean_predicted_swing": 3,
                "observed_swing_rate": 3,
            }
        ).to_string(index=False)
    )

    print()
    print("Player early/late September stability")
    print(
        player_summary.round(
            {
                "pearson_r": 3,
                "spearman_rho": 3,
            }
        ).to_string(index=False)
    )

    print()
    print("Largest count/zone subgroups by sample")
    print(
        subgroups.sort_values("pitches", ascending=False)
        .head(20)
        .round(
            {
                "observed_swing_share": 3,
                "model_swing_share": 3,
                "observed_value": 5,
                "model_dr_value": 5,
                "model_advantage": 5,
                "mean_q_gap": 5,
                "mean_abs_q_gap": 5,
                "optimal_choice_rate": 3,
            }
        )
        .to_string(index=False)
    )

    print()
    print(f"Saved overlap audit:      {DEFAULT_OVERLAP}")
    print(f"Saved propensity summary: {DEFAULT_PROPENSITY}")
    print(f"Saved propensity bins:    {DEFAULT_PROPENSITY_BINS}")
    print(f"Saved subgroup audit:     {DEFAULT_SUBGROUPS}")
    print(f"Saved player stability:   {DEFAULT_PLAYER_STABILITY}")
    print(f"Saved player summary:     {DEFAULT_PLAYER_SUMMARY}")
    print(f"Saved figure:             {DEFAULT_FIGURE}")


if __name__ == "__main__":
    main()
