from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from northstar.validation.execution_constructs import (
    CONTACT_CONSTRUCT_FEATURE_SETS,
    DAMAGE_CONSTRUCT_FEATURE_SETS,
)
from northstar.validation.future_incremental import (
    compare_future_incremental_prediction,
)


DEFAULT_CONTACT_HALF = Path(
    "reports/tables/contact_execution_incremental_half_dataset_2025.csv"
)
DEFAULT_DAMAGE_HALF = Path(
    "reports/tables/damage_execution_incremental_half_dataset_2025.csv"
)

DEFAULT_CONTACT_SUMMARY = Path(
    "reports/tables/contact_execution_future_prediction_2025.csv"
)
DEFAULT_DAMAGE_SUMMARY = Path(
    "reports/tables/damage_execution_future_prediction_2025.csv"
)
DEFAULT_CONTACT_PREDICTIONS = Path(
    "reports/tables/contact_execution_future_predictions_2025.csv"
)
DEFAULT_DAMAGE_PREDICTIONS = Path(
    "reports/tables/damage_execution_future_predictions_2025.csv"
)
DEFAULT_FIGURE = Path(
    "reports/figures/execution_future_incremental_value_2025.png"
)


CONTACT_TARGET = "statcast_contact_rate"
CONTACT_NORTHSTAR = "contact_above_expected_per_100_swings"

DAMAGE_TARGET = "statcast_xwoba_on_contact"
DAMAGE_NORTHSTAR = "damage_xwoba_points_above_expected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether early NorthStar contact/damage adjustments improve "
            "out-of-fold prediction of late conventional hitter skill."
        )
    )
    parser.add_argument(
        "--contact-half",
        type=Path,
        default=DEFAULT_CONTACT_HALF,
    )
    parser.add_argument(
        "--damage-half",
        type=Path,
        default=DEFAULT_DAMAGE_HALF,
    )
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def _save_delta_figure(
    contact: pd.DataFrame,
    damage: pd.DataFrame,
    output: Path,
) -> None:
    pieces: list[pd.DataFrame] = []

    if not contact.empty:
        current = contact[
            [
                "construct_set",
                "delta_oof_r2",
            ]
        ].copy()
        current["label"] = (
            "contact / " + current["construct_set"].astype(str)
        )
        pieces.append(current)

    if not damage.empty:
        current = damage[
            [
                "construct_set",
                "delta_oof_r2",
            ]
        ].copy()
        current["label"] = (
            "damage / " + current["construct_set"].astype(str)
        )
        pieces.append(current)

    if not pieces:
        return

    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.sort_values("delta_oof_r2")

    fig, ax = plt.subplots(
        figsize=(9, max(5.5, len(frame) * 0.65))
    )
    ax.barh(frame["label"], frame["delta_oof_r2"])
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel(
        "Change in out-of-fold R² after adding early NorthStar component"
    )
    ax.set_title(
        "Does NorthStar improve prediction of late conventional skill?"
    )
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _print_summary(
    title: str,
    frame: pd.DataFrame,
) -> None:
    print()
    print(title)
    if frame.empty:
        print("No eligible rows.")
        return

    display = [
        "component",
        "future_target",
        "construct_set",
        "players",
        "base_oof_r2",
        "augmented_oof_r2",
        "delta_oof_r2",
        "base_pearson_r",
        "augmented_pearson_r",
        "delta_pearson_r",
        "base_spearman_rho",
        "augmented_spearman_rho",
        "delta_spearman_rho",
        "base_mae",
        "augmented_mae",
        "delta_mae",
    ]
    print(
        frame[display]
        .round(
            {
                "base_oof_r2": 4,
                "augmented_oof_r2": 4,
                "delta_oof_r2": 4,
                "base_pearson_r": 3,
                "augmented_pearson_r": 3,
                "delta_pearson_r": 3,
                "base_spearman_rho": 3,
                "augmented_spearman_rho": 3,
                "delta_spearman_rho": 3,
                "base_mae": 4,
                "augmented_mae": 4,
                "delta_mae": 4,
            }
        )
        .to_string(index=False)
    )


def main() -> None:
    args = parse_args()

    if args.folds < 3:
        raise ValueError("--folds must be >= 3")
    for path in [args.contact_half, args.damage_half]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required half-season construct dataset: {path}. "
                "Run validate_contact_damage_execution.py first."
            )

    contact_half = pd.read_csv(args.contact_half)
    damage_half = pd.read_csv(args.damage_half)

    contact_predictions, contact_summary = (
        compare_future_incremental_prediction(
            contact_half,
            future_target=CONTACT_TARGET,
            northstar_feature=CONTACT_NORTHSTAR,
            feature_sets=CONTACT_CONSTRUCT_FEATURE_SETS,
            component="contact",
            folds=args.folds,
            seed=81,
        )
    )
    damage_predictions, damage_summary = (
        compare_future_incremental_prediction(
            damage_half,
            future_target=DAMAGE_TARGET,
            northstar_feature=DAMAGE_NORTHSTAR,
            feature_sets=DAMAGE_CONSTRUCT_FEATURE_SETS,
            component="damage",
            folds=args.folds,
            seed=91,
        )
    )

    DEFAULT_CONTACT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    contact_summary.to_csv(DEFAULT_CONTACT_SUMMARY, index=False)
    damage_summary.to_csv(DEFAULT_DAMAGE_SUMMARY, index=False)
    contact_predictions.to_csv(
        DEFAULT_CONTACT_PREDICTIONS,
        index=False,
    )
    damage_predictions.to_csv(
        DEFAULT_DAMAGE_PREDICTIONS,
        index=False,
    )

    _save_delta_figure(
        contact_summary,
        damage_summary,
        DEFAULT_FIGURE,
    )

    print()
    print("NorthStar future-skill incremental audit V1")
    print(
        "Question: does an EARLY NorthStar adjustment improve OOF "
        "prediction of LATE observed conventional skill when EARLY "
        "conventional metrics are already in the model?"
    )
    print()
    print(
        "A positive delta means the early NorthStar component adds "
        "future predictive information beyond the conventional features."
    )

    _print_summary(
        "Contact: predict late observed contact rate",
        contact_summary,
    )
    _print_summary(
        "Damage: predict late xwOBA on contact",
        damage_summary,
    )

    print()
    print(f"Saved contact summary:      {DEFAULT_CONTACT_SUMMARY}")
    print(f"Saved damage summary:       {DEFAULT_DAMAGE_SUMMARY}")
    print(f"Saved contact predictions:  {DEFAULT_CONTACT_PREDICTIONS}")
    print(f"Saved damage predictions:   {DEFAULT_DAMAGE_PREDICTIONS}")
    print(f"Saved delta figure:         {DEFAULT_FIGURE}")


if __name__ == "__main__":
    main()
