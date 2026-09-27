from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("data/processed/plate_appearances_2025.parquet")
DEFAULT_SUMMARY = Path("reports/tables/re24_attribution_audit_2025.csv")
DEFAULT_BY_EVENT = Path("reports/tables/re24_attribution_by_event_2025.csv")
DEFAULT_FIGURE = Path("reports/figures/re24_attribution_difference_2025.png")


def main() -> None:
    if not DEFAULT_INPUT.exists():
        raise FileNotFoundError(
            f"{DEFAULT_INPUT} does not exist. Rebuild the 2025 baseline first."
        )

    frame = pd.read_parquet(DEFAULT_INPUT)
    required = {
        "re24",
        "terminal_event_re24",
        "within_pa_state_re24",
        "runs_scored_before_terminal_event",
        "events",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Terminal-event attribution columns are missing. Rebuild the baseline "
            f"with the current code. Missing: {sorted(missing)}"
        )

    data = frame.copy()
    for column in [
        "re24",
        "terminal_event_re24",
        "within_pa_state_re24",
        "runs_scored_before_terminal_event",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["re24", "terminal_event_re24", "within_pa_state_re24"])
    data["abs_attribution_shift"] = data["within_pa_state_re24"].abs()
    data["attribution_affected"] = data["abs_attribution_shift"].gt(1e-12)

    summary = pd.DataFrame(
        [
            {
                "plate_appearances": len(data),
                "affected_pa": int(data["attribution_affected"].sum()),
                "affected_pa_pct": float(data["attribution_affected"].mean() * 100.0),
                "mean_abs_re24_shift": float(data["abs_attribution_shift"].mean()),
                "median_abs_re24_shift": float(data["abs_attribution_shift"].median()),
                "max_abs_re24_shift": float(data["abs_attribution_shift"].max()),
                "sum_abs_re24_shift": float(data["abs_attribution_shift"].sum()),
                "net_re24_shift": float(data["within_pa_state_re24"].sum()),
                "pa_with_runs_before_terminal_event": int(
                    data["runs_scored_before_terminal_event"].fillna(0).gt(0).sum()
                ),
                "runs_before_terminal_event": float(
                    data["runs_scored_before_terminal_event"].fillna(0).sum()
                ),
            }
        ]
    )

    by_event = (
        data.assign(events=data["events"].fillna("unknown"))
        .groupby("events", as_index=False)
        .agg(
            plate_appearances=("events", "size"),
            affected_pa=("attribution_affected", "sum"),
            mean_abs_re24_shift=("abs_attribution_shift", "mean"),
            sum_abs_re24_shift=("abs_attribution_shift", "sum"),
        )
    )
    by_event["affected_pa_pct"] = (
        by_event["affected_pa"] / by_event["plate_appearances"] * 100.0
    )
    by_event = by_event.sort_values(
        ["sum_abs_re24_shift", "affected_pa"],
        ascending=[False, False],
    ).reset_index(drop=True)

    DEFAULT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DEFAULT_SUMMARY, index=False)
    by_event.to_csv(DEFAULT_BY_EVENT, index=False)

    affected = data.loc[data["attribution_affected"]].copy()
    if not affected.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(affected["within_pa_state_re24"], bins=60)
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel("Whole-PA RE24 minus terminal-event RE24")
        ax.set_ylabel("Plate appearances")
        ax.set_title("2025 RE24 attribution shifts within plate appearances")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        DEFAULT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(DEFAULT_FIGURE, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print()
    print("NorthStar RE24 attribution audit")
    print(summary.round(4).to_string(index=False))
    print()
    print("Largest event groups by total attribution shift")
    print(by_event.head(15).round(4).to_string(index=False))
    print()
    print(f"Saved summary:  {DEFAULT_SUMMARY}")
    print(f"Saved by event: {DEFAULT_BY_EVENT}")
    if DEFAULT_FIGURE.exists():
        print(f"Saved figure:   {DEFAULT_FIGURE}")


if __name__ == "__main__":
    main()
