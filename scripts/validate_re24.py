from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("data/processed/run_expectancy_2025.csv")
FIGURE_OUTPUT = Path("reports/figures/re24_heatmap_2025.png")
TABLE_OUTPUT = Path("reports/tables/re24_matrix_2025.csv")

BASE_ORDER = [
    "000",
    "100",
    "010",
    "001",
    "110",
    "101",
    "011",
    "111",
]

BASE_LABELS = {
    "000": "Bases empty",
    "100": "Runner on 1st",
    "010": "Runner on 2nd",
    "001": "Runner on 3rd",
    "110": "1st + 2nd",
    "101": "1st + 3rd",
    "011": "2nd + 3rd",
    "111": "Bases loaded",
}


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} does not exist. Run build_baseline.py first."
        )

    re24 = pd.read_csv(
        INPUT,
        dtype={"base_state_before": "string"},
    )

    re24["base_state_before"] = re24["base_state_before"].str.zfill(3)
    re24["outs_before"] = pd.to_numeric(re24["outs_before"])

    # Basic validation
    if len(re24) != 24:
        raise ValueError(f"Expected 24 base/out states, found {len(re24)}")

    if re24["base_out_state_before"].duplicated().any():
        raise ValueError("Duplicate base/out states found.")

    if re24["expected_runs"].isna().any():
        missing = re24.loc[
            re24["expected_runs"].isna(), "base_out_state_before"
        ].tolist()
        raise ValueError(f"Missing expected-run values for: {missing}")

    matrix = (
        re24.pivot(
            index="base_state_before",
            columns="outs_before",
            values="expected_runs",
        )
        .reindex(BASE_ORDER)
        .reindex(columns=[0, 1, 2])
    )

    # Baseball sanity check:
    # for the same runner configuration, more outs should generally
    # mean fewer expected runs.
    violations = []

    for state in BASE_ORDER:
        zero_out = matrix.loc[state, 0]
        one_out = matrix.loc[state, 1]
        two_out = matrix.loc[state, 2]

        if not (zero_out >= one_out >= two_out):
            violations.append(state)

    if violations:
        print(
            "Warning: run expectancy did not decrease with outs for:",
            ", ".join(violations),
        )
    else:
        print("✓ Run expectancy decreases as outs increase for all 8 base states.")

    # Loaded bases should be more valuable than empty bases.
    for outs in [0, 1, 2]:
        empty = matrix.loc["000", outs]
        loaded = matrix.loc["111", outs]

        if loaded <= empty:
            raise ValueError(
                f"Unexpected result with {outs} outs: "
                "bases loaded is not above bases empty."
            )

    print("✓ Bases loaded has higher run expectancy than bases empty.")
    print("✓ All 24 RE24 states are present.")
    print()

    display_matrix = matrix.rename(index=BASE_LABELS)

    print("2025 MLB Run Expectancy Matrix")
    print(display_matrix.round(3).to_string())

    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    display_matrix.round(4).to_csv(TABLE_OUTPUT)

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.imshow(display_matrix.values)

    ax.set_xticks(
        range(3),
        labels=["0 outs", "1 out", "2 outs"],
    )

    ax.set_yticks(
        range(len(display_matrix)),
        labels=display_matrix.index,
    )

    ax.set_title("2025 MLB Run Expectancy by Game State")
    ax.set_xlabel("Outs")
    ax.set_ylabel("Runners")

    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            value = display_matrix.iloc[row, column]

            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Expected runs")
    fig.tight_layout()

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)

    print()
    print(f"Saved table:   {TABLE_OUTPUT}")
    print(f"Saved heatmap: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()