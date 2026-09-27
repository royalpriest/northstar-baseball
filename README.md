# NorthStar Baseball

**NorthStar Baseball** is a context-aware baseball analytics project that studies how individual plate
appearances change a team's expected scoring opportunities, with a dedicated focus on the Toronto Blue Jays
and Canadian MLB players.

The long-term goal is to develop an explainable machine-learning metric for **Chance Creation Added (CCA)**.
Before creating a new metric, the project first reproduces established run-expectancy baselines and validates
the data pipeline.

## Phase 1 question

> From a given base/out state, how many runs should a team expect to score before the half-inning ends?

The first pipeline converts pitch-level Statcast data into plate appearances, reconstructs pre/post states,
builds an empirical RE24 table, and calculates Run Expectancy Added for each plate appearance.

## Why this structure?

Statcast data are pitch-level. A plate appearance may contain runner movement before the final pitch, so using
the final pitch's base state as the PA's starting state can introduce errors. NorthStar explicitly separates
first-pitch state from final-pitch outcome.

## Stack

- Python 3.12+
- pandas + Polars
- DuckDB + Parquet
- pybaseball / Baseball Savant
- scikit-learn
- XGBoost
- SHAP
- matplotlib
- pytest + Ruff

Later phases add FastAPI, PostgreSQL, Next.js/TypeScript, and Datadog.

## Setup

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Build the 2025 baseline

The 2025 MLB regular season opened with the Tokyo Series on March 18 and concluded September 28, so the
starter command includes the full regular-season date range.

```bash
python scripts/build_baseline.py --start 2025-03-18 --end 2025-09-28
```

The download is deliberately chunked into seven-day Parquet files so interrupted runs can resume without
starting the season over.

Outputs:

```text
data/processed/plate_appearances_2025.parquet
data/processed/run_expectancy_2025.csv
data/processed/scoring_probability_2025.csv
```

Raw and processed data are intentionally Git-ignored. The future public repository should distribute code,
methodology, small reproducible samples, and derived results where licensing/terms permit rather than blindly
committing the entire raw dataset.

## Build the 2025 hitter run-value leaderboard

After the baseline has produced plate-appearance RE24 values, aggregate them by batter:

```bash
python scripts/build_player_leaderboard.py --min-pa 300 --top 20
```

Use rate value instead of seasonal total with:

```bash
python scripts/build_player_leaderboard.py --min-pa 300 --top 20 --sort rate
```

The leaderboard reports total RE24, RE24 per 100 plate appearances, and the share of plate appearances
with positive RE24. Player names are resolved from MLBAM IDs through pybaseball's Chadwick player registry
lookup. The 300-PA cutoff is an analysis threshold, not an official MLB qualification rule.

Outputs:

```text
reports/tables/player_run_value_2025.csv
reports/figures/player_run_value_2025_top20.png
```

This is a benchmark, not CCA. The later machine-learning metric must be evaluated against this simpler
run-expectancy baseline.

## Validate the player benchmark

NorthStar compares RE24 per 100 plate appearances with established Baseball Savant outcome and expected
metrics before developing CCA. The validation uses Statcast batter IDs directly and compares RE24 with actual
wOBA plus expected contact-quality measures such as xwOBA, xBA, and xSLG.

Run:

```bash
python scripts/validate_player_run_value.py --season 2025 --min-pa 300
```

Outputs:

```text
reports/tables/player_run_value_validation_2025.csv
reports/tables/player_run_value_correlations_2025.csv
reports/figures/re24_vs_xwoba_2025.png
```

The goal is not for RE24 to perfectly duplicate an existing statistic. A strong positive relationship is a
sanity check that the pipeline captures real offensive quality; meaningful differences help identify what
NorthStar may be measuring differently and what the later CCA model needs to explain.

## Train the first ML model

NorthStar's first machine-learning model predicts the probability that at least one run will score from the
start of a plate appearance before the half-inning ends. It deliberately uses only information known before
the plate appearance: base state, outs, inning, and top/bottom of the inning.

The season is split chronologically so future games never leak into training:

- training: before August 1, 2025
- validation: August 2025
- test: September 2025

Run:

```bash
python scripts/train_scoring_model.py
```

The script compares logistic regression against the simpler 24-state empirical probability baseline and
reports ROC AUC, log loss, and Brier score. It also produces a September holdout calibration curve.

Outputs:

```text
reports/tables/scoring_probability_model_metrics_2025.csv
reports/tables/scoring_probability_model_coefficients_2025.csv
reports/figures/scoring_probability_calibration_2025.png
data/processed/scoring_probability_test_predictions_2025.parquet
```

The prediction Parquet file stays local because processed data are Git-ignored. The metrics and calibration
figure can be committed as reproducible model results.

## Project layout

```text
northstar-baseball/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── reports/
├── scripts/
├── src/northstar/
│   ├── data/
│   ├── features/
│   ├── metrics/
│   └── models/
└── tests/
```

## Metric development principle

CCA is **not defined yet**. The project will first establish RE24 and a calibrated scoring-probability model.
Only then will a new player metric be formalized, documented, and tested out-of-time. This avoids inventing a
name first and reverse-engineering a statistic to fit it.

## Data sources

- MLB Baseball Savant / Statcast
- pybaseball as a Python retrieval interface
- Retrosheet is planned for historical validation and complementary play-by-play coverage

See `docs/methodology.md` and `docs/roadmap.md`.
