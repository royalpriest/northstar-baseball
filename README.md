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

## Development and experiment log

NorthStar keeps a visible engineering log in this README. Significant successes, failed experiments, runtime
errors, data-quality issues, and workflow mistakes are recorded rather than silently removed. Future work
should append to this section whenever a meaningful experiment succeeds or fails, an error is encountered,
or a design decision changes because of evidence.

| Date | Type | Event | Outcome / fix |
| --- | --- | --- | --- |
| 2026-09-26 | Error | Initial editable install ran under Python 3.10 even though NorthStar requires Python 3.12+. `pytest` was therefore never installed. | Standardized the local environment on Python 3.13. |
| 2026-09-26 | Error | The Windows Python launcher listed Python 3.13, but the registered `python.exe` path no longer existed. | Reinstalled Python 3.13 and recreated `.venv`. |
| 2026-09-26 | Success | NorthStar virtual environment activated correctly with Python 3.13 and the project dependencies installed. | Local development environment became reproducible and isolated. |
| 2026-09-26 | Workflow issue | Editable installation generated `src/northstar_baseball.egg-info/`, which was accidentally staged for the first commit. | Added `*.egg-info/` and other Python build/cache artifacts to `.gitignore` and removed generated metadata from Git. |
| 2026-09-26 | Success | A short Statcast smoke test completed before the full-season run. | Confirmed ingestion, Parquet writing, plate-appearance reconstruction, and RE24 generation worked end to end. |
| 2026-09-26 | Bug found by tests | Using a pandas grouped `first()`-style approach could skip null base-runner values column-by-column and accidentally combine state from different pitches. | Plate appearances now use the literal first and last pitch rows with `groupby(...).head(1)` / `tail(1)`. |
| 2026-09-26 | Success | Full 2025 regular-season baseline completed from Baseball Savant / Statcast. | Processed **742,080 pitches** into **190,987 plate appearances**. |
| 2026-09-26 | Success | Empirical RE24 matrix was generated for all 24 base/out states. | The table showed expected baseball behavior, including lower run expectancy with more outs and higher expectancy with more runners aboard. |
| 2026-09-26 | Success | RE24 validation and heatmap completed. | All 24 states were present; run expectancy decreased with outs for every base configuration; bases loaded exceeded bases empty at each out count. |
| 2026-09-27 | Success | First 2025 hitter RE24 leaderboard was produced. | Aaron Judge ranked first at 75.08 total RE24 and 10.79 RE24 per 100 PA; the output looked baseball-plausible and became NorthStar's first player-level benchmark. |
| 2026-09-27 | Workflow error | A local `feat/player-run-value` branch was created before fetching the already-created remote branch. The local branch had no tracking information and did not contain the new leaderboard script. | Reset the local branch to `origin/feat/player-run-value` and set the upstream branch. |
| 2026-09-27 | Error | `pybaseball.batting_stats()` failed because FanGraphs' legacy leaderboard endpoint returned HTTP 403. | Replaced the legacy scraper attempt. |
| 2026-09-27 | Error | A direct attempt to use FanGraphs' modern leaderboard API also returned HTTP 403 from the local environment. | Abandoned FanGraphs as a runtime dependency for this validation step and switched to Baseball Savant expected-statistics data. |
| 2026-09-27 | Error | `pyproject.toml` became invalid after a literal `\\n` was accidentally inserted into the dependency list. This caused both `pip install -e` and `pytest` to fail while parsing TOML. | Corrected the dependency syntax; the full test suite returned to passing. |
| 2026-09-27 | Success | RE24 was validated against Baseball Savant outcome and expected metrics across **281 hitters** with at least 300 PA. | RE24/100 correlated strongly with wOBA (**Pearson 0.906, Spearman 0.902**) and SLG (**0.823, 0.810**), while remaining less tightly correlated with xwOBA (**0.768, 0.729**). |
| 2026-09-27 | Visualization issue | Player annotations on the RE24-vs-xwOBA scatter plot overlapped and became unreadable. | Reduced default annotations, enlarged the figure, staggered label offsets, and added callout boxes/connector lines. |
| 2026-09-27 | Git workflow error | The local metric-validation branch and its remote counterpart each gained a unique commit, causing a non-fast-forward push rejection. | Rebased the local result commit on top of the remote branch with `git pull --rebase`, then pushed successfully. |
| 2026-09-27 | Success | Metric-validation work was merged into `main` through PR #3. | RE24 benchmark validation is now part of the main project history. |
| 2026-09-27 | Success | First leakage-safe ML scoring model completed with **11 passing tests**. | Chronological split: 131,045 PA training, 31,877 validation, 28,065 September test. |
| 2026-09-27 | Experiment result | Logistic V1 produced test ROC AUC **0.709**, log loss **0.5082**, and Brier **0.1667**. The 24-state baseline produced ROC AUC **0.707**, log loss **0.5071**, and Brier **0.1663**. | The ML model slightly improved ranking discrimination but did **not** beat the simpler baseline on probability quality. This establishes that base/out state is already a strong benchmark and motivates richer pre-PA batter/pitcher/context features for V2. |
| 2026-09-27 | Success | First scoring-probability ML work was merged into `main` through PR #4. | The leakage-safe V1 model and its holdout evaluation became the official baseline for richer models. |
| 2026-09-27 | Design decision | Context V2 was designed around only information available before the PA: handedness, score state, home/park context, and prior-date batter/pitcher RE24 history. | Same-day player outcomes are excluded from rolling history to avoid ambiguous doubleheader ordering and future leakage. |

### Logging rule going forward

Every meaningful experiment should add an entry here with the date, whether it was a success, failure, error,
workflow issue, or inconclusive result, what happened, and what changed because of it. Failed approaches should
remain documented when they influenced the project architecture or methodology.

## Train the context model V2

V2 adds matchup and backward-looking player context while preserving the same chronological holdout design.
The rebuilt plate-appearance dataset retains batter/pitcher handedness, score differential, home/away context,
and the home team as a park proxy. NorthStar then creates batter and pitcher history features using only
results from dates strictly before each plate appearance.

This same-day exclusion is deliberate: Statcast's processed PA table does not currently preserve a trustworthy
game-time ordering for doubleheaders, so the conservative implementation refuses to let any result from the
current calendar day influence another PA on that date.

V2 compares four models:

- empirical 24-state scoring probability
- logistic regression V1
- context-aware logistic regression V2
- context-aware XGBoost V2

Because new pre-PA fields must be added to the processed plate-appearance file, rebuild the baseline first.
The raw seven-day Statcast Parquet chunks are reused, so this should not redownload the season:

```bash
python scripts/build_baseline.py --start 2025-03-18 --end 2025-09-28
```

Then train V2:

```bash
python scripts/train_context_model.py
```

Outputs:

```text
reports/tables/context_model_metrics_2025.csv
reports/tables/context_logistic_coefficients_2025.csv
reports/tables/context_xgboost_feature_importance_2025.csv
reports/figures/context_model_calibration_2025.png
data/processed/context_model_test_predictions_2025.parquet
```

The processed prediction file remains local. Model metrics, importance tables, and the calibration figure can
be committed after the holdout results are reviewed.

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
