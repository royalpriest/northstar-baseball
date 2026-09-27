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

The baseline builder is season-aware. For external validation, build the complete 2024 regular season into a
separate cache so the existing 2025 files are untouched:

```bash
python scripts/build_baseline.py --season 2024 --start 2024-03-20 --end 2024-09-30
```

That command defaults to `data/raw/statcast_2024/` and writes
`data/processed/plate_appearances_2024.parquet`.

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
| 2026-09-27 | Experiment result | Context V2 completed on the September holdout. Logistic Context V2 reached ROC AUC **0.711**, log loss **0.5076**, and Brier **0.1665** versus the 24-state baseline at **0.707 / 0.5071 / 0.1663**. XGBoost Context V2 reached **0.709 / 0.5089 / 0.1669**. | Context improved ranking slightly, but neither V2 model beat the simple state baseline on probability quality. XGBoost feature importance was dominated by base state and outs, so the project stopped optimizing this target and pivoted toward player-action run value. |
| 2026-09-27 | Success | Context-model V2 and its preserved holdout results were merged into `main` through PR #5. | V2 became the documented evidence for changing the ML target rather than blindly increasing model complexity. |
| 2026-09-27 | Design decision | Traditional whole-PA RE24 can include steals, wild pitches, balks, caught stealing, or runs that occur before the terminal PA-ending pitch. | NorthStar now retains the final-pitch pre-state and computes a separate **terminal-event RE24** so earlier within-PA state changes can be audited instead of automatically credited to the batter's terminal event. |
| 2026-09-27 | Development error | While adding regression metrics, a code edit initially inserted literal `\\n` escapes into a Python import statement. | The malformed import was detected during repository inspection and corrected before the local model run. The error remains logged because it affected development history. |
| 2026-09-27 | Success | Terminal-event attribution, the attribution audit, prior-date terminal-event player context, and Ridge/XGBoost expected-RE24 training pipelines were implemented on `feat/expected-re24-v1`. | The next local step is to rebuild the cached 2025 processed dataset, quantify how many PAs are affected by within-PA state movement, and only then evaluate expected run value on the September holdout. |
| 2026-09-27 | Success | The 2025 processed dataset was rebuilt with terminal-event attribution fields while preserving **190,987 plate appearances** and the established RE24 state table. | Cached Statcast data were successfully reused; the attribution audit could now compare first-pitch and final-pitch pre-states across the full season. |
| 2026-09-27 | Experiment result | The RE24 attribution audit found **5,917 of 190,987 PAs (3.0981%)** had a non-zero within-PA state shift. Total absolute RE24 moved by **1,393.0269 runs**, with a net whole-PA-minus-terminal-event shift of **+657.7306 runs**. **334 PAs** contained runs before the terminal event, totaling **338 runs**. | The attribution problem is material rather than theoretical. Most PAs are unaffected, so the first expected-RE24 model will use the clean subset by default while preserving shifted PAs for sensitivity analysis. |
| 2026-09-27 | Interpretation note | The largest audit groups by terminal PA result were field outs, strikeouts, walks, singles, and doubles. | These labels describe how the PA eventually ended; they do **not** identify the earlier runner event that caused the state shift. High-volume outcomes can accumulate large total attribution shifts simply because they occur often. |
| 2026-09-27 | Experiment result | The clean-attribution expected-RE24 regression produced essentially no useful September holdout signal. State mean: MAE **0.3596**, RMSE **0.4855**, R² **0.0000**. Ridge context: **0.3601 / 0.4852 / 0.0009**. XGBoost context: **0.3643 / 0.4855 / ~0.0000**. | Directly regressing terminal-event RE24 is too noisy with the current pre-PA feature set. The project will not promote residuals from this model to CCA and will instead model discrete PA outcome probabilities before translating them into expected run value. |
| 2026-09-27 | Interpretation note | XGBoost's top expected-RE24 features included several park indicators, handedness, and base/out terms despite near-zero holdout R². | Feature importance from a model with almost no predictive power is not treated as substantive baseball evidence. |
| 2026-09-27 | Git workflow error | The local expected-RE24 results commit was rejected on push because the remote branch had newer implementation/documentation commits. | Rebased the local result commit onto `origin/feat/expected-re24-v1` with `git pull --rebase`, then pushed successfully without force-pushing. |
| 2026-09-27 | Success | Attribution and expected-RE24 work was merged into `main` through PR #6. | The attribution audit remains a useful result; the near-zero direct-regression signal is preserved as a documented failed hypothesis rather than hidden. |
| 2026-09-27 | Design decision | NorthStar moved from direct continuous RE24 regression to a structured terminal-PA outcome model. | V1 predicts eight outcome families first, then converts the full probability distribution into expected terminal-event RE24 using training-only state/outcome run values. |
| 2026-09-27 | Success | Leakage-safe prior-date batter/pitcher outcome rates, multinomial logistic regression, XGBoost multiclass classification, class-prior baseline, confusion-matrix reporting, and probability-to-RE24 translation were implemented on `feat/pa-outcome-model-v1`. | The next local experiment tests whether structured outcome probabilities outperform both a class-frequency baseline and the failed direct regression approach on the September holdout. |
| 2026-09-27 | Experiment result | PA outcome V1 learned modest but real out-of-time outcome signal. On the September holdout, XGBoost improved multiclass log loss from **1.5325** (class prior) to **1.4868**, macro F1 from **0.0764** to **0.0950**, and top-2 accuracy from **0.6733** to **0.6826**. | Modeling discrete outcomes is more learnable than directly regressing RE24, but hard-class accuracy barely moved because field outs dominate the class distribution. Probability quality is the more useful result. |
| 2026-09-27 | Experiment result | Translating PA outcome probabilities back into terminal-event RE24 still produced negligible September holdout signal. XGBoost implied-value R² was only **0.0026** with Pearson **0.061**. | The structured outcome model improves event probabilities, but current state × coarse-outcome values still do not explain enough within-outcome run-value variation to support CCA residuals. |
| 2026-09-27 | Warning | scikit-learn emitted `y_prob values do not sum to one` during multiclass top-k evaluation. | The model outputs were valid probability-like scores, but rows are now explicitly normalized before classification metrics and expected-value translation to remove floating-point / alignment drift and silence the warning. |
| 2026-09-27 | Interpretation note | XGBoost feature importance remained led by base/out state, but batter prior strikeout, field-out, and home-run rates plus pitcher prior strikeout rate entered the top features. | Player history is contributing real classification signal, but importance magnitudes should still be interpreted cautiously because overall multiclass performance remains modest. |
| 2026-09-27 | Success | PA-outcome V1 and its preserved holdout results were merged into `main` through PR #7. | The structured outcome model remains useful for probability modeling, but its implied RE24 still was not strong enough to justify CCA residuals. |
| 2026-09-27 | Design decision | NorthStar moved from whole-PA outcome prediction to pitch-level opportunity and batter decision modeling. | Exact pitch location, movement, velocity, count, handedness, and game state are now treated as the opportunity presented to the hitter; post-contact information remains excluded from the decision model. |
| 2026-09-27 | Success | A first pitch-level swing/take model pipeline was implemented on `feat/pitch-decision-v1`. | It compares a count+zone swing-rate baseline with logistic regression and XGBoost, uses the cached raw Statcast pitch files directly, and preserves out-of-time decision probabilities for later decision-value work. |
| 2026-09-27 | Runtime error | The first full-season pitch-decision run raised `numpy._core._exceptions._ArrayMemoryError` while pandas tried to copy/consolidate the full wide Statcast DataFrame (~740k pitches) during swing/take filtering. | Updated the Parquet reader to project only the pitch-decision columns before concatenation, substantially reducing memory pressure instead of requiring a larger machine. |
| 2026-09-27 | Runtime error | After chunk projection fixed raw-data loading, the next full-season run reached model fitting but failed inside scikit-learn categorical `SimpleImputer` while constructing another large object-array copy. | Replaced the generic scoring preprocessor for pitch models with a dedicated sparse `float32` pipeline: incomplete pitch-opportunity rows are removed before fitting, categories are one-hot encoded directly, the combined matrix stays sparse, logistic regression uses the sparse-friendly `saga` solver, and XGBoost uses histogram trees. |
| 2026-09-27 | Warning | The first sparse pitch-model run emitted a scikit-learn `FutureWarning` because `LogisticRegression(n_jobs=-1)` is ignored/deprecated for this solver configuration. | Removed the obsolete `n_jobs` argument. The warning did not stop training and did not affect model results. |
| 2026-09-27 | Success | Pitch-decision V1 completed across **729,619 physical decisions (98.32% of 742,080 cached pitches)**. On the September holdout, XGBoost reached ROC AUC **0.861**, log loss **0.4620**, and Brier **0.1522**, materially outperforming the count+zone baseline at **0.777 / 0.5610 / 0.1900** and logistic at **0.782 / 0.5601 / 0.1895**. | This is NorthStar's first clearly strong out-of-time ML result. Exact pitch geometry, count, movement, and pitch type add substantial information about hitter swing/take behavior beyond a simple count+zone benchmark. |
| 2026-09-27 | Interpretation note | XGBoost feature importance was led by two-strike count, exact Statcast zone, hitter-perspective horizontal location, normalized vertical location, count state, pitch movement, and pitch type. | The learned ranking is baseball-plausible and supports the decision-model framing, but feature importance is descriptive rather than causal. |
| 2026-09-27 | Warning | The successful full-season run emitted a `ConvergenceWarning` because sparse logistic regression hit its iteration cap. | The logistic model is only a baseline and the XGBoost result is unaffected. The sparse logistic configuration was relaxed to `tol=1e-3` with a higher iteration cap for a cleaner canonical rerun. |
| 2026-09-27 | Success | Pitch-decision V1 and its preserved full-season holdout results were merged into `main` through PR #8. | The strong XGBoost swing/take model became the propensity foundation for the first observational decision-value experiment. |
| 2026-09-27 | Design decision | NorthStar began a counterfactual-style pitch decision-value V1 rather than treating decision surprise as value. | The new experiment estimates separate expected future-PA run values for swing and take, adjusts observed-action training with the swing-propensity model, trims low-overlap opportunities, and uses doubly robust policy estimates as diagnostics. |
| 2026-09-27 | Methodology caution | Swing/take choices are observational rather than randomized, so modeled action gaps can still reflect unmeasured hitter intent, recognition, and other confounding. | Decision-value V1 is explicitly exploratory. No player metric or causal CCA claim will be made unless overlap, factual prediction, and out-of-time policy diagnostics are credible. |
| 2026-09-27 | Experiment result | Pitch decision-value V1 retained **89.18%** of training pitches and **88.94%** of September pitches in common support. On the September holdout, XGBoost action-value prediction improved over the count+zone+action baseline: MAE **0.3263 vs 0.3280**, RMSE **0.4604 vs 0.4625**, R² **0.0338 vs 0.0252**, Pearson **0.184 vs 0.159**, and Spearman **0.233 vs 0.163**. | Pitch-level action value contains measurable out-of-time signal, unlike the earlier PA-level expected-RE24 regressions. |
| 2026-09-27 | Experiment result | The September doubly robust estimates were **-0.02926** for always take, **-0.04653** for always swing, **-0.01968** for observed actions, and **+0.00628** for the model-selected policy. Validation showed the same policy ordering. | The stable direction is encouraging, but the implied improvement over observed decisions is too large to treat literally as a causal run gain. |
| 2026-09-27 | Methodology limitation | Decision-value V1 uses the remaining value of the entire PA as the return for every pitch. Multiple pitches in the same PA therefore inherit overlapping downstream value, and a counterfactual choice on an early pitch can change whether later pitches even exist. | NorthStar will replace the repeated downstream-return target with a one-step pitch-state transition value whose realized values telescope across the PA. |
| 2026-09-27 | Success | Pitch decision-value V1 and its preserved results were merged into `main` through PR #9. | The measurable action-value signal is retained as evidence, while the overlapping-return formulation remains documented as a methodological limitation rather than being promoted to a player metric. |
| 2026-09-27 | Design decision | Pitch-state value V2 replaces the repeated PA-end return with a one-step decision-opportunity transition: `runs during step + V(next state) - V(current state)`. | State values are fit on pre-August training data using count + base/out state. The transition values telescope across each PA, so downstream PA value is not repeatedly assigned to every pitch. |
| 2026-09-27 | Methodology safeguard | V2 flags gaps where the next recognized physical decision is not the immediately following pitch number. | Those gaps stay in the telescoping audit but are excluded from counterfactual swing/take model fitting, reducing attribution of intervening events to the previous batter decision. |
| 2026-09-27 | Experiment result | Pitch-state V2 completed with **zero telescoping error** across train, validation, and test. It retained **99.97%** of clean train transitions and **88.73%** of September pitches in 5%-95% common support. On the September holdout, XGBoost transition value improved over the count+zone baseline: MAE **0.1040 vs 0.1077**, RMSE **0.2452 vs 0.2474**, R² **0.0626 vs 0.0454**, Pearson **0.250 vs 0.213**, and Spearman **0.537 vs 0.503**. | The one-step state-transition formulation preserved predictive signal while fixing the repeated downstream-return problem. |
| 2026-09-27 | Experiment result | September one-step doubly robust estimates were **-0.01546** for always take, **-0.03389** for always swing, **-0.00623** for observed actions, and **+0.02121** for the model-selected action. Validation showed the same ordering. | The direction is stable, but the implied gain is still treated as observational and requires overlap, clustering, subgroup, and player-stability sensitivity checks before any metric is named. |
| 2026-09-27 | Success | Pitch-state value V2 and its preserved holdout results were merged into `main` through PR #10. | The next phase is robustness auditing rather than immediately increasing model complexity. |
| 2026-09-27 | Design decision | NorthStar added a dedicated robustness phase for pitch-state V2. | The audit tightens propensity-overlap windows, uses game-cluster bootstrap uncertainty, checks count/zone subgroup stability, rechecks propensity calibration on the saved holdout, and measures early-vs-late September batter stability. |
| 2026-09-27 | Experiment result | Pitch-state robustness V1 preserved a positive model-action advantage under progressively tighter common support: **+0.02744** at 5%-95% (95% game-cluster bootstrap CI **0.02605 to 0.02879**), **+0.03017** at 10%-90% (**0.02864 to 0.03170**), and **+0.03464** at 20%-80% (**0.03277 to 0.03627**). | The result did not collapse when easy near-deterministic swing/take opportunities were removed; the estimate strengthened on more ambiguous opportunities. |
| 2026-09-27 | Experiment result | Within the saved 5%-95% holdout, propensity ROC AUC was **0.828**, log loss **0.5108**, Brier **0.1696**, mean predicted swing rate **0.515**, and observed swing rate **0.518**. Early/late September player stability across **310 hitters** was moderate: modeled-action-advantage Pearson/Spearman **0.334/0.299** and optimal-choice-rate **0.374/0.305**. | League-level robustness is strong enough to justify cross-fitted nuisance modeling, but player-level reliability is not yet strong enough for a leaderboard or named metric. |
| 2026-09-27 | Success | Pitch-state robustness V1 and its preserved sensitivity results were merged into `main` through PR #11. | The positive one-step advantage survived tighter overlap and game-cluster uncertainty, so the next phase attacks nuisance-model estimation rather than adding model complexity. |
| 2026-09-27 | Design decision | NorthStar started cross-fitted nuisance modeling for pitch-state value. | Swing propensity is now generated out-of-fold with game-grouped folds, calibrated from OOF predictions, and used to fit action-value models under stabilized weight caps of 3, 5, 10, and 20. Effective sample size and clipping rates are reported before interpreting policy sensitivity. |
| 2026-09-27 | Experiment result | Cross-fitted pitch-state V1 remained stable across stabilized IPW caps **3, 5, 10, and 20**. September model-action advantage ranged only from **+0.02846 to +0.02918**, with all game-cluster bootstrap 95% intervals above zero. OOF propensity folds were stable at ROC AUC **0.861–0.862** and test factual transition-value R² remained about **0.065**. | The result does not depend on in-sample propensity predictions or one arbitrary weight cap. Effective sample size remained substantial, and only **0.033%** of rows were clipped at cap 10. |
| 2026-09-27 | Success | Cross-fitted nuisance validation and preserved results were merged into `main` through PR #12. | Internal 2025 methodology is now strong enough to stop iterating on the same season and move to external-season transport validation. |
| 2026-09-27 | Design decision | External-season validation uses **all of 2024 for training** and **all of 2025 as an untouched external season**. | Run expectancy, decision-state values, propensity calibration, overlap weights, and swing/take value models are all learned from 2024 only; 2025 is scored without refitting. |
| 2026-09-27 | Experiment result | The 2024 → 2025 season-transport test held. The 2024 propensity model scored 2025 at ROC AUC **0.861**, calibrated log loss **0.4608**, and Brier **0.1521**. All **729,387** external decision states had exact 2024 state coverage. | The swing/take opportunity structure transports cleanly across seasons, with only modest average swing-rate drift between training and external data. |
| 2026-09-27 | Experiment result | External 2025 transition-value prediction remained useful across all weight caps. At cap 10, MAE was **0.1013**, RMSE **0.2404**, R² **0.0653**, Pearson **0.256**, and Spearman **0.548**. Model-action advantage remained nearly unchanged across caps 3/5/10/20 at about **+0.0285 runs per eligible decision**, with every game-cluster bootstrap interval above zero. | The core one-step signal survived full-season temporal transport and weight sensitivity without 2025 refitting. |
| 2026-09-27 | Interpretation note | The 2024 → 2025 run is a strong model-transport test, but 2025 was used earlier while developing the methodology. | Treat this as external-to-the-fitted-model season validation rather than a pristine never-observed confirmatory dataset. A future untouched season remains valuable for final confirmation. |
| 2026-09-27 | Success | External-season validation and its preserved 2024 → 2025 results were merged into `main` through PR #13. | NorthStar now has full-season transport evidence strong enough to prototype a player-facing descriptive decision metric. |
| 2026-09-27 | Design decision | The first player metric prototype is **not** named CCA. | It compares the modeled value of a hitter's observed swing/take choice with an opportunity-conditioned league-like action mixture, reports missed value versus the modeled optimal action, and requires full-season split-half reliability before any public metric name is considered. |
| 2026-09-27 | Experiment result | The full-season 2025 player prototype passed a meaningful reliability gate across **347 hitters**. Split-half Pearson/Spearman reliability was **0.704/0.690** for decision value per 100, **0.627/0.614** for missed value per 100, **0.699/0.651** for optimal-choice rate, and **0.678/0.652** for value-weighted optimal-choice rate. | Player-level signal is substantially more repeatable over full-season halves than the earlier short September split. |
| 2026-09-27 | Experiment result | The 20%-80% discretionary-opportunity view also remained repeatable: decision value per 100 Pearson/Spearman **0.656/0.639** and discretionary optimal-choice rate **0.677/0.633**. The per-pitch accounting identity had maximum absolute error only **2.98e-8**. | Repeatability is not driven only by obvious near-deterministic swing/take opportunities, but construct validation is still required before naming a public metric. |
| 2026-09-27 | Success | The player decision-value prototype and preserved results were merged into `main` through PR #14. | NorthStar now moves from proving repeatability to testing what baseball behavior the player signal actually represents and stabilizing rankings for unequal sample sizes. |
| 2026-09-27 | Design decision | Player metric validation compares the prototype with independently aggregated Statcast plate discipline plus existing RE24/wOBA/xwOBA references, and adds game-cluster empirical-Bayes-style shrinkage. | The public-facing candidate should be centered on league average, require both total and discretionary opportunity minimums, and avoid treating 500-pitch and 2,000-pitch samples as equally certain. |
| 2026-09-27 | Experiment result | Construct validation showed that raw decision value per 100 is strongly associated with chase behavior: Statcast chase rate Pearson/Spearman **-0.805/-0.793** and two-strike chase rate **-0.795/-0.789**. Plate-discipline features reconstructed **82.0%** of out-of-fold player-level variance; adding RE24/wOBA/xwOBA increased that only to **83.5%**. | The prototype is highly repeatable but is also largely explainable by familiar plate-discipline behavior, so NorthStar should not name it as a novel public metric yet. |
| 2026-09-27 | Experiment result | The discretionary version behaved similarly: plate discipline alone explained **79.7%** of out-of-fold variance and the combined construct set explained **81.3%**. Offensive outcome references alone explained only **13–15%**. | The metric is primarily a context-weighted plate-discipline signal rather than a disguised traditional outcome-value statistic. |
| 2026-09-27 | Experiment result | Sample-aware shrinkage preserved the broad top group while reducing smaller-sample extremes. The pitch-weighted league mean raw decision value was **-0.0552 per 100 decisions** and is explicitly centered to zero for presentation. | Shrinkage improves ranking stability, but novelty now depends on whether a repeatable residual signal remains after removing simpler discipline constructs. |
| 2026-09-27 | Design decision | NorthStar added an incremental-signal audit before any metric naming. | Early-half out-of-fold residuals after chase / plate-discipline models are compared with late-half residuals predicted from early-half construct relationships. If the residual is not repeatable, the player metric should be treated as a sophisticated plate-discipline summary rather than a new standalone statistic. |
| 2026-09-27 | Experiment result | The incremental player-signal audit showed that **chase rate alone** leaves a moderately repeatable residual (early/late Pearson/Spearman **0.497/0.489**), but adding swing rate and two-strike chase collapses residual reliability to **0.178/0.168**. The full plate-discipline residual is only **0.211/0.205**. | Most of the raw ~0.70 split-half reliability belongs to repeatable plate-discipline behavior rather than a distinct NorthStar player trait. |
| 2026-09-27 | Experiment result | Full-season OOF reconstruction reached **R² 0.646** with chase rate alone and **0.818–0.820** with swing/chase or the full discipline set. Residual associations with RE24, wOBA, and xwOBA remained small; the full-discipline residual correlated only **0.190/0.166** with RE24, **0.187/0.154** with wOBA, and **0.222/0.200** with xwOBA (Pearson/Spearman). | The remaining residual is neither strongly repeatable nor strongly tied to offensive outcomes. |
| 2026-09-27 | Methodology decision | NorthStar will **not** promote the current raw player prototype as a novel standalone metric and will not name it CCA. | The validated signal is retained as a context-weighted **decision/plate-discipline component**. The next research phase separates *selection* (swing/take quality) from *execution* (what the hitter does after choosing to swing) instead of forcing novelty from a weak residual. |
| 2026-09-27 | Success | Player construct validation, incremental-signal auditing, and preserved outputs were merged into `main` through PR #15. | The project keeps the validated selection signal but explicitly rejects standalone-novelty claims unsupported by the residual tests. |
| 2026-09-27 | Design decision | Swing execution V1 is defined as **actual one-step transition value minus league-expected swing transition value**, conditional on the pitch and game context. | The execution model is trained on all clean 2024 swings, uses no batter identity and no post-swing/contact variables, and is evaluated on 2025 swings without refitting. This separates *how well the hitter executed a swing* from *whether swinging was the right choice*. |

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

## Audit attribution and train expected RE24

The scoring-probability experiments showed that runners and outs dominate the probability that somebody scores
later in the inning. NorthStar therefore changes the next research target from *future inning scoring* to the
run value of the current plate appearance's terminal event.

Traditional whole-PA RE24 starts at the first pitch. That can accidentally include state changes that happen
before the PA-ending pitch, such as a stolen base, wild pitch, balk, caught stealing, or a run scoring during
the at-bat. The current pipeline now retains two separate states:

- `base_out_state_before`: state at the first pitch of the PA
- `action_base_out_state_before`: state immediately before the final PA-ending pitch

From those states NorthStar calculates:

```text
traditional RE24
terminal-event RE24
within-PA state RE24 = traditional RE24 - terminal-event RE24
```

This does not claim perfect batter attribution; events on the terminal pitch itself can still involve
baserunners and fielding. It is a stricter benchmark that removes state changes known to have happened before
the terminal event.

The processed plate-appearance file must be rebuilt once so it contains the new terminal-event fields. The
cached Statcast Parquet chunks are reused:

```bash
python scripts/build_baseline.py --start 2025-03-18 --end 2025-09-28
```

Audit how much traditional RE24 changes under the stricter attribution rule:

```bash
python scripts/audit_re24_attribution.py
```

Then train the first expected-run-value models:

```bash
python scripts/train_expected_re24_model.py
```

The model asks:

> Given only information known before the plate appearance, how much terminal-event RE24 should we expect?

Models compared:

- starting base/out-state mean
- Ridge regression with context
- XGBoost regression with context

The context models use handedness, inning/score/park context, and prior-date batter/pitcher terminal-event RE24.
By default, the first expected-value model excludes PAs whose base/out state changed before the terminal pitch;
those PAs remain in the attribution audit instead of being mixed into a pre-PA prediction target whose terminal
state was not known at the start of the PA. Use `--include-shifted-pa` only for a sensitivity run.
As before, the season is split chronologically into pre-August training, August validation, and a September
holdout.

Outputs:

```text
reports/tables/re24_attribution_audit_2025.csv
reports/tables/re24_attribution_by_event_2025.csv
reports/figures/re24_attribution_difference_2025.png

reports/tables/expected_re24_model_metrics_2025.csv
reports/tables/expected_re24_ridge_coefficients_2025.csv
reports/tables/expected_re24_xgboost_feature_importance_2025.csv
reports/figures/expected_re24_holdout_calibration_2025.png
data/processed/expected_re24_test_predictions_2025.parquet
```

The local prediction file also contains exploratory residuals:

```text
actual terminal-event RE24 - expected terminal-event RE24
```

Those residuals are **not yet called CCA**. They first need stability, attribution, and player-level validation.

## Train the PA outcome model V1

Directly regressing terminal-event RE24 produced essentially zero holdout R², so NorthStar now decomposes the
problem into a more structured question:

> Given only information known before the PA, what terminal outcome distribution should we expect?

The first outcome model uses eight stable classes:

- strikeout
- walk / hit-by-pitch
- single
- double / triple
- home run
- field out
- sacrifice
- other reach / rare terminal event

Unknown rare Statcast terminal events are folded into the final reach/other bucket so the class IDs remain
stable and contiguous for both scikit-learn and XGBoost.

In addition to game state, handedness, score, and park context, V1 builds prior-date batter and pitcher rates
for every outcome family. Same-day outcomes remain excluded to preserve the project's conservative leakage
rule.

Models compared:

- training class-frequency baseline
- multinomial logistic regression
- XGBoost multiclass classifier

The predicted outcome probabilities are also translated back into expected terminal-event RE24 with a
training-only table of:

```text
starting base/out state × outcome class -> mean terminal-event RE24
```

This lets NorthStar test whether predicting the *distribution of baseball outcomes* is a better route to
expected run value than directly regressing a noisy RE24 number.

Run:

```bash
python scripts/train_pa_outcome_model.py
```

Outputs:

```text
reports/tables/pa_outcome_class_distribution_2025.csv
reports/tables/pa_outcome_model_metrics_2025.csv
reports/tables/pa_outcome_expected_re24_metrics_2025.csv
reports/tables/pa_outcome_xgboost_feature_importance_2025.csv
reports/tables/pa_outcome_xgboost_confusion_2025.csv
reports/figures/pa_outcome_xgboost_confusion_2025.png
data/processed/pa_outcome_test_predictions_2025.parquet
```

Classification is evaluated with multiclass log loss, accuracy, macro F1, and top-2 accuracy. The implied
expected-RE24 predictions are separately evaluated with MAE, RMSE, R², Pearson correlation, and Spearman
correlation. The prediction Parquet stays local.

## Train the pitch-decision model V1

The PA-level experiments showed that exact run value is too noisy to predict before a plate appearance begins.
The next stage moves inside the PA and conditions on the **actual pitch opportunity presented to the hitter**.

NorthStar classifies each physical pitch as:

```text
swing
take
```

Clock violations / automatic balls and strikes are excluded because no physical pitch opportunity was
presented. Bunt attempts, fouls, whiffs, and balls put in play count as swings.

V1 uses only information available as the pitch arrives:

- balls / strikes
- pitch type
- release velocity
- horizontal / vertical movement
- exact plate location
- location normalized to the hitter's strike-zone height
- batter / pitcher handedness
- runner / out state
- inning, home/away, and score differential

Horizontal location and movement are mirrored for left-handed hitters so the model can learn a common
hitter-perspective inside/outside geometry.

This is intentionally **not a pre-pitch forecast**. Plate location is allowed because the model asks how a
hitter responds to the pitch that was actually presented. Post-contact variables such as exit velocity,
launch angle, and terminal event are not used.

Models compared:

- empirical count + Statcast-zone swing-rate baseline
- logistic regression
- XGBoost

Run directly from the cached raw Statcast chunks:

```bash
python scripts/train_pitch_decision_model.py
```

Outputs:

```text
reports/tables/pitch_decision_model_metrics_2025.csv
reports/tables/pitch_decision_xgboost_feature_importance_2025.csv
reports/tables/pitch_decision_batter_holdout_2025.csv
reports/figures/pitch_decision_calibration_2025.png
data/processed/pitch_decision_test_predictions_2025.parquet
```

The local predictions include:

```text
decision_surprise = actual swing - predicted swing probability
```

This is a **behavioral deviation**, not a value metric. A surprising swing is not automatically a good swing.
The next stage will estimate the expected value of swing versus take for comparable pitch opportunities before
NorthStar assigns decision value.

## Train pitch decision-value V1

The swing/take model can tell us what hitters usually do, but **surprise is not value**. Decision-value V1 asks
a harder observational question:

> For this exact pitch opportunity, what future plate-appearance run value is associated with swinging versus taking?

For every physical pitch, NorthStar constructs a training-table run-value return from the current pitch state
through the end of the plate appearance:

```text
runs scored from this pitch through PA end
+ training-only RE(after PA)
- training-only RE(current pitch base/out state)
```

The run-expectancy table is built from the pre-August training partition only.

The strong XGBoost swing/take model from pitch-decision V1 is reused as a **propensity model**. The value
experiment then:

- trims pitches with estimated swing probability below 5% or above 95%
- computes stabilized inverse-propensity weights
- fits one sparse XGBoost value model on observed swings
- fits another sparse XGBoost value model on observed takes
- predicts both action values for every overlap pitch
- compares against a count + zone + action mean baseline
- reports doubly robust estimates for always-take, always-swing, and the model-selected policy

Run:

```bash
python scripts/train_pitch_decision_value.py
```

Outputs:

```text
reports/tables/pitch_decision_value_factual_metrics_2025.csv
reports/tables/pitch_decision_value_policy_estimates_2025.csv
reports/tables/pitch_decision_value_batter_holdout_2025.csv
reports/tables/pitch_value_swing_xgboost_importance_2025.csv
reports/tables/pitch_value_take_xgboost_importance_2025.csv
reports/figures/pitch_decision_value_holdout_2025.png
data/processed/pitch_decision_value_test_predictions_2025.parquet
```

The test predictions include separate `q_take` and `q_swing` estimates plus:

```text
modeled_action_advantage = modeled value of chosen action - modeled value of alternative action
```

This is **not yet a causal decision-value statistic**. Swing/take behavior is observational, propensity models
cannot guarantee that all confounding is measured, and V1 does not yet use cross-fitting. The doubly robust
policy estimates are diagnostics for whether the research direction is credible, not proof that a recommended
action would causally add the estimated number of runs.

## Train pitch-state value V2

Decision-value V1 showed measurable signal, but its target reused the rest of the plate appearance for every
pitch. V2 replaces that overlapping return with a **one-step decision-opportunity transition value**.

NorthStar first fits a training-only value table for discrete decision states:

```text
balls-strikes + base/out state -> expected future PA value
```

where future PA value is:

```text
runs remaining in PA + RE(after PA)
```

For each recognized swing/take opportunity:

```text
transition value
= runs before the next decision opportunity
+ V(next decision state)
- V(current decision state)
```

If the PA ends, `V(next)` is simply the training-only run expectancy of the post-PA base/out state.

The key property is **telescoping**. Summing observed transition values through one PA collapses to:

```text
runs scored
+ terminal run expectancy
- value of the first decision state
```

so the same downstream PA outcome is no longer copied onto every pitch.

V2 also records pitch-number gaps. A non-terminal transition is used for counterfactual swing/take training only
when the next recognized decision is the immediately following pitch number. Gapped transitions remain in the
telescoping audit but are excluded from the action-value model.

The observational swing/take machinery remains conservative:

- XGBoost swing propensity model
- 5%–95% common-support trimming
- stabilized inverse-propensity weights
- separate XGBoost swing and take transition-value models
- one-step doubly robust action estimates on the observed state distribution

Run:

```bash
python scripts/train_pitch_state_value.py
```

Outputs:

```text
reports/tables/pitch_decision_state_values_2025.csv
reports/tables/pitch_state_telescoping_audit_2025.csv
reports/tables/pitch_state_value_factual_metrics_2025.csv
reports/tables/pitch_state_value_one_step_policy_2025.csv
reports/tables/pitch_state_value_batter_holdout_2025.csv
reports/tables/pitch_state_value_swing_importance_2025.csv
reports/tables/pitch_state_value_take_importance_2025.csv
reports/figures/pitch_state_value_advantage_2025.png
data/processed/pitch_state_value_test_predictions_2025.parquet
```

The one-step policy table is **not a full sequential policy evaluation**. It compares actions over the state
distribution that actually occurred; it does not simulate how changing an early decision would alter the
distribution of later pitch states. That requires a later transition/dynamics model or fitted policy
evaluation.

## Audit pitch-state robustness

Pitch-state V2 produced a cleaner target and useful out-of-time signal, but NorthStar does not promote a player
metric from one holdout run. The next phase stress-tests the saved September predictions without refitting the
main model.

Run:

```bash
python scripts/audit_pitch_state_robustness.py
```

The audit uses the local V2 prediction file:

```text
data/processed/pitch_state_value_test_predictions_2025.parquet
```

and performs four checks:

- overlap sensitivity at 5%-95%, 10%-90%, and 20%-80% swing propensity
- game-cluster bootstrap uncertainty for model-action advantage versus observed decisions
- descriptive stability by count and Statcast zone
- early-vs-late September batter stability for modeled action advantage and optimal-choice rate

Propensity calibration is also rechecked on the saved 5%-95% overlap sample.

Outputs:

```text
reports/tables/pitch_state_robustness_overlap_2025.csv
reports/tables/pitch_state_robustness_propensity_2025.csv
reports/tables/pitch_state_robustness_propensity_bins_2025.csv
reports/tables/pitch_state_robustness_subgroups_2025.csv
reports/tables/pitch_state_robustness_player_stability_2025.csv
reports/tables/pitch_state_robustness_player_summary_2025.csv
reports/figures/pitch_state_robustness_overlap_2025.png
```

The cluster bootstrap resamples entire games rather than individual pitches, so its uncertainty estimate better
respects within-game dependence. It is still conditional on the fitted nuisance models and does not replace a
full causal sensitivity analysis.

Tighter overlap windows do **not** retrain the action-value models; they ask whether the saved holdout conclusion
persists when analysis is restricted to increasingly ambiguous swing/take opportunities. If the estimated
advantage changes sign or collapses under tighter overlap, NorthStar should not proceed to a player metric.

## Cross-fit pitch-state nuisance models

The robustness audit showed that pitch-state V2 survives tighter common support and game-cluster uncertainty.
The next phase targets a subtler source of optimism: nuisance models trained on the same observations used to
construct training weights.

Run:

```bash
python scripts/train_pitch_state_crossfit.py
```

The workflow:

- rebuilds the clean pre-August pitch-state training sample from cached Statcast
- creates **game-grouped out-of-fold** XGBoost swing propensities
- fits an isotonic calibrator using only those OOF predictions
- fits one final propensity model on all pre-August data for August/September scoring
- applies the OOF calibrator to validation/test probabilities
- restricts analysis to calibrated 5%-95% common support
- refits separate swing/take transition-value models at stabilized IPW caps **3, 5, 10, and 20**
- reports weight clipping, effective sample size, factual transition prediction, and game-cluster bootstrap
  intervals for model-action advantage

Default cross-fitting uses three folds because each fold trains a full-season pitch XGBoost model. Increase with
`--folds 5` for a heavier sensitivity run.

Outputs:

```text
reports/tables/pitch_state_crossfit_fold_metrics_2025.csv
reports/tables/pitch_state_crossfit_propensity_metrics_2025.csv
reports/tables/pitch_state_crossfit_propensity_bins_2025.csv
reports/tables/pitch_state_crossfit_weight_diagnostics_2025.csv
reports/tables/pitch_state_crossfit_factual_metrics_2025.csv
reports/tables/pitch_state_crossfit_policy_sensitivity_2025.csv
reports/figures/pitch_state_crossfit_weight_sensitivity_2025.png
data/processed/pitch_state_crossfit_test_predictions_2025.parquet
```

The canonical local prediction file uses weight cap 10. As before, it stays local.

Cross-fitting here is aimed at the **training propensity weights**. August and September are already
chronological holdouts, so their propensity and action-value predictions come from models trained only on the
pre-August period.

## Validate transport from 2024 to 2025

After the cross-fitted 2025 experiments survived overlap, calibration, clustering, and weight-cap sensitivity,
NorthStar stops tuning on 2025 and performs a true season-transport test.

First build the 2024 baseline:

```bash
python scripts/build_baseline.py --season 2024 --start 2024-03-20 --end 2024-09-30
```

Then run:

```bash
python scripts/validate_external_season.py
```

The default experiment is deliberately strict:

```text
training season: 2024
external season: 2025
```

Everything that defines the model is learned from 2024:

- empirical run expectancy
- count + base/out decision-state values
- game-grouped OOF swing propensity
- isotonic propensity calibration
- common-support training sample
- stabilized inverse-propensity weights
- separate swing/take transition-value models

The complete 2025 season is then scored without refitting any of those quantities.

The experiment again refits the 2024 action-value models at stabilized weight caps 3, 5, 10, and 20. On 2025
it reports propensity transport, exact training-state coverage, factual transition-value prediction, overlap,
and game-cluster bootstrap intervals for model-action advantage.

Outputs:

```text
reports/tables/external_season_sample_summary_2024_2025.csv
reports/tables/external_season_state_coverage_2024_2025.csv
reports/tables/external_season_propensity_folds_2024_2025.csv
reports/tables/external_season_propensity_metrics_2024_2025.csv
reports/tables/external_season_propensity_bins_2024_2025.csv
reports/tables/external_season_weight_diagnostics_2024_2025.csv
reports/tables/external_season_factual_metrics_2024_2025.csv
reports/tables/external_season_policy_sensitivity_2024_2025.csv
reports/figures/external_season_weight_sensitivity_2024_2025.png
data/processed/external_season_test_predictions_2024_2025.parquet
```

The external prediction Parquet remains local.

Passing this test does not make the one-step action contrast causal or a finished player metric. It establishes
a different property: **transport**. NorthStar must show that relationships learned from one MLB season retain
useful probability calibration, factual value signal, and action-estimate direction in a later season it did
not train on.

## Build the player decision-value prototype

After 2024-trained models transported successfully to the full 2025 season, NorthStar can finally test whether
the pitch-level signal aggregates into a repeatable hitter-level quantity.

The prototype intentionally does **not** use the name CCA yet.

Run:

```bash
python scripts/build_player_decision_metric.py
```

The default input is the canonical cap-10 external-season prediction file:

```text
data/processed/external_season_test_predictions_2024_2025.parquet
```

For each eligible pitch opportunity, NorthStar defines an opportunity-conditioned expected action value:

```text
expected action value
= P(swing | opportunity) * Q_swing
+ P(take | opportunity) * Q_take
```

The hitter's descriptive decision value is:

```text
decision value vs expected
= Q(chosen action) - expected action value
```

The same pitch also records:

```text
missed decision value
= max(Q_take, Q_swing) - Q(chosen action)

opportunity headroom
= max(Q_take, Q_swing) - expected action value
```

which gives the accounting identity:

```text
decision value vs expected + missed decision value = opportunity headroom
```

The leaderboard reports total and per-100 decision value, missed value, ordinary optimal-choice rate, and a
**value-weighted optimal-choice rate** where decisions with a larger modeled swing/take value gap count more.

A second view restricts the sample to 20%-80% calibrated swing propensity. These are called
**discretionary opportunities** because neither swing nor take was near-deterministic under the transported
2024 propensity model.

The default leaderboard requires 500 eligible decisions. Reliability is tested by splitting the 2025 season
at July 1 and requiring at least 200 decisions in both halves.

Outputs:

```text
reports/tables/player_decision_metric_prototype_2025.csv
reports/tables/player_decision_metric_half_splits_2025.csv
reports/tables/player_decision_metric_reliability_2025.csv
reports/figures/player_decision_metric_prototype_2025_top20.png
reports/figures/player_decision_metric_reliability_2025.png
```

This remains a descriptive, model-based prototype. The Q-functions are observational estimates and the metric
does not claim that forcing the modeled alternative action would causally change runs by the quoted amount.

## Validate and stabilize the player metric

The player prototype is repeatable enough to justify construct validation, but repeatability alone does not
tell us what the metric represents. The next workflow asks whether NorthStar is merely repackaging ordinary
plate discipline or whether it contains a distinct context-adjusted signal.

Run:

```bash
python scripts/validate_player_decision_metric.py
```

The workflow uses the cached 2025 Statcast pitch files to build independent full-season hitter summaries for:

- swing rate
- zone swing rate
- chase rate
- whiffs per swing
- contact rate
- first-pitch swing rate
- two-strike swing rate
- two-strike chase rate

It then joins the existing RE24 / Baseball Savant validation table where available, including:

- RE24 per 100 plate appearances
- wOBA
- xwOBA

NorthStar reports both pairwise Pearson/Spearman relationships and player-level out-of-fold Ridge diagnostics
showing how much of decision value can be reconstructed from simple plate discipline, offensive outcomes, or
both together.

### Sample-aware ranking prototype

The same workflow creates an explicitly provisional shrunken ranking.

For each hitter, the uncertainty of mean decision value is estimated with a game-cluster robust standard error.
A method-of-moments between-player variance is then used for normal-normal shrinkage toward the pitch-weighted
league mean.

The default ranking requires:

```text
>= 500 eligible decisions
>= 300 discretionary decisions
```

The displayed candidate is centered so league-average decision value is zero:

```text
centered value / 100
= 100 * (player mean decision value - league mean decision value)
```

and the shrunken version pulls smaller/noisier samples toward zero more strongly.

Outputs:

```text
reports/tables/player_metric_construct_dataset_2025.csv
reports/tables/player_metric_construct_correlations_2025.csv
reports/tables/player_metric_construct_cv_2025.csv
reports/tables/player_metric_shrunk_2025.csv
reports/figures/player_metric_construct_correlations_2025.png
reports/figures/player_metric_shrunk_2025_top20.png
```

Shrinkage is a ranking-stability device, not causal correction. The Q-functions remain observational, and the
normal-normal player model is an approximation rather than a claim that true player ability follows an exact
Gaussian distribution.

## Audit incremental player signal

Construct validation showed that the first player prototype is highly related to existing plate-discipline
behavior. NorthStar therefore tests the **residual**, not the raw leaderboard, before considering any public
metric name.

Run:

```bash
python scripts/audit_incremental_player_signal.py
```

The audit compares three increasingly rich explanations of player decision value:

```text
chase_only
swing_chase
plate_discipline
```

For the full season, each simple construct model is cross-fitted across players and NorthStar saves:

```text
incremental residual
= observed NorthStar decision value / 100
- out-of-fold construct prediction
```

The more important test is temporal. Early-2025 plate-discipline relationships are fit using the early half of
the season. Early player residuals are out-of-fold, while the fitted early relationship is transported to the
late half. NorthStar then asks whether the **same players remain above or below construct expectation** later
in the season.

A repeatable residual would support genuinely incremental player information. A near-zero residual
early/late correlation would mean the original ~0.70 reliability largely belongs to ordinary plate discipline
rather than to a distinct NorthStar player skill.

Outputs:

```text
reports/tables/player_metric_incremental_residuals_2025.csv
reports/tables/player_metric_incremental_summary_2025.csv
reports/tables/player_metric_incremental_half_dataset_2025.csv
reports/tables/player_metric_incremental_half_residuals_2025.csv
reports/tables/player_metric_incremental_reliability_2025.csv
reports/tables/player_metric_incremental_outcome_correlations_2025.csv
reports/figures/player_metric_incremental_reliability_2025.png
```

## Train the swing-execution component

NorthStar now treats hitter behavior as two separate descriptive axes:

```text
selection
= was swing/take the better choice?

execution
= given that the hitter swung, how much value did the swing create
  above or below league expectation for that pitch/context?
```

The selection component is the already-validated context-weighted plate-discipline signal.

Execution V1 is deliberately trained without hitter identity or any post-swing variables such as exit velocity,
launch angle, batted-ball type, or realized event. The model only sees the pitch and pre-/at-pitch context that
was available when the swing occurred.

Run:

```bash
python scripts/train_swing_execution_model.py
```

The default experiment again uses strict season transport:

```text
train expected swing execution: 2024
evaluate player execution:      2025
```

The target is the same telescoping one-step transition value used by pitch-state V2. On observed swing rows:

```text
swing execution value
= actual one-step transition value
- expected swing transition value from the 2024 league model
```

A positive residual means the hitter converted that swing better than the league model expected from pitch
location, movement, velocity, count, handedness, base/out state, inning, and score context. A negative residual
means worse conversion.

The workflow reports:

- count/zone/pitch-type empirical baseline versus XGBoost
- full 2025 external swing-value prediction metrics
- player execution value per 100 swings
- early/late 2025 split-half reliability
- a **two-axis selection/execution profile** without combining the axes into one score

Outputs:

```text
reports/tables/swing_execution_model_metrics_2024_2025.csv
reports/tables/player_swing_execution_2025.csv
reports/tables/player_swing_execution_half_splits_2025.csv
reports/tables/player_swing_execution_reliability_2025.csv
reports/tables/selection_execution_profile_2025.csv
reports/figures/player_swing_execution_reliability_2025.png
reports/figures/selection_execution_profile_2025.png
data/processed/swing_execution_test_predictions_2024_2025.parquet
```

The prediction Parquet remains local.

Execution V1 is still observational. In particular, the pitch itself is conditioned on because the question is
how well a hitter converted the swing that occurred; the model does not claim that the same swing outcome would
have occurred under a different action or pitch sequence.

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
