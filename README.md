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
