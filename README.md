# NorthStar Baseball

**NorthStar Baseball** is a context-aware baseball analytics project that studies hitter decision-making,
contact, damage, and run creation, with a dedicated focus on the Toronto Blue Jays and Canadian MLB players.

The research phase tested whether those components justified a new standalone **Chance Creation Added (CCA)**
style metric. Repeated construct, reliability, external-season, and future-prediction audits did not support
collapsing them into a superior universal score. NorthStar therefore preserves the validated quantities as
transparent contextual diagnostics alongside established baseball measures.

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
- FastAPI
- PostgreSQL + SQLAlchemy
- Next.js 16 / React 19 / TypeScript
- Docker + GitHub Actions

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
| 2026-09-30 | Production hardening | Prepared NorthStar for deployment rather than local-only execution. | Production now requires PostgreSQL, disables CSV fallback and API docs by default, adds `/ready`, generic backend-failure responses, gzip and security headers, frontend/API timeouts, server-only API configuration, a non-root minimal FastAPI Docker image, CI release gates, Dependabot, and a production runbook. |
| 2026-09-30 | Search UX | Player lookup was inconsistent: general search tolerated partial names, while comparison still expected exact full names. | Unified name resolution across the in-memory profile store, PostgreSQL store, API comparison endpoint, CLI/profile selector, homepage search, Players search, and comparison UI. First name, last name, partial name, full name, and MLBAM ID are now supported; accent-insensitive matching is included, exact matches are preferred, and ambiguous short names return a clear request for more detail instead of silently choosing a hitter. |
| 2026-09-30 | Frontend build | The new responsive player-list component failed lint/build with an invalid-character / unterminated-template error. | The generated TSX contained escaped backticks and interpolation markers as literal source characters. Rewrote the component with ordinary string composition so ESLint and Turbopack can parse it cleanly. |
| 2026-09-30 | Mobile UX | Small-screen navigation and long player leaderboards still felt desktop-shaped. | Reworked primary navigation into a fixed glass dock below 820px so it remains available on the homepage and every route, added safe-area spacing, and capped the mobile Players list at 8 rows with an accessible Show more / Show less control while preserving the full list on larger screens. |
| 2026-09-30 | Frontend lint | The first accessibility/responsive pass surfaced 8 ESLint findings: unescaped JSX punctuation in the Guide, synchronous state resets inside a search effect, internal navigation through `window.location.href`, and combobox ARIA attributes applied to the input's implicit textbox role. | Replaced JSX punctuation with typographic text, made short-query search state derived instead of synchronously reset in the effect, switched internal navigation to Next.js `useRouter().push()`, and declared the search input as a proper combobox while preserving its live result list semantics. |
| 2026-09-30 | Product UX | Completed the first site-wide consistency, responsive, and accessibility pass. | Added active navigation states, a keyboard skip link, visible focus treatment, keyboard-readable metric help, improved search/comparison semantics, reduced-motion support, stronger secondary-text contrast, and responsive behavior across navigation, profiles, comparison tables, search results, discovery rows, and team-context surfaces at tablet and mobile widths. |
| 2026-09-30 | Product UX | Added the first plain-language interpretation layer for non-expert users. | Player profiles now explain Selection / Contact / Damage in plain English, percentiles include a direct “higher than about X% of profiled hitters” interpretation, metric labels can expose concise help text, and a new `/how-it-works` guide defines the NorthStar axes plus wOBA, xwOBA, RE24, chase rate, and barrel rate without introducing a composite score. |
| 2026-09-27 | Dev warning | Next.js reported a hydration mismatch on `/compare` because browser-injected `data-new-gr-c-s-check-loaded` and `data-gr-ext-installed` attributes appeared on `<body>` before React hydrated. | These attributes are injected by Grammarly, not NorthStar. Disable Grammarly for localhost or test in an extension-free/incognito window; no application code change is required unless the warning persists without extensions. |
| 2026-09-27 | Frontend visual pass | Canada V2 retains the existing hero and NorthStar palette while adding birthplace/province context, full Selection / Contact / Damage shapes, separate lens leaders, and clearer summary cards. | Percentiles remain league-relative, exact leader ties are retained, missing axes remain unavailable rather than zero, and mobile rows preserve all three axes. Canadian-born still means MLB birth country equals Canada; no citizenship inference or composite score is introduced. |
| 2026-09-27 | Data presentation | Canada RE24 coverage now counts finite values in the displayed cohort instead of calling a separately limited leaderboard and labeling its count qualified. | Zero and negative values count as available. The page explains that RE24 is joined from the separate run-value validation source; a missing value is not zero or evidence of poor performance. No individual missing-value cause or qualification threshold is invented. |
| 2026-09-27 | Frontend validation | Canada V2 passed the production build, four frontend data-handling tests, and ESLint for changed code. Production-browser inspection used the committed Canadian hitter CSV through a temporary API fixture at desktop (1440 px) and mobile (390 px) widths. | Eight rows and 24 axes render without horizontal overflow or browser runtime errors; the RE24 explanation link works. The mobile section heading was stacked after visual review. PostgreSQL integration was not rerun in this environment. |
| 2026-09-27 | Validation warning | Full-project frontend lint remains blocked by the pre-existing synchronous state update in `web/components/search-box.tsx` (`react-hooks/set-state-in-effect`), with an existing internal-navigation warning in the same file. | Canada V2 changed-file lint passes; unrelated search behavior was left unchanged. The initial browser download failed, so visual checks used a temporary npm-provided Chromium runtime without adding a project dependency. |
| 2026-09-27 | Frontend bug | A player-page backend failure was being swallowed by the dynamic route and rendered as the generic **404 / off the board** screen, making an API outage look like an unknown hitter. | Player requests now preserve API status codes: only a real HTTP 404 becomes the hitter-not-found page. Other backend failures surface a dedicated profile-service error state with retry guidance. |
| 2026-09-27 | Frontend visual pass | Blue Jays V2 completed after browser review. | Reworked the page into a team-context surface with Toronto exposure / primary-team / multi-team / Canadian-born summary cards, three separate NorthStar context leaders, richer roster rows with full three-axis shape and exposure flags, and clearer language distinguishing Toronto membership from full-season MLB diagnostics. |
| 2026-09-27 | Frontend visual pass | Player index V2 completed after browser review. | Turned the page from a long RE24 list into a discovery surface with switchable RE24 / Selection / Contact / Damage lenses, richer rows showing each hitter's three-axis NorthStar shape, sample context, clearer hierarchy, and responsive behavior while preserving search as the primary action. |
| 2026-09-27 | Frontend visual pass | Comparison workbench V2 completed after successful Judge vs. Arráez browser test. | Replaced the table-first presentation with visual hitter profile cards, three-axis percentile rails, compact established-outcome summaries, a clearer metric matrix, and refined controls while preserving the no-winner/no-composite interpretation. |
| 2026-09-27 | Frontend bug | The comparison workbench returned **Failed to fetch** in the browser while server-rendered NorthStar pages were working. | Browser interactions now use a same-origin Next.js proxy at `/api/northstar/*`, with the FastAPI upstream URL kept server-side. This removes browser CORS/origin mismatch risk for comparison and live search and returns a clearer 502 message if FastAPI is actually offline. |
| 2026-09-27 | Frontend visual pass | Player profile V2 completed after Aaron Judge browser review. | Rebalanced the oversized empty header with a three-axis player signature, gave Selection / Contact / Damage distinct visual identities, replaced nested notice boxes with compact percentile tracks, added a clearer transition into supporting metrics, refined detail-card hierarchy, and increased disclaimer legibility without introducing a composite score. |
| 2026-09-27 | Frontend visual pass | Homepage V2 polish completed after first browser review. | Increased supporting-text/nav/search scale, tightened hero-to-content spacing, expanded the NorthStar orbital visual to all three axes, added layered signal paths and reduced-motion-safe micro-motion, and gave Selection / Contact / Damage cards distinct visual signatures without adding dashboard clutter. |
| 2026-09-27 | Frontend security | Next.js production build completed successfully, then `npm audit` reported **3 vulnerabilities (2 high, 1 critical)** through `next`, `postcss`, and `sharp`. npm indicated the remediation path is Next.js **16.3.6**. | Updated `next` and `eslint-config-next` from 16.2.6 to 16.3.6. Regenerate the lockfile with `npm install`, then rerun `npm audit` and `npm run build`; do not use `npm audit fix --force`. |
| 2026-09-27 | Frontend setup | `web/` dependency installation completed successfully: **343 packages added / 344 audited**. npm also reported an ESLint deprecation warning plus **3 dependency vulnerabilities (2 high, 1 critical)**. | Installation itself succeeded. Do **not** apply `npm audit fix --force` blindly; inspect `npm audit` first, then update only the affected packages without destabilizing the Next.js stack. |
| 2026-09-27 | Product milestone | Next.js Web UI V1 started on `feat/web-ui-v1`. | Added a responsive dark NorthStar interface with live API-backed landing page, hitter search, individual player profiles, Canadian-born and Toronto views, and an interactive 2-4 player comparison workbench. Visual design keeps contextual diagnostics distinct from established metrics and avoids an overall score. |
| 2026-09-27 | Parity fix | PostgreSQL smoke tests confirmed metadata, player search, Toronto team filtering, and Canadian RE24 leaderboard parity, but the SQL-backed `/api/v1/compare` response exposed the entire stored profile JSON instead of the smaller comparison-field contract used by the CSV backend. | SQL comparison responses now filter through the shared `COMPARISON_FIELDS` contract, with a regression test preventing backend-specific extra fields. |
| 2026-09-27 | Success | PostgreSQL Product Store V1 passed its first real local integration run. Docker Compose started `postgres:17-alpine` healthy on port 5432, the canonical 2025 profile loader inserted **404 profiles**, and FastAPI `/api/v1/meta` returned `storage_backend: postgresql` with **404 players**. | This confirms the application is reading NorthStar profiles from PostgreSQL rather than the CSV fallback. |
| 2026-09-27 | Success | FastAPI V1 was merged into `main` through PR #21 after **117 tests**, Uvicorn runtime checks, live Judge search, Judge/Arráez comparison, and UTF-8 response hardening. | NorthStar now has a stable read-only application API over the validated profile dataset. |
| 2026-09-27 | Product milestone | PostgreSQL Product Store V1 started on `feat/product-db-v1`. | Added SQLAlchemy models, normalized team membership, transactional season loading, a SQL-backed profile-store implementation, local PostgreSQL Compose service, and API backend selection through `NORTHSTAR_DATABASE_URL`. |
| 2026-09-27 | Runtime fix | Live Judge search and Judge/Arráez comparison endpoints returned the expected records, but Windows PowerShell rendered the accented name as `Luis ArrÃ¡ez` even though the stored profile uses `Luis Arráez`. | API JSON responses now explicitly declare `charset=utf-8`, and the test suite includes an accented-name round-trip assertion so browser/CLI clients receive unambiguous UTF-8 metadata. |
| 2026-09-27 | Success | Live API data smoke tests passed for player search and two-player comparison. Judge search returned MLBAM 592450 with the expected 2025 profile; comparison returned Judge and Arráez plus the no-winner/no-composite interpretation. | FastAPI V1 is functionally ready to preserve after the UTF-8 response hardening check. |
| 2026-09-27 | Success | FastAPI V1 passed **116/116 tests with no warning summary**, started successfully under Uvicorn, and returned HTTP **200** from `/health` with service `northstar-baseball-api`, API version `v1`, season `2025`. | Automated and runtime smoke tests now agree. Added a small `/` discovery endpoint so opening the API base URL returns service/docs links instead of the previously observed harmless 404. |
| 2026-09-27 | Warning | FastAPI tests passed **116/116**, but Starlette emitted a deprecation warning because `starlette.testclient` now prefers the successor package `httpx2`. An initial attempted fix incorrectly required nonexistent `httpx>=2` releases. | Corrected the dev dependency to `httpx2>=2,<3`; reinstall the dev environment and rerun pytest to confirm a warning-free suite. |
| 2026-09-27 | Success | Canada Layer V1 was merged into `main` through PR #20 after full identity, historical team-scope, and comparison review. | The project now has validated Canadian-born, Toronto Blue Jays, and multi-player comparison datasets available as product inputs. |
| 2026-09-27 | Product milestone | FastAPI V1 added on `feat/product-api-v1`. | Read-only endpoints now expose player search/detail, Canadian-born and team scopes, named-metric leaderboards, metadata, and 2–4 player comparisons while preserving NorthStar's no-composite-score interpretation. |
| 2026-09-27 | Data-quality fix | Blue Jays review exposed blank `plate_appearances` for some sub-300-PA hitters because that field came from the qualified offensive-reference table rather than the full PA reconstruction. | Canada/team views now expose `season_plate_appearances` from the complete Statcast PA dataset and use it for season/team presentation, while the qualified outcome field remains available separately. |
| 2026-09-27 | Success | Canada layer V1 completed locally with **110 passing tests**. MLB identity resolution covered **404/404** canonical hitters; the automated birthplace rule identified **8 Canadian-born hitters**, and the 2025 PA-derived team view identified **17 hitters with at least one Toronto PA**. | The Canada layer is now end-to-end functional, with cached identities, explicit birthplace scope, historical team exposure, Canadian-born views, and a Blue Jays view. |
| 2026-09-27 | Success | The first multi-player comparison report completed for Aaron Judge and Luis Arráez. | The comparison preserved the opposing profile shapes from the individual cards (Judge: 98th Selection / 4th Contact / 100th Damage; Arráez: 4th / 100th / 6th) without creating an overall score or declaring a winner. |
| 2026-09-27 | Success | NorthStar Hitter Profile V1 was merged into `main` through PR #19 after representative Aaron Judge and Luis Arráez cards validated opposite player shapes. | The canonical profile now exposes Selection, Contact, and Damage alongside established measures without a composite score. |
| 2026-09-27 | Design decision | The Canada layer defines its automated player subset using MLB birthplace data rather than inferring nationality. | `birth_country == Canada` is labeled **Canadian-born**; citizenship and national-team eligibility are intentionally not inferred. Team membership is derived separately from 2025 plate appearances. |
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
| 2026-09-27 | Experiment result | Swing execution V1 improved external 2025 one-step prediction over the count/zone/pitch-type baseline, but absolute predictive signal remained weak: XGBoost R² **0.0207**, Pearson **0.144**, Spearman **0.125** versus baseline R² **0.0041**. | Incoming pitch/context alone explains only a small share of realized one-step swing value, so the residual remains highly outcome-noisy. |
| 2026-09-27 | Experiment result | Player execution value per 100 swings had only **0.248/0.255** early/late Pearson/Spearman reliability across **313 hitters**. Positive-execution rate was more stable at **0.464/0.463**, while expected swing opportunity value itself was **0.555/0.538** reliable. | Hitters receive repeatably different swing opportunities, but the all-in-one realized-value residual is not stable enough to retain as a standalone execution axis. |
| 2026-09-27 | Experiment result | Selection and execution V1 were distinct cross-sectionally (Pearson **-0.141**, Spearman **-0.119** across **384 players**). | Distinctness is useful, but low execution reliability blocks a two-axis public profile from this formulation. |
| 2026-09-27 | Methodology decision | NorthStar will preserve swing execution V1 as an informative mixed/negative result and decompose execution further. | V2 separates **contact execution** (contact made vs expected on an observed swing) from **damage execution** (contact quality above expected conditional on a batted ball) instead of tuning the noisy all-in-one run-value residual. |
| 2026-09-27 | Success | Selection/execution V1 and its preserved external-season results were merged into `main` through PR #16. | The project keeps the negative reliability result visible and advances to a cleaner execution decomposition instead of tuning toward a preferred answer. |
| 2026-09-27 | Design decision | Contact/damage execution V2 uses separate 2024-trained models transported to 2025. | Contact is calibrated probability residual on observed swings; damage is Statcast xwOBA-on-contact residual on balls in play. Neither model uses batter identity or post-outcome variables as features. |
| 2026-09-27 | Experiment result | Contact execution V2 transported strongly from 2024 to 2025. The calibrated contact model reached ROC AUC **0.774**, log loss **0.4390**, and Brier **0.1388** versus the empirical baseline at **0.733 / 0.4974 / 0.1523**. | The incoming-pitch/context model has useful discrimination and calibration for whether a swing produces contact. |
| 2026-09-27 | Experiment result | Player **contact above expected per 100 swings** showed **0.820/0.803** early/late Pearson/Spearman reliability across **313 hitters**, exceeding raw contact-rate reliability (**0.803/0.800**) while expected-contact opportunity itself remained stable (**0.770/0.737**). | Contact execution is a strong candidate player axis and materially outperforms the all-in-one execution V1 reliability. |
| 2026-09-27 | Experiment result | Damage execution V2 also improved substantially over execution V1. External XGBoost contact-quality prediction reached R² **0.0312**, Pearson **0.177**, and Spearman **0.144**; player **damage xwOBA points above expected** showed **0.677/0.645** split-half reliability across **319 hitters**. | Pitch-level contact quality remains difficult to predict, but the player residual aggregates into a repeatable hitter-level signal. |
| 2026-09-27 | Experiment result | The three hitter dimensions are not redundant: selection vs contact Pearson/Spearman **-0.312/-0.302**, selection vs damage **0.067/0.062**, and contact vs damage **-0.461/-0.462**. | Selection, bat-to-ball execution, and damage execution appear to capture meaningfully different aspects of hitter behavior, but construct validation is required before formalizing the profile. |
| 2026-09-27 | Success | Contact/damage execution V2 and its preserved 2024 → 2025 results were merged into `main` through PR #17. | Both contact and damage now advance to the same incremental-validity standard that prevented the earlier selection prototype from being overclaimed. |
| 2026-09-27 | Design decision | Contact/damage construct validation tests the residuals against conventional hitter skills before any profile is formalized. | Contact is challenged with raw, zone, chase, and two-strike contact rates; damage is challenged with raw xwOBA on contact, average exit velocity, hard-hit rate, barrel rate, and sweet-spot rate. |
| 2026-09-27 | Experiment result | Contact construct validation showed very high overlap with conventional contact skill. Raw contact rate reconstructed **84.1%** of full-season contact-above-expected variance and the situational contact set reconstructed **96.1%**. Raw NorthStar split-half reliability was **0.820/0.803** Pearson/Spearman, while the situational-contact residual retained **0.403/0.359**. | Contact execution is mostly conventional contact skill with a smaller repeatable context-adjusted remainder; a stronger future-prediction test is required before claiming incrementality. |
| 2026-09-27 | Experiment result | Damage construct validation showed raw xwOBA on contact reconstructed **96.5%** of full-season damage-above-expected variance. Raw damage reliability was **0.677/0.645**, and the full-contact-quality residual remained **0.648/0.651**. | This apparently strong residual is not sufficient evidence of novelty because damage is algebraically defined as observed xwOBA minus expected xwOBA; residualizing against observed xwOBA can leave a stable negative expected-opportunity component. |
| 2026-09-27 | Validation warning | Residual reliability can be misleading for metrics defined as **observed minus expected**. | NorthStar will therefore compare whether EARLY conventional metrics plus the EARLY NorthStar adjustment improve out-of-fold prediction of the LATE observed conventional skill versus EARLY conventional metrics alone. This targets future skill directly and avoids interpreting stable opportunity/context as new player ability. |
| 2026-09-27 | Experiment result | The direct future-skill audit found **no meaningful incremental predictive value for contact**. Early raw contact predicted late contact at OOF R² **0.6316**; adding NorthStar contact changed this to **0.6310** (Δ **-0.0006**). With situational contact features, R² changed from **0.6264** to **0.6266** (Δ **+0.0002**). | NorthStar contact-above-expected is a highly reliable context-adjusted diagnostic, but it does not materially improve prediction of future contact skill beyond conventional contact metrics. |
| 2026-09-27 | Experiment result | Damage showed a conditional result. Adding NorthStar damage to EV/hard-hit/barrel/sweet-spot features improved late xwOBA-on-contact OOF R² from **0.4653** to **0.4898** (Δ **+0.0245**), but once early xwOBA-on-contact was included the full model moved from **0.4906** to **0.4886** (Δ **-0.0021**). | NorthStar damage contains some information beyond coarse batted-ball-quality summaries, but no incremental future prediction once the stronger conventional xwOBA-on-contact measure is already known. |
| 2026-09-27 | Methodology decision | NorthStar will not promote selection, contact, or damage as novel standalone predictive statistics. | The validated outputs will be retained as **contextual hitter diagnostics** inside a transparent multi-axis player profile alongside established measures. The project now shifts from inventing a single metric toward explaining *why* conventional hitter outcomes look the way they do. |
| 2026-09-27 | Visualization issue | Hitter Profile Card V1 was functionally correct but read like a diagnostic Matplotlib chart: oversized bars dominated the layout, rate statistics were shown as decimals, percentile notation was inconsistent, and the conventional context block had weak visual hierarchy. | Reworked the card into V2 with equal-weight Selection/Contact/Damage diagnostic cards, natural percentage formatting, ordinal percentiles, separate plate-discipline / bat-to-ball / contact-quality panels, compact sample footer, and a single-line interpretation disclaimer. |
| 2026-09-27 | Error | Accented player names produced awkward profile filenames (for example, Luis Arráez became `luis-arr-ez`). | Profile filename slugs now normalize Unicode accents to ASCII before replacing punctuation, producing stable names such as `luis-arraez`. |
| 2026-09-27 | Error | The first contact-model cross-fit failed in scikit-learn's ColumnTransformer because nullable pandas `Float64` columns (for example `release_speed`) still contained `pd.NA`. | Contact/damage chunk loading now passes every pitch frame through the existing `prepare_pitch_decision_model_frame` normalization step before modeling, converting numeric features to ordinary `float32`, categorical features to strings, and dropping incomplete model rows consistently with the validated pitch-decision pipeline. |

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

## Train contact / damage execution V2

Swing-execution V1 was too noisy at the player level, so NorthStar decomposes execution into two narrower
questions instead of trying to tune the same run-value residual.

Run:

```bash
python scripts/train_contact_damage_execution.py
```

The experiment again trains on 2024 and evaluates the complete 2025 season without refitting on 2025.

### Contact execution

Among observed swings:

```text
contact execution residual
= I(contact)
- calibrated P(contact | swing, pitch/context)
```

The 2024 contact model is cross-fitted by game to fit an isotonic calibrator, then refit on all 2024 swings and
transported to 2025. Player output is expressed as **contact above expected per 100 swings**.

### Damage execution

Among balls put into play with a valid Statcast `estimated_woba_using_speedangle` target:

```text
damage execution residual
= observed xwOBA contact quality
- expected xwOBA contact quality from pitch/context
```

The player output is reported in **xwOBA points above expected**:

```text
1000 * mean(damage residual)
```

A value of +20 therefore means roughly +.020 xwOBA contact quality above the 2024-trained league expectation
for the batted-ball opportunities the hitter produced.

Neither model uses batter identity. The contact model never uses the realized pitch description as a feature,
and the damage model never uses exit velocity, launch angle, batted-ball type, or the xwOBA target as features.

The workflow separately reports:

- empirical count/zone/pitch-type baseline versus XGBoost
- external 2025 contact probability metrics
- external 2025 damage regression metrics
- player contact split-half reliability
- player damage split-half reliability
- pairwise relationships among selection, contact execution, and damage execution

Outputs:

```text
reports/tables/contact_execution_model_metrics_2024_2025.csv
reports/tables/damage_execution_model_metrics_2024_2025.csv
reports/tables/player_contact_execution_2025.csv
reports/tables/player_damage_execution_2025.csv
reports/tables/player_contact_execution_half_splits_2025.csv
reports/tables/player_damage_execution_half_splits_2025.csv
reports/tables/player_contact_execution_reliability_2025.csv
reports/tables/player_damage_execution_reliability_2025.csv
reports/tables/selection_contact_damage_profile_2025.csv
reports/figures/player_contact_execution_reliability_2025.png
reports/figures/player_damage_execution_reliability_2025.png
reports/figures/selection_contact_damage_profile_2025.png
```

Local prediction artifacts remain Git-ignored:

```text
data/processed/contact_execution_test_predictions_2024_2025.parquet
data/processed/damage_execution_test_predictions_2024_2025.parquet
```

The two components are not combined into a score. Each must first demonstrate meaningful external-season
prediction quality and player-level repeatability on its own.

## Validate contact and damage beyond conventional metrics

The V2 execution components are repeatable, but NorthStar applies the same novelty test that rejected the first
selection prototype as a standalone metric.

Run:

```bash
python scripts/validate_contact_damage_execution.py
```

No new data download or model retraining is required. The workflow reads the cached 2025 Statcast season plus
the already-preserved contact/damage execution tables.

### Contact challenge set

NorthStar independently aggregates:

```text
overall contact rate
zone contact rate
chase contact rate
two-strike contact rate
whiffs per swing
```

The main incremental tests are:

```text
raw_contact
situational_contact
```

For each construct set, a player-level five-fold OOF Ridge model tries to reconstruct
`contact_above_expected_per_100_swings`.

The stricter temporal test learns the contact/construct relationship from the early half and transports it to
the late half. The early and late **residuals** are then correlated player by player.

### Damage challenge set

NorthStar independently aggregates:

```text
raw xwOBA on contact
average exit velocity
hard-hit rate (>= 95 mph)
barrel rate (Statcast launch_speed_angle == 6)
sweet-spot rate (8°-32° launch angle)
```

The main incremental tests are:

```text
raw_xwoba_contact
batted_ball_quality
full_contact_quality
```

Again, the key test is whether early/late **damage residuals** remain repeatable after these conventional
quality-of-contact summaries explain what they can.

Both residual families are also compared descriptively with RE24/100 PA, wOBA, and xwOBA.

Outputs:

```text
reports/tables/contact_execution_construct_dataset_2025.csv
reports/tables/damage_execution_construct_dataset_2025.csv
reports/tables/contact_execution_construct_correlations_2025.csv
reports/tables/damage_execution_construct_correlations_2025.csv
reports/tables/contact_execution_incremental_residuals_2025.csv
reports/tables/damage_execution_incremental_residuals_2025.csv
reports/tables/contact_execution_incremental_summary_2025.csv
reports/tables/damage_execution_incremental_summary_2025.csv
reports/tables/contact_execution_incremental_half_dataset_2025.csv
reports/tables/damage_execution_incremental_half_dataset_2025.csv
reports/tables/contact_execution_incremental_half_residuals_2025.csv
reports/tables/damage_execution_incremental_half_residuals_2025.csv
reports/tables/contact_execution_incremental_reliability_2025.csv
reports/tables/damage_execution_incremental_reliability_2025.csv
reports/tables/contact_execution_incremental_outcome_correlations_2025.csv
reports/tables/damage_execution_incremental_outcome_correlations_2025.csv
reports/figures/contact_execution_incremental_reliability_2025.png
reports/figures/damage_execution_incremental_reliability_2025.png
```

The decision rule is intentionally asymmetric: contact and damage do not have to rise or fall together. A
component can be retained if it shows meaningful incremental repeatability even if the other component does not.

## Audit future incremental value

Residual reliability is not the final novelty test for an observed-minus-expected metric. For example:

```text
damage adjustment
= observed xwOBA on contact - expected xwOBA on contact
```

If that adjustment is later residualized against observed xwOBA, the remainder can approximate
`-expected xwOBA`. A stable residual may therefore reflect stable opportunity quality rather than a distinct
execution skill.

The stronger audit predicts the **future observed conventional skill directly**.

Run:

```bash
python scripts/audit_future_execution_value.py
```

For contact:

```text
Model A
EARLY conventional contact metrics
        ↓
OOF prediction of LATE observed contact rate

Model B
EARLY conventional contact metrics
+ EARLY NorthStar contact-above-expected
        ↓
OOF prediction of LATE observed contact rate
```

For damage:

```text
Model A
EARLY xwOBA/contact-quality metrics
        ↓
OOF prediction of LATE observed xwOBA on contact

Model B
EARLY xwOBA/contact-quality metrics
+ EARLY NorthStar damage-above-expected
        ↓
OOF prediction of LATE observed xwOBA on contact
```

Both models use the **same players and same cross-validation folds**. The primary diagnostic is:

```text
delta OOF R²
= augmented OOF R² - conventional-only OOF R²
```

The workflow also reports paired changes in Pearson, Spearman, MAE, and RMSE.

A positive out-of-fold improvement supports incremental predictive information in the NorthStar adjustment.
Little or negative improvement means the context-adjusted component may be useful descriptively but does not
add meaningful information about future hitter skill beyond the conventional metrics already available.

Outputs:

```text
reports/tables/contact_execution_future_prediction_2025.csv
reports/tables/damage_execution_future_prediction_2025.csv
reports/tables/contact_execution_future_predictions_2025.csv
reports/tables/damage_execution_future_predictions_2025.csv
reports/figures/execution_future_incremental_value_2025.png
```

## Final interpretation of the hitter research phase

NorthStar's hitter research produced three reliable context-adjusted diagnostics:

```text
selection
contact execution
damage execution
```

The validation program also established an important limit: none currently supports a claim of being a new
standalone predictive hitter statistic once sufficiently strong conventional measures are already known.

The direct future-skill audit is the final gate.

### Contact

Adding early NorthStar contact-above-expected to early conventional contact metrics produced essentially no
change in out-of-fold prediction of late contact rate.

```text
raw contact:
R² 0.6316 -> 0.6310   delta -0.0006

situational contact:
R² 0.6264 -> 0.6266   delta +0.0002
```

NorthStar contact is therefore retained as a **context adjustment / explanatory diagnostic**, not as a new
predictive contact skill.

### Damage

NorthStar damage added modest predictive information beyond coarse batted-ball-quality summaries:

```text
EV + hard-hit + barrel + sweet spot:
R² 0.4653 -> 0.4898   delta +0.0245
```

But once early xwOBA on contact was already included:

```text
full contact-quality set:
R² 0.4906 -> 0.4886   delta -0.0021
```

So NorthStar damage is also retained as a contextual diagnostic rather than promoted as a standalone new
predictive statistic.

The product direction is therefore a **NorthStar Hitter Profile**: show established measures and NorthStar's
context-adjusted diagnostics together, with clear labels separating conventional skill, opportunity/context,
and NorthStar explanatory adjustments.

## Build the NorthStar Hitter Profile

The validated hitter research now feeds a product-facing profile rather than a single invented score.

Run:

```bash
python scripts/build_hitter_profiles.py
```

This creates a canonical 2025 table joining:

```text
NorthStar selection context
NorthStar contact-above-expected context
NorthStar damage-above-expected context

conventional swing / chase / contact measures
xwOBA on contact
exit velocity / hard-hit / barrel measures
wOBA / xwOBA
RE24 / 100 PA
sample sizes and uncertainty fields
```

The profile deliberately has **no overall score**.

Each axis receives its own league-relative percentile so users can compare dimensions without collapsing them
into one ranking. The percentiles describe position within the available 2025 profile sample; they are not
claims that NorthStar has discovered new independent predictive skills.

Output:

```text
reports/tables/hitter_profile_2025.csv
```

To generate an individual profile card:

```bash
python scripts/build_hitter_profiles.py --player "Aaron Judge"
```

or use an MLBAM ID:

```bash
python scripts/build_hitter_profiles.py --player 592450
```

The player card places the three NorthStar contextual axes beside established measures and explicitly states
that no composite overall rating is being produced.

## Build the Canada layer

NorthStar adds a Canada-focused identity and team layer on top of the canonical hitter profile.

Run:

```bash
python scripts/build_canada_layer.py
```

The workflow:

- resolves profile hitters through the MLB Stats API people endpoint
- records public birthplace fields and marks `canadian_born` only when `birth_country == Canada`
- derives each hitter's 2025 batting-team exposure directly from NorthStar's plate-appearance dataset
- creates a Canadian-born hitter view and RE24/100 PA leaderboard
- creates a Toronto Blue Jays view for hitters with at least one 2025 Toronto plate appearance

"Canadian-born" is intentionally a birthplace definition, **not a citizenship or nationality claim**.

The Blue Jays table also keeps an important scope distinction: NorthStar diagnostic fields are full-season MLB
values, while `tor_plate_appearances`, `tor_pa_share`, and team columns describe Toronto exposure. This
avoids silently presenting a traded player's full-season profile as if every observation occurred with Toronto.

Outputs:

```text
reports/tables/player_identity_2025.csv
reports/tables/player_team_context_2025.csv
reports/tables/hitter_profile_with_canada_2025.csv
reports/tables/canadian_hitter_profile_2025.csv
reports/tables/canadian_hitter_leaderboard_2025.csv
reports/tables/blue_jays_hitter_profile_2025.csv
```

Identity records are cached in the reports table so repeated local runs do not need to refetch already-resolved
players. Use `--refresh-identities` when a clean lookup is desired.

## Compare hitter profiles

After building the Canada layer, compare two to four hitters:

```bash
python scripts/compare_hitter_profiles.py --players 592450 650333
```

Names can also be supplied when quoted. The report compares NorthStar Selection / Contact / Damage percentiles
with established measures such as chase avoidance, contact rate, xwOBA on contact, barrel rate, xwOBA, and
RE24 / 100 PA.

The comparison deliberately does **not** compute a winner, grade, or overall score.

## Run the FastAPI service

NorthStar now exposes the canonical hitter profile through a read-only FastAPI service.

Install/update the local environment after pulling this branch:

```bash
python -m pip install -e ".[dev]"
```

Start the API:

```bash
uvicorn northstar.api.app:app --reload
```

Default local URLs:

```text
API:      http://127.0.0.1:8000
OpenAPI:  http://127.0.0.1:8000/docs
Schema:   http://127.0.0.1:8000/openapi.json
```

Core endpoints:

```text
GET /health
GET /api/v1/meta
GET /api/v1/players
GET /api/v1/players/{mlbam_id}
GET /api/v1/canada
GET /api/v1/teams/{team_code}
GET /api/v1/leaderboards
GET /api/v1/compare
```

Examples:

```text
/api/v1/players?query=judge
/api/v1/players?team=TOR
/api/v1/players?canadian_born=true
/api/v1/players/592450
/api/v1/teams/TOR
/api/v1/leaderboards?scope=canadian&sort_by=re24_per_100_pa
/api/v1/leaderboards?scope=team:TOR&sort_by=est_woba&min_pa=200
/api/v1/compare?player=592450&player=650333
```

The API returns the same validated product semantics as the report layer:

- no composite NorthStar score
- Canadian scope means Canadian-born, not inferred citizenship
- team filters use 2025 plate-appearance exposure
- player diagnostics are full-season MLB values unless a future endpoint explicitly says otherwise
- comparison endpoints return profiles without declaring a winner

## Run the PostgreSQL profile store

NorthStar can now serve the same API contract from PostgreSQL instead of reading the canonical CSV directly.

The local development database is defined in `compose.yaml`.

Start it:

```bash
docker compose up -d postgres
```

Load the canonical enriched profile:

```bash
python scripts/load_profiles_to_db.py
```

The loader replaces one season transactionally and creates:

```text
hitter_profiles
    one canonical JSON-backed profile per MLBAM hitter / season
    plus materialized fields used by search and leaderboards

hitter_team_memberships
    normalized hitter / season / team rows used for historical team filtering
```

The default development connection is:

```text
postgresql+psycopg://northstar:northstar@localhost:5432/northstar
```

Set `NORTHSTAR_DATABASE_URL` before starting FastAPI to switch the service from the CSV store to the SQL store:

```powershell
$env:NORTHSTAR_DATABASE_URL="postgresql+psycopg://northstar:northstar@localhost:5432/northstar"
python -m uvicorn northstar.api.app:app --reload
```

If `NORTHSTAR_DATABASE_URL` is absent during local development, the API keeps the CSV-backed store as a fallback.
When `NORTHSTAR_ENV=production`, PostgreSQL is required and the CSV fallback is disabled. Both backends implement
the same player/search/team/leaderboard/comparison semantics.

The first SQL schema keeps frequently queried product fields as typed columns while retaining the complete
canonical profile in JSON. This lets the product migrate away from flat files without discarding validated
research fields that are not yet first-class UI columns.

## Run the Next.js interface

The first NorthStar web client lives in `web/` and consumes the FastAPI service.

It includes:

- a dark, data-forward landing page with live 2025 dataset status
- global hitter search by name or MLBAM ID
- individual hitter profile pages
- separate Selection / Contact / Damage context presentation
- established wOBA, xwOBA, chase, contact, barrel, and RE24 context
- Canadian-born player view
- Toronto Blue Jays 2025 exposure view
- interactive 2-4 hitter comparison
- responsive desktop/mobile layouts
- explicit product language preserving NorthStar's no-composite-score interpretation

Run the API first, then in a second terminal:

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Frontend verification (run from `web/`; the TypeScript test runner requires Node.js 22.6+):

```bash
npm test
npm run lint
npm run build
```

Canada V2 tests cover missing/zero RE24 values, empty cohorts, independent lens leaders and ties,
and incomplete birthplace metadata. Its summary and lens cards use the same returned Canadian-born
cohort as the hitter rows; percentiles retain their existing league-relative meaning.


Local frontend environment variables are documented in `web/.env.example`:

```text
NORTHSTAR_API_URL=http://127.0.0.1:8000
NORTHSTAR_API_TIMEOUT_MS=8000
```

The API URL stays server-side. Interactive browser features such as live player search and the comparison workbench
call a same-origin Next.js proxy, which forwards requests to FastAPI.

Configuration:

```text
NORTHSTAR_PROFILE_PATH
    Optional path to the enriched hitter-profile CSV.

NORTHSTAR_CORS_ORIGINS
    Comma-separated allowed browser origins.
    Defaults to localhost:3000 and 127.0.0.1:3000.
```

## Production

Production hardening now includes strict PostgreSQL-backed runtime configuration, separate liveness and readiness
checks, bounded upstream request timeouts, baseline security headers, a non-root FastAPI container image, automated
dependency updates, and CI release gates for Python, frontend, and container builds.

See `docs/production.md` for the deployment order, environment variables, database-load procedure, release gate,
and post-deploy smoke test.

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
├── web/                  # Next.js product interface
├── src/northstar/
│   ├── data/
│   ├── features/
│   ├── metrics/
│   └── models/
└── tests/
```

## Metric development principle

NorthStar does **not** currently define a composite CCA score. Candidate player metrics must survive
out-of-time validation, construct reconstruction, reliability checks, and direct incremental-prediction tests
before any standalone claim is made. The first research cycle showed why: useful contextual diagnostics can be
highly repeatable without adding meaningful information beyond stronger established measures.

The project therefore keeps Selection, Contact, and Damage separate and presents them as explanatory context
rather than reverse-engineering a single ranking statistic.

## Data sources

- MLB Baseball Savant / Statcast
- pybaseball as a Python retrieval interface
- Retrosheet is planned for historical validation and complementary play-by-play coverage

See `docs/methodology.md` and `docs/roadmap.md`.

