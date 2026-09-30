# Roadmap

## M0 — Foundation
- [x] Package layout
- [x] Statcast chunked ingestion
- [x] Plate-appearance reconstruction
- [x] Empirical RE24 table
- [x] Scoring-probability table
- [x] Unit tests for state reconstruction and RE24

## M1 — 2025 baseline dataset
- [x] Download full 2025 regular season
- [x] Data-quality and state-validation workflow
- [x] Validate all 24 base/out states
- [x] Validate player RE24 against established Baseball Savant offensive measures
- [x] Produce league RE24 visualization

## M2 — First ML model
- [x] Time-based train/validation/test split implemented
- [x] Logistic-regression scoring model implemented
- [x] Calibration curve output implemented
- [x] Brier score / log loss / ROC AUC reporting implemented
- [x] Initial leakage-safe feature audit implemented
- [x] Run the full 2025 model and preserve holdout results

## M3 — Context scoring model
- [x] Leakage-safe prior-date batter features implemented
- [x] Leakage-safe prior-date pitcher features implemented
- [x] Handedness, score, home/away, and park-proxy context implemented
- [x] Context logistic-regression model implemented
- [x] XGBoost context model implemented
- [x] Rebuild 2025 PA dataset with retained context fields
- [x] Run V2 holdout evaluation and preserve results
- [x] Document target limitation: base/out state dominates future-inning scoring probability

## M4 — Expected run value / attribution
- [x] Retain final-pitch pre-state for terminal-event attribution
- [x] Compute terminal-event RE24 and within-PA state RE24
- [x] Add RE24 attribution audit script
- [x] Add prior-date terminal-event batter/pitcher context
- [x] Implement Ridge expected-RE24 baseline
- [x] Implement XGBoost expected-RE24 model
- [x] Rebuild 2025 dataset with terminal-event fields
- [x] Run attribution audit and preserve results
- [x] Run expected-RE24 holdout evaluation and preserve negative result
- [x] Reject direct regression as the primary path after near-zero holdout R²

## M5 — PA outcome model
- [x] Define stable terminal-outcome classes
- [x] Add leakage-safe prior-date batter outcome rates
- [x] Add leakage-safe prior-date pitcher outcome rates
- [x] Add class-frequency baseline
- [x] Add multinomial logistic model
- [x] Add XGBoost multiclass model
- [x] Translate outcome probabilities into expected RE24
- [x] Run September holdout evaluation and preserve results
- [x] Document that outcome probabilities improved while implied RE24 remained weak

## M6 — Pitch decision model
- [x] Define physical-pitch swing/take labels
- [x] Build hitter-perspective pitch opportunity features
- [x] Add count+zone empirical swing baseline
- [x] Add logistic pitch-decision model
- [x] Add XGBoost pitch-decision model
- [x] Add out-of-time calibration and batter behavior summary
- [x] Run September holdout evaluation and preserve strong results
- [ ] Audit unrecognized / excluded Statcast pitch descriptions

## M7 — Pitch decision value
- [x] Define training-only pitch-to-PA-end run-value return
- [x] Reuse validated swing propensity model
- [x] Add common-support trimming and stabilized propensity weights
- [x] Add separate swing/take XGBoost value models
- [x] Add count+zone+action value baseline
- [x] Add doubly robust policy-value diagnostics
- [x] Run September holdout evaluation and preserve results
- [x] Document overlapping downstream-return limitation
- [x] Reject direct summation of V1 pitch advantages as a player metric

## M8 — Telescoping pitch-state value
- [x] Fit training-only count + base/out state values
- [x] Build one-step decision-opportunity transition target
- [x] Add within-PA telescoping audit
- [x] Exclude non-terminal pitch-number gaps from counterfactual fitting
- [x] Add propensity-weighted swing/take transition-value models
- [x] Add one-step doubly robust action diagnostics
- [x] Run September holdout evaluation and preserve results
- [x] Verify zero telescoping error and 99.97% clean-transition retention
- [x] Preserve stable validation/test action-estimate ordering

## M9 — Pitch-state robustness
- [x] Add tighter overlap-window sensitivity
- [x] Add game-cluster bootstrap confidence intervals
- [x] Add propensity calibration recheck on saved holdout
- [x] Add count/zone subgroup stability audit
- [x] Add early/late September batter stability audit
- [x] Run robustness audit and preserve results
- [x] Confirm positive advantage under 5%-95%, 10%-90%, and 20%-80% overlap
- [x] Add game-cluster bootstrap uncertainty and player split-half stability
- [x] Decide that cross-fitted nuisance models and weight-clipping refits are necessary before sequential policy evaluation

## M10 — Cross-fitted pitch-state nuisance models
- [x] Add game-grouped OOF propensity estimation
- [x] Add OOF-fitted isotonic propensity calibration
- [x] Add stabilized weight diagnostics and effective sample size
- [x] Add weight-cap refits at 3, 5, 10, and 20
- [x] Add clustered policy-sensitivity reporting by cap
- [x] Run cross-fit evaluation and preserve results
- [x] Confirm stability across caps 3, 5, 10, and 20
- [x] Advance to external-season validation rather than further 2025 tuning

## M11 — External-season validation
- [x] Make baseline builder season-aware
- [x] Add 2024 -> 2025 transport workflow
- [x] Keep RE/state values, propensity calibration, and action models training-season only
- [x] Add external state-coverage and propensity-transport diagnostics
- [x] Add external factual and clustered policy sensitivity across weight caps
- [x] Build the complete 2024 baseline dataset
- [x] Run 2024 -> 2025 external validation and preserve results
- [x] Confirm complete 2024-state coverage of 2025 external decision states
- [x] Confirm stable external signal across weight caps 3, 5, 10, and 20
- [x] Advance to player-metric prototyping while retaining observational caveats

## M12 — Player decision-value prototype
- [x] Define opportunity-conditioned expected action value
- [x] Add chosen-action decision value and missed-value accounting
- [x] Add value-weighted optimal-choice rate
- [x] Add 20%-80% discretionary-opportunity view
- [x] Add full-season player leaderboard workflow
- [x] Add early/late 2025 split-half reliability analysis
- [x] Run the player prototype and preserve results
- [x] Confirm full-season split-half reliability around 0.70 for primary DV/100
- [x] Confirm discretionary DV/100 reliability remains above 0.65 Pearson
- [x] Advance to construct validation without naming CCA

## M13 — Player metric construct validation
- [x] Add independent Statcast plate-discipline summaries
- [x] Join RE24 / wOBA / xwOBA reference outputs
- [x] Add pairwise construct-correlation diagnostics
- [x] Add out-of-fold construct reconstruction tests
- [x] Add league centering
- [x] Add game-cluster sample-aware shrinkage
- [x] Require minimum total and discretionary opportunities
- [x] Run construct validation
- [x] Confirm plate discipline explains about 82% of raw DV/100 variance
- [x] Reject immediate public naming of the raw prototype

## M14 — Incremental player-signal audit
- [x] Add chase-only residualization
- [x] Add swing/chase residualization
- [x] Add full plate-discipline residualization
- [x] Add early-OOF to late-season residual reliability test
- [x] Add residual association checks against RE24 / wOBA / xwOBA
- [x] Run incremental-signal audit
- [x] Confirm full-discipline residual reliability is only about 0.21
- [x] Reject CCA/public standalone naming for the current player prototype
- [x] Retain the validated quantity as a selection / plate-discipline component

## M15 — Selection / execution decomposition
- [x] Define swing execution as actual minus expected one-step swing value
- [x] Exclude batter identity and post-swing/contact leakage from execution features
- [x] Add 2024 -> 2025 external-season swing-execution model
- [x] Add empirical count/zone/pitch-type execution baseline
- [x] Add player execution value per 100 swings
- [x] Add early/late 2025 execution reliability test
- [x] Add a separate selection / execution two-axis profile
- [x] Run execution model and preserve results
- [x] Confirm execution-value reliability is only about 0.25
- [x] Reject all-in-one swing-value residual as a stable standalone execution axis
- [x] Preserve selection/execution V1 as an informative mixed/negative result

## M16 — Contact / damage execution V2
- [x] Define contact execution among observed swings
- [x] Define damage execution among balls in play
- [x] Add game-grouped OOF contact calibration
- [x] Add 2024 -> 2025 external contact expectation model
- [x] Add 2024 -> 2025 external damage expectation model
- [x] Add empirical count/zone/pitch-type baselines
- [x] Add player contact and damage summaries
- [x] Add independent early/late reliability tests
- [x] Add selection / contact / damage profile without combining scores
- [x] Run V2 and preserve results
- [x] Confirm contact residual reliability around 0.82 Pearson
- [x] Confirm damage residual reliability around 0.68 Pearson
- [x] Advance both components to construct validation without combining them

## M17 — Contact / damage construct validation
- [x] Build conventional contact-skill constructs by hitter
- [x] Build conventional quality-of-contact constructs by hitter
- [x] Add raw and situational contact reconstruction tests
- [x] Add xwOBA-on-contact and batted-ball-quality reconstruction tests
- [x] Add early-OOF to late-season residual reliability tests
- [x] Add incremental residual association checks with RE24 / wOBA / xwOBA
- [x] Run construct validation and preserve results
- [x] Identify algebraic residual-risk for observed-minus-expected metrics
- [x] Defer novelty decision to direct future-skill prediction

## M18 — Future-skill incremental audit
- [x] Add conventional-only vs conventional-plus-NorthStar paired OOF comparison
- [x] Predict late observed contact from early contact constructs
- [x] Predict late xwOBA on contact from early contact-quality constructs
- [x] Use identical player samples and folds for paired model comparisons
- [x] Run future-skill audit and preserve results
- [x] Confirm contact adds effectively zero future predictive information beyond conventional contact
- [x] Confirm damage adds value beyond coarse batted-ball summaries but not beyond xwOBA on contact
- [x] Reject standalone novel-stat claims for current selection/contact/damage components
- [x] Retain all three as transparent contextual diagnostics

## M19 — NorthStar hitter profile
- [x] Build canonical player-profile dataset
- [x] Join NorthStar diagnostics with established offensive measures
- [x] Add percentile / league-relative presentation without implying a single overall score
- [x] Add optional per-player profile-card generation
- [x] Preserve sample sizes and three-axis completeness flags
- [x] Document interpretation and limitations for every displayed axis
- [x] Run profile builder and preserve canonical table
- [x] Review profile-card readability with representative Aaron Judge / Luis Arráez profiles
- [x] Preserve representative profile visualizations
- [x] Preserve research provenance from the validation phases

## M20 — Canada layer
- [x] Add MLBAM identity lookup and birthplace normalization
- [x] Define automated Canadian-born mapping without inferring citizenship
- [x] Derive 2025 player-team exposure from plate appearances
- [x] Add Toronto Blue Jays season view with explicit full-season metric scope
- [x] Add Canadian-born leaderboard with named sorting metric
- [x] Add 2-4 player comparison report without an overall score
- [x] Run the Canada layer and preserve derived outputs
- [x] Review identity coverage: 404 / 404 canonical hitters resolved
- [x] Review Blue Jays and Canadian-born outputs for historical scope

## M21 — Product
- [x] Add FastAPI application and health / metadata endpoints
- [x] Add player search and MLBAM detail endpoints
- [x] Add Canadian-born and historical team-scope endpoints
- [x] Add explicit named-metric leaderboard endpoint
- [x] Add 2-4 player comparison endpoint without an overall winner
- [x] Run API test suite and local smoke test
- [x] Add SQLAlchemy PostgreSQL schema and transactional profile loader
- [x] Add normalized historical team-membership storage
- [x] Add SQL-backed API store with CSV fallback
- [x] Run PostgreSQL integration load and confirm API reads 404 profiles from PostgreSQL
- [x] Run PostgreSQL API parity smoke test for player, team, leaderboard, and comparison endpoints
- [x] Add Next.js interface foundation and responsive visual system
- [x] Add searchable player index and individual player pages
- [x] Add Canadian-born and Toronto Blue Jays product views
- [x] Add interactive 2-4 player comparison workbench
- [x] Add Canada V2 birthplace/context presentation
- [x] Add plain-language player interpretation and metric guide
- [ ] Run Next.js build and local browser review
- [ ] Refine interactive visualizations after visual QA
- [x] Run first site-wide consistency, accessibility, and mobile implementation pass
- [ ] Complete browser QA at 375 / 430 / 768 / 1024 / desktop widths

## M22 — Production engineering
- [ ] Docker
- [ ] GitHub Actions
- [ ] Scheduled ingestion
- [ ] Model versioning
- [ ] Datadog tracing/logging/custom metrics
