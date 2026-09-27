# Roadmap

## M0 — Foundation
- [x] Package layout
- [x] Statcast chunked ingestion
- [x] Plate-appearance reconstruction
- [x] Empirical RE24 table
- [x] Scoring-probability table
- [x] Unit tests for state reconstruction and RE24

## M1 — 2025 baseline dataset
- [ ] Download full 2025 regular season
- [ ] Data-quality report
- [ ] Validate 24 base/out states
- [ ] Compare RE24 values with established public references
- [ ] Produce league RE24 visualization

## M2 — First ML model
- [x] Time-based train/validation/test split implemented
- [x] Logistic-regression scoring model implemented
- [x] Calibration curve output implemented
- [x] Brier score / log loss / ROC AUC reporting implemented
- [x] Initial leakage-safe feature audit implemented
- [ ] Run the full 2025 model and preserve holdout results

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
- [ ] Run September holdout evaluation and preserve results
- [ ] Analyze per-class errors and confusion
- [ ] Decide whether outcome-based expected value is strong enough for player residual analysis
- [ ] SHAP explanations
- [ ] Model-card documentation

## M6 — Canada layer
- [ ] Canadian-born player identity mapping
- [ ] Toronto Blue Jays view
- [ ] Canadian player leaderboard
- [ ] Player comparison report

## M7 — Product
- [ ] FastAPI service
- [ ] PostgreSQL results store
- [ ] Next.js interface
- [ ] Player pages
- [ ] Interactive visualizations

## M8 — Production engineering
- [ ] Docker
- [ ] GitHub Actions
- [ ] Scheduled ingestion
- [ ] Model versioning
- [ ] Datadog tracing/logging/custom metrics
