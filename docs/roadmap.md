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
- [ ] Time-based train/validation/test split
- [ ] Logistic-regression scoring model
- [ ] Calibration curve
- [ ] Brier score / log loss / ROC AUC
- [ ] Leakage audit

## M3 — Context model
- [ ] Rolling batter features
- [ ] Rolling pitcher features
- [ ] Park context
- [ ] XGBoost model
- [ ] SHAP explanations
- [ ] Model-card documentation

## M4 — Canada layer
- [ ] Canadian-born player identity mapping
- [ ] Toronto Blue Jays view
- [ ] Canadian player leaderboard
- [ ] Player comparison report

## M5 — Product
- [ ] FastAPI service
- [ ] PostgreSQL results store
- [ ] Next.js interface
- [ ] Player pages
- [ ] Interactive visualizations

## M6 — Production engineering
- [ ] Docker
- [ ] GitHub Actions
- [ ] Scheduled ingestion
- [ ] Model versioning
- [ ] Datadog tracing/logging/custom metrics
