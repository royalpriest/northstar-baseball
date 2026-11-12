# Experiments and reproduction commands

Every research step in NorthStar, in the order it was run, with the command that reproduces it and the outputs it writes. Run `make baseline` first (see the README); every step below reads that cache.

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

