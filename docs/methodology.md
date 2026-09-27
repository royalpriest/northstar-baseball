# Methodology

## Phase 1: establish a trustworthy baseline

NorthStar starts with the traditional 24 base/out states: 8 runner configurations × 3 out counts.
For each plate appearance, the pipeline calculates the number of runs the batting team scores from the
start of that plate appearance until the end of the half-inning. Averaging that target within each state
produces an empirical run-expectancy table.

### Plate-appearance boundary rule

Statcast is pitch-level. The base/out state on the final pitch is not always the state that existed when the
plate appearance began. A steal, wild pitch, passed ball, balk, or other runner movement can occur during an
at-bat. NorthStar therefore uses the first pitch for the pre-PA state and the last pitch for the PA outcome.

### RE24 baseline

For each plate appearance:

`RE24 = runs scored on PA + RE(after state) - RE(before state)`

This is not the project's novel metric. It is the benchmark NorthStar must reproduce before introducing a
machine-learned chance-creation model.

## Phase 2: context-aware scoring probability

Target:

`P(at least one run scores before the half-inning ends)`

Initial model: logistic regression.

Version 1 deliberately begins with a small leakage-safe feature set available in the current plate-appearance
dataset:
- base state
- outs
- inning
- top/bottom of inning

The 2025 season is split chronologically: training before August 1, validation during August, and a final
September holdout. The model is compared with a 24-state empirical scoring-probability benchmark that is fit
on the training partition only.

Evaluation metrics:
- ROC AUC for ranking discrimination
- log loss for probability quality
- Brier score for probabilistic accuracy
- calibration curve on the September holdout

Later context models will add batter/pitcher quality, handedness, park, score differential, and other
pre-PA information. No post-PA features (event result, current-PA exit velocity, post-play base state, etc.)
may be used in a pre-PA prediction model.

## Phase 2b: context model V2

V2 asks whether player and matchup context adds predictive information beyond the 24-state baseline and the
minimal logistic V1 model.

Additional pre-PA fields:
- batter handedness
- pitcher handedness
- batting-team score differential
- home/away batting indicator
- home team as a park proxy
- batter prior RE24 per PA and positive-RE24 rate
- pitcher prior RE24 allowed per batter faced and positive-RE24-allowed rate
- log-transformed prior opportunity counts for both batter and pitcher

Player-history features are calculated at the date level and only use dates strictly before the current game.
This intentionally gives up same-day updates in exchange for a clear no-leakage guarantee when exact game-time
ordering is unavailable.

Models compared:
- 24-state empirical scoring probability
- logistic V1
- logistic context V2
- XGBoost context V2

The same August validation / September test design is retained so V2 can be compared directly with V1.

## Phase 2c: attribution audit and expected terminal-event RE24

The scoring-probability V2 experiment showed that base/out state overwhelmingly drives the target
`P(at least one run scores before the half-inning ends)`. Richer player context improved ROC AUC only
slightly and did not improve holdout log loss or Brier score over the empirical 24-state baseline.

NorthStar therefore changes the next target to the run value of the current PA-ending event.

### Attribution refinement

Traditional whole-PA RE24 is still retained as a benchmark:

`RE24 = runs during PA + RE(next PA state) - RE(first-pitch state)`

A second quantity begins immediately before the terminal PA-ending pitch:

`terminal-event RE24 = runs on terminal event + RE(next PA state) - RE(final-pitch pre-state)`

Their difference is:

`within-PA state RE24 = whole-PA RE24 - terminal-event RE24`

This difference measures run-expectancy movement that happened before the terminal pitch. It is useful for
auditing steals, wild pitches, balks, caught stealing, and other within-PA state changes that should not be
automatically credited to the batter's terminal event.

Terminal-event RE24 is a stricter attribution target, not a claim of perfect causal batter value. The terminal
play itself can still include baserunning, fielding, and defensive effects.

### Expected run-value model

Target:

`terminal_event_re24`

Pre-PA features:
- starting base state and outs
- inning and top/bottom
- score differential
- home/away indicator and home-team park proxy
- batter/pitcher handedness
- batter prior-date terminal-event RE24 rate
- pitcher prior-date terminal-event RE24 allowed rate
- positive-value rates and prior-opportunity counts

Models:
- training-only base/out-state mean
- Ridge regression
- XGBoost regression

Evaluation:
- MAE
- RMSE
- R²
- Pearson correlation
- Spearman rank correlation
- binned predicted-vs-observed holdout plot

The first model fit excludes PAs with non-zero `within_pa_state_re24` by default. Those observations are still
preserved in the attribution audit, but their terminal event begins from a state that was not known at the
start of the PA. A sensitivity run may include them explicitly.

The same chronological August validation / September holdout is retained. Player-level residual analysis is
performed only after the continuous target model is evaluated out-of-time.

## Phase 2d: structured plate-appearance outcome model

The direct expected-terminal-event-RE24 experiment produced essentially zero out-of-time R². NorthStar
therefore decomposes the target rather than increasing regressor complexity.

Target classes:
- strikeout
- walk / HBP
- single
- double / triple
- home run
- field out
- sacrifice
- reach / other terminal event

Pre-PA features retain the existing game/matchup context and add prior-date batter and pitcher rates for each
outcome family. These expanding rates exclude the current calendar day.

Models:
- training class-frequency prior
- multinomial logistic regression
- XGBoost multiclass classifier

Classification evaluation:
- multiclass log loss
- accuracy
- macro F1
- top-2 accuracy
- normalized confusion matrix

A second evaluation translates predicted class probabilities into run value. On the training partition only,
NorthStar estimates mean terminal-event RE24 for each starting base/out state × outcome-class pair. For each
future PA:

`expected terminal-event RE24 = Σ P(outcome class) × training mean RE24(state, class)`

Sparse state/class combinations fall back to the training-wide mean for that outcome class, then to the
overall training mean if required.

This structured value estimate is compared out-of-time using MAE, RMSE, R², Pearson correlation, and Spearman
correlation. Residuals remain exploratory and are not called CCA unless the model proves stable and useful.

## Phase 2e: pitch opportunity and batter decision model

The PA-outcome model learned modest out-of-time event probability signal, but translating that distribution
into exact terminal-event RE24 still produced negligible R². NorthStar therefore moves to the individual pitch,
where substantially more information about the actual opportunity is observable.

### Decision target

Each physical Statcast pitch is labeled as `swing` or `take` from the pitch description. Automatic
ball/strike clock violations are excluded because they are count changes without a physical pitch.

The target is the hitter's decision, not the terminal PA result.

### Opportunity features

V1 uses:
- pre-pitch balls and strikes
- pitch type
- release speed
- horizontal and vertical pitch movement
- plate location
- plate height normalized to the hitter-specific strike zone
- hitter-perspective horizontal location/movement
- batter and pitcher handedness
- base state and outs
- inning / top-bottom
- score differential and home batting indicator

Exact pitch location is permissible here because the model conditions on the pitch opportunity as presented.
Current-pitch contact quality, batted-ball result, and PA terminal event are prohibited.

### Models and evaluation

Models:
- training-only count + Statcast-zone empirical swing rate
- logistic regression
- XGBoost binary classifier

The same chronological design is retained:
- train before August 1
- validate in August
- final test in September

Evaluation:
- ROC AUC
- log loss
- Brier score
- calibration curve

The out-of-time residual `actual swing - expected swing probability` is stored as **decision surprise**. It
measures unusual aggressiveness/passivity relative to the opportunity, not quality. A later counterfactual
value model must estimate swing and take consequences before any decision-value metric is defined.

## Phase 2f: observational pitch decision value

Pitch-decision V1 produced the first strong out-of-time predictive result in NorthStar. The next step separates
**behavior prediction** from **decision quality**.

### Return target

For each pitch, a training-only run-expectancy table is used to construct a realized return from the current
pitch state through PA completion:

`Y = runs remaining in PA + RE(after PA) - RE(current pitch base/out state)`

This is a contextual action-value return: the consequence includes the rest of the PA after the current
swing/take choice rather than only the terminal pitch result.

### Propensity and overlap

The validated XGBoost swing/take model estimates:

`e(X) = P(swing | observed pitch opportunity X)`

V1 restricts the action-value analysis to common support, by default:

`0.05 <= e(X) <= 0.95`

Observed actions receive stabilized inverse-propensity weights before fitting action-specific value models.

### Action-value models

Two XGBoost regressors estimate:

- `Q_take(X) = E[Y | X, take]`
- `Q_swing(X) = E[Y | X, swing]`

The descriptive action gap is:

`chosen-action advantage = Q(chosen | X) - Q(alternative | X)`

The model-selected policy chooses the larger of `Q_take` and `Q_swing`.

### Evaluation

Out-of-time diagnostics include:
- factual MAE / RMSE / R² / correlations for observed-action value prediction
- overlap share
- doubly robust policy-value estimates for always take, always swing, and model-selected action
- policy swing share
- player-level September summaries

These are observational estimates, not randomized causal effects. Unmeasured pitch recognition, hitter intent,
and model misspecification can bias both propensity and outcome models. V1 also does not cross-fit nuisance
models. No causal player-value claim is made at this stage.

## Phase 2g: telescoping pitch-state value

Decision-value V1 used the rest-of-PA return as the outcome for every pitch. Although predictive signal was
measurable, that formulation duplicates downstream value across sequential decisions.

V2 introduces a training-only decision-state value:

`V(s) = E[runs remaining in PA + RE(after PA) | balls, strikes, base/out state]`

For an observed transition between recognized physical decision opportunities:

`ΔV = runs during step + V(s_next) - V(s_current)`

For the terminal decision of a PA, `V(s_next)` is replaced by the training-only run expectancy of the
post-PA base/out state.

This target telescopes within a PA. Summing all observed transition values gives total PA runs plus terminal
RE minus the first state value, eliminating the repeated downstream-return problem from V1.

### Clean one-step sample

If the next recognized decision opportunity has a pitch number more than one greater than the current pitch,
the transition may span an automatic count change, excluded pitch, or other intervening event. V2 retains such
rows for the telescoping audit but excludes them from swing/take counterfactual model training.

### Action-value experiment

On the clean one-step sample, V2 retains:
- the validated XGBoost swing propensity model
- common-support trimming
- stabilized inverse-propensity weighting
- separate XGBoost models for swing and take transition value
- a count + zone + action empirical baseline
- doubly robust one-step action estimates

The doubly robust numbers average immediate transition value over the **observed** state distribution. They are
not full dynamic policy values because an alternative action can change future state occupancy. A future
sequential model would need explicit transition dynamics or fitted policy evaluation.

## Phase 2h: pitch-state robustness audit

Pitch-state V2 is not promoted directly to a player metric. Its saved September holdout predictions are first
subjected to sensitivity analysis.

### Overlap sensitivity

The primary V2 analysis uses:

`0.05 <= P(swing | X) <= 0.95`

The robustness audit reevaluates the same saved out-of-time predictions under:
- 5%-95%
- 10%-90%
- 20%-80%

These tighter windows reduce reliance on observations where one action was rarely chosen. The action-value
models are not refit in this first audit; the goal is to test whether the holdout conclusion is sensitive to
common-support restrictions.

### Clustered uncertainty

Pitch outcomes within the same game are not independent. The audit therefore computes the model-action minus
observed-action doubly robust score at the pitch level, then resamples whole games with replacement. The
reported 95% interval is a game-cluster bootstrap conditional on the already-fitted propensity and outcome
models.

### Subgroup checks

The saved holdout is summarized by:
- balls-strikes count
- Statcast zone

For sufficiently large subgroups, NorthStar records observed swing share, model-selected swing share, observed
transition value, doubly robust model-action value, action advantage, modeled Q-gap magnitude, and agreement
with the model-selected action.

### Player stability

September is split at September 15. Batters with enough decisions in both halves are compared on:
- mean modeled chosen-action advantage
- optimal-choice rate

Pearson and Spearman correlations across the two halves are reported. This is a short-horizon reliability
diagnostic, not proof of a stable season-level skill.

## Phase 2i: cross-fitted nuisance models and weight sensitivity

The robustness audit established that pitch-state V2 remains positive under tighter propensity overlap. The
next concern is nuisance-model overfitting inside the pre-August training sample.

### Game-grouped cross-fitting

The training period is partitioned with stratified group folds using `game_pk` as the group. Every pitch from
one game remains in the same fold. For each fold, the swing-propensity model is trained on the other games and
predicts only the held-out games.

The combined predictions form:

`e_oof(X) = out-of-fold P(swing | X)`

These OOF probabilities, rather than in-sample predictions, determine training common support and stabilized
inverse-propensity weights.

### Propensity calibration

An isotonic calibrator is fit on the OOF training probabilities and observed swing decisions. A final XGBoost
propensity model is then trained on the complete pre-August sample. Its August/September probabilities are
passed through the OOF-fitted calibrator before overlap trimming or doubly robust evaluation.

Calibration is evaluated with ROC AUC, log loss, Brier score, average predicted swing probability, observed
swing rate, and holdout calibration bins.

### Weight-cap sensitivity

On the calibrated OOF 5%-95% training sample, raw stabilized inverse-propensity weights are computed once.
Separate swing/take XGBoost transition-value models are then refit under caps:

- 3
- 5
- 10
- 20

For every cap NorthStar records:
- percentage of weights clipped
- maximum raw / clipped weight
- total and action-specific effective sample size
- August and September factual transition-value metrics
- one-step model-action advantage
- game-cluster bootstrap standard error and 95% interval
- model-selected swing share and action agreement

A credible result should not depend on one arbitrary weight cap. Large shifts in sign or magnitude, or severe
effective-sample-size collapse, would block progression toward a player metric.

## Phase 3: NorthStar Chance Creation Added (CCA)

CCA will only be formally defined after the probability model is calibrated and validated out-of-time.
The metric should answer how much a player's plate appearances alter future scoring opportunity relative to
league/context expectation while avoiding outcome leakage.
