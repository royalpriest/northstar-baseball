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

## Phase 2j: external-season transport validation

After cross-fitted propensity estimation and weight-cap sensitivity remain stable inside 2025, the next
validation unit is an entirely different season.

### Design

Training:
- complete 2024 regular season

External evaluation:
- complete 2025 regular season

No 2025 observation contributes to:
- the RE24 table used for the pitch-state target
- decision-state values
- propensity-model fitting
- propensity calibration
- common-support training selection
- inverse-propensity training weights
- swing/take value-model fitting

The 2024 propensity model itself is cross-fitted by game to construct OOF training weights. Its isotonic
calibrator is fit only on 2024 OOF predictions. A final propensity model is fit on all 2024 pitches and
transported directly to 2025 before applying that 2024 calibrator.

### State-value transport

2025 pitch transitions are valued with 2024 run expectancy and 2024 count + base/out decision-state values.
Exact state coverage is reported. Any unseen decision state falls back to the 2024 base/out value and then to
the 2024 overall value.

### Action-value transport

On the calibrated 2024 5%-95% overlap sample, stabilized inverse-propensity weights are computed. Separate
swing/take XGBoost transition-value models are fit at caps 3, 5, 10, and 20.

The untouched 2025 season is evaluated with:
- propensity ROC AUC / log loss / Brier score and calibration bins
- exact decision-state coverage
- overlap retention
- factual transition-value MAE / RMSE / R² / Pearson / Spearman
- model-selected swing share
- doubly robust one-step model-action advantage
- game-cluster bootstrap uncertainty
- weight-cap sensitivity

A successful transport result should preserve the broad direction and useful signal seen in 2025 internal
holdouts without depending on a single cap. A failure is also informative: it would indicate temporal drift or
that the learned relationships were more season-specific than internal validation suggested.

## Phase 2k: player decision-value prototype

The 2024-trained pitch-state system transported to 2025 with stable propensity discrimination, transition-value
signal, common support, and weight-cap sensitivity. NorthStar therefore begins player aggregation, while
keeping the metric explicitly provisional.

For opportunity `X`, let:

- `e(X)` be the transported/calibrated probability of swinging
- `Q_swing(X)` be modeled one-step transition value if swinging
- `Q_take(X)` be modeled one-step transition value if taking

The opportunity-conditioned expected action value is:

`Q_expected(X) = e(X) Q_swing(X) + (1 - e(X)) Q_take(X)`

For the hitter's observed action `A`:

`DV(X,A) = Q_A(X) - Q_expected(X)`

The prototype also records:

`Missed(X,A) = max(Q_swing, Q_take) - Q_A`

and:

`Headroom(X) = max(Q_swing, Q_take) - Q_expected`

so:

`DV + Missed = Headroom`

### Player aggregation

For each batter NorthStar reports:
- eligible pitch decisions
- discretionary decisions with `0.20 <= e(X) <= 0.80`
- swing rate and expected swing rate
- total decision value
- decision value per 100 eligible pitches
- missed value per 100 eligible pitches
- ordinary optimal-choice rate
- value-weighted optimal-choice rate using `|Q_swing - Q_take|` as the weight
- discretionary decision value per 100
- discretionary optimal-choice rates

The value-weighted rate emphasizes choices where the model sees a materially larger consequence difference
between swinging and taking.

### Reliability gate

The full 2025 external season is split into early and late samples. Players must meet a minimum pitch-decision
threshold in both halves. Pearson and Spearman reliability are reported for rate metrics rather than totals,
including decision value per 100, missed value per 100, optimal-choice rate, value-weighted optimal-choice
rate, and discretionary equivalents.

A leaderboard can be generated for inspection, but NorthStar should not assign a final public metric name
unless player-level reliability is materially stronger than the short September split-half results and the
rankings remain interpretable under discretionary-opportunity filtering.

## Phase 2l: player construct validation and ranking shrinkage

Full-season player split-half reliability is materially stronger than short-horizon September reliability, so
NorthStar next tests **construct validity** rather than immediately naming the statistic.

### Independent plate-discipline constructs

Full-season 2025 Statcast pitches are independently aggregated by batter. Physical swing/take descriptions use
the same transparent event definitions as the pitch-decision pipeline, but no NorthStar Q-value is involved in
these summaries.

The construct set includes:
- overall swing rate
- zone swing rate
- chase rate on Statcast zones 11-14
- whiffs per swing
- contact rate
- first-pitch swing rate
- two-strike swing rate
- two-strike chase rate

Existing player RE24 / Baseball Savant outputs add outcome-oriented references such as RE24 per 100 PA, wOBA,
and xwOBA.

Pairwise Pearson and Spearman correlations are reported for primary NorthStar player metrics. In addition,
five-fold player-level out-of-fold Ridge models estimate how much of NorthStar decision value can be explained
by:
- plate discipline only
- outcome value only
- their combination

The purpose is diagnostic. Very high reconstruction would suggest NorthStar is mostly a repackaging of simpler
statistics; modest reconstruction alongside stable reliability would support incremental descriptive content.

### League centering

Raw decision value is measured relative to an opportunity-conditioned action mixture and is not mathematically
forced to average zero across player means. A presentation layer therefore centers the player rate on the
pitch-weighted league mean:

`Centered DV/100 = 100 * (player mean DV - league mean DV)`

This changes interpretation, not underlying player ordering.

### Sample-aware shrinkage

For each hitter, NorthStar estimates the standard error of mean decision value with game clusters. Let
`sigma_i²` be the squared cluster standard error for player `i`. A method-of-moments estimate of
between-player variance `tau²` is:

`tau² = max(Var(player means) - mean(sigma_i²), epsilon)`

The provisional shrinkage factor is:

`lambda_i = tau² / (tau² + sigma_i²)`

and:

`Shrunk_i = league mean + lambda_i * (raw_i - league mean)`

This prevents short or noisy samples from receiving the same ranking certainty as large stable samples.
Default eligibility requires at least 500 total decisions and 300 discretionary opportunities.

This is an empirical-Bayes-style approximation for ranking stabilization. It does not remove confounding from
the underlying observational action-value model.

## Phase 2m: incremental player-signal audit

Construct validation indicates that raw NorthStar decision value is largely reconstructable from simple
plate-discipline summaries. The next question is therefore not whether the raw metric is reliable, but whether
its **residual beyond ordinary plate discipline** is reliable.

Three construct models are tested:
- chase rate only
- overall swing rate + chase rate + two-strike chase rate
- the complete plate-discipline construct set

For full-season diagnostics, player-level Ridge models are cross-fitted and the out-of-fold residual is:

`Residual_i = NorthStar DV100_i - Predicted DV100_i(constructs)`

For temporal reliability, the season is split at July 1. Early-half residuals are produced out-of-fold. A model
fit on all early-half players is then transported to late-half plate-discipline summaries, generating late
residuals without learning the late relationship.

The key diagnostic is the early/late Pearson and Spearman correlation of these residuals. This is stricter than
correlating raw decision value because it asks whether a player-specific component remains after accounting for
the simpler behaviors that already explain most of NorthStar's cross-sectional variation.

Residuals are also compared with RE24, wOBA, and xwOBA. These outcome correlations are descriptive and do not
convert the observational decision-value model into a causal statistic.

## Phase 2n: interpretation of the incremental-signal result

The incremental audit provides a clear stopping rule for the first player prototype.

Raw decision value per 100 is strongly reliable across season halves, but that reliability is mostly reproduced
by ordinary swing-selection behavior. Chase rate alone explains a large share of the cross-sectional signal
and leaves a moderately stable residual. Once overall swing tendency and two-strike chase behavior are added,
the residual early/late reliability falls below 0.20. The full plate-discipline residual is only slightly
higher, around 0.21.

NorthStar therefore interprets the current quantity as a **context-weighted plate-discipline / selection-value
component**, not as evidence for a new independent all-purpose hitter metric.

This distinction is intentional:

- the pitch-level transition-value and policy machinery remains useful
- the player aggregation is repeatable
- the player aggregation is not sufficiently distinct from established discipline behavior to justify a new
  standalone name
- a weak residual should not be amplified or rebranded simply to manufacture novelty

The next player-level research question is a decomposition:

1. **Selection value** — was swing/take the better decision given the pitch/context?
2. **Execution value** — conditional on choosing to swing, did the hitter convert that opportunity better or
   worse than expected from pitch/context?

Only after both components are validated separately should NorthStar test whether a combined offensive
decision profile contains stable, incremental information.

## Phase 2o: selection / execution decomposition

The first player decision-value prototype is reliable but largely reconstructable from ordinary plate-discipline
statistics. Rather than force a new aggregate metric, NorthStar decomposes hitter behavior into separate
selection and execution components.

### Selection

Selection remains the validated context-weighted swing/take component. It answers whether the observed action
was favorable relative to the opportunity-conditioned league action mixture.

Because construct validation showed that this component is largely plate discipline, it is retained as a
descriptive selection axis rather than promoted as a novel standalone statistic.

### Execution

For execution, NorthStar conditions on **observed swings only**.

A 2024 league model estimates:

`E[one-step transition value | swing, pitch/context]`

using the same leakage-safe pitch features as the decision system:

- count
- pitch type
- hitter/pitcher handedness
- exact hitter-oriented location
- normalized vertical location
- pitch velocity and movement
- base/out state
- inning / half-inning
- score differential

The model does **not** use:
- batter identity
- realized pitch description
- exit velocity
- launch angle
- batted-ball type
- realized event
- any other post-swing outcome variable

For every clean 2025 swing:

`Execution = realized one-step transition value - expected swing transition value`

The one-step target is defined with 2024 run expectancy and 2024 decision-state values, preserving the same
external-season transport discipline as the earlier validation work.

### Player reliability

Execution is aggregated as value per 100 swings and split at July 1. Early/late Pearson and Spearman
correlations test whether hitters persistently outperform or underperform league expectation after controlling
for the pitch/context they chose to swing at.

The workflow also reports actual swing value and expected swing value separately. This helps distinguish a
stable execution trait from stable differences in the quality of swing opportunities a player receives.

### Two-axis profile

NorthStar joins the existing selection component with execution as two separate coordinates. No combined score
is created at this stage. A combined statistic will only be considered if execution shows meaningful
out-of-time and player-level reliability and later survives construct validation against xwOBA/contact quality.

## Phase 2p: interpretation of swing execution V1

The external 2024-to-2025 swing-execution model improves on the empirical count/zone/pitch-type baseline, but
realized one-step swing value is too noisy to support a stable player execution axis.

The key distinction is between **opportunity stability** and **execution stability**:

- expected swing opportunity value is moderately repeatable across season halves
- actual swing value is weakly repeatable
- actual-minus-expected execution value remains only weakly repeatable

This indicates that the all-in-one target combines several mechanisms with different noise properties:

1. swing-and-miss versus contact
2. foul contact versus ball in play
3. quality of batted-ball contact
4. fielding / baserunning / subsequent state effects embedded in one-step run value

NorthStar therefore does not discard execution. Instead, V2 decomposes it into two cleaner conditional tasks:

### Contact execution

Among observed swings:

`Contact residual = I(contact) - P(contact | swing, pitch/context)`

This asks whether a hitter makes contact more often than a 2024 league model expects on the pitches they
actually swing at.

### Damage execution

Among balls put into play with a valid Statcast contact-quality target:

`Damage residual = observed contact-quality target - E[target | BIP, pitch/context]`

The initial target is Statcast estimated wOBA using launch speed/angle. Exit velocity, launch angle, and the
estimated-wOBA target are **outcomes only** and are never model features.

Each component must pass external-season model checks and player split-half reliability independently before
NorthStar considers retaining it as a player axis.

## Phase 2q: contact / damage execution V2

Execution V2 replaces the single realized run-value residual with two conditional tasks that correspond to
different hitter skills.

### Contact model

Sample:
- observed physical swings only

Target:
- 1 for foul/foul-tip/ball-in-play contact
- 0 for swinging strikes, blocked swinging strikes, swinging pitchouts, and missed bunts

The model uses the existing leakage-safe pitch-opportunity feature set and does not receive the realized pitch
description. A 2024 game-grouped out-of-fold contact model supplies probabilities for isotonic calibration.
The final 2024 contact model is transported to 2025 and calibrated using the 2024 OOF calibrator.

Per swing:

`Contact residual = I(contact) - calibrated P(contact | swing, X)`

Player contact execution is the mean residual times 100, interpreted as contact percentage points above or
below league expectation for the pitches the hitter actually swung at.

### Damage model

Sample:
- balls put into play
- valid Statcast `estimated_woba_using_speedangle`

Target:
- Statcast xwOBA contact quality derived from the realized launch speed / angle

Features:
- the same pitch/context information available on the incoming pitch
- no batter identity
- no exit velocity
- no launch angle
- no batted-ball type
- no realized event
- no target-derived feature

A 2024 regression model is transported to 2025:

`Damage residual = observed xwOBA contact quality - predicted xwOBA contact quality`

Player damage execution is expressed in xwOBA points:

`1000 * mean(Damage residual)`

### Reliability

Both components are split at July 1, 2025 and evaluated independently. Contact requires at least 150 swings in
each half by default; damage requires at least 50 target-eligible batted balls per half.

The decomposition is retained only if the residual player quantities are meaningfully more repeatable than the
all-in-one execution V1 residual. A strong expected-value reliability with weak residual reliability is treated
as opportunity stability rather than player skill.

## Phase 2r: interpretation of contact / damage execution V2

The V2 decomposition succeeds where the all-in-one execution residual did not.

### Contact result

The 2024-trained calibrated contact model transports cleanly to 2025 and produces a highly repeatable player
residual. Contact above expected per 100 swings has stronger early/late reliability than raw observed contact
rate, while expected-contact opportunity itself is also stable.

This supports the interpretation that the adjustment is not merely injecting noise into ordinary contact rate.
However, high reliability alone does not establish incremental value; the next phase explicitly asks whether
zone contact, chase contact, two-strike contact, and whiff/contact summaries can reconstruct the residual.

### Damage result

The pitch-level damage model has low absolute R², which is expected because launch speed and angle are highly
dependent on hitter execution and are intentionally withheld from the feature set. Despite weak individual-
batted-ball prediction, the 2025 player residual aggregates to substantially stronger split-half reliability
than execution V1.

The next phase compares damage above expected with conventional quality-of-contact constructs such as average
exit velocity, hard-hit rate, sweet-spot rate, and raw xwOBA on contact. A useful NorthStar damage axis should
retain repeatable information after those simpler constructs explain what they can.

### Three-axis interpretation

Selection, contact execution, and damage execution are deliberately kept separate. Their observed pairwise
relationships are modest rather than near-duplicates, including a negative contact/damage association that is
consistent with different hitter approaches.

No combined score is justified yet. The project proceeds to construct and incremental-signal validation for
contact and damage independently.

## Phase 2s: contact / damage construct validation

Reliability is necessary but not sufficient for a new player dimension. V2 therefore applies construct and
incremental-signal validation independently to contact and damage.

### Contact constructs

Conventional contact skill is summarized from observed swings:
- overall contact rate
- zone contact rate
- chase contact rate
- two-strike contact rate
- whiffs per swing

Two nested feature sets are used:

`raw_contact = [overall contact rate]`

`situational_contact = [overall, zone, chase, two-strike contact]`

Five-fold player-level OOF Ridge prediction measures how much of NorthStar contact-above-expected can be
reconstructed cross-sectionally.

For temporal incremental validity, early-half residuals are OOF. The early-half construct relationship is then
fit on all eligible early players and transported to late-half constructs. Early/late residual reliability asks
whether the same hitters continue to outperform conventional contact expectation.

### Damage constructs

Conventional quality of contact is summarized from balls in play:
- xwOBA on contact
- average exit velocity
- hard-hit rate at 95+ mph
- Statcast barrel rate using launch-speed/angle category 6
- sweet-spot rate using launch angles 8° through 32°

Three feature sets are tested:

`raw_xwoba_contact`

`batted_ball_quality = [average EV, hard-hit, barrel, sweet-spot]`

`full_contact_quality = [xwOBA contact + batted-ball-quality features]`

The same OOF reconstruction and early-to-late residual reliability design is applied.

### Interpretation

A high raw split-half correlation that collapses after conventional constructs are removed indicates a useful
context-adjusted presentation of an established skill, not a distinct hitter dimension.

A residual that remains meaningfully repeatable after construct removal provides stronger evidence that the
NorthStar context adjustment preserves player information not captured by the simpler summary statistics.

Associations with RE24, wOBA, and xwOBA are reported only as descriptive external checks and are not used to
retrofit the component toward offensive outcomes.

## Phase 2t: future-skill incremental audit

The first construct audit revealed an important identification problem.

Both V2 components are defined as an observed hitter result minus a model-based expected opportunity result.
Residualizing such a metric against the observed conventional result can mechanically leave part of the
expected-opportunity term. If opportunity quality is itself repeatable by hitter, early/late residual
correlation can remain high even without a new execution skill.

NorthStar therefore changes the incremental question from:

`Is (NorthStar - conventional reconstruction) repeatable?`

to:

`Does EARLY NorthStar information improve prediction of LATE observed skill beyond EARLY conventional skill?`

### Contact future-prediction test

Target:
- late-half observed contact rate

Base predictors:
- early overall contact rate
- optionally early zone contact, chase contact, and two-strike contact

Augmented predictor:
- the same conventional set
- plus early contact above expected per 100 swings

### Damage future-prediction test

Target:
- late-half observed xwOBA on contact

Base predictors:
- early xwOBA on contact
- optionally early average exit velocity, hard-hit rate, barrel rate, and sweet-spot rate

Augmented predictor:
- the same conventional set
- plus early damage xwOBA points above expected

For every construct set, both models use the identical complete-case player sample and identical K-fold player
splits. Standardized Ridge models generate out-of-fold late-skill predictions.

The main statistic is:

`Delta OOF R² = R²(augmented) - R²(conventional only)`

Changes in Pearson, Spearman, MAE, and RMSE are reported alongside it.

This design does not prove causality. It asks a narrower and cleaner question: whether the context-adjusted
NorthStar component contains player information that helps predict future observed skill after the conventional
early-season skill measures are already known.

## Phase 3: NorthStar Chance Creation Added (CCA)

CCA will only be formally defined after the probability model is calibrated and validated out-of-time.
The metric should answer how much a player's plate appearances alter future scoring opportunity relative to
league/context expectation while avoiding outcome leakage.
