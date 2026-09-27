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

## Phase 3: NorthStar Chance Creation Added (CCA)

CCA will only be formally defined after the probability model is calibrated and validated out-of-time.
The metric should answer how much a player's plate appearances alter future scoring opportunity relative to
league/context expectation while avoiding outcome leakage.
