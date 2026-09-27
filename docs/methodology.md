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

Candidate pre-PA features:
- base state
- outs
- inning
- score differential
- batter handedness
- pitcher handedness
- batter rolling quality estimates
- pitcher rolling quality estimates
- home/away
- park

No post-PA features (event result, exit velocity from the current PA, post-play base state, etc.) may be used
in a pre-PA prediction model.

## Phase 3: NorthStar Chance Creation Added (CCA)

CCA will only be formally defined after the probability model is calibrated and validated out-of-time.
The metric should answer how much a player's plate appearances alter future scoring opportunity relative to
league/context expectation while avoiding outcome leakage.
