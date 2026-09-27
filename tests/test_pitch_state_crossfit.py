import numpy as np
import pandas as pd
import pytest

from northstar.validation.crossfit import (
    apply_propensity_calibrator,
    cap_weights,
    effective_sample_size,
    fit_isotonic_calibrator,
    stabilized_raw_weights,
    weight_diagnostics,
)


def test_isotonic_propensity_calibration_is_bounded():
    raw = np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    target = np.array([0, 0, 0, 1, 1, 1])

    calibrator = fit_isotonic_calibrator(raw, target)
    calibrated = apply_propensity_calibrator(calibrator, raw)

    assert np.all(calibrated > 0)
    assert np.all(calibrated < 1)
    assert np.all(np.diff(calibrated) >= 0)


def test_stabilized_weights_and_cap_diagnostics():
    action = np.array([0, 0, 1, 1])
    propensity = np.array([0.10, 0.40, 0.60, 0.90])

    raw = stabilized_raw_weights(action, propensity)
    clipped = cap_weights(raw, 1.0)
    diagnostics = weight_diagnostics(action, raw, 1.0)

    assert raw.tolist() == pytest.approx(
        [
            0.5 / 0.9,
            0.5 / 0.6,
            0.5 / 0.6,
            0.5 / 0.9,
        ]
    )
    assert np.all(clipped <= 1.0)
    assert diagnostics.rows == 4
    assert diagnostics.clipped_rows == 0
    assert diagnostics.effective_sample_size <= 4


def test_effective_sample_size_penalizes_uneven_weights():
    equal = effective_sample_size(np.array([1.0, 1.0, 1.0, 1.0]))
    uneven = effective_sample_size(np.array([0.1, 0.1, 0.1, 4.0]))

    assert equal == pytest.approx(4.0)
    assert uneven < equal


def test_weight_cap_counts_clipped_rows():
    action = pd.Series([0, 1, 0, 1])
    raw = np.array([0.8, 2.0, 5.0, 12.0])

    diagnostics = weight_diagnostics(action, raw, 3.0)

    assert diagnostics.clipped_rows == 2
    assert diagnostics.clipped_pct == pytest.approx(50.0)
    assert diagnostics.clipped_max == pytest.approx(3.0)
