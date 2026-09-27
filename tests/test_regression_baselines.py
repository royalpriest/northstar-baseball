import pandas as pd
import pytest

from northstar.models.baselines import evaluate_regression_predictions


def test_regression_metrics_are_computed():
    metrics = evaluate_regression_predictions(
        pd.Series([0.0, 1.0, 2.0, 3.0]),
        pd.Series([0.1, 0.9, 2.1, 2.9]),
    )

    assert metrics.mae == pytest.approx(0.1)
    assert metrics.rmse == pytest.approx(0.1)
    assert metrics.r2 > 0.98
    assert metrics.pearson_r > 0.99
    assert metrics.spearman_rho == pytest.approx(1.0)
