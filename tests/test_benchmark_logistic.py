"""Check deterministic logistic designs and the recorded independent reference."""

import json

import numpy as np
import pytest
from scipy import special

from benchmarks import estimator, logistic, targets


def test_six_parameter_design_reproduces_original_experiment() -> None:
    rng = np.random.default_rng(1)
    matrix = np.column_stack([np.ones(200), rng.standard_normal((200, 5))])
    signal = np.array([-0.5, 1, -1, 0.5, 0, 2])
    outcomes = rng.random(200) < special.expit(matrix @ signal)
    data = logistic.design(6)
    np.testing.assert_array_equal(data.matrix, matrix)
    np.testing.assert_array_equal(data.outcomes, outcomes)


@pytest.mark.parametrize("dimension", [6, 300])
def test_logistic_case_uses_the_independent_reference_for_its_recorded_design(
    dimension: int,
) -> None:
    case = logistic.logistic(dimension)
    assert case.draw_exact is None
    assert case.covariance_source == "independent_gibbs_marginal_variances"
    np.testing.assert_allclose(case.wide.cov, 100 * case.matched.cov)
    assert all(estimand.mean_standard_error > 0 for estimand in case.estimands)
    point = case.matched.mean
    offsets = np.eye(dimension)[[0, 5]] * 1e-5
    central = (case.log_prob(point + offsets) - case.log_prob(point - offsets)) / 2e-5
    np.testing.assert_allclose(case.grad(point)[[0, 5]], central, atol=1e-7, rtol=1e-6)


def test_archived_six_parameter_reference_agrees_with_importance_sampling() -> None:
    # #91's importance reference used seed 9799 and 1,000,000 proposals.
    means = [-0.6401618240399656, 2.905689922799669]
    errors = [0.00029987107756111937, 0.0005871813642581041]
    for estimand, mean, error in zip(logistic.logistic(6).estimands, means, errors, strict=True):
        combined = np.hypot(estimand.mean_standard_error, error)
        assert abs(estimand.mean - mean) < 3 * combined


def test_pooled_z_includes_reference_uncertainty() -> None:
    estimand = targets.Estimand("mean", lambda points: points[..., 0], 1.0, 1.0, 0.5)
    record = estimator.RunRecord(
        100, 0, [], [], {"mean": estimator.IntervalRecord([2.0, 2.0], True)}
    )
    summary = estimator.summarize([record], estimand)
    assert summary["pooled_standard_error"] == 0.0
    assert summary["target_mean_standard_error"] == 0.5
    assert summary["pooled_z"] == 2.0
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize(
    ("name", "value"),
    [("PRIOR_VARIANCE", 1.0), ("ESTIMAND_INDICES", (5, 0))],
)
def test_loader_rejects_reference_after_model_or_estimand_change(monkeypatch, name, value) -> None:
    monkeypatch.setattr(logistic, name, value)
    with pytest.raises(ValueError, match="reference must match"):
        logistic.logistic(6)
