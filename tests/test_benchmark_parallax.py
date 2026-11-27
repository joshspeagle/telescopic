"""Verify the parallax target and its nested numerical reference."""

import numpy as np
import pytest
from scipy import stats

from benchmarks import parallax


@pytest.mark.parametrize("dimension", [51, pytest.param(501, marks=pytest.mark.slow)])
def test_nested_reference_converges_under_inner_and_outer_refinement(dimension: int) -> None:
    for original, repeated in zip(
        parallax._data(dimension - 1), parallax._data(dimension - 1), strict=True
    ):
        np.testing.assert_array_equal(original, repeated)
    coarse = parallax.parallax(dimension)
    fine = parallax.parallax(dimension, order=16, outer_limit=6.0, outer_rtol=1e-10)
    np.testing.assert_allclose(coarse.matched.mean, fine.matched.mean, atol=1e-8, rtol=0)
    np.testing.assert_allclose(coarse.matched.cov, fine.matched.cov, atol=1e-8, rtol=0)
    np.testing.assert_allclose(coarse.wide.cov, 100 * coarse.matched.cov)
    for estimand, reference in zip(coarse.estimands, fine.estimands, strict=True):
        np.testing.assert_allclose(
            [estimand.mean, estimand.variance],
            [reference.mean, reference.variance],
            atol=1e-8,
            rtol=0,
        )


def test_parallax_log_density_and_score_include_log_distance_jacobians() -> None:
    case = parallax.parallax(51)
    baseline = case.matched.mean
    points = np.stack([baseline, baseline + np.linspace(-0.2, 0.2, 51)])
    scale = np.exp(points[:, 0])
    distance = np.exp(points[:, 1:])
    observed, error = parallax._data(50)
    reference = (
        stats.norm.logpdf(points[:, 0], scale=0.5)
        + (stats.gamma.logpdf(distance, a=3, scale=scale[:, None]) + points[:, 1:]).sum(axis=1)
        + stats.norm.logpdf(observed, loc=1 / distance, scale=error).sum(axis=1)
    )
    actual = case.log_prob(points)
    np.testing.assert_allclose(actual - actual[0], reference - reference[0], atol=1e-10, rtol=0)
    step = 1e-5
    offsets = np.eye(51)[[0, 1, 10, 50]] * step
    central = (
        case.log_prob(points[:, None] + offsets) - case.log_prob(points[:, None] - offsets)
    ) / (2 * step)
    np.testing.assert_allclose(case.grad(points)[:, [0, 1, 10, 50]], central, atol=1e-7, rtol=1e-6)


def test_unrepresentably_small_parallax_density_rejects_cleanly() -> None:
    case = parallax.parallax(51)
    points = np.zeros((2, 51))
    points[0, :2] = [-1e308, 1e308]
    points[1, :2] = [1e308, -1e308]
    np.testing.assert_array_equal(case.log_prob(points), -np.inf)
