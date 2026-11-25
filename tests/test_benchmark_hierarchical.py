"""Check the collapsed reference for the centered normal-means benchmark."""

import numpy as np
import pytest
from scipy import stats

from benchmarks import hierarchical


@pytest.mark.parametrize("dimension", [10, 102])
def test_posterior_draws_agree_with_quadrature_at_both_dimensions(dimension: int) -> None:
    case = hierarchical.normal_means(dimension)
    tighter = hierarchical.normal_means(dimension, quadrature_rtol=1e-11)
    np.testing.assert_allclose(case.matched.mean, tighter.matched.mean, atol=1e-9, rtol=0)
    np.testing.assert_allclose(case.matched.cov, tighter.matched.cov, atol=1e-9, rtol=0)
    np.testing.assert_allclose(case.wide.cov, 100 * case.matched.cov)
    draws = case.draw_exact(np.random.default_rng(400), 12_000)
    assert draws.shape == (12_000, dimension)
    for estimand in case.estimands:
        error = abs(estimand.function(draws).mean() - estimand.mean)
        assert error < 5 * np.sqrt(estimand.variance / len(draws))


def test_centered_log_density_matches_normal_terms_and_the_scale_jacobian() -> None:
    dimension = 10
    case = hierarchical.normal_means(dimension)
    points = case.draw_exact(np.random.default_rng(701), 3)
    mean, log_tau, effects = points[:, 0], points[:, 1], points[:, 2:]
    tau = np.exp(log_tau)
    sigma = np.linspace(0.7, 2.0, dimension - 2)
    y = 1 + np.sqrt(4 + sigma**2) * np.random.default_rng(hierarchical.DATA_SEED).standard_normal(
        dimension - 2
    )
    reference = (
        stats.norm.logpdf(y, loc=effects, scale=sigma).sum(axis=1)
        + stats.norm.logpdf(effects, loc=mean[:, None], scale=tau[:, None]).sum(axis=1)
        + stats.norm.logpdf(mean, scale=5)
        + stats.halfnorm.logpdf(tau, scale=5)
        + log_tau
    )
    actual = case.log_prob(points)
    np.testing.assert_allclose(actual - actual[0], reference - reference[0], atol=1e-10, rtol=0)
    step = 1e-5
    offsets = step * np.eye(dimension)
    central = (
        case.log_prob(points[:, None] + offsets) - case.log_prob(points[:, None] - offsets)
    ) / (2 * step)
    np.testing.assert_allclose(case.grad(points), central, atol=1e-7, rtol=1e-6)


def test_extreme_scales_distinguish_exactly_centered_and_off_center_points() -> None:
    case = hierarchical.normal_means(10)
    points = np.zeros((4, 10))
    points[:, 1] = [-800.0, -800.0, -400.0, 800.0]
    points[1:3, 2] = 1.0
    actual = case.log_prob(points)
    assert np.isfinite(actual[0])
    assert np.isfinite(case.grad(points[:1])).all()
    np.testing.assert_array_equal(actual[1:], -np.inf)
