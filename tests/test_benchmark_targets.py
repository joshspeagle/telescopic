"""Check the benchmark oracles independently of the sampling engine."""

import numpy as np
import pytest
from scipy import stats

from benchmarks import targets


@pytest.fixture(params=["rotated_gaussian", "banana", "funnel", "mixture", "four_mode"])
def case(request: pytest.FixtureRequest) -> targets.BenchmarkCase:
    return getattr(targets, request.param)(10)


def test_exact_draws_agree_with_analytic_expectations(case: targets.BenchmarkCase) -> None:
    size = 20_000
    draws = case.draw_exact(np.random.default_rng(91), size)
    assert draws.shape == (size, case.dimension)
    for estimand in case.estimands:
        error = abs(estimand.function(draws).mean() - estimand.mean)
        assert error < 5 * np.sqrt(estimand.variance / size)


def test_wide_start_scales_each_standard_deviation_by_ten(case: targets.BenchmarkCase) -> None:
    np.testing.assert_array_equal(case.matched.mean, case.wide.mean)
    np.testing.assert_allclose(case.wide.cov, 100 * case.matched.cov)


def test_batch_scores_match_finite_differences(case: targets.BenchmarkCase) -> None:
    points = case.draw_exact(np.random.default_rng(7), 3)
    step = 1e-5
    offsets = step * np.eye(case.dimension)
    central = (
        case.log_prob(points[:, None, :] + offsets) - case.log_prob(points[:, None, :] - offsets)
    ) / (2 * step)
    np.testing.assert_allclose(case.grad(points), central, atol=1e-7, rtol=1e-6)


def test_single_component_draws_fail_mixture_region_oracle() -> None:
    case = targets.mixture(10, separation=6.0, narrow_scale=1.0)
    region = case.estimands[1]
    size = 2_000
    draws = np.random.default_rng(23).standard_normal((size, case.dimension))
    draws[:, 0] -= 3.0
    assert region.mean != pytest.approx(0.7)
    assert region.function(draws).mean() - region.mean > 5 * np.sqrt(region.variance / size)


@pytest.mark.parametrize(
    "constructor", [targets.rotated_gaussian, targets.banana, targets.mixture]
)
def test_targets_with_two_estimands_require_two_dimensions(constructor) -> None:
    with pytest.raises(ValueError, match="dimension"):
        constructor(1)


def test_mixture_density_preserves_the_narrow_components_mass() -> None:
    case = targets.mixture(10)
    points = np.zeros((3, 10))
    points[:, 0] = [-6.0, 0.0, 6.0]
    center = points[0]
    left = np.log(0.7) + stats.multivariate_normal.logpdf(points, mean=center)
    right = np.log(0.3) + stats.multivariate_normal.logpdf(points, mean=-center, cov=0.01)
    expected = np.logaddexp(left, right)
    actual = case.log_prob(points)
    np.testing.assert_allclose(actual - actual[0], expected - expected[0], atol=1e-10, rtol=0)


def test_four_mode_curved_density_includes_the_jacobian() -> None:
    case = targets.four_mode(2)
    points = np.array([[-2.0, -3.0], [2.0, 4.0], [6.0, 5.0]])
    linear = np.array(
        [
            np.log(weight) + stats.multivariate_normal.logpdf(points, mean=[first, second])
            for weight, first, second in [(0.4, -4, -4), (0.3, 4, -4), (0.2, -4, 4)]
        ]
    )
    scale = np.exp(0.1 * (points[:, 0] - 4) ** 2)
    unbent = np.column_stack([points[:, 0], points[:, 1] / scale])
    bent = np.log(0.1) + stats.multivariate_normal.logpdf(unbent, mean=[4, 4]) - np.log(scale)
    expected = np.logaddexp.reduce(np.vstack([linear, bent]), axis=0)
    actual = case.log_prob(points)
    np.testing.assert_allclose(actual - actual[0], expected - expected[0], atol=1e-10, rtol=0)
