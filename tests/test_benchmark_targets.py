"""Check the benchmark oracles independently of the sampling engine."""

import numpy as np
import pytest

from benchmarks import targets


@pytest.fixture(params=["rotated_gaussian", "banana", "funnel", "mixture"])
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
