import numpy as np
import pytest
from scipy import stats

from telescopic.initial import Empirical, Normal, Uniform, as_initial, draw


@pytest.mark.parametrize(
    "init",
    [
        lambda rng: rng.standard_normal(2),
        stats.multivariate_normal(np.zeros(2), np.eye(2)),
        np.ones((3, 2)),
        Normal([0.0, 1.0], [1.0, 2.0]),
        Uniform([0.0, 0.0], [1.0, 2.0]),
    ],
)
def test_every_input_form_draws_points_of_the_right_shape(init):
    point = draw(as_initial(init), np.random.default_rng(0), ndim=2)
    assert point.shape == (2,)


def test_a_univariate_scipy_distribution_draws_one_dimensional_points():
    assert draw(as_initial(stats.norm()), np.random.default_rng(0)).shape == (1,)


def test_point_arrays_are_sampled_uniformly_over_their_rows():
    points = np.arange(8.0).reshape(4, 2)
    rng = np.random.default_rng(1)
    rows = [int(Empirical(points).sample(rng)[0]) // 2 for _ in range(4000)]
    assert stats.chisquare(np.bincount(rows, minlength=4)).pvalue > 1e-5


def test_normal_draws_have_the_requested_moments():
    init = Normal([1.0, -1.0], [[2.0, 0.6], [0.6, 1.0]])
    rng = np.random.default_rng(2)
    draws = np.array([init.sample(rng) for _ in range(20_000)])
    np.testing.assert_allclose(draws.mean(axis=0), init.mean, atol=5 * np.sqrt(2 / 20_000))
    np.testing.assert_allclose(np.cov(draws.T), init.cov, atol=0.06)


def test_a_draw_of_the_wrong_shape_raises():
    with pytest.raises(ValueError, match="drew shape"):
        draw(as_initial(lambda rng: np.zeros(3)), np.random.default_rng(0), ndim=2)


def test_a_non_finite_draw_raises():
    with pytest.raises(ValueError, match="non-finite"):
        draw(as_initial(lambda rng: np.array([np.nan])), np.random.default_rng(0))


def test_a_single_point_is_refused_with_guidance():
    with pytest.raises(TypeError, match="Empirical"):
        as_initial(np.zeros(2))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Normal([0.0, 0.0], np.eye(3)), "cov must have shape"),
        (lambda: Uniform([1.0], [0.0]), "low < high"),
        (lambda: Empirical(np.zeros((2, 2, 2))), "shape"),
    ],
)
def test_invalid_helper_parameters_raise(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
