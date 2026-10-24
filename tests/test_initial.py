import numpy as np
import pytest
from scipy import stats

from references import P_MIN
from telescopic.initial import Empirical, Normal, Uniform, as_initial, draw
from telescopic.kernels import MultivariateT


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
    rows = [int(Empirical(points).rvs(random_state=rng)[0]) // 2 for _ in range(4000)]
    assert stats.chisquare(np.bincount(rows, minlength=4)).pvalue > P_MIN


def test_normal_draws_have_the_requested_moments():
    init = Normal([1.0, -1.0], [[2.0, 0.6], [0.6, 1.0]])
    rng = np.random.default_rng(2)
    draws = np.array([init.rvs(random_state=rng) for _ in range(20_000)])
    np.testing.assert_allclose(draws.mean(axis=0), init.mean, atol=5 * np.sqrt(2 / 20_000))
    np.testing.assert_allclose(np.cov(draws.T), init.cov, atol=0.06)


def test_a_draw_of_the_wrong_shape_raises():
    with pytest.raises(ValueError, match="drew shape"):
        draw(as_initial(lambda rng: np.zeros(3)), np.random.default_rng(0), ndim=2)


def test_a_zero_dimensional_draw_raises():
    # It reached tuning, which divided by the dimension.
    with pytest.raises(ValueError, match=r"drew shape \(0,\), not \(d,\) with d >= 1"):
        draw(as_initial(lambda rng: np.zeros(0)), np.random.default_rng(0))


def test_a_draw_of_shape_one_by_d_is_a_point():
    # A frozen SciPy Dirichlet draws shape (1, d) (spec section 4.2).
    point = draw(as_initial(stats.dirichlet([1.0, 2.0, 3.0])), np.random.default_rng(0), 3)
    assert point.shape == (3,)
    np.testing.assert_allclose(point.sum(), 1.0)


def test_a_covariance_symmetric_up_to_rounding_is_accepted_and_symmetrised():
    # As np.linalg.inv returns: an asymmetry of 1e-20 against variances of 1e-4 is
    # rounding, which a test relative to each entry refused.
    cov = np.array([[1e-4, 1e-20], [0.0, 2e-4]])
    np.testing.assert_array_equal(Normal([0.0, 0.0], cov).cov, [[1e-4, 5e-21], [5e-21, 2e-4]])


def test_a_non_finite_draw_raises():
    with pytest.raises(ValueError, match="non-finite"):
        draw(as_initial(lambda rng: np.array([np.nan])), np.random.default_rng(0))


def test_a_single_point_is_refused_with_guidance():
    with pytest.raises(TypeError, match="Empirical"):
        as_initial(np.zeros(2))


def test_an_object_without_rvs_is_not_an_initial_distribution():
    # A proposal samples from buffered draws, not from a generator.
    with pytest.raises(TypeError, match="rvs"):
        as_initial(MultivariateT(np.zeros(2), np.eye(2)))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Normal([0.0, 0.0], np.eye(3)), "cov must have shape"),
        (lambda: Normal([0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]]), "cov must be symmetric"),
        (lambda: Normal([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
        (lambda: Uniform([1.0], [0.0]), "low < high"),
        (lambda: Empirical(np.zeros((2, 2, 2))), "shape"),
        (lambda: Empirical(np.zeros((0, 2))), "n >= 1"),
        (lambda: Normal([0.0, 0.0], [[1e-12, 5e-9], [0.0, 1e-12]]), "symmetric"),
        (lambda: Normal([0.0], [[np.nan]]), "finite"),
        (lambda: Normal([[0.0, 0.0]], np.eye(2)), r"mean must have shape \(d,\)"),
        (lambda: Normal([np.nan], [[1.0]]), "mean must be finite"),
        (lambda: Uniform([-np.inf], [0.0]), "low must be finite"),
        (lambda: Uniform([0.0], [np.inf]), "high must be finite"),
        (lambda: Empirical([[0.0, 1.0], [np.nan, 0.0]]), r"finite, but row 1 is \[nan, 0.0\]"),
        (lambda: Empirical(np.zeros((3, 0))), "d >= 1"),
        (lambda: Empirical([1.0, 2.0]), r"not \(2,\)"),
    ],
)
def test_invalid_helper_parameters_raise(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
