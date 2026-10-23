import numpy as np
import pytest

from telescopic.target import Target


def gaussian(x, scale=1.0):
    return -0.5 * np.sum((x / scale) ** 2, axis=-1)


def test_vectorised_and_point_by_point_targets_agree():
    x = np.random.default_rng(0).standard_normal((5, 3))
    np.testing.assert_allclose(
        Target(gaussian).log_prob(x), Target(gaussian, vectorize=True).log_prob(x), rtol=1e-15
    )


def test_args_and_kwargs_reach_the_log_density():
    x = np.ones((2, 2))
    target = Target(gaussian, args=(2.0,))
    np.testing.assert_allclose(target.log_prob(x), [-0.25, -0.25])
    np.testing.assert_allclose(
        Target(gaussian, kwargs={"scale": 2.0}).log_prob(x), target.log_prob(x)
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nan_and_plus_infinity_raise_with_the_point(bad):
    target = Target(lambda x: bad if x[0] > 0 else 0.0)
    with pytest.raises(ValueError, match=r"at x = \[1.0\]"):
        target.log_prob(np.array([[-1.0], [1.0]]))


def test_minus_infinity_marks_zero_density():
    target = Target(lambda x: -np.inf if x[0] > 0 else 0.0)
    np.testing.assert_array_equal(target.log_prob(np.array([[-1.0], [1.0]])), [0.0, -np.inf])


def test_a_log_density_that_modifies_its_input_raises():
    def careless(x):
        x[0] = 0.0
        return 0.0

    x = np.ones((1, 2))
    with pytest.raises(ValueError, match="read-only"):
        Target(careless).log_prob(x)
    np.testing.assert_array_equal(x, 1.0)


def test_counters_count_points():
    target = Target(gaussian, grad=lambda x: -x)
    target.log_prob(np.zeros((3, 2)))
    target.log_prob_and_grad(np.zeros((4, 2)))
    assert (target.n_log_prob, target.n_grad) == (7, 4)


def test_gradients_come_with_the_log_densities():
    x = np.random.default_rng(1).standard_normal((4, 3))
    values, gradients = Target(gaussian, grad=lambda x: -x).log_prob_and_grad(x)
    np.testing.assert_allclose(values, gaussian(x))
    np.testing.assert_allclose(gradients, -x)


def test_a_target_without_gradient_refuses_gradient_requests():
    with pytest.raises(ValueError, match="no gradient"):
        Target(gaussian).log_prob_and_grad(np.zeros((1, 2)))


def test_a_wrong_output_shape_raises():
    with pytest.raises(ValueError, match="expected"):
        Target(lambda x: np.zeros(3), vectorize=True).log_prob(np.zeros((2, 1)))


@pytest.mark.parametrize(("log_prob", "grad"), [(1.0, None), (gaussian, 1.0)])
def test_non_callables_are_rejected(log_prob, grad):
    with pytest.raises(TypeError, match="must be callable"):
        Target(log_prob, grad)
