import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from telescopic.kernels import MALA
from telescopic.target import Target
from telescopic.testing import random_draws
from telescopic.warnings import InputWarning


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


def test_the_gradient_is_evaluated_only_where_the_log_density_is_finite():
    # A gradient written for the support raised outside it.
    target = Target(
        lambda x: -2 * math.sqrt(x[0]) if x[0] > 0 else -np.inf,
        grad=lambda x: np.array([-1 / math.sqrt(x[0])]),
    )
    values, gradients = target.log_prob_and_grad(np.array([[4.0], [-1.0]]))
    np.testing.assert_array_equal(values, [-4.0, -np.inf])
    np.testing.assert_array_equal(gradients, [[-0.5], [0.0]])
    assert target.n_grad == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_gradient_where_the_log_density_is_finite_is_zero_and_counted(bad):
    # Not finite in one coordinate only: the whole row is zeroed, and counted once.
    gradients = np.array([[1.0, 2.0], [bad, 3.0]])
    target = Target(gaussian, grad=lambda x: gradients, vectorize=True)
    values, given = target.log_prob_and_grad(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    np.testing.assert_array_equal(values, [-0.5, -0.5])
    np.testing.assert_array_equal(given, [[1.0, 2.0], [0.0, 0.0]])
    assert (target.n_grad, target.n_grad_zeroed) == (2, 1)
    assert gradients[1, 0] != 0  # the function's own array is left alone


def test_a_non_finite_gradient_at_a_start_raises_with_the_point():
    target = Target(gaussian, grad=lambda x: np.where(x > 0, np.nan, -x), vectorize=True)
    with pytest.raises(ValueError, match=r"at x = \[1.0\], where log_prob is finite: chains"):
        target.log_prob_and_grad(np.array([[-1.0], [1.0]]), start=True)


@pytest.mark.parametrize("vectorize", [False, True])
def test_a_function_that_returns_none_is_named(vectorize):
    # NumPy read None as NaN, and the error blamed the value.
    def log_prob(x):
        pass

    with pytest.raises(ValueError, match="log_prob returned None"):
        Target(log_prob, vectorize=vectorize).log_prob(np.zeros((2, 1)))


def test_vectorize_must_be_a_bool():
    # "no" is true, so the functions were called on batches.
    with pytest.raises(TypeError, match="vectorize must be True or False"):
        Target(gaussian, vectorize="no")


@pytest.mark.parametrize("args", [np.zeros(2), "ab", 5])
def test_target_refuses_args_that_are_not_a_tuple(args):
    # An array's rows, or a string's letters, became extra positional arguments.
    with pytest.raises(TypeError, match="args must be a tuple of extra arguments"):
        Target(gaussian, args=args)


def test_a_target_without_gradient_refuses_gradient_requests():
    with pytest.raises(ValueError, match="no gradient"):
        Target(gaussian).log_prob_and_grad(np.zeros((1, 2)))


def test_a_wrong_output_shape_raises():
    with pytest.raises(ValueError, match=r"returned shape \(3,\) for 2 points"):
        Target(lambda x: np.zeros(3), vectorize=True).log_prob(np.zeros((2, 1)))


def test_a_vectorized_log_prob_that_squeezes_one_point_is_accepted():
    # SciPy's multivariate_normal.logpdf returns shape () for one point, which stopped
    # every run at the first batch of one row.
    target = Target(lambda x: np.squeeze(gaussian(x)), vectorize=True)
    values = target.log_prob(np.ones((1, 2)))
    assert values.shape == (1,)
    assert values[0] == -1.0


def test_a_log_prob_that_returns_one_element_arrays_point_by_point_is_accepted():
    values = Target(lambda x: np.array([gaussian(x)])).log_prob(np.ones((3, 2)))
    np.testing.assert_array_equal(values, [-1.0, -1.0, -1.0])


def test_a_gradient_of_the_wrong_shape_still_raises():
    target = Target(gaussian, grad=lambda x: -x.T, vectorize=True)
    with pytest.raises(ValueError, match=r"returned shape \(2, 3\) for 3 points"):
        target.log_prob_and_grad(np.ones((3, 2)))


@pytest.mark.parametrize("buffered", ["log_prob", "grad"])
def test_a_target_function_that_reuses_its_buffer_does_not_alias_states(buffered):
    # The start states aliased the buffer, so the next evaluation, at the proposals,
    # overwrote them, and a MALA step kept the proposals' values on rejected rows.
    buffers = {}

    def reusing(function):
        def wrapped(x):
            values = function(x)
            out = buffers.setdefault((function, values.shape), np.empty(values.shape))
            out[...] = values
            return out

        return wrapped

    functions = {"log_prob": gaussian, "grad": np.negative}
    functions[buffered] = reusing(functions[buffered])
    target = Target(functions["log_prob"], functions["grad"], vectorize=True)
    n = 2000
    kernel = MALA(1.6, cov=[[1.0]])
    start = kernel.init_state(target, np.random.default_rng(0).standard_normal((n, 1)))
    new, _ = kernel.step(target, start, random_draws(n, 0))
    for states in (start, new):
        np.testing.assert_array_equal(states.log_prob, gaussian(states.x))
        np.testing.assert_array_equal(states.grad, -states.x)


@pytest.mark.parametrize(("log_prob", "grad"), [(1.0, None), (gaussian, 1.0)])
def test_non_callables_are_rejected(log_prob, grad):
    with pytest.raises(TypeError, match="must be callable"):
        Target(log_prob, grad)


def test_a_target_without_a_log_density_gives_zero_and_takes_no_gradient():
    target = Target(None)
    np.testing.assert_array_equal(target.log_prob(np.ones((3, 2))), np.zeros(3))
    assert not target.has_log_prob
    with pytest.raises(ValueError, match="grad needs log_prob"):
        Target(None, grad=lambda x: -x)


@pytest.mark.parametrize(("bad", "returned"), [(None, "None"), (np.zeros(1), r"shape \(1,\)")])
def test_a_pointwise_grad_returning_none_or_a_wrong_shape_at_one_point_names_it_and_the_point(
    bad, returned
):
    # One bad row among good ones raised NumPy's "inhomogeneous shape" ValueError, which
    # named neither grad nor the point.
    def grad(x):
        return bad if x[0] > 0 else -x

    target = Target(gaussian, grad)
    with pytest.raises(ValueError, match=f"grad returned {returned} at x = \\[1.0, 0.0\\]"):
        target.log_prob_and_grad(np.array([[-1.0, 0.0], [1.0, 0.0]]))


@pytest.mark.parametrize(
    ("output", "returned"),
    [([[0.0], [0.0, 1.0]], "a ragged sequence"), (np.array(["a", "b"]), "array")],
)
def test_a_vectorized_output_that_is_not_numbers_is_named(output, returned):
    with pytest.raises(ValueError, match=f"<lambda> returned {returned}.* for 2 points"):
        Target(lambda x: output, vectorize=True).log_prob(np.zeros((2, 1)))


def test_a_log_prob_returning_emcee_blobs_is_told_where_derived_quantities_go():
    with pytest.raises(ValueError, match=r"computed after the run with Result\.estimate"):
        Target(lambda x: (gaussian(x), x[0])).log_prob(np.zeros((2, 1)))


@pytest.mark.parametrize("vectorize", [False, True])
def test_a_complex_grad_with_an_imaginary_part_raises_naming_grad(vectorize):
    # It was cast to its real part with only a ComplexWarning.
    def grad(x):
        return -x + 1j * (x > 0)

    target = Target(gaussian, grad, vectorize=vectorize)
    with pytest.raises(ValueError, match=r"grad returned \[\(-1\+1j\)\] at x = \[1.0\]"):
        target.log_prob_and_grad(np.array([[-1.0], [1.0]]))


@pytest.mark.parametrize("vectorize", [False, True])
def test_a_complex_output_with_a_zero_imaginary_part_is_its_real_part(vectorize):
    # A complex log_prob raised TypeError, and a complex grad warned (ComplexWarning).
    def log_prob(x):
        return gaussian(x) + 0j

    def grad(x):
        return -x + 0j

    x = np.array([[1.0], [2.0]])
    values, gradients = Target(log_prob, grad, vectorize=vectorize).log_prob_and_grad(x)
    np.testing.assert_array_equal(values, [-0.5, -2.0])
    np.testing.assert_array_equal(gradients, -x)
    assert gradients.dtype == np.float64


@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("narrow", ["log_prob", "grad"])
def test_a_float32_output_warns_once(vectorize, narrow):
    # float32 values, as JAX returns by default, were accepted silently.
    def log_prob(x):
        return gaussian(x).astype(np.float32 if narrow == "log_prob" else np.float64)

    def grad(x):
        return -x.astype(np.float32 if narrow == "grad" else np.float64)

    target = Target(log_prob, grad, vectorize=vectorize)
    with pytest.warns(InputWarning, match=f"^{narrow} returned float32 values"):
        target.log_prob_and_grad(np.ones((2, 1)))
    target.log_prob_and_grad(np.ones((1, 1)))  # filterwarnings = error


def test_a_complex64_output_is_named_complex64_in_the_warning():
    # The warning read "float32", the dtype of the real part.
    target = Target(lambda x: np.complex64(gaussian(x)))
    with pytest.warns(InputWarning, match="returned complex64 values.*rounded to float32"):
        target.log_prob(np.ones((2, 1)))


def test_an_integer_log_prob_does_not_warn():
    values = Target(lambda x: 0).log_prob(np.ones((2, 1)))  # filterwarnings = error
    assert values.dtype == np.float64


def test_a_pointwise_grad_on_no_points_returns_an_empty_array():
    # The empty list of outputs stacked as shape (0,), which raised StopIteration.
    values, gradients = Target(gaussian, np.negative).log_prob_and_grad(np.empty((0, 2)))
    assert values.shape == (0,)
    assert gradients.shape == (0, 2)


def test_pointwise_outputs_that_differ_only_by_axes_of_length_one_are_accepted():
    # A float on one branch and SciPy's logpdf, shape (1,), on the other raised, naming
    # the float as the wrong shape.
    target = Target(
        lambda x: -np.inf if x[0] < 0 else stats.norm.logpdf(x),
        lambda x: -x[None] if x[0] < 1 else -x,
    )
    x = np.array([[2.0], [-1.0], [0.5]])
    values, gradients = target.log_prob_and_grad(x)
    np.testing.assert_array_equal(
        values, [stats.norm.logpdf(2.0), -np.inf, stats.norm.logpdf(0.5)]
    )
    np.testing.assert_array_equal(gradients, [[-2.0], [0.0], [-0.5]])


@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("number", [Fraction, Decimal, float])
def test_real_numbers_held_as_objects_convert_as_floats(vectorize, number):
    # Fractions, Decimals and object arrays of floats, which NumPy's float conversion took,
    # were refused as not numbers.
    def log_prob(x):
        values = np.array([number(str(-row @ row / 2)) for row in np.atleast_2d(x)], object)
        return values if vectorize else values[:1]

    values = Target(log_prob, vectorize=vectorize).log_prob(np.array([[1.0], [3.0]]))
    np.testing.assert_array_equal(values, [-0.5, -4.5])
    assert values.dtype == np.float64


@pytest.mark.parametrize("vectorize", [False, True])
def test_none_among_real_numbers_held_as_objects_is_refused(vectorize):
    def log_prob(x):
        values = np.array([None if row[0] > 0 else 0.0 for row in np.atleast_2d(x)], object)
        return values if vectorize else values[0]

    with pytest.raises(ValueError, match="log_prob returned"):
        Target(log_prob, vectorize=vectorize).log_prob(np.array([[-1.0], [1.0]]))
