import warnings

import numpy as np
import pytest

from telescopic._gradient_check import check_gradient
from telescopic._random import root_seed
from telescopic.initial import Normal, as_initial
from telescopic.warnings import InputWarning

MEAN = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
INIT = Normal(np.zeros(5), 4 * np.eye(5))


def log_prob(x):
    return -0.5 * np.sum((x - MEAN) ** 2, axis=-1)


def grad(x):
    return -(x - MEAN)


class Counted:
    # A vectorized function that counts the points it is evaluated at.
    def __init__(self, function):
        self.function, self.points = function, 0

    def __call__(self, x):
        self.points += len(x)
        return self.function(x)


def checked(gradient=grad, function=log_prob, init=INIT, rng=1):
    return check_gradient(
        function, gradient, as_initial(init), root_seed(rng), args=(), kwargs=None, vectorize=True
    )


def quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return checked(*args, **kwargs)


def test_the_check_is_silent_on_a_correct_gradient():
    check = quiet()
    assert len(check.gap) == 8
    assert check.ratio.max() < 1


@pytest.mark.parametrize(
    ("wrong", "relative"),
    [
        (lambda x: x - MEAN, "2 times"),  # a wrong sign
        (lambda x: -2 * (x - MEAN), "0.5 times"),  # a factor of 2
        (lambda x: -x, ""),  # the mean left out
    ],
)
def test_the_check_warns_on_a_wrong_gradient_with_the_size_of_the_disagreement(wrong, relative):
    with pytest.warns(InputWarning, match="grad disagrees") as caught:
        check = checked(wrong)
    message = str(caught[0].message)
    line = int(np.argmax(check.ratio))
    assert f"by {check.gap[line]:.3g} nats, {relative}" in message
    assert f"draw {check.point[line] + 1} of 8" in message
    assert "check_grad=False" in message


@pytest.mark.parametrize("d", [5, 1000])
def test_the_check_costs_40_log_densities_and_8_gradients_at_any_dimension(d):
    counted = Counted(lambda x: -0.5 * np.sum(x * x, axis=-1)), Counted(lambda x: -x)
    quiet(counted[1], counted[0], Normal(np.zeros(d), np.eye(d)))
    assert (counted[0].points, counted[1].points) == (40, 8)


def test_an_error_inside_the_check_names_the_check():
    def failing(x):
        raise ZeroDivisionError("a gradient that fails")

    with pytest.raises(RuntimeError, match=r"the gradient check.*check_grad=False") as raised:
        checked(failing)
    assert isinstance(raised.value.__cause__, ZeroDivisionError)


def test_a_check_that_compares_nothing_warns():
    # A point start on the boundary of the support: every line reaches zero density.
    def half_normal(x):
        return np.where(x[:, 0] < 0, -np.inf, -0.5 * x[:, 0] ** 2)

    with pytest.warns(InputWarning, match="compared no line, so grad is unchecked"):
        check = checked(lambda x: -x, half_normal, np.zeros((1, 1)))
    assert (len(check.gap), check.n_unusable, check.n_skipped) == (0, 8, 0)


def test_the_check_skips_draws_of_zero_density_and_compares_the_others():
    def half_normal(x):
        return np.where(x[:, 0] < 0, -np.inf, -0.5 * np.sum(x * x, axis=1))

    check = quiet(lambda x: -x, half_normal, Normal(np.zeros(2), np.eye(2)), rng=6)
    assert check.n_skipped > 0
    assert len(check.gap) + check.n_skipped + check.n_unusable == 8
    assert check.ratio.max() < 1


def test_a_float32_gradient_of_a_float64_log_density_is_silent():
    # Its probe adds an evaluation of grad at each draw.
    counted = Counted(lambda x: grad(x).astype(np.float32))
    assert quiet(counted).ratio.max() < 1
    assert counted.points == 16


@pytest.mark.parametrize(
    ("dtype", "always"), [(np.float32, True), (np.complex64, True), (np.float32, False)]
)
def test_a_float32_log_density_with_its_gradient_is_silent(dtype, always):
    # Float32 values from the first draw, or only where every point has x_0 > 0, which
    # makes the check start again at the narrower dtype once it sees them. Complex64
    # values are float32 ones to the Target, which takes their real part.
    def log_prob32(x):
        values = log_prob(x)
        return values.astype(dtype) if always or np.all(x[:, 0] > 0) else values

    counted = Counted(log_prob32)
    check = quiet(function=counted, rng=5)
    assert len(check.gap) == 8
    assert check.ratio.max() < 1
    if always:
        assert counted.points == 48  # each draw evaluated again once rounded


def test_the_check_changes_no_warning_filter_and_hides_no_warning():
    # It ignored its own precision warnings with a filter, which hid a user's warning
    # that mentions a returned float and, from two threads, outlived the check.
    def warning_log_prob(x):
        seen.append(list(warnings.filters))
        warnings.warn("the solver returned float values", UserWarning, stacklevel=1)
        return log_prob(x)

    seen = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filters = list(warnings.filters)
        checked(function=warning_log_prob)
        seen.append(list(warnings.filters))
    assert all(during == filters for during in seen)
    assert len(caught) == len(seen) - 1
