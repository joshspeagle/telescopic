import warnings

import numpy as np
import pytest

from references import Counted, gaussian_log_prob
from telescopic._gradient_check import check_gradient
from telescopic._random import replicate_generator, root_seed
from telescopic.diagnostics import GradientCheck
from telescopic.initial import Normal, as_initial, draw
from telescopic.warnings import InputWarning

MEAN = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
INIT = Normal(np.zeros(5), 4 * np.eye(5))


def log_prob(x):
    return -0.5 * np.sum((x - MEAN) ** 2, axis=-1)


def grad(x):
    return -(x - MEAN)


def checked(gradient=grad, function=log_prob, init=INIT, rng=1):
    return check_gradient(
        function, gradient, as_initial(init), root_seed(rng), args=(), kwargs=None, vectorize=True
    )


def quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return checked(*args, **kwargs)


@pytest.mark.parametrize(
    ("function", "gradient"),
    [
        (log_prob, grad),
        # Not a polynomial, so that central differences carry a truncation error.
        (lambda x: np.sum(np.cos(3 * x), axis=-1), lambda x: -3 * np.sin(3 * x)),
    ],
    ids=["Gaussian", "cosine"],
)
def test_the_check_is_silent_on_a_correct_gradient(function, gradient):
    for rng in range(1, 6):
        check = quiet(gradient, function, rng=rng)
        assert len(check.gap) == 8


def test_a_constant_log_density_with_a_zero_gradient_is_silent():
    # Its gaps and tolerances are 0: a line disagrees only when its gap exceeds its
    # tolerance strictly.
    check = quiet(lambda x: np.zeros(x.shape), lambda x: np.zeros(len(x)))
    assert check.gap.max() == check.tolerance.max() == 0


def test_the_ratio_is_infinite_for_a_gap_above_a_tolerance_of_0():
    lines = np.array([0.0, 1.0, 1.0]), np.array([0.0, 0.0, 2.0])
    check = GradientCheck(np.arange(1, 4), *lines, np.ones(3), n_skipped=0, n_unusable=0)
    np.testing.assert_array_equal(check.ratio, [0.0, np.inf, 0.5])


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
    assert (check.gap > check.tolerance).all()
    message = str(caught[0].message)
    line = int(np.argmax(check.ratio))
    assert f"by {check.gap[line]:.3g} nats, {relative}" in message
    assert f"draw {check.draw[line]} of 8" in message
    assert "check_grad=False" in message


@pytest.mark.parametrize("d", [5, 1000])
def test_the_check_costs_40_log_densities_and_8_gradients_at_any_dimension(d):
    counted = Counted(gaussian_log_prob), Counted(lambda x: -x)
    quiet(counted[1], counted[0], Normal(np.zeros(d), np.eye(d)))
    assert (counted[0].points, counted[1].points) == (40, 8)


def failing(x):
    raise ZeroDivisionError("a gradient that fails")


@pytest.mark.parametrize(
    ("gradient", "error", "message"),
    [
        (failing, ZeroDivisionError, "a gradient that fails"),
        (lambda x: None, ValueError, "returned None for 1 points"),
        (lambda x: -x[:, :1], ValueError, r"returned shape \(1, 1\)"),
    ],
)
def test_an_error_inside_the_check_keeps_its_type_and_notes_the_check(gradient, error, message):
    # It was raised again as a RuntimeError that advised check_grad=False, which only
    # moved the package's own complaints about the functions' outputs to the run.
    with pytest.raises(error, match=message) as raised:
        checked(gradient)
    (note,) = raised.value.__notes__
    assert note.startswith("raised in the gradient check, which evaluates log_prob and grad")
    assert "check_grad" not in note


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

    init = Normal(np.zeros(2), np.eye(2))
    check = quiet(lambda x: -x, half_normal, init, rng=6)
    # The draws x_p of its stream, namespace 3 of the seed, each followed by a_p and b_p.
    generator = replicate_generator(root_seed(6), 3, 0)
    starts = [draw(init, generator, 2) for _ in range(24)][::3]
    inside = [p + 1 for p, x in enumerate(starts) if x[0] >= 0]
    assert check.n_skipped == 8 - len(inside) > 0
    assert check.draw.tolist() == inside


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

    counted = Counted(log_prob32), Counted(grad)
    check = quiet(*counted[::-1], rng=5)
    assert len(check.gap) == 8
    # Each draw evaluated again once rounded; the sweep at float64 first (40 and 8) when
    # the dtype narrows after the first draw.
    assert (counted[0].points, counted[1].points) == ((48, 16) if always else (88, 24))


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


def test_a_check_whose_steps_round_away_in_a_float32_log_density_names_that_cause():
    # The steps along lines from a start of spread 1e-3 at 1e6 are below float32's
    # resolution there, about 0.06; the warning said only that lines reached zero density.
    def log_prob32(x):
        return (-0.5 * np.sum((x - 1e6) ** 2, axis=-1)).astype(np.float32)

    with pytest.warns(InputWarning, match="steps that round away in the dtype of log_prob"):
        check = checked(
            lambda x: -(x - 1e6), log_prob32, Normal(np.full(3, 1e6), 1e-6 * np.eye(3))
        )
    assert (len(check.gap), check.n_unusable, check.n_skipped) == (0, 8, 0)


def test_a_float32_log_density_with_a_wrong_gradient_warns():
    def log_prob32(x):
        return log_prob(x).astype(np.float32)

    def wrong(x):
        return grad(x).astype(np.float32) * np.array([-1.0, 1, 1, 1, 1], dtype=np.float32)

    with pytest.warns(InputWarning, match="grad disagrees"):
        checked(wrong, log_prob32)


# A start on three points, whose lines have directions of 1 or 2 (or the fallback, the
# range 2) from x of at most 3. Tolerance = 10 (T + R_f + R_g), divided by h as recorded.
POINTS3 = np.array([[1.0], [2.0], [3.0]])
EPS32 = float(np.finfo(np.float32).eps)


def test_the_tolerance_of_a_float32_gradient_is_640_units_of_its_last_place():
    # f = 3x and a float32 gradient 3: R_g / h = 64 eps32 |g u| = 64 eps32 |predicted|
    # exactly in d = 1, and T + R_f, of the order of eps64 |f| <= 9 eps64 over
    # 64 eps32 h |g u| >= 64 eps32 h 3, add at most 1e-3 of it.
    check = quiet(lambda x: np.full(x.shape, 3, dtype=np.float32), lambda x: 3 * x[:, 0], POINTS3)
    ratio = check.tolerance / (10 * 64 * EPS32 * np.abs(check.predicted))
    assert np.all((ratio >= 1) & (ratio <= 1 + 2e-3))


def test_the_tolerance_of_a_float32_log_density_is_640_units_of_its_last_place():
    # f = float32(3x + 1000): R_f = (2 / w_1) 64 eps32 L, with L between 1002.9 and 1009.1
    # on every line and w_1 = 2 up to 1e-4; T and |F4| / sqrt(70), each about eps32 L,
    # add at most 3 percent, and R_g, of the order of eps64, nothing.
    check = quiet(
        lambda x: np.full(x.shape, 3.0), lambda x: (3 * x[:, 0] + 1000).astype(np.float32), POINTS3
    )
    largest = check.tolerance * EPS32 ** (1 / 3) / (10 * 64 * EPS32)
    assert np.all((largest >= 1000) & (largest <= 1009.1 * 1.03))


def test_an_empirical_start_with_repeated_rows_compares_along_its_range():
    # Where a_p = b_p, the direction is a random sign times the range of the draws, 10,
    # not |x_p|, which is 0 or 10; f = 2x predicts 20 on every line.
    check = quiet(
        lambda x: np.full(x.shape, 2.0), lambda x: 2 * x[:, 0], np.array([[0.0], [10.0]])
    )
    np.testing.assert_allclose(np.abs(check.predicted), 20.0)


@pytest.mark.parametrize(("sign", "warns"), [(1.0, False), (-1.0, True)])
def test_a_point_start_off_any_boundary_checks_the_sign_of_grad(sign, warns):
    # Every direction is the fallback, a random sign times |x_p|.
    point = (MEAN + 1.0)[None]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check = checked(lambda x: sign * grad(x), init=point)
    assert len(check.gap) == 8
    assert bool(caught) == warns


def test_a_start_far_narrower_than_its_distance_from_0_is_compared_on_every_line():
    # Steps of about 1e-6 h = 6e-12, below half the spacing of float64 at 1e6, 1.2e-10,
    # would round away; each direction is raised to sqrt(eps) |x|.
    def far(x):
        return -0.5 * np.sum((x - 1e6) ** 2, axis=-1)

    check = quiet(lambda x: -(x - 1e6), far, Normal(np.full(3, 1e6), 1e-12 * np.eye(3)))
    assert len(check.gap) == 8


def never_called(x):
    raise AssertionError("log_prob was called")


def checked_combined(log_prob_and_grad, rng=1):
    # The check of a log_prob_and_grad, with a log_prob that it must not call.
    return check_gradient(
        never_called,
        None,
        as_initial(INIT),
        root_seed(rng),
        args=(),
        kwargs=None,
        vectorize=True,
        log_prob_and_grad=log_prob_and_grad,
    )


def float32(function, where=lambda x: True):
    def narrow(x):
        values = function(x)
        return values.astype(np.float32) if where(x) else values

    return narrow


@pytest.mark.parametrize(
    ("function", "gradient"),
    [
        (log_prob, grad),
        (log_prob, float32(grad)),
        (float32(log_prob), float32(grad)),
        (float32(log_prob, lambda x: np.all(x[:, 0] > 0)), grad),
        (log_prob, lambda x: x - MEAN),
    ],
    ids=["float64", "float32 gradient", "float32", "narrowing log-density", "wrong sign"],
)
def test_a_combined_callable_gets_the_record_and_cost_of_its_two_functions(function, gradient):
    # Its gradient is compared with differences of its own values, at the same points.
    separate = Counted(function), Counted(gradient)
    combined = Counted(lambda x: (function(x), gradient(x)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected = checked(separate[1], separate[0], rng=5)
        record = checked_combined(combined, rng=5)
    for name, value in vars(expected).items():
        np.testing.assert_array_equal(getattr(record, name), value, err_msg=name)
    assert len(caught) == 2 * (expected.gap > expected.tolerance).any()
    assert combined.points == separate[0].points


def test_the_check_of_a_combined_callable_names_it_in_its_warning_and_errors():
    def wrong_sign(x):
        return log_prob(x), x - MEAN

    with pytest.warns(
        InputWarning,
        match="the gradient of log_prob_and_grad disagrees with "
        "differences of the log-density of log_prob_and_grad",
    ):
        checked_combined(wrong_sign)
    with pytest.raises(ValueError, match="returned None") as raised:
        checked_combined(lambda x: None)
    assert raised.value.__notes__[0].startswith(
        "raised in the gradient check, which evaluates log_prob_and_grad at draws"
    )
