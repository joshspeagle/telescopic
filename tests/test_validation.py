import numpy as np
import pytest

from telescopic import diagnostics, engine, estimators, initial, testing, tuning
from telescopic.kernels import MALA, Mixture, MultivariateT, RandomWalk
from telescopic.target import Target

KERNEL = RandomWalk(np.eye(1))
TARGET = Target(lambda x: -0.5 * float(x @ x))


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("mean", lambda: initial.Normal("abc", 1.0)),
        ("cov", lambda: initial.Normal(0.0, "abc")),
        ("mean", lambda: initial.Normal(1j, 1.0)),
        ("mean", lambda: initial.Normal(None, 1.0)),
        ("low", lambda: initial.Uniform(["a"], [1.0])),
        ("points", lambda: initial.Empirical([[1, 2], [3]])),
        (
            "a draw of the initial distribution",
            lambda: initial.draw(initial.as_initial(lambda rng: "abc"), None),
        ),
        ("mean", lambda: MultivariateT("abc", np.eye(1))),
        ("cov", lambda: RandomWalk([["a"]])),
        ("cov", lambda: MALA(cov=[[1.0], [2.0, 3.0]])),
        ("weights", lambda: Mixture([KERNEL, KERNEL], ["a", "b"])),
        ("x", lambda: testing.check_coupling(KERNEL, TARGET, "a", [1.0])),
        ("normals", lambda: testing.fixed_draws(normals=[["a"]])),
        ("meeting_times", lambda: diagnostics.tv_bound("abc", 1, 0)),
        ("t", lambda: diagnostics.tv_bound([1, 2], 1, "abc")),
        ("distance", lambda: diagnostics.w1_bound([1], "abc", 1, 1, 0)),
        ("values", lambda: diagnostics.hill([1.0, None, 2.0], 1)),
        ("values", lambda: diagnostics.variance_share([[1.0], [2.0, 3.0]])),
        ("q", lambda: estimators.weighted_quantile(np.zeros(2), np.full(2, 0.5), "q")),
        ("t", lambda: estimators.correction_coefficients("t", 0, 1, 1, 1)),
        ("meeting_times", lambda: tuning.choose_k("abc")),
        (
            "indices",
            lambda: engine.run(TARGET, KERNEL, initial.Normal(0, 1), ["a"], k=0, m=0, lag=1),
        ),
    ],
)
def test_array_arguments_that_numpy_cannot_convert_raise_naming_the_argument(name, call):
    # NumPy's message, "could not convert string to float: 'abc'", named no argument, a
    # complex number raised TypeError and a string ValueError, and None became NaN.
    with pytest.raises(TypeError, match=f"^{name} must be "):
        call()


def test_a_ragged_array_argument_chains_numpy_error():
    with pytest.raises(TypeError, match="points must be") as raised:
        initial.Empirical([[1, 2], [3]])
    assert isinstance(raised.value.__cause__, ValueError)


def test_as_initial_raises_its_type_error_for_a_ragged_init():
    # np.asarray raised NumPy's "inhomogeneous shape" ValueError first.
    with pytest.raises(TypeError, match="init must have rvs"):
        initial.as_initial([[1, 2], [3]])
