import numpy as np
import pytest

from references import synthetic
from telescopic import (
    CoupledSampler,
    diagnostics,
    engine,
    estimators,
    initial,
    sample,
    testing,
    tuning,
)
from telescopic.kernels import MALA, Mixture, MultivariateT, RandomWalk
from telescopic.target import Target

KERNEL = RandomWalk(np.eye(1))


def log_prob(x):
    return -0.5 * float(x @ x)


TARGET = Target(log_prob)
INIT = initial.Normal(0.0, 1.0)


def run_engine(**options):
    return engine.run(TARGET, KERNEL, INIT, [0], k=0, m=0, lag=1, **options)


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
            lambda: engine.run(TARGET, KERNEL, INIT, ["a"], k=0, m=0, lag=1),
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


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("progress", lambda: CoupledSampler(log_prob, INIT, progress="no")),
        (
            "progress",
            lambda: CoupledSampler(log_prob, INIT, rng=0).run(2, k=1, progress="no"),
        ),
        ("progress", lambda: sample(log_prob, INIT, 2, k=1, progress=1)),
        ("strict", lambda: synthetic([1, 1], [0.0, 1.0]).check(strict="no")),
        ("store", lambda: run_engine(store="no")),
        ("keep_paths", lambda: run_engine(keep_paths=1)),
    ],
)
def test_flags_that_are_not_bool_raise_naming_the_flag(name, call):
    # progress="no" drew a progress bar, strict="no" raised, and store="no" was stored.
    with pytest.raises(TypeError, match=f"^{name} must be True or False"):
        call()


@pytest.mark.parametrize(
    ("kernels", "given"),
    [([RandomWalk], "the class RandomWalk"), (["rw", "rw"], "str"), ([1, 2], "int")],
)
def test_mixture_refuses_components_that_are_not_kernel_instances(kernels, given):
    # The class was accepted and failed at first use; a string raised AttributeError.
    with pytest.raises(TypeError, match=f"kernels\\[0\\] must be a Kernel instance, not {given}"):
        Mixture(kernels, np.ones(len(kernels)))


def test_check_coupling_refuses_a_target_that_is_not_a_target():
    with pytest.raises(TypeError, match="target must be a Target instance, not function"):
        testing.check_coupling(KERNEL, log_prob, [0.0], [1.0])
    with pytest.raises(TypeError, match="kernel must be a Kernel instance, not function"):
        testing.check_coupling(log_prob, TARGET, [0.0], [1.0])


def test_engine_run_and_tune_refuse_a_callable_target():
    # Both raised AttributeError, on n_log_prob and has_log_prob.
    with pytest.raises(TypeError, match="target must be a Target instance, not function"):
        engine.run(log_prob, KERNEL, INIT, [0], k=0, m=0, lag=1)
    with pytest.raises(TypeError, match="kernel must be a Kernel instance, not str"):
        engine.run(TARGET, "rwmh", INIT, [0], k=0, m=0, lag=1)
    with pytest.raises(TypeError, match="target must be a Target instance, not function"):
        RandomWalk().tune(log_prob, INIT, 0)
