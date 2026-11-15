import copy
import importlib
import inspect
import io
import itertools
import os
import pkgutil
import re
import runpy
import subprocess
import sys
import tempfile
import warnings
from dataclasses import replace
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

import telescopic
from references import (
    Counted,
    Failing,
    assert_same_outputs,
    assert_same_replicates,
    gaussian_log_prob,
    never_called,
    synthetic,
)
from references import Normal as Conditional
from telescopic import CoupledSampler, Result, __version__, engine, sample
from telescopic._parallel import Interrupt
from telescopic.engine import Settings, Status
from telescopic.initial import Empirical, Normal
from telescopic.kernels import HMC, MALA, Gibbs, Mixture, RandomWalk
from telescopic.sampler import _SlowPairs
from telescopic.warnings import ConvergenceWarning, InputWarning

INIT = Normal([2.0, -1.0], [4.0, 4.0])
KERNEL = RandomWalk(np.eye(2))


log_prob = gaussian_log_prob


def test_a_run_tunes_runs_the_pilot_and_records_both():
    result = sample(log_prob, INIT, 20, rng=1)
    assert result.tuning is not None
    assert result.kernel.is_tuned
    assert len(result.pilot) == 100
    assert (result.lag, result.m) == (result.k, 10 * result.k)
    np.testing.assert_array_equal(result.index, np.arange(20))


def test_two_runs_of_a_sampler_equal_one_run_of_both():
    # Point by point, so each replicate is bitwise independent of the batch.
    sampler = CoupledSampler(log_prob, INIT, rng=2)
    sampler.run(10)
    split = sampler.run(15)
    whole = CoupledSampler(log_prob, INIT, rng=2).run(25)
    np.testing.assert_array_equal(split.meeting_times, whole.meeting_times)
    np.testing.assert_array_equal(split.measure.atoms, whole.measure.atoms)


def test_a_sampler_keeps_its_settings():
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3)
    assert sampler.settings is None
    sampler.run(5, k=40, thin=2)
    assert sampler.run(5).thin == 2
    for change, message in (({"k": 41}, "k was 40"), ({"thin": 1}, "thin was 2")):
        with pytest.raises(ValueError, match=message):
            sampler.run(5, **change)
    with pytest.raises(AttributeError):
        sampler.settings = sampler.settings._replace(lag=2)


@pytest.fixture(scope="module")
def first_run():
    # The first 10 replicates of a run with everything automatic, and 15 of the same run.
    whole = CoupledSampler(log_prob, INIT, rng=4).run(15)
    return CoupledSampler(log_prob, INIT, rng=4).run(10), whole


def resumed(result, *arguments, **options):
    sampler = CoupledSampler(log_prob, INIT, *arguments, **options)
    sampler.resume(result)
    return sampler


def test_resumed_run_equals_the_uninterrupted_runs_replicates(first_run):
    first, whole = first_run
    more = resumed(first, kernel=first.kernel).run(5)
    assert_same_replicates(more._replicates, whole._replicates)
    assert more.tuning is first.tuning
    np.testing.assert_array_equal(more.pilot, first.pilot)


def test_resumed_run_from_a_loaded_result_equals_the_uninterrupted_run(first_run):
    first, whole = first_run
    buffer = io.BytesIO()
    first.save(buffer)
    buffer.seek(0)
    # kernel "auto" is tuned again from the result's seed and matches its tuning, so no warning.
    sampler = resumed(Result.load(buffer))
    np.testing.assert_array_equal(sampler.tuning.cov, first.tuning.cov)
    assert_same_replicates(sampler.run(5)._replicates, whole._replicates)


def test_resume_warns_when_retuning_differs_from_the_results_tuning(first_run):
    sampler = CoupledSampler(lambda x: log_prob(x / 2), INIT)
    with pytest.warns(InputWarning, match="the sampler's kernel differs from the result's"):
        sampler.resume(first_run[0])
    assert not np.array_equal(sampler.tuning.cov, first_run[0].tuning.cov)


def test_a_resume_whose_tuning_raises_leaves_the_sampler_unchanged(first_run):
    calls = []

    def failing(x):
        calls.append(1)
        if len(calls) > 50:
            raise RuntimeError("the target failed")
        return log_prob(x)

    sampler = CoupledSampler(failing, INIT)
    seed = sampler.seed
    with pytest.raises(RuntimeError, match="the target failed"):
        sampler.resume(first_run[0])
    assert sampler.seed is seed
    assert (sampler.tuning, sampler.pilot, sampler.settings) == (None, None, None)


def test_resume_warns_when_a_given_kernel_differs_from_the_results(first_run):
    kernel = RandomWalk(cov=first_run[0].tuning.cov, coupling="maximal")
    with pytest.warns(InputWarning, match="the sampler's kernel differs from the result's"):
        resumed(first_run[0], kernel=kernel)


def test_a_resume_whose_warning_is_an_error_has_already_resumed(first_run):
    # The suite turns warnings into errors (pyproject.toml), as a user's filter can.
    sampler = CoupledSampler(lambda x: log_prob(x / 2), INIT)
    with pytest.raises(InputWarning):
        sampler.resume(first_run[0])
    assert sampler.settings == first_run[0]._replicates.settings
    assert engine._seed_key(sampler.seed) == engine._seed_key(first_run[0].seed)


def test_resumed_vectorized_run_equals_the_samplers_next_call():
    sampler = CoupledSampler(log_prob, INIT, vectorize=True, rng=5)
    first = sampler.run(10)
    following = sampler.run(6)
    more = resumed(first, vectorize=True, kernel=first.kernel).run(6)
    assert_same_replicates(more._replicates, following._replicates)


def test_resumed_run_completes_an_interrupted_result_and_clears_the_flag():
    first = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=6).run(8, k=20)
    # Replicates 8 to 11 were requested but never started: the first two runs complete
    # half of them, the next the rest and two new ones.
    interrupted = Result(replace(first._replicates, requested=np.arange(12)), kernel=KERNEL)
    assert interrupted.interrupted
    sampler = resumed(interrupted, kernel=KERNEL)
    assert sampler.run(2).interrupted
    more = sampler.run(4)
    assert not more.interrupted
    whole = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=6).run(14, k=20)
    assert_same_replicates(more._replicates, whole._replicates)


def test_resume_of_a_merged_result_starts_above_its_highest_requested_index():
    first = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=7).run(5, k=20)
    apart = engine.run(
        telescopic.Target(log_prob),
        KERNEL,
        INIT,
        np.arange(10, 15),
        rng=first.seed,
        **first._replicates.settings._asdict(),
    )
    merged = first.merge(Result(apart, kernel=KERNEL))
    more = resumed(merged, kernel=KERNEL).run(3)
    np.testing.assert_array_equal(more.index, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17])


def test_resume_never_reruns_censored_replicates():
    kernel = RandomWalk(np.eye(2), scale=0.1)
    with pytest.warns(ConvergenceWarning):
        first = sample(log_prob, INIT, 10, kernel=kernel, k=1, m=2, max_iter=3, rng=8)
    censored = first.index[np.isinf(first.meeting_times)]
    assert len(censored)
    with pytest.warns(ConvergenceWarning):
        more = resumed(first, kernel=kernel).run(4)
    np.testing.assert_array_equal(more.index, np.arange(14))
    np.testing.assert_array_equal(more.meeting_times[:10], first.meeting_times)
    assert more.max_iter == 3


def test_resumed_sampler_refuses_a_change_of_k(first_run):
    first = first_run[0]
    sampler = resumed(first, kernel=first.kernel)
    assert sampler.settings == first._replicates.settings
    with pytest.raises(ValueError, match=f"k was {first.k} in the earlier runs"):
        sampler.run(2, k=first.k + 1)


def used(kind):
    kernel = KERNEL if kind == "replicates" else "auto"
    sampler = CoupledSampler(log_prob, INIT, kernel=kernel, rng=11)
    if kind == "replicates":
        sampler.run(2, k=20)
    elif kind == "pilot":
        sampler.meeting_times(2)
    else:
        sampler.tune()
    return sampler


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("replicates", "has already run replicates"),
        ("pilot", "has already run pilot pairs"),
        ("tuning", "has already tuned its kernel"),
    ],
)
def test_resume_refuses_a_used_sampler(first_run, kind, message):
    with pytest.raises(ValueError, match=f"resume needs a new sampler: this one {message}"):
        used(kind).resume(first_run[0])


def test_resume_refuses_another_dimension(first_run):
    sampler = CoupledSampler(log_prob, Normal(np.zeros(3), np.ones(3)))
    with pytest.raises(ValueError, match="the result has dimension 2, but the initial"):
        sampler.resume(first_run[0])


@pytest.mark.parametrize("loaded", [False, True])
def test_resume_refuses_another_kernel_class(first_run, loaded):
    first = first_run[0]
    if loaded:
        first = Result(first._replicates, kernel=str(first.kernel), tuning=first.tuning)
    sampler = CoupledSampler(log_prob, INIT, kernel="independence")
    with pytest.raises(ValueError, match="run with RandomWalk, but the sampler's kernel is Ind"):
        sampler.resume(first)


def test_resume_refuses_a_disagreeing_rng(first_run):
    with pytest.raises(ValueError, match="rng was given, but the result has another seed"):
        resumed(first_run[0], rng=5)
    # The result's own seed is accepted.
    assert resumed(first_run[0], kernel=first_run[0].kernel, rng=4).settings is not None


def test_resume_refuses_different_names(first_run):
    with pytest.raises(ValueError, match=r"names \['a', 'b'\] differ from the result's"):
        resumed(first_run[0], names=["a", "b"])


def test_resume_refuses_an_untuned_kernel_without_tuning():
    first = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=9).run(2, k=20)
    with pytest.raises(ValueError, match="the result holds no tuning report"):
        resumed(first)


def test_resume_refuses_a_non_result(first_run):
    with pytest.raises(TypeError, match=r"result must be a telescopic\.Result, not Replicates"):
        CoupledSampler(log_prob, INIT).resume(first_run[0]._replicates)


def test_resume_without_a_pilot_runs_no_pilot_pairs():
    first = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=10).run(3, k=20)
    sampler = resumed(first, kernel=KERNEL)
    sampler.run(2)
    assert sampler.pilot is None


def test_resume_does_not_modify_the_result(first_run):
    first = first_run[0]
    before = copy.deepcopy((first._replicates, first.pilot, first.names))
    sampler = resumed(first, kernel=first.kernel)
    sampler.run(3)
    sampler.meeting_times(2)
    assert_same_replicates(before[0], first._replicates)
    np.testing.assert_array_equal(before[1], first.pilot)
    assert before[2] == first.names


def unevaluated(x):
    raise AssertionError("the log-density was evaluated")


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"n_replicates": 10.0}, TypeError, "n_replicates must be an integer, not 10.0"),
        ({"n_replicates": True}, TypeError, "n_replicates must be an integer, not True"),
        ({"k": 5.5}, TypeError, "k must be an integer or 'auto', not 5.5"),
        ({"k": np.array([3, 4])}, TypeError, "k must be an integer or 'auto'"),
        ({"thin": 2.0}, TypeError, "thin must be an integer or None, not 2.0"),
        ({"max_iter": "9"}, TypeError, "max_iter must be an integer or None"),
        ({"lag": 0}, ValueError, "lag must be at least 1, not 0"),
        ({"thin": 0}, ValueError, "thin must be at least 1, not 0"),
        ({"m": -1}, ValueError, "m must be at least 0, not -1"),
        ({"max_iter": 1}, ValueError, r"max_iter must exceed max\(lag, m\) = 1, not 1"),
        ({"k": -1}, ValueError, "k must be at least 0, not -1"),
        ({"k": 5, "m": 3}, ValueError, "m must be at least k = 5, not 3"),
        ({"k": 40, "max_iter": 100}, ValueError, "max_iter must exceed"),
    ],
)
def test_invalid_settings_raise_before_tuning(arguments, error, message):
    sampler = CoupledSampler(unevaluated, INIT, rng=3)
    with pytest.raises(error, match=message):
        sampler.run(**{"n_replicates": 5} | arguments)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"n": 2.5}, TypeError, "n must be an integer, not 2.5"),
        ({"max_iter": 1}, ValueError, "max_iter must exceed"),
    ],
)
def test_invalid_pilot_settings_raise_before_tuning(arguments, error, message):
    sampler = CoupledSampler(unevaluated, INIT, rng=3)
    with pytest.raises(error, match=message):
        sampler.meeting_times(**arguments)


def test_invalid_first_settings_are_not_kept():
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3)
    with pytest.raises(ValueError, match="m must be at least k = 40"):
        sampler.run(5, k=40, m=5)
    assert sampler.run(5, k=40, m=50).m == 50


@pytest.mark.parametrize(
    ("given", "expected"),
    [({"k": 40}, (40, 40, 400)), ({"k": 40, "m": 50, "lag": 2}, (40, 2, 50))],
)
def test_an_explicit_k_needs_no_pilot_and_sets_the_lag_and_m(given, expected):
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=4)
    result = sampler.run(5, **given)
    assert (result.k, result.lag, result.m) == expected
    assert sampler.pilot is None


@pytest.mark.parametrize(
    "init",
    [lambda rng: rng.normal(size=2), np.random.default_rng(5).standard_normal((6, 2))],
    ids=["callable", "points"],
)
def test_initial_distributions_without_rvs_run(init):
    assert sample(log_prob, init, 4, kernel=KERNEL, k=40, rng=6).n_replicates == 4


def test_a_seed_reproduces_a_run_and_a_generator_moves_on():
    first, second = (sample(log_prob, INIT, 5, kernel=KERNEL, k=40, rng=7) for _ in range(2))
    np.testing.assert_array_equal(first.measure.atoms, second.measure.atoms)
    rng = np.random.default_rng(8)
    third, fourth = (sample(log_prob, INIT, 5, kernel=KERNEL, k=40, rng=rng) for _ in range(2))
    assert not np.array_equal(third.measure.atoms, fourth.measure.atoms)


def test_the_pilot_draws_from_streams_of_its_own():
    # Sharing the main run's streams, a lag-1 run would repeat its meeting times (R-TUNE-3).
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3)
    pilot = sampler.meeting_times(10)
    with pytest.warns(ConvergenceWarning, match="k may be too small"):
        result = sampler.run(10, k=1, m=2, lag=1)
    assert not np.array_equal(pilot, result.meeting_times)


def test_a_run_completes_a_pilot_of_fewer_than_100_pairs():
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3)
    sampler.meeting_times(30)
    result = sampler.run(20)
    whole = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3).meeting_times()
    np.testing.assert_array_equal(result.pilot, whole)


def test_an_explicit_k_of_zero_gets_a_lag_of_one():
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=4)
    with pytest.warns(ConvergenceWarning, match="met after k \\+ lag"):  # with k = 0, all do
        sampler.run(2, k=0)
    assert (sampler.settings.lag, sampler.settings.m) == (1, 0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: CoupledSampler(log_prob, INIT, kernel=KERNEL).meeting_times(0), "n must be"),
        (
            lambda: CoupledSampler(log_prob, INIT, kernel=RandomWalk(np.eye(3))),
            "the kernel works in 3 dimensions, but the initial points have 2",
        ),
    ],
)
def test_invalid_arguments_raise_before_any_run(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_sample_refuses_fewer_than_two_replicates():
    # One replicate was tuned and run, and its result could give no estimate.
    def never(x):
        raise AssertionError("sample() did work before checking n_replicates")

    with pytest.raises(ValueError, match="n_replicates must be at least 2, not 1"):
        sample(never, INIT, 1)


def test_unknown_kernel_names_are_listed():
    with pytest.raises(ValueError, match="'auto', 'rwmh', 'mala', 'hmc', 'independence'"):
        CoupledSampler(log_prob, INIT, kernel="nuts")
    with pytest.raises(TypeError, match="kernel must be a Kernel"):
        CoupledSampler(log_prob, INIT, kernel=RandomWalk)


@pytest.mark.parametrize(("kernel", "kind"), [("mala", MALA), ("hmc", HMC)])
def test_gradient_kernels_are_tuned_by_name(kernel, kind):
    result = sample(log_prob, INIT, 5, grad=lambda x: -x, kernel=kernel, k=100, rng=13)
    assert isinstance(result.kernel, kind)
    assert result.kernel.is_tuned
    assert result.n_replicates == 5


def conditional(point, block):
    return Conditional(0.8 * point[1 - block], 0.6)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"kernel": "mala"}, "needs the gradient"),
        ({"log_prob": None}, "needs a log-density"),
        (
            {
                "log_prob": None,
                "kernel": Mixture(
                    [Gibbs(conditional, [[0], [1]]), RandomWalk(np.eye(2))], [0.5, 0.5]
                ),
            },
            "needs a log-density",
        ),
    ],
)
def test_kernels_without_what_they_need_raise(arguments, message):
    with pytest.raises(ValueError, match=message):
        CoupledSampler(**({"log_prob": log_prob, "init": INIT} | arguments))


@pytest.mark.parametrize(
    "kernel",
    [
        Gibbs(conditional, [[0], [1]]),
        Mixture(
            [Gibbs(conditional, [[0], [1]]), Gibbs(conditional, [[0], [1]], coupling="crn")],
            [0.5, 0.5],
        ),
    ],
    ids=["gibbs", "mixture"],
)
def test_gibbs_samplers_and_their_mixtures_run_without_a_log_density(kernel):
    assert sample(None, INIT, 5, kernel=kernel, k=20, rng=14).n_replicates == 5


def test_a_slow_pair_warns_once_when_unmet_at_the_threshold():
    # A pair still unmet at t = 10^4 meets after it, as diagnostics.issues counts slow.
    slow = _SlowPairs(lag=1, m=10, kind="replicate")
    slow(Status(0, 3, 9_999))
    with pytest.warns(ConvergenceWarning, match="replicate 3 has not met after 10000"):
        slow(Status(0, 3, 10_000))
    slow(Status(0, 3, 10_001))  # a second warning would fail under -W error


def test_a_slow_pair_seen_as_it_happens_says_that_the_run_waits_for_it():
    with pytest.warns(ConvergenceWarning, match="pilot pair 3 .* the run waits for it; the"):
        _SlowPairs(lag=1, m=10, kind="pilot pair")(Status(0, 3, 10_000))


def test_a_slow_replicate_names_ctrl_c_only_while_its_handler_listens():
    interrupt = Interrupt()
    clause = "waits for it: a first Ctrl-C starts no new replicate but lets this one finish"
    with interrupt.catching(), pytest.warns(ConvergenceWarning, match=clause):
        _SlowPairs(1, 10, "replicate", interrupt)(Status(0, 3, 10_000))
    # Outside its block, as in a thread or under a host that keeps its own handler, and
    # after a first Ctrl-C, the clause would not be true.
    with pytest.warns(ConvergenceWarning) as caught:
        _SlowPairs(1, 10, "replicate", interrupt)(Status(0, 3, 10_000))
    interrupt.stopping = True
    with interrupt.catching(), pytest.warns(ConvergenceWarning) as stopped:
        _SlowPairs(1, 10, "replicate", interrupt)(Status(0, 3, 10_000))
    assert "Ctrl-C" not in str(caught[0].message) + str(stopped[0].message)


def test_a_pair_censored_at_the_threshold_warns_as_slow_after_a_run():
    # The engine frees a censored pair's lane before its status is read, so only the
    # check after the run sees a pair censored at t = 10^4.
    replicates = synthetic([2, -1], [0.0], m=10, max_iter=10_000)._replicates
    with pytest.warns(ConvergenceWarning, match="replicate 1 has not met after 10000"):
        _SlowPairs(lag=1, m=10, kind="replicate").after(replicates)


def test_tuning_chains_in_separate_modes_warn():
    starts = itertools.cycle([[-20.0], [20.0]])

    def two_modes(x):
        return np.logaddexp(-0.5 * (x[0] + 20) ** 2, -0.5 * (x[0] - 20) ** 2)

    sampler = CoupledSampler(two_modes, lambda rng: next(starts), rng=12)
    with pytest.warns(ConvergenceWarning, match="or the chains may not have converged"):
        sampler.tune()


def test_a_run_with_censored_pairs_warns():
    kernel = RandomWalk(np.eye(2), scale=0.1)
    with pytest.warns(ConvergenceWarning) as record:  # censored pairs are also late
        sample(log_prob, INIT, 20, kernel=kernel, k=1, m=2, max_iter=3, rng=17)
    assert any("reached max_iter = 3" in str(warning.message) for warning in record)


def half_normal(x):
    return -0.5 * x[0] ** 2 if x[0] > 0 else -np.inf


@pytest.mark.parametrize(
    ("call", "kind"),
    [
        (lambda sampler: sampler.run(6, k=40), "replicates"),
        (lambda sampler: sampler.meeting_times(6), "pilot pairs"),
    ],
    ids=["replicates", "pilot"],
)
def test_starts_outside_the_support_warn(call, kind):
    sampler = CoupledSampler(
        half_normal, Empirical([[-1.0], [1.0]]), kernel=RandomWalk(np.eye(1)), rng=9
    )
    with pytest.warns(InputWarning, match=f"{kind} start where the target density is zero"):
        call(sampler)


def test_a_zero_density_start_warns_before_the_run_ends():
    # The log-density counts its calls, and the warning records the count when it comes.
    calls = []

    def counting(x):
        calls.append(1)
        return half_normal(x)

    seen = []
    sampler = CoupledSampler(
        counting, Empirical([[-1.0], [1.0]]), kernel=RandomWalk(np.eye(1)), rng=9
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = lambda message, *_, **__: seen.append((str(message), len(calls)))
        sampler.run(6, k=40)
    outside = [n for message, n in seen if "start where the target density is zero" in message]
    assert outside
    assert outside[0] < len(calls)


def test_choose_k_failure_names_zero_density_starts():
    # From -50 a random-walk chain on the half-normal never proposes a point of positive
    # density, so the pairs that start there never meet within the pilot's cap.
    sampler = CoupledSampler(
        half_normal, Empirical([[-50.0], [1.0]]), kernel=RandomWalk(np.eye(1)), rng=9
    )
    with pytest.warns(InputWarning):
        sampler.meeting_times(100, max_iter=20)
    with pytest.raises(ValueError, match=r"pilot pairs never met .* of the 100 started where"):
        sampler.run(2)


def tuned_sampler(kernel):
    sampler = CoupledSampler(
        log_prob, INIT, grad=lambda x: -x, kernel=kernel, check_grad=False, rng=0
    )
    sampler.tune()
    return sampler


@pytest.mark.parametrize(
    ("kernel", "noted"),
    [("hmc", True), ("mala", True), ("rwmh", False), (MALA(0.5, cov=np.eye(2)), False)],
    ids=["hmc", "mala", "rwmh", "mala with its parameters"],
)
def test_choose_k_failure_points_tuned_gradient_kernels_to_their_step_size(kernel, noted):
    # The automatic steps of MALA and HMC can fail to couple; a kernel without a gradient,
    # or given with every parameter, has none.
    with pytest.raises(ValueError, match="never met within their cap") as raised:
        tuned_sampler(kernel)._choose_k(np.full(100, np.inf))
    notes = getattr(raised.value, "__notes__", [])
    assert any("step_size checked on pilot pairs" in note for note in notes) == noted


def test_choose_k_failure_at_zero_density_starts_carries_no_note_on_the_step_size():
    sampler = tuned_sampler("hmc")
    sampler._pilot_zero_density = 1
    with pytest.raises(ValueError, match="started where the target density is zero") as raised:
        sampler._choose_k(np.full(100, np.inf))
    assert not hasattr(raised.value, "__notes__")
    assert not hasattr(raised.value.__cause__, "__notes__")


def slow_pair_from_zero_density_as_it_happens(slow):
    slow(Status(0, 1, 10_000, 1, 1, True))


def slow_pair_from_zero_density_after_a_run(slow):
    replicates = synthetic([2, 20_000], [0.0, 1.0])._replicates
    replicates.zero_density = np.array([1])
    slow.after(replicates)


@pytest.mark.parametrize(
    "seen",
    [slow_pair_from_zero_density_as_it_happens, slow_pair_from_zero_density_after_a_run],
    ids=["as it happens", "after a run"],
)
def test_a_slow_pair_that_started_at_zero_density_says_so(seen):
    with pytest.warns(ConvergenceWarning, match="replicate 1 .* it started where the target"):
        seen(_SlowPairs(lag=1, m=10, kind="replicate"))


def test_a_run_that_would_store_gigabytes_warns():
    # Without building the averaging set, whose 10^18 times would not fit in memory.
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=10)
    with pytest.warns(InputWarning, match="GB of atoms"):
        sampler._check_memory(Settings(0, 10**18, 1), 1)


def test_the_memory_warning_compares_the_peak_with_the_threshold():
    # A pilot pair meeting at 1001 stands for 1000 distances of 24 bytes and 2001 atoms
    # of 36 bytes at k = 0: 0.48 GB stored for 5,000 replicates, of which the atoms are
    # 0.36, and a peak of three times that, up to twice as much in two dimensions.
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=10)
    sampler.pilot = np.full(100, 1001.0)
    with pytest.warns(InputWarning, match="store about 0.5 GB .* take about 1.4 to 2.9 GB"):
        sampler._check_memory(Settings(0, 0, 1), 5_000)


def test_the_readme_example_runs_without_a_warning(capsys):
    # Its code block as written; an earlier test ran the first draft's example, which
    # the README and the spec replaced (DECISIONS #35). Warnings are errors here.
    pytest.importorskip("matplotlib").use("Agg")
    text = (Path(__file__).parents[1] / "README.md").read_text()
    code = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        script = Path(directory) / "readme.py"
        script.write_text(code)
        runpy.run_path(str(script))
    pytest.importorskip("matplotlib.pyplot").close("all")
    assert "slope" in capsys.readouterr().out


def test_citation_version_matches_package_version():
    # The version is written in both files; the test group has no YAML parser.
    text = (Path(__file__).parents[1] / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*[\"']?([^\"'\s]+)", text, re.MULTILINE)
    assert match, "CITATION.cff has no version line"
    assert match.group(1) == __version__


def test_the_package_exports_what_the_readme_uses():
    # A name left out of __all__ is left out of `from telescopic import *`.
    text = (Path(__file__).parents[1] / "README.md").read_text()
    code = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1)
    used = set(re.findall(r"\btelescopic\.(\w+)", code))
    assert used, "the README's example takes no name from telescopic"
    assert used | {"__version__"} <= set(telescopic.__all__)


# The rows of the autosummary tables of docs/api.md, which the tests read; the strict build
# of the documentation checks that each row generates a page. A module's page shows the
# rest of its __all__ and links the pages of its classes.
API_PAGES = re.findall(
    r"^ {3}~?(telescopic[\w.]*)$",
    (Path(__file__).parents[1] / "docs" / "api.md").read_text(),
    re.MULTILINE,
)
SUBMODULES = {
    f"telescopic.{module.name}"
    for module in pkgutil.iter_modules(telescopic.__path__)
    if not module.name.startswith("_")
}
DOCUMENTED_MODULES = [name for name in API_PAGES if name in SUBMODULES]


def _objects_with_pages():
    return [
        getattr(importlib.import_module(module), attribute)
        for module, _, attribute in (
            page.rpartition(".") for page in API_PAGES if page not in SUBMODULES
        )
    ]


def test_every_public_module_has_a_page_or_its_names_have_pages():
    # A module without a row, such as telescopic.results, is shown by the pages of its
    # names, which telescopic exports; a name of its __all__ without one is missing.
    pages = _objects_with_pages()
    missing = []
    for name in sorted(SUBMODULES - set(API_PAGES)):
        module = importlib.import_module(name)
        missing += [
            f"{name}.{key}"
            for key in module.__all__
            if not any(getattr(module, key) is page for page in pages)
        ]
    assert not missing


@pytest.mark.parametrize("name", DOCUMENTED_MODULES)
def test_a_documented_module_shows_every_class_and_function_it_defines(name):
    # A name left out of __all__ is missing from its module's page, and a class of
    # __all__ left out of the index has no page.
    try:
        module = importlib.import_module(name)
    except ImportError as error:  # telescopic.plotting without matplotlib
        pytest.skip(str(error))
    defined = {
        key
        for key, value in vars(module).items()
        if not key.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__ == name
    }
    if name == "telescopic.warnings":
        # The helper that emits the package's warnings is internal, and stays out of __all__.
        defined.remove("warn")
    assert defined <= set(module.__all__) <= set(vars(module))
    classes = {f"{name}.{key}" for key in module.__all__ if inspect.isclass(vars(module)[key])}
    assert classes <= set(API_PAGES)


def test_the_api_reference_has_a_page_for_every_public_name():
    pages = _objects_with_pages()
    public = [getattr(telescopic, name) for name in telescopic.__all__ if name != "__version__"]
    assert {value.__name__ for value in public if inspect.ismodule(value)} <= set(API_PAGES)
    assert all(value in pages for value in public if not inspect.ismodule(value))


def test_the_package_is_marked_as_typed():
    # PEP 561: without the marker, a user's type checker treats telescopic as untyped.
    assert (resources.files("telescopic") / "py.typed").is_file()


def test_import_does_not_load_scipy_stats_optimize_or_integrate():
    # Each worker of a process pool imports the package before its first chunk, and these
    # three subpackages took most of that import's time. A fresh interpreter, since this
    # one has imported them, on this one's copy of the package. It also couples a pair,
    # whose test for SciPy's Dirichlet must not import scipy.stats.
    code = """
import sys
from telescopic import couplings, examples, testing
gamma = examples._Gamma(2.0, 1.0)
couplings.maximal([gamma], [gamma], testing.random_draws(1, 0))
print(sorted(set(sys.modules) & set(sys.argv[1:])))
"""
    loaded = subprocess.run(
        [sys.executable, "-c", code, "scipy.stats", "scipy.optimize", "scipy.integrate"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(telescopic.__file__).parents[1])},
    ).stdout
    assert loaded == "[]\n"


def test_chains_cannot_start_where_the_gradient_is_not_finite():
    # MALA chains that started there stayed at its edge: on a mixture whose gradient
    # underflows beyond 40, from N(45, 1), E[x] was 40.10 +- 0.03 (DECISIONS #67).
    def grad(x):
        return np.where(x > 1, np.nan, -x)

    # The gradient check, which skips such points, says first that it compared nothing.
    with (
        pytest.raises(ValueError, match="where the log-density is finite: chains cannot start"),
        pytest.warns(InputWarning, match="compared no line"),
    ):
        sample(
            log_prob, Normal([3.0], [0.01]), 10, grad=grad, kernel=MALA(1.0, cov=np.eye(1)), k=5
        )


def test_a_run_warns_when_it_first_takes_a_gradient_as_zero():
    # It warned after the run, so that a run whose pairs never met gave no sign: the count
    # in the warning is the run's so far.
    def grad(x):
        return np.where(x > 1, np.nan, -x)

    kernel = MALA(1.0, cov=np.eye(1))
    pattern = r"taken as 0 in \d+ of (\d+) evaluations of the replicates"
    with pytest.warns(InputWarning, match=pattern) as caught:
        result = sample(log_prob, Normal([0.0], [0.01]), 20, grad=grad, kernel=kernel, k=10, rng=1)
    message = next(str(w.message) for w in caught if "taken as 0" in str(w.message))
    evaluated = int(re.search(pattern, message).group(1))
    assert 0 < evaluated < result.n_grad
    assert result.n_grad_zeroed > 0


def test_a_sampler_with_a_gradient_kernel_checks_grad_unless_told_not_to():
    def wrong(x):
        return np.asarray(x)  # the sign of the standard normal's gradient flipped

    kernel = MALA(0.5, cov=np.eye(2))
    counted = Counted(log_prob), Counted(wrong)
    with pytest.warns(InputWarning, match="grad disagrees"):
        sampler = CoupledSampler(
            counted[0], INIT, grad=counted[1], vectorize=True, kernel=kernel, rng=7
        )
    assert sampler.gradient_check.ratio.max() > 1
    assert (counted[0].points, counted[1].points) == (40, 8)
    counted = Counted(log_prob), Counted(wrong)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = CoupledSampler(
            counted[0],
            INIT,
            grad=counted[1],
            vectorize=True,
            kernel=kernel,
            check_grad=False,
            rng=7,
        )
    assert sampler.gradient_check is None
    assert (counted[0].points, counted[1].points) == (0, 0)


@pytest.mark.parametrize(
    "kernel",
    [MALA(), HMC(), MALA(0.5, cov=np.eye(2)), HMC(0.3, cov=np.eye(2))],
    ids=["MALA", "HMC", "MALA given", "HMC given"],
)
def test_the_gradient_check_changes_no_result(kernel):
    runs, counts = [], []
    for check_grad in (True, False):
        counted = Counted(log_prob), Counted(lambda x: -x)
        runs.append(
            sample(
                counted[0],
                INIT,
                10,
                grad=counted[1],
                vectorize=True,
                kernel=kernel,
                k=50,
                check_grad=check_grad,
                rng=np.random.default_rng(3),
            )
        )
        counts.append(np.array([counted[0].points, counted[1].points]))
    np.testing.assert_array_equal(runs[0].meeting_times, runs[1].meeting_times)
    np.testing.assert_array_equal(runs[0].measure.atoms, runs[1].measure.atoms)
    np.testing.assert_array_equal(runs[0].measure.weights, runs[1].measure.weights)
    np.testing.assert_array_equal(counts[0] - counts[1], [40, 8])  # the check, once


def test_the_gradient_check_passes_args_and_kwargs():
    def shifted(x, shift, *, scale):
        return -0.5 * np.sum(((x - shift) / scale) ** 2, axis=-1)

    def shifted_grad(x, shift, *, scale):
        return -(x - shift) / scale**2

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = CoupledSampler(
            shifted,
            INIT,
            grad=shifted_grad,
            args=(np.array([1.0, 2.0]),),
            kwargs={"scale": 3.0},
            vectorize=True,
            kernel=MALA(0.5, cov=np.eye(2)),
            rng=9,
        )
    assert len(sampler.gradient_check.gap) == 8


def test_the_gradient_check_runs_when_the_kernel_evaluates_grad():
    given = MALA(0.5, cov=np.eye(2))
    for kernel, vectorize in ((Mixture([KERNEL, given], [0.5, 0.5]), True), (given, False)):
        sampler = CoupledSampler(
            log_prob, INIT, grad=lambda x: -x, vectorize=vectorize, kernel=kernel, rng=8
        )
        assert len(sampler.gradient_check.gap) == 8
    assert CoupledSampler(log_prob, INIT, grad=lambda x: -x, kernel=KERNEL).gradient_check is None


def test_meeting_times_refuses_max_iter_none():
    # None ran an uncapped pilot, whose pairs then chose k.
    with pytest.raises(TypeError, match="max_iter must be an integer, not None"):
        CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=0).meeting_times(5, max_iter=None)


def test_a_names_length_mismatch_names_the_initial_dimension():
    # It blamed names, "names must have 1 entries, not 2", for an initial distribution of
    # the wrong dimension.
    with pytest.raises(ValueError, match="draws points of dimension 1, but names gives 2"):
        CoupledSampler(log_prob, Normal([0.0], [1.0]), names=["p", "lam"])


def truncated_log_prob(x):
    # The standard normal on x0 > -1, of a point or of rows of points.
    return np.where(x[..., 0] > -1, -0.5 * np.sum(np.square(x), axis=-1), -np.inf)


def truncated_grad(x):
    # Not finite where x1 > 2, so that some gradients are taken as 0 and counted.
    return np.where(x[..., 1:] > 2, np.nan, -x)


def truncated_log_prob_and_grad(x):
    return truncated_log_prob(x), truncated_grad(x)


@pytest.mark.filterwarnings("ignore::telescopic.warnings.TelescopicWarning")
@pytest.mark.parametrize("vectorize", [True, False], ids=["vectorized", "point by point"])
@pytest.mark.parametrize("kernel", ["mala", "hmc"])
def test_combined_callable_matches_separate_log_prob_and_grad(kernel, vectorize):
    # Tuned in the run, so that tuning's pilot chains, which call log_prob alone, and its
    # dual averaging, which calls log_prob_and_grad, are compared too.
    runs = [
        CoupledSampler(
            truncated_log_prob,
            Normal([0.0, 0.0], [1.0, 0.5]),
            vectorize=vectorize,
            kernel=kernel,
            rng=3,
            **gradient,
        )
        for gradient in (
            {"grad": truncated_grad},
            {"log_prob_and_grad": truncated_log_prob_and_grad},
        )
    ]
    for name, value in vars(runs[0].gradient_check).items():
        np.testing.assert_array_equal(getattr(runs[1].gradient_check, name), value, err_msg=name)
    separate, combined = (sampler.run(10, k=3, m=10, lag=3) for sampler in runs)
    assert str(combined.kernel) == str(separate.kernel)
    assert_same_replicates(combined._replicates, separate._replicates)
    # Points of zero density, whose gradients are not counted, and gradients taken as 0.
    assert combined.n_log_prob > combined.n_grad
    assert combined.n_grad_zeroed > 0


# Pairs this short may meet late, which warns.
@pytest.mark.filterwarnings("ignore::telescopic.warnings.ConvergenceWarning")
@pytest.mark.parametrize("check_grad", [False, True], ids=["unchecked", "checked"])
@pytest.mark.parametrize(
    "kernel",
    [
        MALA(0.8, cov=np.eye(2)),
        HMC(0.3, 5, cov=np.eye(2)),
        Mixture([MALA(0.8, cov=np.eye(2)), RandomWalk(np.eye(2), scale=1.0)], [0.5, 0.5]),
    ],
    ids=["mala", "hmc", "mixture"],
)
def test_a_run_with_log_prob_and_grad_takes_its_values_from_it_alone(kernel, check_grad):
    # Kernels given with their parameters, and k, m and lag, so that nothing is tuned:
    # tuning's pilot chains are the one part of a run that calls log_prob. Every other
    # evaluation, the gradient check's included, comes from log_prob_and_grad.
    result = sample(
        never_called,
        Normal([0.0, 0.0], [1.0, 1.0]),
        4,
        log_prob_and_grad=lambda x: (gaussian_log_prob(x), -x),
        kernel=kernel,
        k=2,
        m=4,
        lag=2,
        check_grad=check_grad,
        rng=0,
    )
    assert result.n_log_prob == result.n_grad > 0


def failed_sampler(at=30_000, n=600):
    # A sampler whose run of n replicates raised at evaluation `at`, and that error.
    failing = Failing(at)
    sampler = CoupledSampler(failing, INIT, kernel=KERNEL, rng=7)
    with pytest.raises(ValueError, match=f"failed at evaluation {at}") as caught:
        sampler.run(n, k=40, m=40)
    assert caught.value is failing.error
    return sampler, caught.value


def test_sampler_keeps_finished_replicates_and_runs_the_missing_ones_next():
    sampler, error = failed_sampler()
    kept = error.telescopic_result
    assert 0 < len(kept.index) < 600
    assert_same_replicates(kept._replicates, sampler._replicates)
    notes = [note for note in error.__notes__ if note.startswith("telescopic:")]
    assert len(notes) == 1
    assert f"next run starts with the {600 - len(kept.index)} missing ones" in notes[0]
    completed = sampler.run(600 - len(kept.index))
    whole = sample(gaussian_log_prob, INIT, 600, kernel=KERNEL, k=40, m=40, rng=7)
    assert_same_outputs(completed._replicates, whole._replicates)


def test_abandoned_replicates_are_flagged_and_cleared_when_run():
    sampler, error = failed_sampler()
    kept = error.telescopic_result
    assert len(kept.abandoned)
    assert kept.interrupted
    summary = kept.summary()
    assert summary.footer["abandoned"] == len(kept.abandoned)
    assert f"an error abandoned {len(kept.abandoned)} replicates in progress" in repr(summary)
    assert "interrupted" not in repr(summary)
    completed = sampler.run(600 - len(kept.index))
    assert not len(completed.abandoned)
    assert not completed.interrupted
    assert completed.summary().footer["abandoned"] == 0


@pytest.mark.parametrize("stage", ["meeting_times", "run"])
def test_a_pilot_error_keeps_no_replicates(stage):
    # Evaluation 1,000 falls inside the 100 lag-1 pilot pairs of an automatic k.
    failing = Failing(1_000)
    sampler = CoupledSampler(failing, INIT, kernel=KERNEL, rng=7)
    with pytest.raises(ValueError, match="failed at evaluation") as caught:
        getattr(sampler, stage)(100)
    assert caught.value is failing.error
    assert not hasattr(caught.value, "telescopic_result")
    assert not getattr(caught.value, "__notes__", [])
    assert sampler.pilot is None
    assert sampler.settings is None


def test_keep_is_fixed_by_the_first_run():
    sampler = CoupledSampler(gaussian_log_prob, INIT, kernel=KERNEL, rng=3)
    first = sampler.run(30, k=40, m=40, keep=10)
    assert (first.keep, first.n_kept) == (10, 10)
    second = sampler.run(30)
    assert second.keep == 10
    assert second.n_kept == 10
    assert second.n_replicates == 60
    with pytest.raises(ValueError, match=r"keep was 10 in the earlier runs; .* for keep=5"):
        sampler.run(10, keep=5)
    with pytest.raises(TypeError, match="keep must be None or an integer >= 0"):
        CoupledSampler(gaussian_log_prob, INIT, kernel=KERNEL).run(10, k=40, keep=1.5)
    reduced = sample(gaussian_log_prob, INIT, 30, kernel=KERNEL, k=40, m=40, keep=10, rng=3)
    assert_same_replicates(reduced._replicates, first._replicates)


def test_the_memory_warning_counts_the_moments_of_every_replicate():
    # At d = 100, 8 (2 d + d (d + 1) / 2) = 42,000 bytes of moments per replicate: 4.2 GB
    # for 100,000 replicates, though keep=0 stores no atom.
    sampler = CoupledSampler(gaussian_log_prob, Normal(np.zeros(100), np.eye(100)), rng=10)
    with pytest.warns(InputWarning, match=r"store about 4\.2 GB of atoms, distances and moments"):
        sampler._check_memory(Settings(0, 0, 1, keep=0), 100_000)
