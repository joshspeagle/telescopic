import itertools

import numpy as np
import pytest
from scipy import stats

import telescopic
from telescopic import CoupledSampler, sample
from telescopic.engine import Status
from telescopic.initial import Empirical, Normal
from telescopic.kernels import RandomWalk
from telescopic.sampler import _slow_pairs
from telescopic.warnings import ConvergenceWarning, InputWarning

INIT = Normal([2.0, -1.0], [4.0, 4.0])
KERNEL = RandomWalk(np.eye(2))


def log_prob(x):
    return -0.5 * float(np.sum(x**2))


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
    sampler.run(5, k=4)
    sampler.run(5)
    with pytest.raises(ValueError, match="k was 4"):
        sampler.run(5, k=5)


@pytest.mark.parametrize(
    ("given", "expected"), [({"k": 6}, (6, 6, 60)), ({"k": 6, "m": 40, "lag": 2}, (6, 2, 40))]
)
def test_an_explicit_k_needs_no_pilot_and_sets_the_lag_and_m(given, expected):
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=4)
    result = sampler.run(5, **given)
    assert (result.k, result.lag, result.m) == expected
    assert sampler.pilot is None


@pytest.mark.parametrize(
    "init",
    [
        INIT,
        stats.multivariate_normal([0.0, 0.0], np.eye(2)),
        lambda rng: rng.normal(size=2),
        np.random.default_rng(5).standard_normal((6, 2)),
    ],
    ids=["helper", "scipy", "callable", "points"],
)
def test_every_form_of_initial_distribution_runs(init):
    assert sample(log_prob, init, 4, kernel=KERNEL, k=2, rng=6).n_replicates == 4


def test_a_seed_reproduces_a_run_and_a_generator_moves_on():
    first, second = (sample(log_prob, INIT, 5, kernel=KERNEL, k=3, rng=7) for _ in range(2))
    np.testing.assert_array_equal(first.measure.atoms, second.measure.atoms)
    rng = np.random.default_rng(8)
    third, fourth = (sample(log_prob, INIT, 5, kernel=KERNEL, k=3, rng=rng) for _ in range(2))
    assert not np.array_equal(third.measure.atoms, fourth.measure.atoms)


def test_unknown_kernel_names_are_listed():
    with pytest.raises(ValueError, match="'auto', 'rwmh', 'independence'"):
        CoupledSampler(log_prob, INIT, kernel="nuts")


def test_a_slow_pair_warns_once():
    monitor = _slow_pairs(lag=1, m=10)
    assert not monitor(Status(1, 0, 3, 10_000))
    with pytest.warns(ConvergenceWarning, match="replicate 3 has not met after 10000"):
        monitor(Status(1, 0, 3, 10_001))
    monitor(Status(1, 0, 3, 10_002))  # a second warning would fail under -W error


def test_tuning_chains_in_separate_modes_warn():
    starts = itertools.cycle([[-20.0], [20.0]])

    def two_modes(x):
        return np.logaddexp(-0.5 * (x[0] + 20) ** 2, -0.5 * (x[0] - 20) ** 2)

    sampler = CoupledSampler(two_modes, lambda rng: next(starts), rng=12)
    with pytest.warns(ConvergenceWarning, match="tuning chains disagree"):
        sampler.tune()


def test_starts_outside_the_support_warn():
    def half_line(x):
        return 0.0 if x[0] > 0 else -np.inf

    with pytest.warns(InputWarning, match="target density is zero"):
        sample(half_line, Empirical([[-1.0], [1.0]]), 6, kernel=RandomWalk(np.eye(1)), k=2, rng=9)


def test_a_run_that_would_store_gigabytes_warns():
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=10)
    with pytest.warns(InputWarning, match="GB of atoms"):
        sampler._check_memory({"k": 0, "m": 10**6, "thin": 1}, 1000)


def test_the_specification_example_runs():
    # Spec section 8.3, with synthetic data.
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(0, 10, 20))
    yerr = 0.5 + rng.random(20)
    y = 1.5 * x - 2.0 + yerr * rng.standard_normal(20)

    def line(theta, x, y, yerr):
        slope, intercept = theta
        return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2)

    init = telescopic.initial.Normal(mean=[0.0, 0.0], cov=np.diag([10.0, 10.0]))
    result = telescopic.sample(line, init, args=(x, y, yerr), names=["slope", "intercept"], rng=42)
    assert "slope" in repr(result.summary())
    assert np.isfinite(result.estimate(lambda t: t[:, 0] * t[:, 1]).value)
