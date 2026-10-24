import numpy as np
import pytest

import references
from telescopic.initial import Normal
from telescopic.kernels import HMC, MALA, Independence, Kernel, Mixture, RandomWalk
from telescopic.target import Target
from telescopic.testing import random_draws
from telescopic.tuning import (
    TARGET_ACCEPTANCE,
    PilotRun,
    adaptive_metropolis,
    choose_k,
    dual_averaging,
)

# A correlated Gaussian with condition number 100.
_q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((5, 5)))
COV = _q @ np.diag(np.logspace(0, 2, 5)) @ _q.T
PRECISION = np.linalg.inv(COV)
TARGET = Target(
    lambda x: -0.5 * np.einsum("ij,jk,ik->i", x, PRECISION, x),
    grad=lambda x: -x @ PRECISION,
    vectorize=True,
)
START = Normal(np.zeros(5), 4 * np.eye(5))


def test_adaptive_metropolis_learns_the_shape_and_the_acceptance_rate():
    pilot = adaptive_metropolis(TARGET, START, 1)
    # Whitened by the pilot's covariance, the target's condition number falls from 100.
    inverse = np.linalg.inv(np.linalg.cholesky(pilot.cov))
    eigenvalues = np.linalg.eigvalsh(inverse @ COV @ inverse.T)
    assert eigenvalues.max() / eigenvalues.min() < 4
    np.testing.assert_allclose(pilot.acceptance, TARGET_ACCEPTANCE, atol=0.05)
    assert pilot.rhat.max() < 1.1


def test_the_tuning_chains_agree_and_find_the_covariance_in_ten_dimensions():
    # Chains adapting alone at the rate t^-0.6 gave R-hat 2 to 3 and a smallest
    # eigenvalue of 0.25 here; over 20 seeds the pooled chains gave at most 1.09 and at
    # least 0.58.
    target = Target(lambda x: -0.5 * np.sum(x**2, axis=1), vectorize=True)
    pilot = adaptive_metropolis(target, Normal(np.full(10, 3.0), np.eye(10)), 4)
    assert pilot.rhat.max() < 1.15
    assert np.linalg.eigvalsh(pilot.cov).min() > 0.4


def test_n_steps_is_checked_before_drawing_from_the_initial_distribution():
    calls = []

    def initial(rng):
        calls.append(rng)
        return rng.normal(size=5)

    with pytest.raises(ValueError, match="n_steps must be at least 3"):
        adaptive_metropolis(TARGET, initial, n_steps=2)
    assert calls == []


def test_chains_that_never_move_raise():
    nowhere = Target(lambda x: np.full(len(x), -np.inf), vectorize=True)
    with pytest.raises(ValueError, match="accepted no proposal"):
        adaptive_metropolis(nowhere, START, 1, n_steps=10)


def test_reports_of_different_pilots_compare_without_error():
    first, second = (adaptive_metropolis(TARGET, START, seed, n_steps=10) for seed in (5, 6))
    assert first.report() != second.report()


def test_tuning_freezes_kernels_with_the_pilots_mean_and_covariance():
    pilot = adaptive_metropolis(TARGET, START, 2, n_steps=500)
    walk, _ = RandomWalk().tune(TARGET, START, pilot=pilot)
    np.testing.assert_array_equal(walk.cov, pilot.cov)
    assert RandomWalk(scale=0.1).tune(TARGET, START, pilot=pilot)[0].scale == 0.1
    independence, report = Independence().tune(TARGET, START, pilot=pilot)
    np.testing.assert_array_equal(independence.proposal.mean, pilot.mean)
    np.testing.assert_array_equal(independence.proposal.cov, pilot.cov)
    assert report.max_weight_ratio >= 1
    mixture, mixed = Mixture([RandomWalk(), Independence()], [0.5, 0.5]).tune(
        TARGET, START, pilot=pilot
    )
    np.testing.assert_array_equal(mixture.kernels[0].cov, pilot.cov)
    np.testing.assert_array_equal(mixture.kernels[1].proposal.mean, pilot.mean)
    assert mixed.max_weight_ratio == report.max_weight_ratio


def test_a_tuned_kernel_returns_itself():
    kernel = RandomWalk(np.eye(5))
    assert kernel.tune(TARGET, START) == (kernel, None)


class Exact(Kernel):
    # Accepts with probability exp(-step_size), so that dual averaging is deterministic.
    def __init__(self, step_size):
        self.step_size = step_size

    def step(self, target, state, draws):
        return state, np.full(len(state), np.exp(-self.step_size))

    def coupled_step(self, target, state_x, state_y, draws):
        return state_x, state_y


def test_dual_averaging_matches_hoffman_and_gelman():
    pilot = PilotRun(np.zeros((4, 1, 5)), np.zeros((4, 1)), np.ones(4), 1)
    step_size = dual_averaging(Exact, TARGET, pilot, 7, 0.574, 0.9)
    expected = references.dual_averaging(lambda epsilon: np.exp(-epsilon), 0.9, 0.574)
    np.testing.assert_allclose(step_size, expected, rtol=1e-12)


@pytest.mark.parametrize(("kernel", "goal"), [(MALA(), 0.574), (HMC(), 0.8)])
def test_dual_averaging_reaches_the_target_acceptance(kernel, goal):
    # The bound is dual averaging's spread over seeds, up to 0.034 above the goal for
    # MALA (DECISIONS #13), plus 3 standard errors of this mean of 4,000 probabilities
    # (0.0055 for MALA, 0.0036 for HMC). The reference test above pins its arithmetic.
    tuned, _ = kernel.tune(TARGET, START, 7)
    trajectory = tuned.kernels[0] if isinstance(tuned, HMC) else tuned
    points = np.random.default_rng(8).multivariate_normal(np.zeros(5), COV, size=4000)
    start = trajectory.init_state(TARGET, points)
    _, acceptance = trajectory.step(TARGET, start, random_draws(4000, 9))
    assert abs(acceptance.mean() - goal) < 0.05


@pytest.mark.parametrize("lowest", [1, 4])
def test_choose_k_takes_the_inverted_cdf_quantile_of_tau_minus_one(lowest):
    # Of 100 pilot pairs, the 99% quantile is the 99th smallest meeting time, 30.
    assert choose_k(np.concatenate([np.full(98, lowest), [30, 50]])) == 29
    assert choose_k(np.full(100, lowest), quantile=0.5) == max(1, lowest - 1)


def test_choose_k_refuses_no_meeting_times():
    with pytest.raises(ValueError, match="non-empty"):
        choose_k([])


def test_choose_k_refuses_a_quantile_beyond_the_pilot():
    with pytest.raises(ValueError, match="run more pilot pairs"):
        choose_k(np.arange(1, 101), quantile=0.999)


def test_choose_k_refuses_a_quantile_on_a_censored_pair():
    tau = np.concatenate([np.arange(1, 99), [np.inf, np.inf]])
    with pytest.raises(ValueError, match="never met within their cap"):
        choose_k(tau)
