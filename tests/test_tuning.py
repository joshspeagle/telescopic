import numpy as np
import pytest

from telescopic.initial import Normal
from telescopic.kernels import Independence, Mixture, RandomWalk
from telescopic.target import Target
from telescopic.tuning import TARGET_ACCEPTANCE, adaptive_metropolis, choose_kml

# A correlated Gaussian with condition number 100.
_q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((5, 5)))
COV = _q @ np.diag(np.logspace(0, 2, 5)) @ _q.T
PRECISION = np.linalg.inv(COV)
TARGET = Target(lambda x: -0.5 * np.einsum("ij,jk,ik->i", x, PRECISION, x), vectorize=True)
START = Normal(np.zeros(5), 4 * np.eye(5))


def test_adaptive_metropolis_learns_the_shape_and_the_acceptance_rate():
    pilot = adaptive_metropolis(TARGET, START, 1)
    # Whitened by the pilot's covariance, the target's condition number falls from 100.
    inverse = np.linalg.inv(np.linalg.cholesky(pilot.cov))
    eigenvalues = np.linalg.eigvalsh(inverse @ COV @ inverse.T)
    assert eigenvalues.max() / eigenvalues.min() < 4
    np.testing.assert_allclose(pilot.acceptance, TARGET_ACCEPTANCE, atol=0.05)
    assert pilot.rhat.max() < 1.1


def test_tuning_freezes_a_kernel_with_the_pilots_covariance_and_scale():
    pilot = adaptive_metropolis(TARGET, START, 2, n_steps=500)
    for kernel in (RandomWalk(), Mixture([RandomWalk(), RandomWalk(scale=0.1)], [0.5, 0.5])):
        tuned, report = kernel.tune(TARGET, START, pilot=pilot)
        assert tuned.is_tuned
        np.testing.assert_array_equal(report.cov, pilot.cov)
    walk, _ = RandomWalk(scale=0.1).tune(TARGET, START, pilot=pilot)
    assert walk.scale == 0.1  # a given scale is kept
    tuned, report = Independence().tune(TARGET, START, pilot=pilot)
    np.testing.assert_array_equal(tuned.proposal.cov, pilot.cov)
    assert report.max_weight_ratio >= 1
    _, mixed = Mixture([RandomWalk(), Independence()], [0.5, 0.5]).tune(TARGET, START, pilot=pilot)
    assert mixed.max_weight_ratio == report.max_weight_ratio


def test_a_tuned_kernel_returns_itself():
    kernel = RandomWalk(np.eye(5))
    assert kernel.tune(TARGET, START) == (kernel, None)


@pytest.mark.parametrize("lowest", [1, 4])
def test_choose_kml_follows_the_recipe(lowest):
    tau = np.concatenate([np.full(98, lowest), [30, 50]])
    k = max(1, int(np.quantile(tau - 1, 0.99, method="inverted_cdf")))
    assert k == 29
    assert choose_kml(tau) == (k, 10 * k, k)
    assert choose_kml(np.full(100, lowest), quantile=0.5)[0] == max(1, lowest - 1)


def test_choose_kml_refuses_a_quantile_beyond_the_pilot():
    with pytest.raises(ValueError, match="run more pilot pairs"):
        choose_kml(np.arange(1, 101), quantile=0.999)


def test_choose_kml_refuses_a_quantile_on_a_censored_pair():
    tau = np.concatenate([np.arange(1, 99), [np.inf, np.inf]])
    with pytest.raises(ValueError, match="never met within their cap"):
        choose_kml(tau)
