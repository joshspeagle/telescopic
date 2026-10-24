import dataclasses
import functools
import itertools
from typing import Any, NamedTuple

import numpy as np
import pytest
from scipy import integrate, stats

from references import assert_z_scores, gaussian_log_prob
from telescopic import diagnostics, examples
from telescopic.couplings import reflection_maximal
from telescopic.engine import run
from telescopic.initial import Empirical, Normal
from telescopic.kernels import (
    HMC,
    MALA,
    Gibbs,
    Independence,
    Kernel,
    Mixture,
    MultivariateT,
    RandomWalk,
    State,
)
from telescopic.kernels.base import evaluate, evaluate_pair
from telescopic.results import Result
from telescopic.target import Target

pytestmark = pytest.mark.slow

G1, G5, G20 = (examples.gaussian(d, 0.5) for d in (1, 5, 20))
BANANA, PUMPS, MIXTURE = examples.banana(), examples.pumps(), examples.mixture()
# Far along the banana's ridge, x_1 = x_0^2, three standard deviations from E[x_0]. From
# U[-5, 5]^2, gradients near 6,000 make HMC's step of 0.02 unstable, Heng & Jacob's step
# of 0.002 needs 500 leapfrog steps, and the automatic step does not couple (QUESTIONS,
# "The automatic steps of HMC and MALA").
RIDGE = Normal([3.0, 9.0], 0.01 * np.eye(2))
# N(0, 1) whose gradient is NaN beyond 1, where the target has mass 0.16. Kernels that
# rejected proposals there stayed below 1 from -1 (DECISIONS #67).
BROKEN_GRADIENT = dataclasses.replace(G1, grad=lambda x: np.where(x > 1, np.nan, G1.grad(x)))


def far(d):
    # Centred three standard deviations from the target's mean in every coordinate.
    return Normal(np.full(d, 3.0), np.eye(d))


def point(d):
    return Empirical([[3.0] * d])


class Case(NamedTuple):
    # A run whose estimates of E[x] and E[x^2] are checked against the exact moments. The
    # replicates make a missing bias correction show at |z| of 5 or more in at least one
    # coordinate (notes/experiments/m9_unbiasedness.py).
    example: examples.Example
    kernel: Kernel
    initial: Any
    k: int
    m: int
    lag: int = 1
    thin: int = 1
    replicates: int = 1000


# Small k, so that the bias correction matters (test plan 2.12).
CASES = {
    "rwmh-d1": Case(G1, RandomWalk(G1.cov), far(1), k=1, m=10, replicates=1600),
    "rwmh-d5-point-lag7": Case(G5, RandomWalk(G5.cov), point(5), k=5, m=30, lag=7),
    "rwmh-d20-lag90-thin3": Case(
        G20, RandomWalk(G20.cov), far(20), k=90, m=300, lag=90, thin=3, replicates=1200
    ),
    "maximal-d1-point": Case(G1, RandomWalk(G1.cov, coupling="maximal"), point(1), k=1, m=10),
    "mala-d5-lag7": Case(G5, MALA(1.0, cov=G5.cov), far(5), k=2, m=20, lag=7),
    "hmc-banana-lag10": Case(BANANA, HMC(0.02, cov=np.eye(2)), RIDGE, k=10, m=100, lag=10),
    "gibbs-pumps": Case(
        PUMPS, Gibbs(PUMPS.conditional, PUMPS.blocks), PUMPS.initial, k=1, m=10, replicates=3200
    ),
    "independence-d5-point": Case(
        G5, Independence(MultivariateT(np.zeros(5), 2 * G5.cov)), point(5), k=1, m=10
    ),
    "mixture-d5": Case(
        G5, Mixture([RandomWalk(G5.cov), MALA(1.0, cov=G5.cov)], [0.5, 0.5]), far(5), k=2, m=20
    ),
    "mala-nan-gradient": Case(
        BROKEN_GRADIENT, MALA(1.0, cov=G1.cov), Empirical([[-1.0]]), k=5, m=50, lag=5
    ),
    "hmc-nan-gradient": Case(
        BROKEN_GRADIENT, HMC(0.5, 4, cov=G1.cov), Empirical([[-1.0]]), k=5, m=50, lag=5
    ),
}


def moments(x):
    return np.concatenate([x, x**2], axis=1)


def run_case(case, n, rng, **options):
    target = Target(case.example.log_prob, grad=case.example.grad, vectorize=True)
    settings = {"k": case.k, "m": case.m, "lag": case.lag, "thin": case.thin}
    return run(
        target, case.kernel, case.initial, range(n), rng=rng, lanes=256, **settings | options
    )


@pytest.mark.parametrize("name", CASES)
def test_estimates_are_unbiased(name):
    case = CASES[name]
    example = case.example
    truth = np.concatenate([example.mean, np.diag(example.cov) + example.mean**2])

    def z_scores(n, rng):
        estimate = Result(run_case(case, n, rng)).estimate(moments)
        return (estimate.value - truth) / estimate.standard_error

    assert_z_scores(z_scores, case.replicates, seed=list(CASES).index(name))


class AR1(Kernel):
    # x' = rho x + sqrt(1 - rho^2) xi leaves N(0, 1) invariant. A coupled step draws the
    # two moves from the reflection-maximal coupling of N(rho x, s^2) and N(rho y, s^2).
    def __init__(self, rho):
        self.rho, self.scale = rho, np.sqrt(1 - rho**2)
        self.chol, self.inverse = np.array([[self.scale]]), np.array([[1 / self.scale]])

    def step(self, target, state, draws):
        x = self.rho * state.x + self.scale * draws.normal(1)
        return State(x, target.log_prob(x)), np.ones(len(x))

    def coupled_step(self, target, state_x, state_y, draws):
        means = self.rho * state_x.x, self.rho * state_y.x
        x, y, identical = reflection_maximal(*means, self.chol, self.inverse, draws)
        return evaluate_pair(lambda points: evaluate(target, points, False), x, y, identical)


STANDARD_NORMAL = Target(gaussian_log_prob, vectorize=True)


def test_intervals_cover_at_the_reference_rate():
    # Intervals of R = 200 from a stationary start, under the recipe for this chain
    # (k = L = 38, m = 380). The reference coverages of x and x^2 come from 10^4 intervals
    # of an implementation written apart from the package
    # (notes/experiments/m9_coverage_reference.py).
    reference, reference_error = np.array([0.9509, 0.9489]), 0.0022
    per_interval, batch = 200, 4000

    def z_scores(n, rng):
        # Runs of 4,000 replicates bound the memory of the stored atoms, until the streaming
        # mode on the roadmap (spec section 6.6).
        seed = int(rng.integers(2**63))
        values = np.concatenate(
            [
                Result(
                    run(
                        STANDARD_NORMAL,
                        AR1(0.9),
                        Normal([0.0], [[1.0]]),
                        range(start, start + batch),
                        k=38,
                        m=380,
                        lag=38,
                        rng=seed,
                        lanes=1024,
                    )
                ).measure.per_replicate(moments)
                for start in range(0, n * per_interval, batch)
            ]
        ).reshape(n, per_interval, 2)
        estimate = values.mean(axis=1)
        error = stats.sem(values, axis=1)
        inside = np.abs(estimate - [0.0, 1.0]) <= stats.norm.ppf(0.975) * error
        coverage = inside.mean(axis=0)
        return (coverage - reference) / np.sqrt(coverage * (1 - coverage) / n + reference_error**2)

    assert_z_scores(z_scores, 2000, seed=20)


@functools.cache
def exact_tv_curve(rho):
    # The total variation distance of X_t ~ N(3 rho^t, 0.25 rho^(2t) + 1 - rho^(2t)), the
    # AR(1) chain from N(3, 0.5^2), to N(0, 1), by quadrature, for t = 0, 1, ... while it is
    # at least 0.01. A Markov chain's distance to its stationary law never increases.
    def gap(x, mean, sd):
        return abs(stats.norm.pdf(x, mean, sd) - stats.norm.pdf(x))

    curve = []
    for t in itertools.count():
        mean, sd = 3 * rho**t, np.sqrt(0.25 * rho ** (2 * t) + 1 - rho ** (2 * t))
        distance = 0.5 * integrate.quad(gap, -np.inf, np.inf, args=(mean, sd), limit=200)[0]
        if distance < 0.01:
            return np.array(curve)
        curve.append(distance)


def pairs(target, kernel, initial, n, rng, lag=1, **options):
    # Pairs run to their meeting, without averaging or storing anything.
    return run(
        target, kernel, initial, range(n), k=0, m=0, lag=lag, rng=rng, store=False, **options
    )


@pytest.mark.parametrize("lag", [1, 10])
def test_tv_bounds_exceed_the_exact_distance(lag):
    # On the times where the exact distance is at least 0.01, the estimated bound is above
    # it, up to 5 of its standard errors.
    rho = 0.9
    tau = pairs(STANDARD_NORMAL, AR1(rho), Normal([3.0], [[0.25]]), 10_000, 21, lag).meeting_time
    exact = exact_tv_curve(rho)
    t = np.arange(len(exact))
    bound = diagnostics.tv_bound(tau, lag, t)
    # Biswas, Jacob & Vanetti (2019), Theorem 2.5, replicate by replicate.
    terms = np.maximum(0, np.ceil((tau[:, None] - lag - t) / lag))
    assert np.all(bound >= exact - 5 * stats.sem(terms))


def test_random_walk_meeting_times_match_oleary():
    # O'Leary (2021), Table 1: N(0, I_10), proposal variance 2.38^2/10, both chains from the
    # target, reflection-maximal proposals and a common uniform: mean meeting time 30
    # (standard error 0.8) without a lag. With a lag of 1, tau - 1 has that law. This test
    # and the next three give the papers' settings even where they are the defaults.
    kernel = RandomWalk(np.eye(10), scale=2.38 / np.sqrt(10))
    initial = Normal(np.zeros(10), np.eye(10))
    tau = pairs(STANDARD_NORMAL, kernel, initial, 4000, 30).meeting_time
    assert 24 < np.mean(tau - 1) < 36


def test_bimodal_meeting_times_match_jacob_oleary_atchade():
    # Jacob, O'Leary & Atchade (2020), section 5.1: proposal variance 9 from N(10, 10^2),
    # mean meeting time 20 and 99% quantile 105.
    target = Target(MIXTURE.log_prob, vectorize=True)
    kernel = RandomWalk(np.eye(1), scale=3.0)
    tau = pairs(target, kernel, MIXTURE.initial, 4000, 31).meeting_time
    assert 15 < np.mean(tau) < 25
    assert 70 < np.quantile(tau, 0.99) < 150


def test_banana_meeting_times_match_heng_jacob():
    # Heng & Jacob (2019), section 5.2: HMC with step 1/500 and 500 leapfrog steps, and a
    # random-walk meeting step of scale 1e-3 with probability 1/20, from U[-5, 5]^2: mean
    # meeting time 158 with synchronous momenta.
    target = Target(BANANA.log_prob, grad=BANANA.grad, vectorize=True)
    kernel = HMC(1 / 500, 500, cov=np.eye(2), meeting_weight=1 / 20, meeting_scale=1e-3)
    tau = pairs(target, kernel, BANANA.initial, 200, 32).meeting_time
    assert 118 < np.mean(tau) < 198


def test_a_wide_initial_distribution_exposes_the_second_mode():
    # Jacob, O'Leary & Atchade (2020), section 5.1, proposal variance 1: from N(10, 1) the
    # pairs meet fast in the right mode and miss the left one; from N(10, 10^2), pairs that
    # start in different modes take far longer, which the meeting times show.
    target = Target(MIXTURE.log_prob, vectorize=True)
    kernel = RandomWalk(np.eye(1), scale=1.0)
    narrow, wide = (
        pairs(target, kernel, initial, 500, 33, max_iter=50_000).stopped
        for initial in (Normal([10.0], [[1.0]]), MIXTURE.initial)
    )
    assert np.quantile(wide, 0.99) > 10 * np.quantile(narrow, 0.99)


class Conjugate(NamedTuple):
    # A conjugate model: theta ~ prior, five observations y = observe(theta, rng), the
    # posterior's log-density and standard deviation, and
    # phi(theta, y) = (theta, theta^2, 1(theta <= threshold), log-likelihood).
    prior: Any
    observe: Any
    log_likelihood: Any
    log_posterior: Any
    posterior_sd: Any
    threshold: float

    def phi(self, theta, y):
        below = theta <= self.threshold
        return np.stack([theta, theta**2, below, self.log_likelihood(theta, y)], axis=-1)


def normal_log_posterior(x, y):
    return -0.5 * x[:, 0] ** 2 - 0.5 * np.sum((y - x) ** 2, axis=1)


def poisson_log_posterior(x, y):
    theta = x[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (1.0 + y.sum()) * np.log(theta) - 6.0 * theta
    return np.where(theta > 0, value, -np.inf)


MODELS = {
    # theta ~ N(0, 1), y_i ~ N(theta, 1); the posterior is N(sum(y) / 6, 1/6).
    "normal": Conjugate(
        stats.norm(),
        lambda theta, rng: rng.normal(theta, 1.0, 5),
        lambda theta, y: stats.norm.logpdf(y[:, None], theta).sum(axis=0),
        normal_log_posterior,
        lambda y: 1 / np.sqrt(6),
        0.0,
    ),
    # theta ~ Gamma(2, rate 1), y_i ~ Poisson(theta); the posterior is
    # Gamma(2 + sum(y), rate 6).
    "gamma-poisson": Conjugate(
        stats.gamma(2.0),
        lambda theta, rng: rng.poisson(theta, 5),
        lambda theta, y: stats.poisson.logpmf(y[:, None], theta).sum(axis=0),
        poisson_log_posterior,
        lambda y: np.sqrt(2 + y.sum()) / 6,
        2.0,
    ),
}


@pytest.mark.parametrize("name", MODELS)
def test_estimates_are_calibrated_over_the_prior(name):
    # E[estimate of E[phi | y] - phi(theta, y)] = 0 when theta and y come from the model,
    # for phi = theta, theta^2, an indicator and the log-likelihood (test plan 2.12).
    model = MODELS[name]

    def z_scores(n, rng):
        differences = []
        for _ in range(n):
            theta = model.prior.rvs(random_state=rng)
            y = model.observe(theta, rng)
            target = Target(model.log_posterior, args=(y,), vectorize=True)
            kernel = RandomWalk(np.eye(1), scale=2.38 * model.posterior_sd(y))
            replicates = run(target, kernel, model.prior, [0], k=2, m=20, lag=1, rng=rng, lanes=1)
            values = Result(replicates).measure.per_replicate(lambda x, y=y: model.phi(x[:, 0], y))
            differences.append(values[0] - model.phi(np.array([theta]), y)[0])
        differences = np.array(differences)
        return differences.mean(axis=0) / stats.sem(differences)

    assert_z_scores(z_scores, 2000, seed=40)
