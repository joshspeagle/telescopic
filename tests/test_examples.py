import numpy as np
import pytest
from scipy import optimize, stats

from references import assert_z_scores
from telescopic import examples

EXAMPLES = {
    "gaussian": examples.gaussian(4, 0.7),
    "banana": examples.banana(),
    "mixture": examples.mixture(),
    "funnel": examples.funnel(4),
    "pumps": examples.pumps(),
}


def exact_draws(name, n, rng):
    # Independent draws from each target's generative description, written apart from the
    # package; the pump model's by its Gibbs sampler, which mixes within a few sweeps.
    if name == "gaussian":
        # x_i = 0.7 x_{i-1} + sqrt(1 - 0.7^2) z_i gives Cov(x_i, x_j) = 0.7^|i-j|.
        x = rng.standard_normal((n, 4))
        for i in range(1, 4):
            x[:, i] = 0.7 * x[:, i - 1] + np.sqrt(1 - 0.7**2) * x[:, i]
        return x
    if name == "banana":
        first = rng.normal(1.0, np.sqrt(0.5), n)
        return np.column_stack([first, rng.normal(first**2, np.sqrt(0.05))])
    if name == "mixture":
        return rng.normal(rng.choice([-4.0, 4.0], n), 1.0)[:, None]
    if name == "funnel":
        v = rng.normal(0.0, 3.0, n)
        return np.column_stack([v, rng.normal(0.0, np.exp(v / 2)[:, None], (n, 3))])
    lam, beta = np.ones((n, 10)), np.ones(n)
    for _ in range(20):
        lam = rng.gamma(
            examples.ALPHA + examples.PUMP_FAILURES, 1 / (beta[:, None] + examples.PUMP_TIMES)
        )
        beta = rng.gamma(
            examples.GAMMA + 10 * examples.ALPHA, 1 / (examples.DELTA + lam.sum(axis=1))
        )
    return np.column_stack([lam, beta])


def z_scores(values, expected):
    # The z-score of the mean of each column of values against its expected value.
    n = len(values)
    return (values.mean(axis=0) - expected) / (values.std(axis=0, ddof=1) / np.sqrt(n))


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_exact_means_match_independent_draws(name):
    def z(n, rng):
        return z_scores(exact_draws(name, n, rng), EXAMPLES[name].mean)

    assert_z_scores(z, 20_000, seed=1)


@pytest.mark.parametrize("name", ["gaussian", "banana", "mixture", "pumps"])
def test_the_exact_covariances_match_independent_draws(name):
    # The funnel is left out: its fourth moments, about 3 e^18, leave no usable error bar.
    def z(n, rng):
        draws = exact_draws(name, n, rng)
        centred = draws - EXAMPLES[name].mean
        return z_scores(centred[:, :, None] * centred[:, None, :], EXAMPLES[name].cov)

    assert_z_scores(z, 20_000, seed=2)


def reference_log_density(name, x):
    # Each target's log-density at a point, from its generative description by SciPy.
    norm = stats.norm.logpdf
    if name == "gaussian":
        # x_0 ~ N(0, 1) and x_i | x_{i-1} ~ N(0.7 x_{i-1}, 1 - 0.7^2).
        return norm(x[0]) + norm(x[1:], 0.7 * x[:-1], np.sqrt(1 - 0.7**2)).sum()
    if name == "banana":
        return norm(x[0], 1.0, np.sqrt(0.5)) + norm(x[1], x[0] ** 2, np.sqrt(0.05))
    if name == "mixture":
        return np.logaddexp(norm(x[0], -4.0), norm(x[0], 4.0)) + np.log(0.5)
    if name == "funnel":
        return norm(x[0], 0.0, 3.0) + norm(x[1:], 0.0, np.exp(x[0] / 2)).sum()
    lam, beta = x[:10], x[10]
    return (
        stats.poisson.logpmf(examples.PUMP_FAILURES, lam * examples.PUMP_TIMES).sum()
        + stats.gamma.logpdf(lam, examples.ALPHA, scale=1 / beta).sum()
        + stats.gamma.logpdf(beta, examples.GAMMA, scale=1 / examples.DELTA)
    )


@pytest.mark.parametrize("name", EXAMPLES)
def test_log_densities_match_their_generative_descriptions_up_to_a_constant(name):
    example = EXAMPLES[name]
    rng = np.random.default_rng(3)
    points = [example.initial.rvs(random_state=rng) for _ in range(5)]
    reference = np.array([reference_log_density(name, point) for point in points])
    difference = np.array([example.log_prob(point) for point in points]) - reference
    np.testing.assert_allclose(difference, difference[0], atol=1e-12 * np.max(np.abs(reference)))


def test_the_banana_moments_follow_from_its_conditionals():
    # x_0 ~ N(1, 1/2) and x_1 | x_0 ~ N(x_0^2, 1/20).
    first = stats.norm(1.0, np.sqrt(0.5))
    second, third, fourth = (first.expect(lambda x, p=p: x**p) for p in (2, 3, 4))
    covariance = third - first.mean() * second
    expected = [[first.var(), covariance], [covariance, 0.05 + fourth - second**2]]
    np.testing.assert_allclose(EXAMPLES["banana"].mean, [first.mean(), second], rtol=1e-10)
    np.testing.assert_allclose(EXAMPLES["banana"].cov, expected, rtol=1e-10)


def test_the_funnel_variances_are_those_of_its_description():
    # x_i | v ~ N(0, e^v) with v ~ N(0, 9): Var(x_i) = E[e^v], the mean of a lognormal.
    expected = np.diag([9.0] + [stats.lognorm(3.0).mean()] * 3)
    np.testing.assert_allclose(EXAMPLES["funnel"].cov, expected, rtol=1e-12)


@pytest.mark.parametrize("name", EXAMPLES)
def test_log_densities_give_a_point_the_same_bits_alone_and_in_a_batch(name):
    # So that runs do not depend on the engine's lanes or a pool's chunks (spec D-20).
    example = EXAMPLES[name]
    rng = np.random.default_rng(4)
    batch = np.stack([example.initial.rvs(random_state=rng) for _ in range(64)])
    assert batch.shape == (64, example.ndim)
    np.testing.assert_array_equal(
        example.log_prob(batch), [example.log_prob(point) for point in batch]
    )
    np.testing.assert_array_equal(example.grad(batch), [example.grad(point) for point in batch])


@pytest.mark.parametrize("name", EXAMPLES)
def test_gradients_match_finite_differences(name):
    example = EXAMPLES[name]
    rng = np.random.default_rng(5)
    batch = np.stack([example.initial.rvs(random_state=rng) for _ in range(5)])
    for point in batch:
        numeric = optimize.approx_fprime(point, example.log_prob, 1e-7)
        np.testing.assert_allclose(example.grad(point), numeric, rtol=1e-4, atol=1e-4)


def test_the_pump_conditionals_are_the_gamma_laws_of_the_model():
    example = EXAMPLES["pumps"]
    point = example.mean
    rate = np.append(point[10] + examples.PUMP_TIMES, examples.DELTA + point[:10].sum())
    shape = np.append(
        examples.ALPHA + examples.PUMP_FAILURES, examples.GAMMA + 10 * examples.ALPHA
    )
    for block in range(11):
        conditional = example.conditional(point, block)
        reference = stats.gamma(shape[block], scale=1 / rate[block])
        np.testing.assert_allclose(conditional.logpdf(0.7), reference.logpdf(0.7), rtol=1e-12)
        np.testing.assert_allclose(conditional.ppf(0.3), reference.ppf(0.3), rtol=1e-10)
    assert example.log_prob(np.append(point[:10], -1.0)) == -np.inf


def test_an_uncorrelated_gaussian_has_a_float_covariance():
    cov = examples.gaussian(3, 0).cov
    assert cov.dtype == np.float64
    np.testing.assert_array_equal(cov, np.eye(3))


@pytest.mark.parametrize(
    ("make", "kwargs"),
    [
        (examples.gaussian, {"d": 0}),
        (examples.gaussian, {"rho": 1.0}),
        (examples.funnel, {"d": 1}),
    ],
)
def test_invalid_example_parameters_raise(make, kwargs):
    with pytest.raises(ValueError, match="must be"):
        make(**kwargs)
