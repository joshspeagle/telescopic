import numpy as np
import pytest
from scipy import optimize, stats

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
        return rng.multivariate_normal(np.zeros(4), EXAMPLES[name].cov, size=n)
    if name == "banana":
        first = rng.normal(1.0, np.sqrt(0.5), n)
        return np.column_stack([first, rng.normal(first**2, np.sqrt(0.05))])
    if name == "mixture":
        return rng.normal(rng.choice([-4.0, 4.0], n), 1.0)[:, None]
    if name == "funnel":
        v = rng.normal(0.0, 3.0, n)
        return np.column_stack([v, rng.normal(0.0, np.exp(v / 2)[:, None], (n, 3))])
    lam, beta = np.ones((n, 10)), np.ones(n)
    for _ in range(50):
        lam = rng.gamma(
            examples.ALPHA + examples.PUMP_FAILURES, 1 / (beta[:, None] + examples.PUMP_TIMES)
        )
        beta = rng.gamma(
            examples.GAMMA + 10 * examples.ALPHA, 1 / (examples.DELTA + lam.sum(axis=1))
        )
    return np.column_stack([lam, beta])


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_exact_means_match_independent_draws(name):
    n = 20_000
    draws = exact_draws(name, n, np.random.default_rng(1))
    z = (draws.mean(axis=0) - EXAMPLES[name].mean) / (draws.std(axis=0) / np.sqrt(n))
    assert np.max(np.abs(z)) < 5


@pytest.mark.parametrize("name", ["gaussian", "banana", "mixture", "pumps"])
def test_the_exact_covariances_match_independent_draws(name):
    # The funnel is left out: its fourth moments, about 3 e^18, leave no usable error bar.
    n = 20_000
    draws = exact_draws(name, n, np.random.default_rng(2))
    centred = draws - draws.mean(axis=0)
    products = centred[:, :, None] * centred[:, None, :]
    z = (products.mean(axis=0) - EXAMPLES[name].cov) / (products.std(axis=0) / np.sqrt(n))
    assert np.max(np.abs(z)) < 5


@pytest.mark.parametrize("name", EXAMPLES)
def test_log_densities_take_points_and_batches_and_gradients_match_finite_differences(name):
    example = EXAMPLES[name]
    rng = np.random.default_rng(3)
    batch = np.stack([example.initial.rvs(random_state=rng) for _ in range(5)])
    assert batch.shape == (5, example.ndim)
    np.testing.assert_allclose(
        example.log_prob(batch), [example.log_prob(point) for point in batch], rtol=1e-12
    )
    np.testing.assert_allclose(example.grad(batch), [example.grad(point) for point in batch])
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
