import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats
from scipy.integrate import quad

from references import reflection
from telescopic._random import MAIN, Streams, replicate_generator, root_seed
from telescopic.couplings import discrete_maximal, maximal, reflection_maximal
from telescopic.testing import fixed_draws

# One-sample tests use a p-value threshold that keeps the family-wise false-positive rate
# of this module below about 1e-3 (CLAUDE.md: tolerances from the Monte Carlo error).
P_MIN = 1e-5


def dense_cholesky(d, rng):
    a = rng.standard_normal((d, d))
    return np.linalg.cholesky(a @ a.T + d * np.eye(d))


def per_row_draws(n, d, seed):
    streams = Streams(n, d)
    root = root_seed(seed)
    for row in range(n):
        streams.assign(row, replicate_generator(root, MAIN, row))
    return streams.view(np.arange(n))


class Normal:
    """A minimal normal distribution: faster than a frozen SciPy one in long loops."""

    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale

    def rvs(self, *, random_state):
        return self.loc + self.scale * random_state.standard_normal()

    def logpdf(self, x):
        return -0.5 * ((x - self.loc) / self.scale) ** 2 - np.log(self.scale * np.sqrt(2 * np.pi))


@pytest.mark.parametrize("meet", [True, False])
def test_reflection_maximal_matches_an_independent_reference(meet):
    rng = np.random.default_rng(1)
    d = 4
    chol = dense_cholesky(d, rng)
    mu_x, mu_y, z = rng.standard_normal(d), rng.standard_normal(d), rng.standard_normal(d)
    delta = np.linalg.solve(chol, mu_x - mu_y)
    threshold = -(z @ delta) - delta @ delta / 2
    u = np.exp(threshold) * (0.5 if meet else 1.5)
    u = min(u, 0.999)
    expected = reflection(mu_x, mu_y, chol, z, u)
    draws = fixed_draws(normals=[z], uniforms=[[u]])
    x, y, identical = reflection_maximal(mu_x[None], mu_y[None], chol, draws)
    assert identical[0] == expected[2] == (u < np.exp(threshold))
    np.testing.assert_allclose(x[0], expected[0], atol=1e-12)
    np.testing.assert_allclose(y[0], expected[1], atol=1e-12)


@given(seed=st.integers(0, 2**32 - 1), d=st.integers(1, 6), scale=st.floats(0.0, 4.0))
def test_reflection_maximal_flags_exactly_the_equal_pairs(seed, d, scale):
    rng = np.random.default_rng(seed)
    n = 64
    chol = dense_cholesky(d, rng)
    mu_x = rng.standard_normal((n, d))
    mu_y = mu_x + scale * rng.standard_normal((n, d))
    draws = fixed_draws(rng.standard_normal((n, d)), rng.random((n, 1)))
    x, y, identical = reflection_maximal(mu_x, mu_y, chol, draws)
    np.testing.assert_array_equal(identical, np.all(x == y, axis=1))
    # Off the diagonal the reflection preserves the norm of the whitened draw.
    zx = np.linalg.solve(chol, (x - mu_x).T).T
    zy = np.linalg.solve(chol, (y - mu_y).T).T
    np.testing.assert_allclose(
        np.linalg.norm(zx[~identical], axis=1), np.linalg.norm(zy[~identical], axis=1), rtol=1e-9
    )


def test_reflection_maximal_with_equal_means_always_meets():
    rng = np.random.default_rng(2)
    mu = rng.standard_normal((100, 3))
    draws = fixed_draws(rng.standard_normal((100, 3)), rng.random((100, 1)))
    _, _, identical = reflection_maximal(mu, mu.copy(), np.eye(3), draws)
    assert identical.all()


@pytest.mark.parametrize("d", [1, 3, 10])
@pytest.mark.parametrize("distance", [0.3, 1.0, 2.5])
def test_reflection_maximal_meets_with_the_maximal_probability(d, distance):
    # A coupling that recomputed y on the meeting branch would meet only up to rounding;
    # this binomial test catches it (spec Appendix A, E-R2).
    rng = np.random.default_rng([d, int(10 * distance)])
    n = 200_000
    chol = dense_cholesky(d, rng)
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    mu_x = np.tile(rng.standard_normal(d), (n, 1))
    mu_y = mu_x - chol @ (distance * direction)
    draws = fixed_draws(rng.standard_normal((n, d)), rng.random((n, 1)))
    _, _, identical = reflection_maximal(mu_x, mu_y, chol, draws)
    p = 2 * stats.norm.cdf(-distance / 2)
    assert abs(identical.mean() - p) < 5 * np.sqrt(p * (1 - p) / n)


def test_reflection_maximal_draws_have_exact_marginals():
    rng = np.random.default_rng(3)
    n, d = 200_000, 3
    chol = dense_cholesky(d, rng)
    mu_x = np.tile(rng.standard_normal(d), (n, 1))
    mu_y = mu_x + 0.7
    draws = fixed_draws(rng.standard_normal((n, d)), rng.random((n, 1)))
    x, y, _ = reflection_maximal(mu_x, mu_y, chol, draws)
    for draw, mu in ((x, mu_x), (y, mu_y)):
        whitened = np.linalg.solve(chol, (draw - mu).T).T
        for column in whitened.T:
            assert stats.kstest(column, "norm").pvalue > P_MIN
        assert stats.kstest((whitened**2).sum(axis=1), stats.chi2(d).cdf).pvalue > P_MIN


class Prescribed:
    """A distribution whose draws are fixed in advance: for oracle tests of `maximal`."""

    def __init__(self, loc, values):
        self.loc, self.values = loc, list(values)

    def rvs(self, *, random_state):
        return self.values.pop(0)

    def logpdf(self, x):
        return stats.norm.logpdf(x, self.loc)


class PrescribedUniforms:
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0)


@pytest.mark.parametrize(
    ("u", "q_draws", "expected_y", "expected_draws"),
    [
        # log q(0.2) - log p(0.2) = -0.3 for p = N(0, 1), q = N(1, 1): u = 0.5 meets.
        ([0.5], [], 0.2, 1),
        # u = 0.9 does not; Y = 0 is rejected (log p - log q = 0.5), Y = 2 accepted at u = 0.5.
        ([0.9, 0.99, 0.5], [0.0, 2.0], 2.0, 3),
    ],
)
def test_maximal_follows_thorissons_algorithm(u, q_draws, expected_y, expected_draws):
    p, q = Prescribed(0.0, [0.2]), Prescribed(1.0, q_draws)
    draws = fixed_draws(generators=[PrescribedUniforms(u)])
    x, y, identical, n_draws = maximal([p], [q], draws)
    assert x[0] == 0.2
    assert y[0] == expected_y
    assert identical[0] == (expected_draws == 1)
    assert n_draws[0] == expected_draws


def test_maximal_with_eta_caps_the_meeting_probability_of_identical_distributions():
    # With p = q, the first test is u <= min(eta, 1): a draw of u above eta must not meet.
    p, q = Prescribed(0.0, [0.3]), Prescribed(0.0, [1.5])
    draws = fixed_draws(generators=[PrescribedUniforms([0.95, 0.9])])
    _, y, identical, n_draws = maximal([p], [q], draws, eta=0.9)
    assert not identical[0]
    assert (y[0], n_draws[0]) == (1.5, 2)


@pytest.mark.parametrize("eta", [1.0, 0.8])
def test_maximal_has_exact_marginals_and_the_documented_meeting_rate(eta):
    p, q = Normal(0.0, 1.0), Normal(1.0, 1.5)
    n = 20_000
    x, y, identical, n_draws = maximal([p] * n, [q] * n, per_row_draws(n, 1, 7), eta=eta)
    rate = quad(lambda t: min(eta * stats.norm.pdf(t), stats.norm.pdf(t, 1.0, 1.5)), -20, 20)[0]
    assert abs(identical.mean() - rate) < 5 * np.sqrt(rate * (1 - rate) / n)
    assert stats.kstest(x, "norm").pvalue > P_MIN
    assert stats.kstest(y, stats.norm(1.0, 1.5).cdf).pvalue > P_MIN
    # The number of draws has mean 2 and variance 2 c / (1 - c), with c the meeting rate.
    assert abs(n_draws.mean() - 2) < 5 * np.sqrt(2 * rate / (1 - rate) / n)
    np.testing.assert_array_equal(identical, x == y)


def test_maximal_accepts_frozen_scipy_distributions_continuous_and_discrete():
    draws = per_row_draws(4, 1, 11)
    _, _, identical, _ = maximal([stats.norm(0, 1)] * 4, [stats.norm(0, 1)] * 4, draws)
    assert identical.all()
    x, y, _, _ = maximal([stats.poisson(3)] * 4, [stats.poisson(5)] * 4, per_row_draws(4, 1, 12))
    assert x.dtype == y.dtype == np.float64
    assert np.all(x == np.round(x))


def test_maximal_rejects_eta_outside_the_unit_interval():
    with pytest.raises(ValueError, match="eta must be in"):
        maximal([], [], fixed_draws(uniforms=[[0.5]]), eta=0.0)


def test_discrete_maximal_matches_the_exact_coupling_table():
    rng = np.random.default_rng(5)
    p, q = np.array([0.5, 0.3, 0.2, 0.0]), np.array([0.1, 0.3, 0.2, 0.4])
    n = 200_000
    with np.errstate(divide="ignore"):
        log_p, log_q = np.tile(np.log(p), (n, 1)), np.tile(np.log(q), (n, 1))
    i, j = discrete_maximal(log_p, log_q, fixed_draws(uniforms=rng.random((n, 3))))
    overlap = np.minimum(p, q)
    alpha = overlap.sum()
    table = np.diag(overlap) + np.outer(p - overlap, q - overlap) / (1 - alpha)
    counts = np.zeros((4, 4))
    np.add.at(counts, (i, j), 1)
    support = table > 0
    assert counts[~support].sum() == 0
    assert stats.chisquare(counts[support], n * table[support]).pvalue > P_MIN


def test_discrete_maximal_handles_peaked_unnormalised_log_probabilities():
    log_p = np.array([[0.0, -800.0, -800.0]])
    log_q = np.array([[-800.0, 3.0, -800.0]])
    i, j = discrete_maximal(log_p, log_q, fixed_draws(uniforms=[[0.3, 0.6, 0.2]]))
    assert (i[0], j[0]) == (0, 1)


def test_discrete_maximal_of_identical_distributions_always_meets():
    rng = np.random.default_rng(6)
    log_p = np.log(rng.dirichlet(np.ones(5), size=1000))
    i, j = discrete_maximal(log_p, log_p.copy(), fixed_draws(uniforms=rng.random((1000, 3))))
    np.testing.assert_array_equal(i, j)
