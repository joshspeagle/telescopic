import numpy as np
import pytest
from scipy import integrate, stats

from references import metropolis, reflection
from telescopic.kernels import Independence, Kernel, Mixture, MultivariateT, RandomWalk, State
from telescopic.target import Target
from telescopic.testing import fixed_draws, random_draws

P_MIN = 1e-5


def gaussian_log_prob(x):
    return -0.5 * np.sum(x**2, axis=-1)


TARGET = Target(gaussian_log_prob, vectorize=True)


def state(points):
    points = np.array(points, dtype=np.float64, ndmin=2)
    return State(points, gaussian_log_prob(points))


def kernels(d):
    return {
        "reflection": RandomWalk(np.eye(d), scale=1.0),
        "maximal": RandomWalk(np.eye(d), scale=1.0, coupling="maximal"),
        "independence": Independence(MultivariateT(np.zeros(d), 1.5 * np.eye(d), df=None)),
        "mixture": Mixture(
            [RandomWalk(np.eye(d), scale=0.5), RandomWalk(np.eye(d), scale=2.0)], [0.3, 0.7]
        ),
    }


# Oracle tests: draw for draw against independent references.


@pytest.mark.parametrize("accept", [True, False])
def test_random_walk_step_matches_the_reference(accept):
    rng = np.random.default_rng(1)
    cov = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.5]])
    chol = np.linalg.cholesky(cov)
    x, z = rng.standard_normal(3), rng.standard_normal(3)
    ratio = np.exp(gaussian_log_prob(x + chol @ z) - gaussian_log_prob(x))
    u = min(0.999, ratio * (0.5 if accept else 1.5))
    new, accepted = RandomWalk(cov, scale=1.0).step(TARGET, state(x), fixed_draws([z], [[u]]))
    expected = metropolis(x, x + chol @ z, gaussian_log_prob, u)
    assert accepted[0] == (u < ratio)
    np.testing.assert_allclose(new.x[0], expected, atol=1e-12)
    np.testing.assert_allclose(new.log_prob[0], gaussian_log_prob(expected), atol=1e-12)


@pytest.mark.parametrize("gap", [3.0, 0.05, 0.0])
@pytest.mark.parametrize("u_accept", [0.01, 0.6, 0.999])
def test_random_walk_reflection_coupling_matches_the_reference(gap, u_accept):
    rng = np.random.default_rng(2)
    x, z, u_meet = rng.standard_normal(2), rng.standard_normal(2), 0.3
    y = x + gap
    px, py, _ = reflection(x, y, 0.8 * np.eye(2), z, u_meet)
    kernel = RandomWalk(np.eye(2), scale=0.8)
    new_x, new_y, met = kernel.coupled_step(
        TARGET, state(x), state(y), fixed_draws([z], [[u_meet, u_accept]])
    )
    np.testing.assert_allclose(
        new_x.x[0], metropolis(x, px, gaussian_log_prob, u_accept), atol=1e-12
    )
    np.testing.assert_allclose(
        new_y.x[0], metropolis(y, py, gaussian_log_prob, u_accept), atol=1e-12
    )
    assert met[0] == np.array_equal(new_x.x[0], new_y.x[0])


def test_independence_kernels_share_the_proposal_and_meet_when_both_accept():
    # With a N(0, I) proposal on a N(0, I) target the weights are constant, so both chains
    # accept the shared proposal x' = z whatever u < 1 is.
    proposal = MultivariateT(np.zeros(2), np.eye(2), df=None)
    z = np.array([0.3, -0.4])
    new_x, new_y, met = Independence(proposal).coupled_step(
        TARGET, state([0.1, 0.2]), state([2.5, -1.0]), fixed_draws([z], [[0.5]])
    )
    np.testing.assert_array_equal(new_x.x[0], z)
    np.testing.assert_array_equal(new_y.x[0], z)
    assert met[0]


class Shift(Kernel):
    """A deterministic test kernel that moves every chain by a fixed amount."""

    def __init__(self, amount):
        self.amount = amount

    def step(self, target, state, draws):
        return State(state.x + self.amount, state.log_prob), np.ones(len(state), bool)

    def coupled_step(self, target, state_x, state_y, draws):
        moved_x, moved_y = (
            self.step(target, state_x, draws)[0],
            self.step(target, state_y, draws)[0],
        )
        return moved_x, moved_y, np.zeros(len(state_x), bool)


def test_mixture_gives_both_chains_the_component_its_uniform_selects():
    n = 4000
    uniforms = np.random.default_rng(3).random((n, 1))
    kernel = Mixture([Shift(1.0), Shift(100.0)], [0.25, 0.75])
    zeros = State(np.zeros((n, 1)), np.zeros(n))
    ones = State(np.ones((n, 1)), np.zeros(n))
    new_x, new_y, _ = kernel.coupled_step(TARGET, zeros, ones, fixed_draws(uniforms=uniforms))
    np.testing.assert_array_equal(new_y.x - new_x.x, 1.0)
    np.testing.assert_array_equal(new_x.x[:, 0] == 100.0, uniforms[:, 0] >= 0.25)
    single, _ = kernel.step(TARGET, zeros, fixed_draws(uniforms=uniforms))
    np.testing.assert_array_equal(single.x, new_x.x)


# Marginals: each chain of a coupled step has exactly the law of one step.


def one_step_law(kernel, x):
    """For a 1-D kernel on N(0, 1): the probability of staying at x, and the CDF of moves."""

    def moved(k, t):
        if isinstance(k, Mixture):
            return sum(w * moved(c, t) for w, c in zip(k.weights, k.kernels, strict=True))
        if isinstance(k, RandomWalk):
            return stats.norm.pdf(t, x, k.scale) * np.minimum(1.0, np.exp(-0.5 * (t * t - x * x)))
        # Independence with q = N(0, 1.5): log pi - log q = -t^2 / 6 + constant.
        return stats.norm.pdf(t, 0, np.sqrt(1.5)) * np.minimum(1.0, np.exp((x * x - t * t) / 6))

    # The moved density is negligible beyond 15 units; a fine grid makes the trapezoid
    # error (about 1e-6) far below the Monte Carlo error of the tests.
    grid = np.linspace(x - 15, x + 15, 60_001)
    cdf = integrate.cumulative_trapezoid(moved(kernel, grid), grid, initial=0)
    return 1 - cdf[-1], lambda t: np.interp(t, grid, cdf / cdf[-1])


@pytest.mark.parametrize("name", ["reflection", "maximal", "independence", "mixture"])
@pytest.mark.parametrize("gap", [0.0, 0.1, 3.0])
def test_each_chain_of_a_coupled_step_has_the_exact_one_step_law(name, gap):
    kernel = kernels(1)[name]
    n = 20_000
    x0, y0 = 0.4, 0.4 + gap
    start_x = State(np.full((n, 1), x0), np.full(n, gaussian_log_prob(np.array([x0]))))
    start_y = State(np.full((n, 1), y0), np.full(n, gaussian_log_prob(np.array([y0]))))
    new_x, new_y, _ = kernel.coupled_step(TARGET, start_x, start_y, random_draws(n, 4))
    for start, new in ((x0, new_x.x[:, 0]), (y0, new_y.x[:, 0])):
        stay, cdf = one_step_law(kernel, start)
        stayed = new == start
        assert abs(stayed.mean() - stay) < 5 * np.sqrt(stay * (1 - stay) / n)
        assert stats.kstest(new[~stayed], cdf).pvalue > P_MIN


@pytest.mark.parametrize("name", ["reflection", "independence", "mixture"])
def test_coupled_marginals_match_single_steps_in_three_dimensions(name):
    kernel = kernels(3)[name]
    n = 20_000
    rng = np.random.default_rng(5)
    x0, y0 = rng.standard_normal(3), rng.standard_normal(3)
    starts = [state(np.tile(p, (n, 1))) for p in (x0, y0)]
    new_x, new_y, _ = kernel.coupled_step(TARGET, *starts, random_draws(n, 6))
    directions = rng.standard_normal((3, 3))
    for start, coupled, seed in ((starts[0], new_x, 7), (starts[1], new_y, 8)):
        single, _ = kernel.step(TARGET, start, random_draws(n, seed))
        for direction in directions:
            p = stats.ks_2samp(coupled.x @ direction, single.x @ direction).pvalue
            assert p > P_MIN / 3


# Faithfulness and the flag contract.


@pytest.mark.parametrize("name", ["reflection", "maximal", "independence", "mixture"])
def test_equal_states_stay_equal_and_met(name):
    kernel = kernels(2)[name]
    points = np.random.default_rng(9).standard_normal((50, 2))
    state_x, state_y = state(points), state(points.copy())
    for step in range(100):
        state_x, state_y, met = kernel.coupled_step(
            TARGET, state_x, state_y, random_draws(50, step)
        )
        assert met.all()
        np.testing.assert_array_equal(state_x.x, state_y.x)
        np.testing.assert_array_equal(state_x.log_prob, state_y.log_prob)


@pytest.mark.parametrize("name", ["reflection", "maximal", "independence", "mixture"])
def test_met_holds_exactly_for_bitwise_equal_pairs(name):
    kernel = kernels(2)[name]
    rng = np.random.default_rng(10)
    state_x, state_y = (
        state(rng.standard_normal((400, 2))),
        state(rng.standard_normal((400, 2)) * 0.2),
    )
    new_x, new_y, met = kernel.coupled_step(TARGET, state_x, state_y, random_draws(400, 11))
    np.testing.assert_array_equal(met, np.all(new_x.x == new_y.x, axis=1))
    np.testing.assert_array_equal(new_x.log_prob[met], new_y.log_prob[met])
    assert met.any()


# Validation and the proposal distribution.


def test_an_untuned_random_walk_refuses_to_step():
    with pytest.raises(ValueError, match="needs cov"):
        RandomWalk().step(TARGET, state([0.0]), fixed_draws([[0.0]], [[0.5]]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cov": [[1.0, 2.0], [2.0, 1.0]]}, "positive definite"),
        ({"cov": [[1.0, 0.5], [0.0, 1.0]]}, "symmetric"),
        ({"cov": np.ones((2, 3))}, "square"),
        ({"cov": np.eye(2), "scale": 0.0}, "scale must be positive"),
        ({"cov": np.eye(2), "coupling": "independent"}, "coupling must be"),
    ],
)
def test_random_walk_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RandomWalk(**kwargs)


@pytest.mark.parametrize("df", [None, 1, 5])
def test_multivariate_t_log_density_is_normalised(df):
    mean, cov = np.array([0.5, -1.0]), np.array([[2.0, 0.3], [0.3, 0.5]])
    x = np.random.default_rng(12).standard_normal((5, 2))
    reference = (
        stats.multivariate_normal(mean, cov)
        if df is None
        else stats.multivariate_t(mean, cov, df=df)
    )
    np.testing.assert_allclose(
        MultivariateT(mean, cov, df).log_density(x), reference.logpdf(x), rtol=1e-12
    )


def test_multivariate_t_draws_have_the_right_covariance():
    mean, cov, df = np.array([1.0, 0.0]), np.array([[1.0, 0.4], [0.4, 2.0]]), 7
    rng = np.random.default_rng(13)
    n = 50_000
    draws = MultivariateT(mean, cov, df).sample(fixed_draws(rng.standard_normal((n, 2 + df))))
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.03)
    np.testing.assert_allclose(np.cov(draws.T), cov * df / (df - 2), atol=0.08)
