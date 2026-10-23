import numpy as np
import pytest
from scipy import integrate, special, stats

from references import P_MIN, gaussian_log_prob, metropolis, reflection
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
from telescopic.target import Target
from telescopic.testing import fixed_draws, random_draws

# N(0, I); gradient-free kernels evaluate the gradient only for states that carry one.
TARGET = Target(gaussian_log_prob, grad=lambda x: -x, vectorize=True)
NAMES = ["reflection", "maximal", "independence", "mixture", "mala", "hmc", "gibbs"]


def state(points, grad=False):
    points = np.array(points, dtype=np.float64, ndmin=2)
    return State(points, gaussian_log_prob(points), -points if grad else None)


class Normal:
    """A univariate normal with the interface of a frozen SciPy one, and faster to build."""

    def __init__(self, mean, sd):
        self.mean, self.sd = mean, sd

    def rvs(self, *, random_state):
        return self.mean + self.sd * random_state.standard_normal()

    def logpdf(self, value):
        return -0.5 * ((value - self.mean) / self.sd) ** 2 - np.log(self.sd * np.sqrt(2 * np.pi))

    def ppf(self, u):
        return self.mean + self.sd * special.ndtri(u)


def standard_normal(point, block):
    # The full conditional of any coordinate of N(0, I).
    return Normal(0.0, 1.0)


def kernels(cov):
    d = len(cov)
    return {
        "reflection": RandomWalk(cov, scale=1.0),
        "maximal": RandomWalk(cov, scale=1.0, coupling="maximal"),
        "independence": Independence(MultivariateT(np.zeros(d), 1.5 * cov, df=None)),
        "mixture": Mixture([RandomWalk(cov, scale=0.5), RandomWalk(cov, scale=2.0)], [0.3, 0.7]),
        "mala": MALA(0.9, cov=cov),
        "hmc": HMC(0.3, 4, cov=cov, meeting_weight=0.3, meeting_scale=0.5),
        "gibbs": Gibbs(standard_normal, [[i] for i in range(d)], eta=0.5),
    }


def dense(d):
    # A dense covariance near the identity, so that most proposals are accepted.
    a = np.random.default_rng(14).standard_normal((d, d))
    return (a @ a.T + d * np.eye(d)) / (2 * d)


# Oracle tests: draw for draw against independent references.


@pytest.mark.parametrize("accept", [True, False])
def test_random_walk_step_matches_the_reference(accept):
    rng = np.random.default_rng(1)
    cov = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.5]])
    chol = np.linalg.cholesky(cov)
    x, z = rng.standard_normal(3), rng.standard_normal(3)
    ratio = np.exp(gaussian_log_prob(x + chol @ z) - gaussian_log_prob(x))
    u = min(0.999, ratio * (0.5 if accept else 1.5))
    new, acceptance = RandomWalk(cov, scale=1.0).step(TARGET, state(x), fixed_draws([z], [[u]]))
    expected = metropolis(x, x + chol @ z, gaussian_log_prob, u)
    np.testing.assert_allclose(acceptance[0], min(1.0, ratio), atol=1e-12)
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
    new_x, new_y = kernel.coupled_step(
        TARGET, state(x), state(y), fixed_draws([z], [[u_meet, u_accept]])
    )
    np.testing.assert_allclose(
        new_x.x[0], metropolis(x, px, gaussian_log_prob, u_accept), atol=1e-12
    )
    np.testing.assert_allclose(
        new_y.x[0], metropolis(y, py, gaussian_log_prob, u_accept), atol=1e-12
    )


def test_independence_kernels_share_the_proposal_and_meet_when_both_accept():
    # With a N(0, I) proposal on a N(0, I) target the weights are constant, so both chains
    # accept the shared proposal x' = z whatever u < 1 is.
    proposal = MultivariateT(np.zeros(2), np.eye(2), df=None)
    z = np.array([0.3, -0.4])
    new_x, new_y = Independence(proposal).coupled_step(
        TARGET, state([0.1, 0.2]), state([2.5, -1.0]), fixed_draws([z], [[0.5]])
    )
    np.testing.assert_array_equal(new_x.x[0], z)
    np.testing.assert_array_equal(new_y.x[0], z)


class Shift(Kernel):
    """A deterministic test kernel that moves every chain by a fixed amount."""

    def __init__(self, amount):
        self.amount = amount

    def step(self, target, state, draws):
        return State(state.x + self.amount, state.log_prob), np.full(len(state), self.amount)

    def coupled_step(self, target, state_x, state_y, draws):
        return self.step(target, state_x, draws)[0], self.step(target, state_y, draws)[0]


def test_mixture_gives_both_chains_the_component_its_uniform_selects():
    n = 4000
    uniforms = np.random.default_rng(3).random((n, 1))
    kernel = Mixture([Shift(1.0), Shift(100.0)], [0.25, 0.75])
    zeros = State(np.zeros((n, 1)), np.zeros(n))
    ones = State(np.ones((n, 1)), np.zeros(n))
    new_x, new_y = kernel.coupled_step(TARGET, zeros, ones, fixed_draws(uniforms=uniforms))
    np.testing.assert_array_equal(new_y.x - new_x.x, 1.0)
    np.testing.assert_array_equal(new_x.x[:, 0] == 100.0, uniforms[:, 0] >= 0.25)
    single, acceptance = kernel.step(TARGET, zeros, fixed_draws(uniforms=uniforms))
    np.testing.assert_array_equal(single.x, new_x.x)
    np.testing.assert_array_equal(acceptance, single.x[:, 0])  # each component's own


# Marginals: each chain of a coupled step has exactly the law of one step.


def one_step_law(kernel, x):
    """For a 1-D kernel on N(0, 1): the probability of staying at x, and the CDF of moves."""

    def moved(k, t):
        if isinstance(k, Mixture):  # HMC included: a leapfrog and a random-walk component
            return sum(w * moved(c, t) for w, c in zip(k.weights, k.kernels, strict=True))
        if isinstance(k, RandomWalk):
            return stats.norm.pdf(t, x, k.scale) * np.minimum(1.0, np.exp(-0.5 * (t * t - x * x)))
        if isinstance(k, Independence):
            # q = N(0, 1.5): log pi - log q = -t^2 / 6 + constant.
            return stats.norm.pdf(t, 0, np.sqrt(1.5)) * np.minimum(
                1.0, np.exp((x * x - t * t) / 6)
            )
        if isinstance(k, MALA):
            # q(. | y) = N(y (1 - e^2 / 2), e^2).
            e = k.step_size
            log_q = stats.norm(0, e).logpdf
            log_alpha = -0.5 * (t * t - x * x) + log_q(x - t * (1 - e * e / 2))
            log_alpha -= log_q(t - x * (1 - e * e / 2))
            return np.exp(log_q(t - x * (1 - e * e / 2)) + np.minimum(0.0, log_alpha))
        if isinstance(k, Gibbs):
            return stats.norm.pdf(t)
        # A leapfrog trajectory is linear on N(0, 1): (x', v') = P (x, v), P = M^n.
        e = k.step_size
        one = np.array([[1 - e * e / 2, e], [-e * (1 - e * e / 4), 1 - e * e / 2]])
        (a, b), (c, d) = np.linalg.matrix_power(one, k.n_steps)
        v = (t - a * x) / b
        energy = 0.5 * (x * x + v * v) - 0.5 * (t * t + (c * x + d * v) ** 2)
        return stats.norm.pdf(v) / abs(b) * np.minimum(1.0, np.exp(energy))

    # The moved density is negligible beyond 15 units; a fine grid makes the trapezoid
    # error (about 1e-6) far below the Monte Carlo error of the tests.
    grid = np.linspace(x - 15, x + 15, 60_001)
    cdf = integrate.cumulative_trapezoid(moved(kernel, grid), grid, initial=0)
    return max(0.0, 1 - cdf[-1]), lambda t: np.interp(t, grid, cdf / cdf[-1])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("gap", [0.0, 0.1, 3.0])
def test_each_chain_of_a_coupled_step_has_the_exact_one_step_law(name, gap):
    kernel = kernels(np.eye(1))[name]
    n = 20_000
    x0, y0 = 0.4, 0.4 + gap
    start_x = state(np.full((n, 1), x0), kernel.needs_grad)
    start_y = state(np.full((n, 1), y0), kernel.needs_grad)
    new_x, new_y = kernel.coupled_step(TARGET, start_x, start_y, random_draws(n, 4))
    for start, new in ((x0, new_x.x[:, 0]), (y0, new_y.x[:, 0])):
        stay, cdf = one_step_law(kernel, start)
        stayed = new == start
        assert stats.binomtest(int(stayed.sum()), n, stay).pvalue > P_MIN
        assert stats.kstest(new[~stayed], cdf).pvalue > P_MIN


@pytest.mark.parametrize("name", ["reflection", "independence", "mixture", "mala", "hmc"])
def test_coupled_marginals_match_single_steps_in_three_dimensions(name):
    kernel = kernels(dense(3))[name]
    n = 20_000
    rng = np.random.default_rng(5)
    x0, y0 = rng.standard_normal(3), rng.standard_normal(3)
    starts = [state(np.tile(p, (n, 1)), kernel.needs_grad) for p in (x0, y0)]
    new_x, new_y = kernel.coupled_step(TARGET, *starts, random_draws(n, 6))
    directions = rng.standard_normal((3, 3))
    for start, coupled, seed in ((starts[0], new_x, 7), (starts[1], new_y, 8)):
        single, _ = kernel.step(TARGET, start, random_draws(n, seed))
        for direction in directions:
            p = stats.ks_2samp(coupled.x @ direction, single.x @ direction).pvalue
            assert p > P_MIN / 3


# Faithfulness and batch invariance.


@pytest.mark.parametrize("name", NAMES)
def test_equal_states_stay_equal(name):
    kernel = kernels(np.eye(2))[name]
    points = np.random.default_rng(9).standard_normal((50, 2))
    state_x, state_y = state(points, kernel.needs_grad), state(points.copy(), kernel.needs_grad)
    for step in range(100):
        state_x, state_y = kernel.coupled_step(TARGET, state_x, state_y, random_draws(50, step))
        np.testing.assert_array_equal(state_x.x, state_y.x)
        np.testing.assert_array_equal(state_x.log_prob, state_y.log_prob)
        if kernel.needs_grad:
            np.testing.assert_array_equal(state_x.grad, state_y.grad)


@pytest.mark.parametrize("name", NAMES)
def test_kernels_leave_their_input_states_unchanged(name):
    kernel = kernels(np.eye(2))[name]
    rng = np.random.default_rng(17)
    x, y = (state(rng.standard_normal((30, 2)), kernel.needs_grad) for _ in range(2))
    before = [a.copy() for a in (x.x, x.log_prob, y.x, y.log_prob)]
    kernel.step(TARGET, x, random_draws(30, 18))
    kernel.coupled_step(TARGET, x, y, random_draws(30, 19))
    for array, copy in zip((x.x, x.log_prob, y.x, y.log_prob), before, strict=True):
        np.testing.assert_array_equal(array, copy)


@pytest.mark.parametrize("name", NAMES)
def test_states_that_carry_gradients_get_them_at_every_new_position(name):
    # A gradient kernel mixed with this one must find valid gradients in the states.
    kernel = kernels(np.eye(2))[name]
    rng = np.random.default_rng(20)
    x, y = (state(rng.standard_normal((40, 2)), grad=True) for _ in range(2))
    moved, _ = kernel.step(TARGET, x, random_draws(40, 21))
    new_x, new_y = kernel.coupled_step(TARGET, x, y, random_draws(40, 22))
    for result in (moved, new_x, new_y):
        np.testing.assert_array_equal(result.grad, -result.x)


class NeedsGradient(RandomWalk):
    @property
    def needs_grad(self):
        return True


def test_a_non_finite_gradient_at_an_initial_state_raises():
    target = Target(gaussian_log_prob, grad=lambda x: np.full_like(x, np.nan), vectorize=True)
    with pytest.raises(ValueError, match="grad returned"):
        NeedsGradient(np.eye(2)).init_state(target, np.zeros((1, 2)))


@pytest.mark.parametrize("name", NAMES)
def test_a_rows_transition_does_not_depend_on_the_rest_of_the_batch(name):
    # Spec R-KER-6: a BLAS product gives most rows other bits in a batch of one.
    d, n = 10, 16
    kernel = kernels(0.25 * dense(d))[name]
    rng = np.random.default_rng(15)
    x, y = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    normals, uniforms = rng.standard_normal((n, 2 * d + 5)), rng.random((n, 3))

    def draws(rows):
        generators = [np.random.default_rng([16, row]) for row in rows]
        return fixed_draws(normals[rows], uniforms[rows], generators)

    def transition(rows):
        grad = kernel.needs_grad
        moved, _ = kernel.step(TARGET, state(x[rows], grad), draws(rows))
        new_x, new_y = kernel.coupled_step(
            TARGET, state(x[rows], grad), state(y[rows], grad), draws(rows)
        )
        return np.stack([moved.x, new_x.x, new_y.x], axis=1)

    batch = transition(np.arange(n))
    for row in range(n):
        np.testing.assert_array_equal(transition(np.array([row]))[0], batch[row])


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


@pytest.mark.parametrize(
    ("weights", "message"),
    [([np.nan, 1.0], "finite, non-negative"), ([0.0, 0.0], "not all zero"), ([1.0], "per kernel")],
)
def test_mixture_rejects_invalid_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        Mixture([RandomWalk(np.eye(1)), RandomWalk(np.eye(1))], weights)


def test_mixture_refuses_components_of_different_dimensions():
    with pytest.raises(ValueError, match="different dimensions"):
        Mixture([RandomWalk(np.eye(1)), RandomWalk(np.eye(2))], [0.5, 0.5])


def test_a_non_finite_gradient_at_an_accepted_proposal_raises():
    # A gradient-free kernel whose states carry gradients must not cache a bad one.
    target = Target(gaussian_log_prob, grad=lambda x: np.where(x > 0, np.nan, -x), vectorize=True)
    start = State(np.full((50, 1), -1.0), np.full(50, -0.5), np.full((50, 1), 1.0))
    with pytest.raises(ValueError, match="grad returned"):
        RandomWalk(np.eye(1), scale=3.0).step(target, start, random_draws(50, 23))


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


# MALA: oracle tests on a non-Gaussian target, where the proposal densities matter.


def banana_log_prob(x):
    return -0.5 * (x[..., 0] ** 2 + (x[..., 1] - x[..., 0] ** 2) ** 2)


def banana_grad(x):
    shift = x[..., 1] - x[..., 0] ** 2
    return np.stack([-x[..., 0] + 2 * x[..., 0] * shift, -shift], axis=-1)


BANANA = Target(banana_log_prob, grad=banana_grad, vectorize=True)
CHOL = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 0.5]]))


def banana_state(points):
    points = np.array(points, dtype=np.float64, ndmin=2)
    return State(points, banana_log_prob(points), banana_grad(points))


def langevin(point, epsilon):
    return point + 0.5 * epsilon**2 * CHOL @ CHOL.T @ banana_grad(point)


def mala_reference(x, proposal, u, epsilon):
    # Metropolis-Hastings with both Langevin proposal densities; the constants cancel.
    def log_q(to, start):
        r = np.linalg.solve(epsilon * CHOL, to - langevin(start, epsilon))
        return -0.5 * r @ r

    log_alpha = banana_log_prob(proposal) - banana_log_prob(x)
    log_alpha += log_q(x, proposal) - log_q(proposal, x)
    return (proposal if np.log(u) < log_alpha else x), min(1.0, np.exp(log_alpha))


@pytest.mark.parametrize("u", [0.4, 0.7])
def test_mala_step_matches_the_reference_with_both_proposal_densities(u):
    # The acceptance probability is 0.52 here, and 1 without the proposal densities, so
    # u = 0.7 rejects only when they are included.
    x, z, epsilon = np.array([0.8, -0.4]), np.array([-1.0, 1.5]), 0.9
    proposal = langevin(x, epsilon) + epsilon * CHOL @ z
    expected, alpha = mala_reference(x, proposal, u, epsilon)
    kernel = MALA(epsilon, cov=CHOL @ CHOL.T)
    new, acceptance = kernel.step(BANANA, banana_state(x), fixed_draws([z], [[u]]))
    np.testing.assert_allclose(new.x[0], expected, atol=1e-12)
    np.testing.assert_allclose(new.grad[0], banana_grad(expected), atol=1e-12)
    np.testing.assert_allclose(acceptance[0], alpha, atol=1e-12)


@pytest.mark.parametrize("gap", [2.0, 0.05, 0.0])
@pytest.mark.parametrize("u_accept", [0.05, 0.6])
def test_mala_reflection_coupling_matches_the_reference(gap, u_accept):
    x, z, u_meet, epsilon = np.array([0.5, 0.2]), np.array([0.3, -0.7]), 0.4, 0.8
    y = x + gap
    px, py, _ = reflection(langevin(x, epsilon), langevin(y, epsilon), epsilon * CHOL, z, u_meet)
    kernel = MALA(epsilon, cov=CHOL @ CHOL.T)
    new_x, new_y = kernel.coupled_step(
        BANANA, banana_state(x), banana_state(y), fixed_draws([z], [[u_meet, u_accept]])
    )
    for start, proposal, new in ((x, px, new_x), (y, py, new_y)):
        expected, _ = mala_reference(start, proposal, u_accept, epsilon)
        np.testing.assert_allclose(new.x[0], expected, atol=1e-12)


# HMC.


def leapfrog_reference(x, momentum, epsilon, n_steps):
    momentum = momentum + 0.5 * epsilon * CHOL.T @ banana_grad(x)
    for step in range(n_steps):
        x = x + epsilon * CHOL @ momentum
        kick = epsilon if step < n_steps - 1 else 0.5 * epsilon
        momentum = momentum + kick * CHOL.T @ banana_grad(x)
    return x, momentum


@pytest.mark.parametrize("u", [0.2, 0.9])
def test_hmc_trajectories_match_the_reference_and_share_their_momentum(u):
    x, y, momentum = np.array([0.5, 0.1]), np.array([-0.3, 0.6]), np.array([0.7, -0.4])
    kernel = HMC(0.25, 5, cov=CHOL @ CHOL.T)
    # A component uniform of 0.1 < 1 - meeting_weight chooses the trajectory.
    draws = fixed_draws([momentum], [[0.1, u]])
    new_x, new_y = kernel.coupled_step(BANANA, banana_state(x), banana_state(y), draws)
    for start, new in ((x, new_x), (y, new_y)):
        end, end_momentum = leapfrog_reference(start, momentum, 0.25, 5)
        log_alpha = banana_log_prob(end) - 0.5 * end_momentum @ end_momentum
        log_alpha -= banana_log_prob(start) - 0.5 * momentum @ momentum
        expected = end if np.log(u) < log_alpha else start
        np.testing.assert_allclose(new.x[0], expected, atol=1e-12)
    single, _ = kernel.step(BANANA, banana_state(x), fixed_draws([momentum], [[0.1, u]]))
    np.testing.assert_array_equal(single.x, new_x.x)


def test_shared_momenta_contract_hmc_chains():
    n = 200
    rng = np.random.default_rng(31)
    kernel = HMC(0.3, 4, cov=np.eye(3), meeting_weight=1e-9)
    x, y = state(rng.standard_normal((n, 3)), True), state(rng.standard_normal((n, 3)), True)
    start = np.linalg.norm(x.x - y.x, axis=1)
    for step in range(20):
        x, y = kernel.coupled_step(TARGET, x, y, random_draws(n, 40 + step))
    assert np.median(np.linalg.norm(x.x - y.x, axis=1)) < 1e-3 * np.median(start)


def test_the_hmc_meeting_step_meets_with_the_reflection_probability():
    # Steps of 1e-4 near the mode are accepted almost surely, so a pair 0.5 apart in units
    # of the step meets in a meeting step with probability 2 Phi(-0.25).
    n, scale = 20_000, 1e-4
    kernel = HMC(0.3, 4, cov=np.eye(2), meeting_weight=0.5, meeting_scale=scale)
    x = state(np.zeros((n, 2)), True)
    y = state(np.tile([0.5 * scale, 0.0], (n, 1)), True)
    new_x, new_y = kernel.coupled_step(TARGET, x, y, random_draws(n, 32))
    met = np.all(new_x.x == new_y.x, axis=1).sum()
    assert stats.binomtest(int(met), n, 0.5 * 2 * stats.norm.cdf(-0.25)).pvalue > P_MIN


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"step_size": 0.0}, "step_size must be positive"),
        ({"n_steps": 0}, "n_steps must be a positive integer"),
        ({"meeting_weight": 0.0}, "meeting_weight must be in"),
        ({"meeting_scale": -1.0}, "scale must be positive"),
    ],
)
def test_hmc_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HMC(cov=np.eye(2), **kwargs)


def test_gradient_kernels_reject_proposals_without_a_finite_gradient():
    # The gradient is NaN beyond x = 1; proposals there are rejected, never raised on.
    target = Target(gaussian_log_prob, grad=lambda x: np.where(x > 1, np.nan, -x), vectorize=True)
    start = State(np.zeros((500, 1)), np.zeros(500), np.zeros((500, 1)))
    for kernel in (MALA(2.0, cov=np.eye(1)), HMC(0.5, 4, cov=np.eye(1))):
        moved, _ = kernel.step(target, start, random_draws(500, 33))
        assert np.all(moved.x <= 1)
        assert np.any(moved.x != 0)


def test_gradient_kernels_refuse_starts_outside_the_support():
    target = Target(
        lambda x: np.where(x[:, 0] > 0, 0.0, -np.inf), grad=np.zeros_like, vectorize=True
    )
    with pytest.raises(ValueError, match="gradient kernel cannot start"):
        MALA(0.5, cov=np.eye(1)).init_state(target, np.array([[-1.0]]))


# Gibbs, on a bivariate Gaussian with correlation 0.8.

RHO = 0.8


def conditional(point, block):
    return Normal(RHO * point[1 - block], np.sqrt(1 - RHO**2))


def bivariate(points):
    points = np.array(points, dtype=np.float64, ndmin=2)
    return State(points, np.zeros(len(points)))


def test_maximal_gibbs_couplings_draw_each_chain_from_its_conditionals():
    n = 4000
    kernel = Gibbs(conditional, [[0], [1]], eta=0.7)
    x, y = bivariate(np.tile([0.5, -1.0], (n, 1))), bivariate(np.tile([-0.5, 1.5], (n, 1)))
    new_x, new_y = kernel.coupled_step(Target(None), x, y, random_draws(n, 34))
    sd = np.sqrt(1 - RHO**2)
    for start, new in ((x, new_x), (y, new_y)):
        first = (new.x[:, 0] - RHO * start.x[0, 1]) / sd
        second = (new.x[:, 1] - RHO * new.x[:, 0]) / sd
        for residual in (first, second):
            assert stats.kstest(residual, "norm").pvalue > P_MIN


def test_crn_gibbs_couplings_push_one_uniform_through_both_quantile_functions():
    x, y, u = np.array([0.5, -1.0]), np.array([-0.5, 1.5]), [0.3, 0.8]
    kernel = Gibbs(conditional, [[0], [1]], coupling="crn")
    new_x, new_y = kernel.coupled_step(
        Target(None), bivariate(x), bivariate(y), fixed_draws(uniforms=[u])
    )
    sd = np.sqrt(1 - RHO**2)
    for start, new in ((x, new_x), (y, new_y)):
        first = stats.norm(RHO * start[1], sd).ppf(u[0])
        np.testing.assert_allclose(new.x[0], [first, stats.norm(RHO * first, sd).ppf(u[1])])


def test_two_scale_gibbs_couplings_switch_at_the_threshold():
    kernel = Gibbs(conditional, [[0], [1]], coupling="two-scale", threshold=1.0)
    far, near = (np.array([[0.5, -1.0], [0.6, -1.1]]), np.array([[-0.5, 1.5], [0.6, -1.0]]))
    generators = [np.random.default_rng(35), np.random.default_rng(36)]
    draws = fixed_draws(uniforms=[[0.3, 0.8], [0.3, 0.8]], generators=generators)
    new_x, _ = kernel.coupled_step(Target(None), bivariate(far), bivariate(near), draws)
    assert new_x.x[0, 0] == conditional(far[0], 0).ppf(0.3)  # far apart: common numbers
    assert new_x.x[1, 0] != conditional(far[1], 0).ppf(0.3)  # close: the maximal coupling
    with pytest.raises(ValueError, match="threshold is required"):
        Gibbs(conditional, [[0], [1]], coupling="two-scale")


def test_identical_conditionals_give_both_chains_the_same_draw():
    # The chains agree outside block 0, so its conditionals are identical: both chains get
    # one draw, then agree outside block 1 too, and meet whatever eta is.
    kernel = Gibbs(conditional, [[0], [1]], eta=0.3)
    x, y = bivariate(np.tile([0.5, -1.0], (50, 1))), bivariate(np.tile([2.0, -1.0], (50, 1)))
    new_x, new_y = kernel.coupled_step(Target(None), x, y, random_draws(50, 37))
    np.testing.assert_array_equal(new_x.x, new_y.x)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"blocks": [[0], [0, 1]]}, "partition"),
        ({"blocks": [[0], [2]]}, "partition"),
        ({"blocks": [[0.0], [1.0]]}, "partition"),
        ({"coupling": "independent"}, "coupling must be one of"),
        ({"threshold": 1.0}, "threshold is required"),
        ({"eta": 0.0}, "eta must be in"),
    ],
)
def test_gibbs_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Gibbs(conditional, **({"blocks": [[0], [1]]} | kwargs))


# Draws per step (spec R-KER-5): exactly what each kernel documents, checked by running
# with that many numbers and failing with one fewer.


@pytest.mark.parametrize(
    ("kernel", "step", "coupled"),
    [
        (MALA(0.5, cov=np.eye(2)), (2, 1), (2, 2)),
        (HMC(0.5, 2, cov=np.eye(2), meeting_weight=0.5), (2, 2), (2, 2)),
    ],
    ids=["mala", "hmc trajectory"],
)
def test_kernels_consume_the_draws_they_document(kernel, step, coupled):
    # The first uniform 0.1 chooses HMC's trajectory component.
    start = state([0.1, -0.2], True)
    for (normals, uniforms), call in ((step, "step"), (coupled, "coupled_step")):
        args = (start,) if call == "step" else (start, state([0.3, 0.4], True))
        exact = fixed_draws([np.full(normals, 0.1)], [np.full(uniforms, 0.1)])
        getattr(kernel, call)(TARGET, *args, exact)
        short = fixed_draws([np.full(normals, 0.1)], [np.full(uniforms - 1, 0.1)])
        with pytest.raises(IndexError, match="exhausted"):
            getattr(kernel, call)(TARGET, *args, short)
