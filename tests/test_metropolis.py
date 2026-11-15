import pickle

import numpy as np
import pytest
from scipy import stats

from references import dense
from telescopic.couplings import Maximal, Reflection, Shared
from telescopic.kernels import (
    HMC,
    MALA,
    Independence,
    IndependenceProposal,
    LangevinProposal,
    MetropolisHastings,
    Mixture,
    MultivariateT,
    RandomWalk,
    RandomWalkProposal,
    State,
)
from telescopic.target import Target
from telescopic.testing import fixed_draws, random_draws


def gaussian(x):
    return -0.5 * np.sum(x * x, axis=-1)


def gaussian_grad(x):
    return -x


def twin_draws(n, d, seed):
    # Two copies of the same draws, as a run reads them: each row's normals and uniforms
    # are two sequences of their own, so a change in the order of either kind changes
    # the numbers, and each row has a generator of its own for the maximal coupling.
    rng = np.random.default_rng(seed)
    normals, uniforms = rng.standard_normal((n, 4 * d + 25)), rng.random((n, 8))
    return [
        fixed_draws(normals, uniforms, [np.random.default_rng([seed, row]) for row in range(n)])
        for _ in range(2)
    ]


def starts(n, d, seed, grad, target):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    # A tenth of the pairs equal, and a tenth close, so that every branch of a coupling
    # runs.
    y = rng.standard_normal((n, d))
    y[: n // 10] = x[: n // 10]
    y[n // 10 : n // 5] = x[n // 10 : n // 5] + 1e-3 * rng.standard_normal((n // 10, d))
    if grad:
        return State(x, *target.log_prob_and_grad(x)), State(y, *target.log_prob_and_grad(y))
    return State(x, target.log_prob(x)), State(y, target.log_prob(y))


def assert_same_states(new, old):
    for a, b in zip(new, old, strict=True):
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.log_prob, b.log_prob)
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            np.testing.assert_array_equal(a.grad, b.grad)


def assert_same_next_numbers(draws, reference):
    # Both read the same count of each kind: their next numbers agree.
    np.testing.assert_array_equal(draws.normal(1), reference.normal(1))
    np.testing.assert_array_equal(draws.uniform(), reference.uniform())
    assert draws.generator(0).random() == reference.generator(0).random()


def bounded_target(d, vectorize):
    # N(0, Sigma) cut at x_0 = -1: a log-density of -inf on part of the space, and a BLAS
    # product in a vectorized call.
    precision = np.linalg.inv(dense(d))

    def log_prob(x):
        return np.where(x[..., 0] > -1.0, -0.5 * np.sum((x @ precision) * x, axis=-1), -np.inf)

    def grad(x):
        return np.where(x[..., :1] > -1.0, -(x @ precision), 0.0)

    return Target(log_prob, grad, vectorize=vectorize)


@pytest.mark.parametrize(("name", "coupling"), [("reflection", Reflection), ("maximal", Maximal)])
def test_random_walk_gives_the_same_bits_with_a_coupling_object_and_its_string(name, coupling):
    d, n = 3, 200
    target = bounded_target(d, False)
    by_name = RandomWalk(dense(d), coupling=name)
    by_object = RandomWalk(dense(d), coupling=coupling())
    state_x, state_y = starts(n, d, 6, False, target)
    draws, reference = twin_draws(n, d, 7)
    for _ in range(3):
        expected = by_name.coupled_step(target, state_x, state_y, reference)
        state_x, state_y = by_object.coupled_step(target, state_x, state_y, draws)
        assert_same_states((state_x, state_y), expected)
    assert_same_next_numbers(draws, reference)
    assert isinstance(by_object.coupling, coupling)
    assert by_name.coupling == name
    assert repr(coupling()) == f"{coupling.__name__}()"


class ShiftedWalk(RandomWalkProposal):
    """A subclass whose draw differs from its parent's: Reflection's coupled step would
    not have its law."""

    def sample(self, target, state, draws):
        x, forward = super().sample(target, state, draws)
        return x + 1.0, forward


@pytest.mark.parametrize(
    ("proposal", "coupling"),
    [
        (LangevinProposal(np.eye(2), np.eye(2)), Maximal()),
        (LangevinProposal(np.eye(2), np.eye(2)), Shared()),
        (RandomWalkProposal(np.eye(2), np.eye(2)), Shared()),
        (IndependenceProposal(MultivariateT(np.zeros(2), np.eye(2))), Reflection()),
        (IndependenceProposal(MultivariateT(np.zeros(2), np.eye(2))), Maximal()),
        (ShiftedWalk(np.eye(2), np.eye(2)), Reflection()),
        (ShiftedWalk(np.eye(2), np.eye(2)), Maximal()),
    ],
)
def test_metropolis_hastings_refuses_a_coupling_for_a_proposal_or_subclass_it_does_not_support(
    proposal, coupling
):
    with pytest.raises(ValueError, match="does not couple"):
        MetropolisHastings(proposal, coupling)


class OwnReflection(Reflection):
    """A subclass, which could couple other laws than the random walk's."""


@pytest.mark.parametrize("coupling", [Shared(), OwnReflection(), "independent"])
def test_random_walk_refuses_a_coupling_other_than_its_two(coupling):
    for cov in (np.eye(2), None):  # untuned too, where no proposal is there to check
        with pytest.raises(ValueError, match="coupling must be"):
            RandomWalk(cov, coupling=coupling)


def test_kernels_with_coupling_objects_survive_pickling():
    d, n = 2, 50
    target = bounded_target(d, False)
    q = MultivariateT(np.zeros(d), 2 * np.eye(d))
    kernels = [
        RandomWalk(np.eye(d), coupling=Maximal()),
        RandomWalk(np.eye(d), coupling=Reflection()),
        MALA(0.5, cov=np.eye(d)),
        Independence(q),
        MetropolisHastings(RandomWalkProposal(np.eye(d), np.eye(d)), Maximal()),
        MetropolisHastings(IndependenceProposal(q), Shared()),
    ]
    for kernel in kernels:
        copy = pickle.loads(pickle.dumps(kernel))
        assert repr(copy) == repr(kernel)
        grad = kernel.needs_grad
        state_x, state_y = starts(n, d, 8, grad, target)
        draws, reference = twin_draws(n, d, 9)
        expected = kernel.coupled_step(target, state_x, state_y, reference)
        assert_same_states(copy.coupled_step(target, state_x, state_y, draws), expected)


@pytest.mark.parametrize("proposal", [RandomWalkProposal, LangevinProposal])
@pytest.mark.parametrize(
    ("factor", "inverse", "error", "message"),
    [
        # An inverse that is not B's inverse would bias every coupled step silently.
        ([[2.0, 0.0], [1.0, 1.0]], np.eye(2), ValueError, "inverse must be the inverse"),
        (np.ones((2, 3)), np.ones((2, 3)), ValueError, "factor must be a square"),
        (np.eye(2), np.eye(3), ValueError, "inverse must have the shape"),
        ([[np.inf, 0.0], [0.0, 1.0]], np.eye(2), ValueError, "must be finite"),
        ("eye", np.eye(2), TypeError, "factor must be real numbers"),
    ],
)
def test_gaussian_proposals_refuse_a_factor_and_inverse_that_do_not_match(
    proposal, factor, inverse, error, message
):
    with pytest.raises(error, match=message):
        proposal(factor, inverse)


@pytest.mark.parametrize(
    "proposal",
    [
        RandomWalkProposal(np.eye(2), np.eye(2)),
        IndependenceProposal(MultivariateT(np.zeros(2), np.eye(2))),
    ],
)
def test_metropolis_hastings_refuses_starts_of_another_dimension_than_its_proposal(proposal):
    coupling = Reflection() if isinstance(proposal, RandomWalkProposal) else Shared()
    kernel = MetropolisHastings(proposal, coupling)
    with pytest.raises(ValueError, match="works in 2 dimensions"):
        kernel.init_state(bounded_target(3, False), np.zeros((4, 3)))


def test_independence_proposal_refuses_a_distribution_without_sample_and_log_density():
    with pytest.raises(TypeError, match="must have sample"):
        IndependenceProposal(stats.multivariate_normal(np.zeros(2)))


def test_a_gaussian_proposal_keeps_its_matrices_when_the_caller_changes_theirs():
    factor = np.asfortranarray([[1.0, 0.0], [0.5, 1.0]])
    inverse = np.linalg.inv(factor)
    proposal = RandomWalkProposal(factor, inverse)
    factor[0, 0] = inverse[1, 0] = 9.0
    np.testing.assert_array_equal(proposal.factor, [[1.0, 0.0], [0.5, 1.0]])
    np.testing.assert_array_equal(proposal.inverse @ proposal.factor, np.eye(2))
    assert proposal.factor.flags.f_contiguous  # the order the caller gave, as einsum reads it


# The cache of log q in Independence.


class Counting:
    """A proposal that counts the rows at which its log-density is evaluated."""

    def __init__(self, q):
        self.q, self.mean, self.rows = q, q.mean, 0

    def sample(self, draws):
        return self.q.sample(draws)

    def log_density(self, x):
        self.rows += len(x)
        return self.q.log_density(x)


def cleared(state):
    return State(state.x, state.log_prob, state.grad)


def assert_same_chains(new, old):
    assert_same_states((new,), (old,))
    for name in old.extra:
        np.testing.assert_array_equal(new.extra[name], old.extra[name])


def test_independence_cache_matches_the_uncached_kernel_bitwise_in_a_mixture_with_reshuffled_lanes():
    # Permuting the rows moves each cache with its chain, so the permutation alone leaves
    # every cache valid; the random walk and HMC rows move chains and carry the old
    # caches, which the guard must recompute.
    d, n = 3, 64
    cov = dense(d)
    target = Target(gaussian, gaussian_grad, vectorize=True)
    proposals = [Counting(MultivariateT(np.zeros(d), 1.5 * cov)) for _ in range(2)]
    cached, reference = (
        Mixture(
            [RandomWalk(cov, scale=1.0), Independence(q), HMC(0.3, 4, cov=cov)], [0.3, 0.4, 0.3]
        )
        for q in proposals
    )
    rng = np.random.default_rng(10)
    pair = [cached.init_state(target, rng.standard_normal((n, d))) for _ in "xy"]
    assert set(pair[0].extra) == {"1/independence.point", "1/independence.log_q"}
    expected = [cleared(state) for state in pair]
    proposals[0].rows = proposals[1].rows = 0
    for t in range(200):
        order = rng.permutation(n)
        pair = [state.take(order) for state in pair]
        expected = [cleared(state.take(order)) for state in expected]
        pair = list(cached.coupled_step(target, *pair, random_draws(n, t)))
        expected = list(reference.coupled_step(target, *expected, random_draws(n, t)))
        for new, old in zip(pair, expected, strict=True):
            assert_same_states((new,), (old,))
        moved, probability = cached.step(target, pair[0], random_draws(n, 1000 + t))
        expected_moved, expected_probability = reference.step(
            target, cleared(expected[0]), random_draws(n, 1000 + t)
        )
        assert_same_states((moved,), (expected_moved,))
        np.testing.assert_array_equal(probability, expected_probability)
        pair[0], expected[0] = moved, expected_moved
    assert 0 < proposals[0].rows < proposals[1].rows


def planted(kernel, target, n, d, wrong):
    # States whose cache holds the log-density of q + 5 at a point that is not the
    # position: moved by 1 in the first coordinate, or 0.0 where the cache holds -0.0.
    rng = np.random.default_rng(11)
    x = rng.standard_normal((n, d))
    if wrong == "signed zero":
        x[:, 0] = 0.0
    state = kernel.init_state(target, x)
    point = state.extra["independence.point"]
    point[:, 0] = -0.0 if wrong == "signed zero" else point[:, 0] + 1.0
    state.extra["independence.log_q"] = state.extra["independence.log_q"] + 5.0
    return state


@pytest.mark.parametrize("wrong", ["moved", "signed zero"])
def test_independence_recomputes_a_cached_value_whose_point_is_not_the_position_bit_for_bit(
    wrong,
):
    d, n = 2, 400
    target = Target(gaussian, vectorize=True)
    q = MultivariateT(np.zeros(d), 1.5 * np.eye(d))
    kernel = Independence(q)
    state_x, state_y = planted(kernel, target, n, d, wrong), planted(kernel, target, n, d, wrong)
    moved, probability = kernel.step(target, state_x, random_draws(n, 1))
    expected, expected_probability = kernel.step(target, cleared(state_x), random_draws(n, 1))
    assert_same_states((moved,), (expected,))
    np.testing.assert_array_equal(probability, expected_probability)
    pair = kernel.coupled_step(target, state_x, state_y, random_draws(n, 2))
    expected_pair = kernel.coupled_step(
        target, cleared(state_x), cleared(state_y), random_draws(n, 2)
    )
    assert_same_states(pair, expected_pair)
    # Every returned cache is right, at a point that is a copy of the position.
    for new in (moved, *pair):
        point = new.extra["independence.point"]
        assert not np.shares_memory(point, new.x)
        np.testing.assert_array_equal(point.view(np.uint64), new.x.view(np.uint64))
        np.testing.assert_array_equal(new.extra["independence.log_q"], q.log_density(new.x))


def test_two_independence_components_of_a_mixture_keep_separate_caches():
    d, n = 2, 64
    target = Target(gaussian, vectorize=True)
    proposals = [MultivariateT(np.zeros(d), np.eye(d)), MultivariateT(np.ones(d), 3 * np.eye(d))]
    kernel = Mixture([Independence(q) for q in proposals], [0.5, 0.5])
    rng = np.random.default_rng(12)
    pair = [kernel.init_state(target, rng.standard_normal((n, d))) for _ in "xy"]
    expected = [cleared(state) for state in pair]
    for t in range(50):
        pair = kernel.coupled_step(target, *pair, random_draws(n, t))
        expected = kernel.coupled_step(target, *map(cleared, expected), random_draws(n, t))
        assert_same_states(pair, expected)
        for state in pair:
            for i, q in enumerate(proposals):
                point = state.extra[f"{i}/independence.point"]
                log_q = state.extra[f"{i}/independence.log_q"]
                np.testing.assert_array_equal(log_q, q.log_density(point))


def test_independence_proposal_cannot_be_reassigned():
    kernel = Independence(MultivariateT(np.zeros(2), np.eye(2)))
    with pytest.raises(AttributeError):
        kernel.proposal = MultivariateT(np.ones(2), np.eye(2))


def test_a_proposals_public_log_alpha_returns_the_ratios_alone():
    # Its private hook carries the states, so that user code calling log_alpha keeps
    # working.
    target = bounded_target(2, False)
    state_x, state_y = starts(20, 2, 13, False, target)
    q = MultivariateT(np.zeros(2), np.eye(2))
    for proposal in (RandomWalkProposal(np.eye(2), np.eye(2)), IndependenceProposal(q)):
        ratio = proposal.log_alpha(target, state_x, state_y, None)
        assert isinstance(ratio, np.ndarray)
        assert ratio.shape == (20,)
